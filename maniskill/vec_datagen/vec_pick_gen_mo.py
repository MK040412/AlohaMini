"""GPU-BATCHED *MULTI-OBJECT* pick data generator (physx_cuda, N envs at once).

Each env has the SAME 3 coloured cubes (fixed at scene load -> physx_cuda stable, no
reconfigure) at per-env random positions; per env we pick a RANDOM target colour and
CuRobo-plan a top-down grasp of THAT cube (the other two are distractors). This grounds
language -> the correct object, exactly like the CPU run_instruction_episodes gen but
N-at-a-time on the GPU (the max-parallel engine the user asked for -- no CPU workers).

Human-like COORD: settle at the station (unrecorded) to get an accurate plan frame,
reset the chassis to the ORIGIN, then RECORD the coordinated approach where the base
ramps origin->station WHILE the arm follows its hover plan (base+arm move together, so
the policy can learn nav+manip -- the fix for the months-long 0/6). Grasp-dwell (ramp
close + hold) teaches STAY+close. Successful (grasp+lift the TARGET) episodes are saved
in the instr_out multi-object format so the existing 5-cam render/train/eval reuses them.

Usage: <basicrl_python> vec_pick_gen_mo.py [n_batches] [N_envs] [ep_base] [c0 c1 c2]
Env: INSTR_OUT_DIR override, PLAN_CAP (default 160).
"""
import os, sys, time, numpy as np, torch, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agents.aloha_mini, vec_datagen.vec_env  # noqa
import vec_datagen.curobo_pickplace as cpp
from vec_datagen.curobo_pickplace import (CFG_YML, TIP_Z, HOVER_DZ, LIFT, LB0, DES_BX, DES_BY,
                                          Frame, topdown_quat_world)
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.types.math import Pose as CuPose
from curobo.util_file import load_yaml
from curobo.geom.types import WorldConfig, Cuboid
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

TABLE_TOP_Z = 0.70
LEFT_ARM_QIDX = [4, 6, 8, 10, 12, 14]
ARM_LO = np.array([-1.92, -3.32, -0.17, -1.66, -2.79, -2.84]); ARM_HI = np.array([1.92, 0.17, 3.14, 1.66, 2.79, 2.84])
GRIP_OPEN, GRIP_CLOSED = 0.037, 0.0; RIGHT_FOLD = np.array([0.0, -1.5, 2.5, 0.0, 0.0, 0.0])
# station_for centres EVERY env's target at the same base-frame spot [DES_BX,DES_BY], so
# the goal pose is identical across envs -> a single wrist yaw that plans works for all.
# yaw=0 is IK-unreachable at this config (verified); 0.7854 is the first TOPDOWN_YAWS hit.
GRASP_YAW = float(os.environ.get("GRASP_YAW", "0.7854"))
OUT_DIR = os.environ.get("INSTR_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "instr_out"))
dev = "cuda"


def station_for(p_w):
    return np.array([p_w[0] - LB0[0] - DES_BX, p_w[1] - LB0[1] - DES_BY, 0.0], np.float32)


def main(n_batches=7, N=16, ep_base=500000, colors=("red", "green", "blue"), cube_half=0.02):
    os.makedirs(OUT_DIR, exist_ok=True)
    colors = list(colors); K = len(colors)
    rng = np.random.default_rng(2026 + ep_base)
    tda = TensorDeviceType()
    robot_cfg = RobotConfig.from_dict(load_yaml(CFG_YML)["robot_cfg"], tda)
    # floor far below only (same as vec_pick_gen): a top-down grasp never hits the table,
    # and a mis-placed table cuboid rejects EVERY plan as in-collision.
    world0 = WorldConfig(cuboid=[Cuboid(name="floor", pose=[0, 0, -1.2, 1, 0, 0, 0], dims=[4.0, 4.0, 0.1])])
    t0 = time.time()
    mg = MotionGen(MotionGenConfig.load_from_robot_config(
        robot_cfg, world0, tensor_args=tda, num_ik_seeds=30, num_trajopt_seeds=6, interpolation_dt=0.02,
        collision_checker_type=CollisionCheckerType.PRIMITIVE, use_cuda_graph=False))
    mg.warmup(enable_graph=False, batch=N, warmup_js_trajopt=False)
    jn = robot_cfg.kinematics.cspace.joint_names
    PC = MotionGenPlanConfig(max_attempts=6, enable_graph=False, timeout=15.0)
    print(f"[MOGEN] warmup(batch={N}) {time.time()-t0:.1f}s", flush=True)

    cz = TABLE_TOP_Z + cube_half                                # cube-top height for the grasp target
    env = gym.make("AM2MultiObject-v1", num_envs=N, sim_backend="physx_cuda", obs_mode="state",
                   control_mode="pd_joint_pos_fixed_base", render_mode=None, reward_mode="none",
                   obj_colors=tuple(colors), cube_half=cube_half)
    u = env.unwrapped
    for ln in ("base_link", "vertical_link", "Link2_dp", "Link3_dp", "Link4_dp"):
        L = {l.name: l for l in u.agent.robot.get_links()}.get(ln)
        if L is not None: cpp.disable_link_collisions(L)
    adim = env.action_space.shape[-1]

    def act(arm6, grip, base3):                                 # arm6:(N,6) grip:(N,) or scalar base3:(N,3)
        a = torch.zeros(N, adim, device=dev)
        a[:, 0:3] = torch.as_tensor(base3, dtype=torch.float32, device=dev)
        a[:, 3] = LIFT
        a[:, 4:10] = torch.as_tensor(arm6, dtype=torch.float32, device=dev)
        a[:, 10] = torch.as_tensor(grip, dtype=torch.float32, device=dev) if np.ndim(grip) else grip
        a[:, 11:17] = torch.as_tensor(RIGHT_FOLD, dtype=torch.float32, device=dev)
        a[:, 17] = GRIP_CLOSED
        return a

    saved = 0; wall0 = time.time()
    for bi in range(n_batches):
        seed = ep_base + bi * 1000
        env.reset(seed=seed)
        cubes_w = np.stack([c.pose.p.cpu().numpy() for c in u.cubes], axis=1)   # (N, K, 3)
        tgt_idx = rng.integers(0, K, N)                                          # per-env target cube
        target_w = cubes_w[np.arange(N), tgt_idx]                                # (N, 3)
        station = np.stack([station_for(target_w[j]) for j in range(N)]).astype(np.float32)  # (N,3)
        # DIVERSITY (env-gated, default off): STATION_NOISE puts the cube OFF-centre in the
        # base frame so the arm reaches varied spots (not one canonical config); YAW_RAND
        # picks a per-env grasp yaw from the verified-reachable set.
        sn = float(os.environ.get("STATION_NOISE", "0"))
        if sn > 0: station[:, 0:2] += rng.normal(0, sn, (N, 2)).astype(np.float32)
        if os.environ.get("YAW_RAND", "0") == "1":
            yaws = rng.choice(np.array([0.7854, -0.7854, 1.5708, -1.5708], np.float32), N)
        else:
            yaws = np.full(N, GRASP_YAW, np.float32)
        origin = np.zeros((N, 3), np.float32)
        rest_c = np.clip(u.agent.robot.get_qpos().cpu().numpy()[:, LEFT_ARM_QIDX], ARM_LO + 0.02, ARM_HI - 0.02)
        # PROPRIO DR (env-gated): randomize the per-env ARM start config so the recorded
        # initial proprioception varies (robustness / less over-fit to one canonical start).
        # The 45-step settle below drives the arm to this perturbed start, so the recorded
        # locate frames + hover plan begin from it. Base start stays origin (eval resets base
        # to origin -> no train/eval mismatch).
        ain = float(os.environ.get("ARM_INIT_NOISE", "0"))
        if ain > 0:
            rest_c = np.clip(rest_c + rng.normal(0, ain, rest_c.shape).astype(np.float32), ARM_LO + 0.02, ARM_HI - 0.02)

        rec_q, rec_a, rec_ph, rec_o = [], [], [], []

        def step_rec(arm6, grip, base3, phase):
            env.step(act(arm6, grip, base3))
            rec_q.append(u.agent.robot.get_qpos().cpu().numpy().astype(np.float32))            # (N,dof)
            A = np.zeros((N, 18), np.float32)
            A[:, 0:3] = base3; A[:, 3] = LIFT; A[:, 4:10] = arm6
            A[:, 10] = grip if np.ndim(grip) else grip; A[:, 11:17] = RIGHT_FOLD; A[:, 17] = GRIP_CLOSED
            rec_a.append(A); rec_ph.append(phase)
            rec_o.append(np.stack([c.pose.p.cpu().numpy() for c in u.cubes], axis=1).astype(np.float32))  # (N,K,3) per step

        # settle at station (UNRECORDED) to get an accurate per-env left_Base plan frame
        for _ in range(45): env.step(act(rest_c, GRIP_OPEN, station))
        lbp = u.agent.robot.links_map["left_Base"].pose
        frames = [Frame(lbp.p[j].cpu().numpy(), lbp.q[j].cpu().numpy()) for j in range(N)]

        def goal_at(dz):
            P = np.zeros((N, 3), np.float32); Q = np.zeros((N, 4), np.float32)
            for j in range(N):
                P[j] = frames[j].pos_to_base(np.array([target_w[j, 0], target_w[j, 1], cz + TIP_Z + dz]))
                Q[j] = frames[j].quat_to_base(topdown_quat_world(float(yaws[j])))
            return CuPose(torch.tensor(P, device=dev).contiguous(), torch.tensor(Q, device=dev).contiguous())

        def plan_batch(start_arm, dz):
            hold = np.ascontiguousarray(np.clip(start_arm, ARM_LO + .02, ARM_HI - .02), np.float32)
            s = JointState.from_position(torch.tensor(hold, dtype=torch.float32, device=dev).contiguous(), joint_names=jn)
            res = mg.plan_batch(s, goal_at(dz), PC.clone())
            ok = res.success.view(-1).cpu().numpy().astype(bool)
            if res.interpolated_plan is None:
                return ok, np.tile(hold[:, None], (1, 8, 1))
            plan = res.interpolated_plan.position.cpu().numpy()
            if plan.ndim == 2: plan = np.tile(plan[None], (N, 1, 1))
            H = min(plan.shape[1], int(os.environ.get("PLAN_CAP", "160")))
            plan = plan[:, :H]
            okm = ok[:, None]
            for t in range(plan.shape[1]):                      # hold the un-planned envs at their start pose
                plan[:, t] = np.where(okm, plan[:, t], hold)
            return ok, plan

        # PHASE A: plan hover (over target) from rest, at the station frame (captured above)
        okH, planH = plan_batch(rest_c, HOVER_DZ)
        # DRIVE the chassis back to ORIGIN (unrecorded) so the RECORDED episode starts at
        # base=0 -- matching the eval reset. (set_qpos does NOT hold under physx_cuda: it
        # moves position but not the pd drive target, so the base drifts back.)
        for _ in range(45): env.step(act(rest_c, GRIP_OPEN, origin))
        for _ in range(5): step_rec(rest_c, GRIP_OPEN, origin, "locate")
        TH = planH.shape[1]
        for t in range(TH):
            base_t = origin + (station - origin) * (t + 1) / TH
            step_rec(planH[:, t], GRIP_OPEN, base_t, "approach")
        for _ in range(6): step_rec(planH[:, -1], GRIP_OPEN, station, "approach")

        # PHASE B: descend to the grasp (base held at station)
        armH = u.agent.robot.get_qpos().cpu().numpy()[:, LEFT_ARM_QIDX]
        okD, planD = plan_batch(armH, 0.0)
        for t in range(planD.shape[1]): step_rec(planD[:, t], GRIP_OPEN, station, "descend")
        desct = planD[:, -1]
        for _ in range(6): step_rec(desct, GRIP_OPEN, station, "descend")

        # grasp-dwell: ramp-close then hold (teaches STAY + close)
        for i in range(30): step_rec(desct, GRIP_OPEN + (GRIP_CLOSED - GRIP_OPEN) * (i + 1) / 30.0, station, "grasp")
        for _ in range(14): step_rec(desct, GRIP_CLOSED, station, "grasp")
        # grasp-DART (env-gated, default off): knock the arm OFF the grasp (unrecorded) ->
        # CuRobo re-plans back -> RECORD the re-align + re-close. Teaches 'off-course ->
        # servo back to the grasp AND close' (the covariate-shift fix eval needs). Adds arm-
        # approach diversity around the grasp that a single canonical descent lacks.
        for _ in range(int(os.environ.get("GRASP_DART_N", "0"))):
            cur = u.agent.robot.get_qpos().cpu().numpy()[:, LEFT_ARM_QIDX]
            pert = np.clip(cur + rng.normal(0, 0.06, (N, 6)), ARM_LO + 0.02, ARM_HI - 0.02)
            for _ in range(6): env.step(act(pert, GRIP_OPEN, station))            # knock off + reopen (unrecorded)
            _, planr = plan_batch(u.agent.robot.get_qpos().cpu().numpy()[:, LEFT_ARM_QIDX], 0.0)
            for t in range(planr.shape[1]): step_rec(planr[:, t], GRIP_OPEN, station, "descend")   # re-align (recorded)
            desct = planr[:, -1]
            for i in range(30): step_rec(desct, GRIP_OPEN + (GRIP_CLOSED - GRIP_OPEN) * (i + 1) / 30.0, station, "grasp")
            for _ in range(14): step_rec(desct, GRIP_CLOSED, station, "grasp")   # re-close (recorded)
        grasped_c = [u.agent.is_grasping(c, arm_id=1).cpu().numpy().astype(bool) for c in u.cubes]

        # PHASE C: lift (base held at station)
        okL, planL = plan_batch(desct, 0.12)
        for t in range(planL.shape[1]): step_rec(planL[:, t], GRIP_CLOSED, station, "lift")
        for _ in range(6): step_rec(planL[:, -1], GRIP_CLOSED, station, "lift")

        # per-env success on the CHOSEN cube
        czf = np.stack([c.pose.p[:, 2].cpu().numpy() for c in u.cubes], axis=1)   # (N,K)
        grasped = np.array([grasped_c[tgt_idx[j]][j] for j in range(N)])
        cubez = np.array([czf[j, tgt_idx[j]] for j in range(N)])
        success = grasped & (cubez > cz + 0.05)

        Q = np.stack(rec_q, 1)     # (N, T, dof)
        A = np.stack(rec_a, 1)     # (N, T, 18)
        O = np.stack(rec_o, 1)     # (N, T, K, 3) per-step cube positions
        PH = np.array(rec_ph)
        for j in np.where(success)[0]:
            gid = ep_base + bi * N + int(j)
            tp = u.target.pose.p[j].cpu().numpy()
            instr = f"pick up the {colors[tgt_idx[j]]} cube and place it on the marker"
            np.savez_compressed(os.path.join(OUT_DIR, f"ep_{gid}.npz"),
                                qpos=Q[j], action=A[j], phase=PH,
                                obj_positions=O[j],
                                target_pos=np.tile(tp[None], (Q.shape[1], 1)).astype(np.float32),
                                colors=np.array(colors), target_color=colors[tgt_idx[j]],
                                target_idx=np.int64(tgt_idx[j]), instruction=instr,
                                seed=int(gid), cube_half=float(cube_half))
            saved += 1
        rate = saved / (time.time() - wall0) * 3600
        print(f"[MOGEN] batch {bi+1}/{n_batches} {colors}: grasp+lift {int(success.sum())}/{N} "
              f"(H={int(okH.sum())} D={int(okD.sum())} L={int(okL.sum())}) | saved={saved} ({rate:.0f} ep/h)", flush=True)
    print(f"[MOGEN] DONE {saved} episodes in {(time.time()-wall0)/60:.1f}min -> {OUT_DIR}", flush=True)
    env.close()


if __name__ == "__main__":
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    eb = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
    cols = tuple(sys.argv[4:]) if len(sys.argv) > 4 else ("red", "green", "blue")   # 3 or 4 cubes
    main(nb, N, eb, cols)
