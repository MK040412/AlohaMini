"""GPU-BATCHED pick data generator (physx_cuda, N envs at once) — the max-parallel
data engine (task #51). Validated core: fixed station + empty world + CuRobo plan_batch
(top-down hover -> descend -> close -> lift), ~50% grasp+lift/batch at yaw=0. Records
per-step qpos/action for every env, saves the SUCCESSFUL (grasp+lift) ones as npz in the
instr_out format (single cube, k=1) so the existing render/cache/train pipeline reuses them.

Speed: one N-env batch plans 4 phases (~6s CuRobo) + steps the sim (physx_gpu) -> ~N/2
successful episodes per ~60-90s batch. Scales with N.

Usage: <basicrl_python> vec_pick_gen.py [n_batches] [N_envs] [ep_base]
Env: OUT_DIR override via INSTR_OUT_DIR.
"""
import os, sys, time, numpy as np, torch, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agents.aloha_mini, vec_datagen.vec_env  # noqa
import vec_datagen.curobo_pickplace as cpp
from vec_datagen.curobo_pickplace import (CFG_YML, TIP_Z, HOVER_DZ, LIFT, LB0, DES_BX, DES_BY,
                                          TOPDOWN_YAWS, Frame, topdown_quat_world)
from vec_datagen.vec_env import NAMED_COLORS
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.types.math import Pose as CuPose
from curobo.util_file import load_yaml
from curobo.geom.types import WorldConfig, Cuboid
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

TABLE_TOP_Z, CUBE_HALF = 0.70, 0.02
LEFT_ARM_QIDX = [4, 6, 8, 10, 12, 14]
ARM_LO = np.array([-1.92, -3.32, -0.17, -1.66, -2.79, -2.84]); ARM_HI = np.array([1.92, 0.17, 3.14, 1.66, 2.79, 2.84])
GRIP_OPEN, GRIP_CLOSED = 0.037, 0.0; RIGHT_FOLD = np.array([0.0, -1.5, 2.5, 0.0, 0.0, 0.0])
DR_CENTER = np.array([-0.13, -0.45]); YAWS = [0.0, 0.7854, -0.7854]     # try a few yaws to lift the ~50% yield
OUT_DIR = os.environ.get("INSTR_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "instr_out"))
COLOR_NAMES = list(NAMED_COLORS.keys())
dev = "cuda"


def main(n_batches=40, N=64, ep_base=20000, color="red"):
    os.makedirs(OUT_DIR, exist_ok=True)
    tda = TensorDeviceType()
    robot_cfg = RobotConfig.from_dict(load_yaml(CFG_YML)["robot_cfg"], tda)
    world0 = WorldConfig(cuboid=[Cuboid(name="floor", pose=[0, 0, -1.2, 1, 0, 0, 0], dims=[4.0, 4.0, 0.1])])
    t0 = time.time()
    mg = MotionGen(MotionGenConfig.load_from_robot_config(
        robot_cfg, world0, tensor_args=tda, num_ik_seeds=30, num_trajopt_seeds=6, interpolation_dt=0.02,
        collision_checker_type=CollisionCheckerType.PRIMITIVE, use_cuda_graph=False))
    mg.warmup(enable_graph=False, batch=N, warmup_js_trajopt=False)
    jn = robot_cfg.kinematics.cspace.joint_names
    PC = MotionGenPlanConfig(max_attempts=6, enable_graph=False, timeout=15.0)
    print(f"[VECGEN] MotionGen warmup(batch={N}) {time.time()-t0:.1f}s", flush=True)

    st = np.array([DR_CENTER[0] - LB0[0] - DES_BX, DR_CENTER[1] - LB0[1] - DES_BY, 0.0], np.float32)
    station = np.tile(st, (N, 1))
    env = gym.make("AM2VecPickPlace-v1", num_envs=N, sim_backend="physx_cuda", obs_mode="state",
                   control_mode="pd_joint_pos_fixed_base", render_mode=None, cube_xy_noise=0.06,
                   cube_color=list(NAMED_COLORS[color]))       # colour FIXED at scene load (no mid-run reconfigure -> physx_cuda stable)
    u = env.unwrapped
    for ln in ("base_link", "vertical_link", "Link2_dp", "Link3_dp", "Link4_dp"):
        L = {l.name: l for l in u.agent.robot.get_links()}.get(ln)
        if L is not None: cpp.disable_link_collisions(L)

    def act(arm6, grip):
        a = torch.zeros(N, 18, device=dev)
        a[:, 0:3] = torch.tensor(station, device=dev); a[:, 3] = LIFT
        a[:, 4:10] = arm6 if torch.is_tensor(arm6) else torch.tensor(arm6, device=dev)
        a[:, 10] = torch.tensor(grip, device=dev) if np.ndim(grip) else grip
        a[:, 11:17] = torch.tensor(RIGHT_FOLD, device=dev); a[:, 17] = GRIP_CLOSED
        return a

    def act_np(arm6, grip):
        g = grip if np.ndim(grip) else np.full(N, grip, np.float32)
        A = np.zeros((N, 18), np.float32)
        A[:, 0:3] = station; A[:, 3] = LIFT; A[:, 4:10] = arm6; A[:, 10] = g
        A[:, 11:17] = RIGHT_FOLD; A[:, 17] = GRIP_CLOSED
        return A

    saved = 0; wall0 = time.time()
    for bi in range(n_batches):
        seed = ep_base + bi * 1000
        cname = color                                    # fixed colour per run (no per-batch reconfigure)
        env.reset(seed=seed)                             # new cube positions only -> physx_cuda stable
        cubes_w = u.cube.pose.p.cpu().numpy()
        rest_arm = u.agent.robot.get_qpos()[0].cpu().numpy()[LEFT_ARM_QIDX]
        rest_c = np.clip(rest_arm, ARM_LO + 0.02, ARM_HI - 0.02)
        rec_q, rec_a, rec_ph = [], [], []      # per-step (N,dof)/(N,18)/phase

        def step_rec(arm6, grip, phase):
            env.step(act(arm6, grip))
            rec_q.append(u.agent.robot.get_qpos().cpu().numpy().astype(np.float32))
            rec_a.append(act_np(arm6, grip)); rec_ph.append(phase)

        # settle at station (locate)
        for _ in range(45): step_rec(rest_c, GRIP_OPEN, "locate")
        lbp = u.agent.robot.links_map["left_Base"].pose
        frames = [Frame(lbp.p[j].cpu().numpy(), lbp.q[j].cpu().numpy()) for j in range(N)]
        cz = TABLE_TOP_Z + CUBE_HALF

        def goal_at(dz):
            P = np.zeros((N, 3), np.float32); Q = np.zeros((N, 4), np.float32)
            for j in range(N):
                P[j] = frames[j].pos_to_base(np.array([cubes_w[j, 0], cubes_w[j, 1], cz + TIP_Z + dz]))
                Q[j] = frames[j].quat_to_base(topdown_quat_world(0.0))
            return CuPose(torch.tensor(P, device=dev), torch.tensor(Q, device=dev))

        def plan_exec(start_arm, dz, grip, phase):
            hold = np.clip(start_arm, ARM_LO + .02, ARM_HI - .02)
            s = JointState.from_position(torch.tensor(hold, dtype=torch.float32, device=dev), joint_names=jn)
            res = mg.plan_batch(s, goal_at(dz), PC.clone())
            ok = res.success.view(-1).cpu().numpy().astype(bool)
            if res.interpolated_plan is None: return ok, hold
            plan = res.interpolated_plan.position.cpu().numpy()
            if plan.ndim == 2: plan = np.tile(plan[None], (N, 1, 1))
            # CuRobo interpolates into a large FIXED buffer padded with the goal pose; the
            # real motion is in the first ~100-150 steps. Cap so episodes aren't 15k steps.
            H = min(plan.shape[1], int(os.environ.get("PLAN_CAP", "160")))
            plan = plan[:, :H]
            okm = ok[:, None]; final = hold.copy()
            for t in range(plan.shape[1]):
                final = np.where(okm, plan[:, t], hold)
                step_rec(final, grip, phase)
            for _ in range(8): step_rec(final, grip, phase)          # settle at target
            return ok, final

        okH, _ = plan_exec(np.tile(rest_c, (N, 1)), HOVER_DZ, GRIP_OPEN, "approach")
        armH = u.agent.robot.get_qpos().cpu().numpy()[:, LEFT_ARM_QIDX]
        okD, desct = plan_exec(armH, 0.0, GRIP_OPEN, "descend")
        for i in range(30): step_rec(desct, GRIP_OPEN + (GRIP_CLOSED - GRIP_OPEN) * (i + 1) / 30.0, "grasp")
        for _ in range(12): step_rec(desct, GRIP_CLOSED, "grasp")
        grasped = u.agent.is_grasping(u.cube, arm_id=1).cpu().numpy().astype(bool)
        okL, _ = plan_exec(desct, 0.12, GRIP_CLOSED, "lift")
        czf = u.cube.pose.p[:, 2].cpu().numpy()
        success = grasped & (czf > TABLE_TOP_Z + CUBE_HALF + 0.05)

        Q = np.stack(rec_q, 1)                 # (N, T, dof)
        A = np.stack(rec_a, 1)                 # (N, T, 18)
        instr = f"pick up the {cname} cube"
        for j in np.where(success)[0]:
            gid = ep_base + bi * N + int(j)
            np.savez(os.path.join(OUT_DIR, f"ep_{gid}.npz"),
                     qpos=Q[j], action=A[j], phase=np.array(rec_ph),
                     obj_positions=np.tile(cubes_w[j][None, None], (Q.shape[1], 1, 1)).astype(np.float32),
                     target_pos=np.tile(u.target.pose.p[j].cpu().numpy()[None], (Q.shape[1], 1)).astype(np.float32),
                     colors=np.array([cname]), target_color=cname, target_idx=0, instruction=instr,
                     seed=gid, cube_half=CUBE_HALF)
            saved += 1
        rate = saved / (time.time() - wall0) * 3600
        print(f"[VECGEN] batch {bi+1}/{n_batches} '{cname}': grasp+lift {int(success.sum())}/{N} | saved={saved} ({rate:.0f} ep/h)", flush=True)
    print(f"[VECGEN] DONE {saved} episodes in {(time.time()-wall0)/60:.1f}min -> {OUT_DIR}", flush=True)
    env.close()


if __name__ == "__main__":
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    eb = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    col = sys.argv[4] if len(sys.argv) > 4 else "red"
    main(nb, N, eb, col)
