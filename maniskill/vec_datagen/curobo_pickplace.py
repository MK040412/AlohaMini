"""Unified CuRobo MotionGen pick-place for the AlohaMini 2 Pro left arm, single
ManiSkill env, state-only (no render), 8 GB-lean.

Proves the CuRobo path end-to-end:
  * base station via analytic offset (arm-only fixed-base curobo config, so the
    heavy base/vertical mega-hulls never enter the collision model),
  * MotionGen plans the 6 arm joints in the left_Base frame,
  * phases: reach->hover (top-down) -> descend to grasp -> close gripper ->
    lift -> (bonus) carry to target -> place -> open,
  * validates ONE cube grasp+lift (cube_z > 0.70 + 0.02 + 0.06).

Run:
  python vec_datagen/curobo_pickplace.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- curobo / torch-2.11 compat --------------------------------------------
# torch.Size never accepted device/dtype kwargs; curobo.util.sample_lib passes
# them. Shim the module-global torch (localized, non-invasive; the shared curobo
# source is left untouched -- do not commit a patch there).
import curobo.util.sample_lib as _sl  # noqa: E402


class _TorchShim:
    def __getattr__(self, n):
        return getattr(torch, n)

    def Size(self, shape, *a, **k):
        return torch.Size(list(shape))


_sl.torch = _TorchShim()

import gymnasium as gym  # noqa: E402
import sapien  # noqa: E402
from scipy.spatial.transform import Rotation as Rt  # noqa: E402

import vec_datagen.vec_env  # noqa: E402  registers AM2VecPickPlace-v1

from curobo.types.base import TensorDeviceType  # noqa: E402
from curobo.types.math import Pose as CuPose  # noqa: E402
from curobo.types.robot import RobotConfig  # noqa: E402
from curobo.types.state import JointState  # noqa: E402
from curobo.util_file import load_yaml  # noqa: E402
from curobo.geom.types import WorldConfig, Cuboid  # noqa: E402
from curobo.geom.sdf.world import CollisionCheckerType  # noqa: E402
from curobo.wrap.reacher.motion_gen import (  # noqa: E402
    MotionGen, MotionGenConfig, MotionGenPlanConfig,
)


def disable_link_collisions(link):
    """Free the vertical_move lift: the base_link/vertical_link collision hulls
    (the 290k-pt chassis/column mega-shells) wedge the prismatic lift joint so it
    is rigidly pinned at ~0.013 and set_qpos springs back. They are irrelevant to
    a tabletop pick (chassis sits below/behind the arm), so drop their collision
    shapes -> the lift tracks its command and the arm can be raised to a station."""
    n = 0
    for body in link._objs:
        for cs in body.get_collision_shapes():
            g = cs.get_collision_groups()
            cs.set_collision_groups([0, 0, g[2], g[3]])
            n += 1
    return n

CFG_YML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curobo_am2pro_left.yml")


def robot_cfg_dict():
    """CuRobo robot_cfg from the yml, with ~ in urdf_path expanded (portable)."""
    from curobo.util_file import load_yaml
    cfg = load_yaml(CFG_YML)["robot_cfg"]
    kin = cfg["kinematics"]
    kin["urdf_path"] = os.path.expanduser(kin["urdf_path"])
    return cfg

# TCP (Fixed_Jaw) -> grasp point offset, Isaac/sim-validated: grasp center sits
# at local +Z 0.1056, jaws separate along local +X (see curobo_am2pro_left.yml).
TIP_Z = 0.1056
TABLE_TOP_Z = 0.70
CUBE_HALF = 0.02
CUBE_TOP_Z = TABLE_TOP_Z + 2 * CUBE_HALF   # 0.74
CUBE_CENTER_Z = TABLE_TOP_Z + CUBE_HALF    # 0.72

# left_Base world pose at base-station zero (measured): pure translation, identity
# orientation (root_z=0, all intermediate joints rpy 0). Lift adds to z only.
LB0 = np.array([0.18726, -0.05990, 0.57370])

# Active-joint (qpos, 20-dim) indices of the LEFT arm (SAPIEN interleaves L/R).
LEFT_ARM_QIDX = [4, 6, 8, 10, 12, 14]
# 18-dim action layout: [root_x,root_y,root_z, lift, Larm(6), Lgrip, Rarm(6), Rgrip]
ACT_BASE = slice(0, 3)
ACT_LIFT = 3
ACT_LARM = slice(4, 10)
ACT_LGRIP = 10
ACT_RARM = slice(11, 17)
ACT_RGRIP = 17
GRIP_OPEN, GRIP_CLOSED = 0.037, 0.0
RIGHT_FOLD = np.array([0.0, -1.5, 2.5, 0.0, 0.0, 0.0])  # keep right arm tucked

# Desired cube position expressed in the left_Base frame at the station (a solidly
# reachable top-down pocket, from the IK reachability sweep) + lift so the grasp
# lands near base-frame z ~ 0.08.
DES_BX, DES_BY = 0.05, -0.26
# Station lift near the top of travel: raises left_Base to ~0.824 so the grasp
# sits at base-frame z ~ 0 and the whole lift stays inside the arm's reachable
# top-down band (z <~ 0.15 base-frame; higher targets have no IK solution).
LIFT = 0.25
HOVER_DZ = 0.07
LIFT_DZ_CANDIDATES = [0.10, 0.09, 0.08, 0.07]  # cube must clear 0.78 (needs >= 0.06)
TOPDOWN_YAWS = [0.0, 0.7854, -0.7854, 1.5708, -1.5708]


def topdown_quat_world(yaw):
    """World quat (wxyz) for a top-down palm: local +Z -> world -Z (approach down),
    local +X -> horizontal, rotated by yaw about world Z."""
    R = Rt.from_euler("z", yaw).as_matrix() @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    q = Rt.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


class Frame:
    """world <-> left_Base frame transform (from the actual sim link pose)."""

    def __init__(self, p_lb, q_lb_wxyz):
        self.p = np.asarray(p_lb, float)
        self.R = Rt.from_quat([q_lb_wxyz[1], q_lb_wxyz[2], q_lb_wxyz[3], q_lb_wxyz[0]]).as_matrix()

    def pos_to_base(self, p_w):
        return self.R.T @ (np.asarray(p_w, float) - self.p)

    def quat_to_base(self, q_w_wxyz):
        Rw = Rt.from_quat([q_w_wxyz[1], q_w_wxyz[2], q_w_wxyz[3], q_w_wxyz[0]]).as_matrix()
        Rb = self.R.T @ Rw
        q = Rt.from_matrix(Rb).as_quat()
        return np.array([q[3], q[0], q[1], q[2]])


def cu_pose(p_base, q_base_wxyz, tda):
    return CuPose(
        torch.tensor([p_base], device=tda.device, dtype=torch.float32),
        torch.tensor([q_base_wxyz], device=tda.device, dtype=torch.float32),
    )


def main():
    tda = TensorDeviceType()
    # DR disabled for this single-env validation: AM2VecPickPlace randomizes the
    # cube xy but its supporting pedestal is a static actor fixed at the nominal
    # xy, so any DR noise > the 2 cm pedestal half-width drops the cube off the
    # edge (it free-falls). Nominal (noise=0) keeps the cube on the pedestal.
    # (Scaling to DR will require moving the pedestal with the cube.)
    env = gym.make("AM2VecPickPlace-v1", num_envs=1, sim_backend="physx_cuda",
                   obs_mode="state", control_mode="pd_joint_pos_fixed_base",
                   reward_mode="none", render_mode=None,
                   cube_xy_noise=0.0, target_xy_noise=0.0)
    env.reset(seed=0)
    u = env.unwrapped
    ag = u.agent
    act_dim = env.action_space.shape[-1]

    # Free the lift (see disable_link_collisions docstring) so the base can raise
    # left_Base to a station where the cube is a comfortable top-down reach.
    links = {l.name: l for l in ag.robot.get_links()}
    nshapes = disable_link_collisions(links["base_link"]) + disable_link_collisions(links["vertical_link"])
    print(f"[LIFT-FIX] dropped {nshapes} chassis/column collision shapes to unjam vertical_move")

    cube = u.cube.pose.p[0].cpu().numpy()
    target = u.target.pose.p[0].cpu().numpy()
    print(f"[SCENE] cube={np.round(cube,4).tolist()} target={np.round(target,4).tolist()} act_dim={act_dim}")

    # ---- base station so the cube lands at the reachable base-frame pocket ----
    bx = float(cube[0] - LB0[0] - DES_BX)
    by = float(cube[1] - LB0[1] + (-DES_BY))
    station = np.array([bx, by, 0.0])
    print(f"[STATION] root_x={bx:.3f} root_y={by:.3f} lift={LIFT:.3f}")

    def make_action(arm6, grip):
        a = np.zeros(act_dim, dtype=np.float32)
        a[ACT_BASE] = station
        a[ACT_LIFT] = LIFT
        a[ACT_LARM] = arm6
        a[ACT_LGRIP] = grip
        a[ACT_RARM] = RIGHT_FOLD
        a[ACT_RGRIP] = GRIP_CLOSED
        return torch.tensor(a, device=u.device).unsqueeze(0)

    def cur_arm():
        q = ag.robot.get_qpos()[0].cpu().numpy()
        return q[LEFT_ARM_QIDX]

    def step_hold(arm6, grip, n):
        a = make_action(arm6, grip)
        for _ in range(n):
            env.step(a)

    # drive the base to station (+ open gripper) while holding the rest fold
    rest_arm = cur_arm()
    for _ in range(40):
        env.step(make_action(rest_arm, GRIP_OPEN))
    lb = ag.robot.links_map["left_Base"].pose
    fr = Frame(lb.p[0].cpu().numpy(), lb.q[0].cpu().numpy())
    print(f"[STATION] left_Base world after settle = {np.round(fr.p,4).tolist()}")

    # ---- build the collision world in the left_Base frame ----
    ped_c_w = np.array([cube[0], cube[1], 0.345])        # pedestal, top at 0.69
    ped_c_b = fr.pos_to_base(ped_c_w)
    floor_c_b = fr.pos_to_base(np.array([cube[0], cube[1], -0.05]))
    world = WorldConfig(cuboid=[
        Cuboid(name="pedestal", pose=[*ped_c_b.tolist(), *fr.quat_to_base([1, 0, 0, 0]).tolist()],
               dims=[0.05, 0.05, 0.69]),
        Cuboid(name="floor", pose=[*floor_c_b.tolist(), 1, 0, 0, 0], dims=[3.0, 3.0, 0.1]),
    ])

    # ---- MotionGen (lean: small seed counts, no cuda graph) ----
    robot_cfg = RobotConfig.from_dict(robot_cfg_dict(), tda)
    t0 = time.time()
    mg = MotionGen(MotionGenConfig.load_from_robot_config(
        robot_cfg, world, tensor_args=tda,
        num_ik_seeds=30, num_trajopt_seeds=6, interpolation_dt=0.02,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,  # cuboid world; avoids warp mesh path
        use_cuda_graph=False,
    ))
    mg.warmup(warmup_js_trajopt=False)
    print(f"[MOTIONGEN] init+warmup {time.time()-t0:.1f}s  VRAM={int(torch.cuda.max_memory_allocated()/1e6)}MB")

    jn = robot_cfg.kinematics.cspace.joint_names
    plan_cfg = MotionGenPlanConfig(max_attempts=6, enable_graph=False, timeout=15.0)

    def start_state():
        return JointState.from_position(
            torch.tensor(cur_arm()[None], device=tda.device, dtype=torch.float32), joint_names=jn)

    def plan_to(world_p, world_q):
        pb = fr.pos_to_base(world_p)
        qb = fr.quat_to_base(world_q)
        res = mg.plan_single(start_state(), cu_pose(pb, qb, tda), plan_cfg.clone())
        return res

    def execute(res, grip):
        traj = res.get_interpolated_plan()
        pos = traj.position.cpu().numpy()
        for wp in pos:
            env.step(make_action(wp, grip))
        step_hold(pos[-1], grip, 8)  # settle at the final waypoint

    def tcp_err(world_p):
        fj = ag.robot.links_map["left_Fixed_Jaw"].pose
        p = fj.p[0].cpu().numpy()
        q = fj.q[0].cpu().numpy()
        R = Rt.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        grasp = p + R @ np.array([0, 0, TIP_Z])
        return float(np.linalg.norm(grasp - world_p))

    ok = {}
    # ============================== PHASE A: hover ==============================
    grasp_w = np.array([cube[0], cube[1], CUBE_CENTER_Z])          # cube center
    fj_grasp_w = np.array([cube[0], cube[1], CUBE_CENTER_Z + TIP_Z])
    hover_w = fj_grasp_w + np.array([0, 0, HOVER_DZ])
    chosen_yaw, resA = None, None
    for yaw in TOPDOWN_YAWS:
        qy = topdown_quat_world(yaw)
        r = plan_to(hover_w, qy)
        if bool(r.success.item()):
            chosen_yaw, resA = yaw, r
            break
    if resA is None or not bool(resA.success.item()):
        print("[PHASE A] hover PLAN FAILED for all yaws -> abort")
        env.close(); return
    print(f"[PHASE A] hover plan OK  yaw={chosen_yaw:.3f}  "
          f"traj_len={resA.get_interpolated_plan().position.shape[0]}")
    execute(resA, GRIP_OPEN)
    ok["A_hover"] = tcp_err(hover_w)
    print(f"[PHASE A] executed; grasp-point err to hover target = {ok['A_hover']*1000:.1f} mm")

    qy = topdown_quat_world(chosen_yaw)
    # ============================== PHASE B: descend ===========================
    resB = plan_to(fj_grasp_w, qy)
    print(f"[PHASE B] descend plan success={bool(resB.success.item())}")
    if bool(resB.success.item()):
        execute(resB, GRIP_OPEN)
    else:
        # fall back: linearly step the arm down via repeated IK-free hold is not
        # possible; instead re-plan grasp directly from current state already done.
        pass
    ok["B_grasp_reach"] = tcp_err(grasp_w)
    print(f"[PHASE B] grasp-point err to cube center = {ok['B_grasp_reach']*1000:.1f} mm")

    # ============================== PHASE C: close =============================
    hold = cur_arm()
    for i in range(25):
        g = GRIP_OPEN + (GRIP_CLOSED - GRIP_OPEN) * (i + 1) / 25.0
        env.step(make_action(hold, g))
    step_hold(hold, GRIP_CLOSED, 10)
    grasped = bool(ag.is_grasping(u.cube, arm_id=1).item())
    ok["C_grasp"] = grasped
    print(f"[PHASE C] gripper closed; is_grasping={grasped}")

    # ============================== PHASE D: lift ==============================
    lift_dz = None
    for dz in LIFT_DZ_CANDIDATES:
        resD = plan_to(fj_grasp_w + np.array([0, 0, dz]), qy)
        if bool(resD.success.item()):
            lift_dz = dz
            break
    print(f"[PHASE D] lift plan success={lift_dz is not None} (dz={lift_dz})")
    if lift_dz is not None:
        execute(resD, GRIP_CLOSED)
    else:
        print("[PHASE D] lift plan failed for all dz; holding")
    cube_z = float(u.cube.pose.p[0, 2].item())
    lifted = cube_z > (CUBE_CENTER_Z + 0.06)
    ok["D_cube_z"] = cube_z
    ok["D_lifted"] = lifted
    print(f"[PHASE D] cube_z={cube_z:.4f}  LIFTED={lifted}  (threshold {CUBE_CENTER_Z+0.06:.3f})")

    # ===================== PHASE E-G (bonus): carry + place ====================
    if lifted:
        carry_w = np.array([target[0], target[1], CUBE_CENTER_Z + TIP_Z + (lift_dz or 0.10)])
        resE = plan_to(carry_w, qy)
        print(f"[PHASE E] carry plan success={bool(resE.success.item())}")
        if bool(resE.success.item()):
            execute(resE, GRIP_CLOSED)
            place_w = np.array([target[0], target[1], CUBE_CENTER_Z + TIP_Z + 0.02])
            resF = plan_to(place_w, qy)
            print(f"[PHASE F] place plan success={bool(resF.success.item())}")
            if bool(resF.success.item()):
                execute(resF, GRIP_CLOSED)
            step_hold(cur_arm(), GRIP_OPEN, 20)  # release
            cp = u.cube.pose.p[0].cpu().numpy()
            placed = float(np.linalg.norm(cp[:2] - target[:2]))
            ok["G_place_xy_err"] = placed
            print(f"[PHASE G] released; cube-target xy err = {placed*1000:.1f} mm")

    print("\n===== SUMMARY =====")
    for k, v in ok.items():
        print(f"  {k}: {v}")
    print(f"  PICK+LIFT SUCCESS: {ok.get('D_lifted', False)}")
    env.close()


if __name__ == "__main__":
    main()
