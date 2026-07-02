"""NAV/MANIP-separated pick-and-place for the Pro robot (ASPIRE-style).

The user's insight (== ASPIRE's "Multi-Angle Approach" skill, Fig.2): instead of
contorting the arm from a fixed base (link overlaps, awkward IK branches), MOVE THE
BASE to a *manipulation station* where the arm's desired action is feasible, and only
then manipulate. Phases:

  [FEASIBILITY] sample candidate base stations around the target; teleport-check each
                (set base qpos, solve tilted full-pose IK, restore) and pick the best
  [NAV-1]  drive base (x, y) to the pick station    — arm tucked at rest, above table
  [MANIP-1] validated grasp recipe (desc-first branch, Cartesian tilted descent)
  [NAV-2]  drive base to the place station          — object held, arm frozen lifted
  [MANIP-2] Cartesian descent to the place point, open, retreat up

Success = cube released within tol of the place target, back on the table.
"""
import sys, os, numpy as np, torch, gymnasium as gym
sys.path.insert(0, "/home/perelman/AlohaMini/maniskill")
sys.path.insert(0, "/tmp/claude-1000/-home-perelman-AlohaMini/2d745dbb-3484-4eaa-af51-d8284661b2bf/scratchpad")
import mani_skill.envs, data_gen
from grasp_demo_v2 import (SlowGrasp, _best_full_pose, interp,
                           V_ARM, V_ARM_DESCEND, V_LIFT, CLOSE_STEPS, SETTLE, HOLD, FPS, FRAMES_TMP)
from data_gen.intern_engine.skills.ik import (actor_position, resolve_actor,
                                              desired_approach_dir, solve_arm_ik_full_pose)

# fast render settings (user preference)
import sapien.render as _R
_R.set_ray_tracing_samples_per_pixel(8)
_R.set_ray_tracing_path_depth(4)
W, H = 1280, 720

OBJ = "077_rubiks_cube"
PITCH = 60.0
PLACE_XY = np.array([0.10, -0.42], np.float32)   # place target on the table (far from slot 0)
V_BASE = 0.010     # m/step  (~0.2 m/s)  base translation cap
OUT = "/home/perelman/AlohaMini/maniskill/data_gen/output/demo_pro"
os.makedirs(OUT, exist_ok=True)


def encode(frames, path):
    import shutil, subprocess
    import imageio.v2 as imageio
    if os.path.isdir(FRAMES_TMP):
        shutil.rmtree(FRAMES_TMP)
    os.makedirs(FRAMES_TMP)
    for i, f in enumerate(frames):
        imageio.imwrite(os.path.join(FRAMES_TMP, f"{i:05d}.png"), f)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i",
                    os.path.join(FRAMES_TMP, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-preset", "veryfast", path],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(FRAMES_TMP, ignore_errors=True)


def to_u8(fr):
    a = fr[0]
    a = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
    return a.astype(np.uint8) if a.dtype == np.uint8 else np.clip(a, 0, 255).astype(np.uint8)


def main():
    env = gym.make("AlohaMiniMultiYCB-v1", num_envs=1, obs_mode="state",
                   control_mode="pd_joint_pos_fixed_base", render_mode="rgb_array",
                   reward_mode="none", sim_backend="physx_cpu", object_ids=[OBJ],
                   robot_uid="aloha_mini_pro_v2", base_xy=(-0.29, -0.10),  # start AWAY from the table
                   render_eye=[0.55, -1.0, 1.15], render_target=[-0.05, -0.40, 0.80],
                   human_render_camera_configs=dict(shader_pack="rt-fast", width=W, height=H))
    be = env.unwrapped
    env.reset(seed=0)
    robot = be.agent.robot
    names = [j.name for j in robot.active_joints]
    idx = {n: i for i, n in enumerate(names)}
    BASE_IDS = [idx["root_x_axis_joint"], idx["root_y_axis_joint"], idx["root_z_rotation_joint"]]
    name = be.object_actor_names[0]
    obj0 = actor_position(resolve_actor(env, name)).copy()
    g = SlowGrasp(env)
    skill, lay = g.skill, g.lay
    op, cl = skill.open_gripper, skill.closed_gripper

    def qnow():
        q = robot.get_qpos()
        return (q[0].cpu().numpy() if hasattr(q, "cpu") else np.asarray(q).reshape(-1)).copy()

    def set_q(q):
        robot.set_qpos(torch.as_tensor(q[None], dtype=torch.float32))

    # ---------------- FEASIBILITY: ASPIRE multi-angle station selection ----------------
    def select_station(target_pt, label):
        """Teleport-check candidate base stations; return the one whose tilted IK is best."""
        appr_dir = desired_approach_dir(target_pt, PITCH).astype(np.float32)
        jaw_dir = np.array([1.0, 0.0, 0.0], np.float32)
        q0 = qnow()
        best = None
        for dx in (-0.15, -0.075, 0.0, 0.075, 0.15):
            for by in (-0.32, -0.26, -0.20):
                bx = float(target_pt[0] - 0.156 + dx)   # arm base sits ~+0.156 in x from root
                q = q0.copy()
                q[BASE_IDS[0]], q[BASE_IDS[1]], q[BASE_IDS[2]] = bx, by, 0.0
                set_q(q)
                r = solve_arm_ik_full_pose(env, target_pt, appr_dir, jaw_dir,
                                           arm="left", lift_position=0.0,
                                           shoulder_lift_seed=1.0, max_iters=120)
                score = r.error
                if best is None or score < best[0]:
                    best = (score, bx, by, r.arm_qpos.copy())
        set_q(q0)
        print(f"[FEAS {label}] station=({best[1]:+.3f},{best[2]:+.3f}) ik_err={best[0]:.4f}", flush=True)
        return best[1], best[2]

    frames = [to_u8(env.render())]
    marks = {}

    def act(base_xy_t, arm_q, grip, lift):
        a = skill.current_action_template(env)
        a[0], a[1], a[2] = float(base_xy_t[0]), float(base_xy_t[1]), 0.0
        a[3] = float(lift)
        a[lay["right_grip"]] = op
        skill.set_arm_action(a, "left", arm_q, grip)
        return a.astype(np.float32)

    def run(actions):
        for a in actions:
            env.step(torch.as_tensor(a)[None])
            frames.append(to_u8(env.render()))

    rest_q = skill.current_action_template(env)[lay["left_arm"]].astype(np.float32)
    base0 = qnow()[BASE_IDS][:2]

    # ---------------- NAV-1: drive to the pick station (arm at rest, above the table) ---
    grasp_pt = obj0.astype(np.float32)
    st1 = np.array(select_station(grasp_pt, "pick"), np.float32)
    marks["NAV-1"] = len(frames)
    nav1 = [act(b, rest_q, op, 0.0) for b in interp(base0, st1, V_BASE)]
    run(nav1)

    # ---------------- MANIP-1: validated grasp from the settled station ----------------
    marks["PICK"] = len(frames)
    appr_dir = desired_approach_dir(grasp_pt, PITCH).astype(np.float32)
    jaw_dir = np.array([1.0, 0.0, 0.0], np.float32)
    pre_pt = (grasp_pt - appr_dir * 0.11).astype(np.float32)
    desc = _best_full_pose(env, grasp_pt, appr_dir, jaw_dir, "left", 0.0)
    appr = _best_full_pose(env, pre_pt, appr_dir, jaw_dir, "left", 0.0, seed=desc.arm_qpos)
    print(f"[PICK] ik appr={appr.error:.4f} desc={desc.error:.4f}", flush=True)
    descent = [appr.arm_qpos]; seedq = appr.arm_qpos
    for s in np.linspace(0.11, 0.0, 9)[1:]:
        w = _best_full_pose(env, (grasp_pt - appr_dir * s).astype(np.float32),
                            appr_dir, jaw_dir, "left", 0.0, seed=seedq)
        descent.append(w.arm_qpos); seedq = w.arm_qpos
    desc_q = descent[-1]
    acts = []
    for q in interp(rest_q, appr.arm_qpos, V_ARM):       acts.append(act(st1, q, op, 0.0))
    for q0_, q1_ in zip(descent[:-1], descent[1:]):
        for q in interp(q0_, q1_, V_ARM_DESCEND):        acts.append(act(st1, q, op, 0.0))
    for k in range(1, CLOSE_STEPS + 1):                  acts.append(act(st1, desc_q, op + (cl - op) * k / CLOSE_STEPS, 0.0))
    for _ in range(SETTLE):                              acts.append(act(st1, desc_q, cl, 0.0))
    for lz in interp([0.0], [0.16], V_LIFT):             acts.append(act(st1, desc_q, cl, float(lz[0])))
    run(acts)
    held = actor_position(resolve_actor(env, name)).copy()
    print(f"[PICK] lifted obj z={held[2]:.3f} (start {obj0[2]:.3f})", flush=True)

    # ---------------- NAV-2: carry to the place station (arm frozen, gripper closed) ---
    place_pt = np.array([PLACE_XY[0], PLACE_XY[1], obj0[2]], np.float32)
    st2 = np.array(select_station(place_pt, "place"), np.float32)
    marks["NAV-2"] = len(frames)
    nav2 = [act(b, desc_q, cl, 0.16) for b in interp(st1, st2, V_BASE)]
    run(nav2)

    # ---------------- MANIP-2: Cartesian descent to the place point, release, retreat --
    marks["PLACE"] = len(frames)
    appr_dir2 = desired_approach_dir(place_pt, PITCH).astype(np.float32)
    desc2 = _best_full_pose(env, place_pt, appr_dir2, jaw_dir, "left", 0.0)
    appr2 = _best_full_pose(env, (place_pt - appr_dir2 * 0.11).astype(np.float32),
                            appr_dir2, jaw_dir, "left", 0.0, seed=desc2.arm_qpos)
    print(f"[PLACE] ik appr={appr2.error:.4f} desc={desc2.error:.4f}", flush=True)
    descent2 = [appr2.arm_qpos]; seedq = appr2.arm_qpos
    for s in np.linspace(0.11, 0.0, 9)[1:]:
        w = _best_full_pose(env, (place_pt - appr_dir2 * s).astype(np.float32),
                            appr_dir2, jaw_dir, "left", 0.0, seed=seedq)
        descent2.append(w.arm_qpos); seedq = w.arm_qpos
    desc2_q = descent2[-1]
    acts = []
    # lower the lift back down while moving to the pre-place config
    for lz, q in zip(interp([0.16], [0.0], V_LIFT),
                     interp(desc_q, appr2.arm_qpos, V_ARM) or [appr2.arm_qpos]):
        acts.append(act(st2, q, cl, float(lz[0])))
    # make sure both finish
    for q in interp(desc_q, appr2.arm_qpos, V_ARM):      acts.append(act(st2, q, cl, 0.0))
    for q0_, q1_ in zip(descent2[:-1], descent2[1:]):
        for q in interp(q0_, q1_, V_ARM_DESCEND):        acts.append(act(st2, q, cl, 0.0))
    for k in range(1, CLOSE_STEPS + 1):                  acts.append(act(st2, desc2_q, cl + (op - cl) * k / CLOSE_STEPS, 0.0))
    for q in interp(desc2_q, appr2.arm_qpos, V_ARM):     acts.append(act(st2, q, op, 0.0))
    for _ in range(HOLD):                                acts.append(act(st2, appr2.arm_qpos, op, 0.0))
    run(acts)

    objf = actor_position(resolve_actor(env, name))
    err = float(np.linalg.norm(objf[:2] - place_pt[:2]))
    placed = err < 0.06 and abs(objf[2] - obj0[2]) < 0.03
    print(f"RESULT placed={placed} obj_final={np.round(objf,3).tolist()} "
          f"place_target={np.round(place_pt,3).tolist()} xy_err={err*1000:.0f}mm", flush=True)
    env.close()

    path = os.path.join(OUT, "pro_nav_pick_place.mp4")
    encode(frames, path)
    print("VIDEO", path, os.path.getsize(path) // 1024, "KB", flush=True)
    print("PHASES", marks, "total_frames", len(frames), flush=True)


if __name__ == "__main__":
    main()
