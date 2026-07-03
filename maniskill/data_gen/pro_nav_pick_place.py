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
import agents.aloha_mini  # noqa: F401  (registers all AlohaMini agent uids incl. pro_v3)
from grasp_demo_v2 import (SlowGrasp, _best_full_pose, interp,
                           V_ARM, V_ARM_DESCEND, V_LIFT, CLOSE_STEPS, SETTLE, HOLD, FPS, FRAMES_TMP)
from data_gen.aspire_engine.nav_planner import plan_path
from data_gen.intern_engine.skills.ik import (actor_position, resolve_actor,
                                              desired_approach_dir, solve_arm_ik_full_pose)

# fast render settings (user preference)
import sapien.render as _R
_R.set_ray_tracing_samples_per_pixel(8)
_R.set_ray_tracing_path_depth(4)
W, H = 1280, 720

OBJ = "077_rubiks_cube"
PITCH = 60.0
# Base<->table collision is now ON (no more bit-24 hack), so the base must park OUTSIDE
# the table footprint (x[-0.43,0.17], y[-0.69,-0.21]). Put the pick object and the place
# target near the NORTH edge so they are reachable from legal stations.
PICK_XY = np.array([-0.13, -0.31], np.float32)
PLACE_XY = np.array([0.06, -0.31], np.float32)
V_BASE = 0.010     # m/step  (~0.2 m/s)  base translation cap
NAV_OBSTACLES = [{"center": [-0.13, -0.45], "half": [0.30, 0.24], "margin": 0.02}]
NAV_ROBOT_RADIUS = 0.22
NAV_BOUNDS = (-1.2, 1.0, -1.0, 1.0)
NOREN = bool(os.environ.get("NORENDER"))
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
                   control_mode="pd_joint_pos_fixed_base", render_mode=None if NOREN else "rgb_array",
                   reward_mode="none", sim_backend="physx_cpu", object_ids=[OBJ],
                   robot_uid=os.environ.get("PRO_UID", "aloha_mini_pro_v2"), base_xy=(-0.85, -0.75),
                   slot_override_xy=[tuple(PICK_XY)],
                   render_eye=[0.60, -1.05, 1.20], render_target=[-0.05, -0.30, 0.80],
                   human_render_camera_configs={} if NOREN else dict(shader_pack="rt-fast", width=W, height=H),
                   render_backend="none" if NOREN else "gpu")
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

    # The root_x/root_y/root_z joints are RELATIVE to the robot's root pose (env base_xy),
    # not world coordinates. Convert world station coords -> joint targets.
    rp = robot.pose.p
    rp = (rp[0].cpu().numpy() if hasattr(rp, "cpu") else np.asarray(rp).reshape(-1))[:2]
    ROOT_XY = rp.astype(np.float64)

    def w2j(world_xy_yaw):
        out = np.array(world_xy_yaw, np.float64).copy()
        out[0] -= ROOT_XY[0]
        out[1] -= ROOT_XY[1]
        return out

    def set_q(q):
        robot.set_qpos(torch.as_tensor(q[None], dtype=torch.float32))

    # ---------------- FEASIBILITY: ASPIRE multi-angle station selection ----------------
    # Two gates per candidate now that base<->table collision is ON:
    #   (1) PHYSICS: teleport the base there, hold for a few steps — if the table (or
    #       anything) shoves the base off the target, the station is physically invalid.
    #   (2) IK: the tilted full-pose IK for the target must converge from that station.
    def armbase_xy():
        for l in be.agent.robot.get_links():
            if l.name == "left_base":
                p = l.pose.p
                return ((p[0].cpu().numpy() if hasattr(p, "cpu") else np.asarray(p).reshape(-1))[:2]).astype(np.float64)
        return None

    # occupancy grid reused for FREE-SPACE station sampling (same params as A* nav)
    from data_gen.aspire_engine.nav_planner import OccupancyGrid
    _g = OccupancyGrid((NAV_BOUNDS[0], NAV_BOUNDS[2]),
                       (NAV_BOUNDS[1] - NAV_BOUNDS[0], NAV_BOUNDS[3] - NAV_BOUNDS[2]))
    for ob in NAV_OBSTACLES:
        _g.add_box(ob["center"], ob["half"], ob.get("margin", 0.0))
    _g.inflate(NAV_ROBOT_RADIUS)

    def _free(xy):
        ij = _g.world_to_cell(xy)
        return _g.in_bounds(ij) and not _g.is_occupied(ij)

    def select_station(target_pt, label, rng=None):
        """FACING-based station sampling (user request): candidates on a ring around
        the target, in FREE space, with the robot FRONT (-y at yaw 0, between the two
        arms) turned toward the object plus a random frontal jitter (~±30 deg) — so
        the robot approaches face-on from varied frontal angles instead of always
        grasping over its side."""
        rng = rng or np.random.default_rng(0)
        jaw_dir = np.array([1.0, 0.0, 0.0], np.float32)
        q0 = qnow()
        rest = skill.current_action_template(env)[lay["left_arm"]].astype(np.float32)
        best = None
        cands = []
        for radius in (0.34, 0.40, 0.46):
            for k in range(12):
                th = 2 * np.pi * k / 12
                bx = float(target_pt[0] + radius * np.cos(th))
                by = float(target_pt[1] + radius * np.sin(th))
                if not _free((bx, by)):
                    continue
                d = np.array([target_pt[0] - bx, target_pt[1] - by], np.float64)
                d /= np.linalg.norm(d)
                yaw_face = float(np.arctan2(d[0], -d[1]))   # front(yaw)=R(yaw)@(0,-1)
                jit = float(rng.choice([-0.5, -0.25, 0.0, 0.25, 0.5]))
                cands.append((abs(jit), bx, by, yaw_face + jit))
        cands.sort(key=lambda c: c[0])   # small jitter (most face-on) first
        for _, bx, by, yaw in cands[:14]:
            if True:
                if True:
                    j = w2j((bx, by, yaw))
                    q = q0.copy()
                    q[BASE_IDS[0]], q[BASE_IDS[1]], q[BASE_IDS[2]] = j[0], j[1], j[2]
                    set_q(q)
                    # physics settle: does the base actually HOLD this station?
                    hold = act((bx, by, yaw), rest, op, 0.0)
                    for _ in range(6):
                        env.step(torch.as_tensor(hold)[None])
                    qs = qnow()
                    base_err = float(np.hypot(qs[BASE_IDS[0]] - j[0], qs[BASE_IDS[1]] - j[1]))
                    if base_err > 0.02:
                        continue   # table (or clutter) rejected this station
                    # approach dir must lean from the CANDIDATE's actual arm base — the
                    # helper's default base_xy=(-0.35,0) is only right for the original
                    # fixed-base layout and rakes the object from any other station.
                    ab = armbase_xy()
                    appr_dir = desired_approach_dir(target_pt, PITCH, base_xy=tuple(ab)).astype(np.float32)
                    r = solve_arm_ik_full_pose(env, target_pt, appr_dir, jaw_dir,
                                               arm="left", lift_position=0.0,
                                               shoulder_lift_seed=1.0, max_iters=120)
                    # comfort tiebreak: prefer stations whose arm base sits in the sweet
                    # range (~0.16-0.25 m) over marginal far-reach ones with equal IK.
                    dist = float(np.linalg.norm(ab - target_pt[:2]))
                    comfort = abs(dist - 0.20)
                    score = r.error + 0.02 * comfort
                    if os.environ.get("FEASDBG"):
                        lb = be.agent.robot.links_map.get("left_base") if hasattr(be.agent.robot, "links_map") else None
                        lbp = None
                        for l in be.agent.robot.get_links():
                            if l.name == "left_base":
                                p = l.pose.p
                                lbp = (p[0].cpu().numpy() if hasattr(p, "cpu") else np.asarray(p).reshape(-1))[:3]
                        d = float(np.linalg.norm(lbp[:2] - target_pt[:2])) if lbp is not None else -1
                        print(f"    cand yaw={np.degrees(yaw):+.0f} bx={bx:+.3f} by={by:+.3f} "
                              f"armbase=({lbp[0]:+.3f},{lbp[1]:+.3f}) dist_xy={d:.3f} ik={r.error:.4f}", flush=True)
                    if best is None or score < best[0]:
                        best = (score, bx, by, yaw, r.arm_qpos.copy(), r.error)
                    if best is not None and best[0] < 0.004:
                        break   # face-on station with converged IK — good enough
        set_q(q0)
        if best is None:
            raise RuntimeError(f"no physically-valid station found for {label}")
        print(f"[FEAS {label}] station=({best[1]:+.3f},{best[2]:+.3f},yaw={np.degrees(best[3]):.0f}deg) "
              f"ik_err={best[5]:.4f}", flush=True)
        # return the winning arm config too — it seeds the MANIP IK after NAV (cold-start
        # sweeps can miss at far-reach stations that the FEAS solve already cracked).
        return np.array([best[1], best[2], best[3]], np.float32), best[4]

    frames = [] if NOREN else [to_u8(env.render())]
    marks = {}

    def act(base_t, arm_q, grip, lift):
        """base_t = WORLD (x, y) or (x, y, yaw) — converted to root-relative joint targets."""
        a = skill.current_action_template(env)
        yaw = float(base_t[2]) if len(base_t) > 2 else 0.0
        j = w2j((float(base_t[0]), float(base_t[1]), yaw))
        a[0], a[1], a[2] = float(j[0]), float(j[1]), float(j[2])
        a[3] = float(lift)
        a[lay["right_grip"]] = op
        skill.set_arm_action(a, "left", arm_q, grip)
        return a.astype(np.float32)

    def run(actions):
        for a in actions:
            env.step(torch.as_tensor(a)[None])
            if not NOREN:
                frames.append(to_u8(env.render()))

    def drive_path_actions(path_xy, start_pose, target_pose, arm_q, grip, lift, vmax):
        actions = []
        cur = np.asarray(start_pose, dtype=np.float32).copy()
        target_pose = np.asarray(target_pose, dtype=np.float32)
        points = [np.asarray(p, dtype=np.float32) for p in path_xy]
        drive_points = points.copy()
        if drive_points and float(np.linalg.norm(drive_points[0] - cur[:2])) <= 0.03:
            drive_points = drive_points[1:]
        if not drive_points or float(np.linalg.norm(drive_points[-1] - target_pose[:2])) > 1e-6:
            drive_points.append(target_pose[:2].copy())
        for xy in drive_points:
            tgt = np.array([xy[0], xy[1], target_pose[2]], np.float32)
            actions.extend(act(b, arm_q, grip, lift) for b in interp(cur, tgt, vmax))
            cur = tgt
        if not actions:
            actions.append(act(target_pose, arm_q, grip, lift))
        return actions

    rest_q = skill.current_action_template(env)[lay["left_arm"]].astype(np.float32)
    base0 = qnow()[BASE_IDS]   # joint coords (x, y, yaw)
    base0[0] += ROOT_XY[0]; base0[1] += ROOT_XY[1]   # -> WORLD

    # ---------------- FEASIBILITY (both stations UPFRONT, while the scene is static —
    # the place feasibility must not teleport the base around while the cube is held) --
    grasp_pt = obj0.astype(np.float32)
    place_pt = np.array([PLACE_XY[0], PLACE_XY[1], obj0[2]], np.float32)
    st1, st1_arm_seed = select_station(grasp_pt, "pick")
    st2, st2_arm_seed = select_station(place_pt, "place")
    if os.environ.get("FEASDBG"):
        env.close(); return
    marks["NAV-1"] = len(frames)
    path1 = plan_path(NAV_OBSTACLES, base0[:2], st1[:2], NAV_ROBOT_RADIUS, NAV_BOUNDS)
    if path1 is None:
        raise RuntimeError("A* failed for start->pick")
    print(f"[NAV] path waypoints={len(path1)} (start->pick)", flush=True)
    nav1 = drive_path_actions(path1, base0, st1, rest_q, op, 0.0, V_BASE)
    nav1 += [nav1[-1]] * 20   # settle: let the base PD fully converge on the station
    run(nav1)

    # ---------------- MANIP-1: validated grasp from the settled station ----------------
    marks["PICK"] = len(frames)
    grasp_pt = actor_position(resolve_actor(env, name)).astype(np.float32)  # re-perceive
    qpn = qnow()
    lbw = None
    for l in be.agent.robot.get_links():
        if l.name == "left_base":
            p = l.pose.p
            lbw = (p[0].cpu().numpy() if hasattr(p, "cpu") else np.asarray(p).reshape(-1))[:3]
    print(f"[NAVCHK] base joints=({qpn[BASE_IDS[0]]+ROOT_XY[0]:+.3f},{qpn[BASE_IDS[1]]+ROOT_XY[1]:+.3f},"
          f"yaw={np.degrees(qpn[BASE_IDS[2]]):+.0f}deg) target st1=({st1[0]:+.3f},{st1[1]:+.3f},"
          f"yaw={np.degrees(st1[2]):+.0f}deg) left_base_world={np.round(lbw,3).tolist()} "
          f"obj={np.round(grasp_pt,3).tolist()}", flush=True)
    appr_dir = desired_approach_dir(grasp_pt, PITCH, base_xy=tuple(lbw[:2])).astype(np.float32)
    jaw_dir = np.array([1.0, 0.0, 0.0], np.float32)
    pre_pt = (grasp_pt - appr_dir * 0.11).astype(np.float32)
    desc = _best_full_pose(env, grasp_pt, appr_dir, jaw_dir, "left", 0.0, seed=st1_arm_seed)
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
    marks["NAV-2"] = len(frames)
    # gentle carry: half speed — the 0.2 m/s velocity step of full V_BASE jerks the
    # held object out of the 15 N gripper.
    path2 = plan_path(NAV_OBSTACLES, st1[:2], st2[:2], NAV_ROBOT_RADIUS, NAV_BOUNDS)
    if path2 is None:
        raise RuntimeError("A* failed for pick->place")
    print(f"[NAV] path waypoints={len(path2)} (pick->place)", flush=True)
    nav2 = drive_path_actions(path2, st1, st2, desc_q, cl, 0.16, V_BASE * 0.5)
    nav2 += [nav2[-1]] * 20   # settle on the place station
    run(nav2)
    carried = actor_position(resolve_actor(env, name))
    print(f"[CARRY] obj after NAV-2 = {np.round(carried,3).tolist()} "
          f"({'STILL HELD' if carried[2] > obj0[2] + 0.05 else 'DROPPED DURING CARRY!'})", flush=True)
    # The object is not gripped exactly at the TCP; aim the place descent at
    # (target - grasp_offset) so the OBJECT (not the TCP) lands on the target.
    t1p = be.agent.left_finger1_tip.pose.p; t2p = be.agent.left_finger2_tip.pose.p
    t1p = (t1p[0].cpu().numpy() if hasattr(t1p, "cpu") else np.asarray(t1p).reshape(-1))[:3]
    t2p = (t2p[0].cpu().numpy() if hasattr(t2p, "cpu") else np.asarray(t2p).reshape(-1))[:3]
    grasp_off = (carried[:2] - (t1p + t2p)[:2] / 2).astype(np.float32)
    print(f"[CARRY] grasp offset obj-TCP = {np.round(grasp_off*1000,0).tolist()} mm", flush=True)

    # ---------------- MANIP-2 (ARM FROZEN): base fine-align + lift-only descent ------
    # Every attempt to reconfigure the arm while holding the cube slung it out of the
    # 15 N grip (TRK traces). So: do not move the arm at all. NAV-2 already parks the
    # cube ~1-2 cm from the target; close the rest with the BASE (closed-loop), descend
    # with the LIFT to 2 cm above the table, release, and retreat with the lift.
    marks["PLACE"] = len(frames)
    cur_base = st2.copy()
    for _ in range(3):
        obj_now = actor_position(resolve_actor(env, name))
        delta = np.array([PLACE_XY[0] - obj_now[0], PLACE_XY[1] - obj_now[1]], np.float32)
        if float(np.linalg.norm(delta)) < 0.008:
            break
        tgt = cur_base.copy(); tgt[0] += delta[0]; tgt[1] += delta[1]
        print(f"[PLACE] base fine-align {np.round(delta*1000,0).tolist()} mm", flush=True)
        acts = [act(b, desc_q, cl, 0.16) for b in interp(cur_base, tgt, V_BASE * 0.5)]
        acts += [act(tgt, desc_q, cl, 0.16)] * 10
        run(acts)
        cur_base = tgt
    # lift-only descent: lower until the cube hangs ~2 cm above the table
    obj_now = actor_position(resolve_actor(env, name))
    drop = float(obj_now[2] - (obj0[2] + 0.02))
    lift_lo = max(0.16 - drop, -0.1)
    print(f"[PLACE] lift descent 0.16 -> {lift_lo:.3f} (cube {obj_now[2]:.3f} -> ~{obj0[2]+0.02:.3f})", flush=True)
    acts = [act(cur_base, desc_q, cl, float(lz[0])) for lz in interp([0.16], [lift_lo], V_LIFT)]
    acts += [act(cur_base, desc_q, cl, lift_lo)] * 15
    run(acts)
    o = actor_position(resolve_actor(env, name))
    print(f"    [TRK pre-release] obj={np.round(o,3).tolist()} held={o[2] > obj0[2] + 0.03}", flush=True)
    # release + retreat up with the lift (fingers never translate over the placed cube)
    acts = []
    for k in range(1, 11):                               acts.append(act(cur_base, desc_q, cl + (op - cl) * k / 10, lift_lo))
    for _ in range(10):                                  acts.append(act(cur_base, desc_q, op, lift_lo))
    for lz in interp([lift_lo], [0.16], V_LIFT):         acts.append(act(cur_base, desc_q, op, float(lz[0])))
    for _ in range(HOLD):                                acts.append(act(cur_base, desc_q, op, 0.16))
    run(acts)

    objf = actor_position(resolve_actor(env, name))
    err = float(np.linalg.norm(objf[:2] - PLACE_XY))
    placed = err < 0.06 and abs(objf[2] - obj0[2]) < 0.03
    print(f"RESULT placed={placed} obj_final={np.round(objf,3).tolist()} "
          f"place_target={np.round(PLACE_XY,3).tolist()} xy_err={err*1000:.0f}mm", flush=True)
    env.close()

    if not NOREN:
        path = os.path.join(OUT, "pro_nav_pick_place.mp4")
        encode(frames, path)
        print("VIDEO", path, os.path.getsize(path) // 1024, "KB", flush=True)
    print("PHASES", marks, "total_frames", len(frames), flush=True)


if __name__ == "__main__":
    main()
