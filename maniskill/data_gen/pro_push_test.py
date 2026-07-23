"""AlohaMini Pro linear-push validation harness.

ASPIRE-style primitive: choose a feasible base station, close the gripper, place the
TCP just behind a tabletop cube, then drive a horizontal Cartesian line through the
cube toward a target XY. The script intentionally prints cube trajectory and contact
pairs so failures can be debugged from stdout without rendering.
"""

import os
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, "/home/perelman/AlohaMini/maniskill")
sys.path.insert(0, "/tmp/claude-1000/-home-perelman-AlohaMini/2d745dbb-3484-4eaa-af51-d8284661b2bf/scratchpad")

import mani_skill.envs  # noqa: F401
import data_gen  # noqa: F401
from grasp_demo_v2 import SlowGrasp, _best_full_pose, interp, V_ARM, V_ARM_DESCEND
from data_gen.intern_engine.skills.ik import (
    actor_position,
    desired_approach_dir,
    resolve_actor,
    solve_arm_ik_full_pose,
)


OBJ = "077_rubiks_cube"
PICK_XY = np.array([-0.13, -0.31], np.float32)
TABLE_X = (-0.43, 0.17)
TABLE_Y = (-0.69, -0.21)
TABLE_Z = 0.70
CUBE_HALF = 0.028

PUSH_DIR = np.array([0.0, -1.0, 0.0], np.float32)
PUSH_DIR_XY = PUSH_DIR[:2]
PUSH_DIST = 0.11
PUSH_TARGET_XY = PICK_XY + PUSH_DIR_XY * PUSH_DIST
PUSH_Z_OFFSET = 0.000
START_BACK = 0.065
# The Pro gripper's side/finger geometry continues transferring momentum after the
# TCP line passes the cube center; stop the commanded TCP north of the target center.
END_OVERDRIVE = -0.070
HOVER = 0.080
WAYPOINT_SPACING = 0.010

V_BASE = 0.010
IK_ERR_LIMIT = 0.025
CONTACT_IMPULSE_EPS = 1e-5


def as_np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def vec3(value) -> np.ndarray:
    a = as_np(value)
    if a.ndim == 2:
        a = a[0]
    return np.asarray(a, dtype=np.float64).reshape(-1)[:3]


def cart_line(start: np.ndarray, end: np.ndarray, spacing: float) -> list[np.ndarray]:
    start = np.asarray(start, np.float32).reshape(3)
    end = np.asarray(end, np.float32).reshape(3)
    dist = float(np.linalg.norm(end - start))
    n = max(1, int(np.ceil(dist / max(spacing, 1e-6))))
    return [(start + (end - start) * (k / n)).astype(np.float32) for k in range(n + 1)]


def main() -> None:
    env = gym.make(
        "AlohaMiniMultiYCB-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_pos_fixed_base",
        render_mode=None,
        reward_mode="none",
        sim_backend="physx_cpu",
        render_backend="none",
        object_ids=[OBJ],
        robot_uid="aloha_mini_2",
        base_xy=(-0.40, 0.18),
        slot_override_xy=[tuple(PICK_XY)],
    )
    be = env.unwrapped
    env.reset(seed=0)

    robot = be.agent.robot
    names = [j.name for j in robot.active_joints]
    idx = {n: i for i, n in enumerate(names)}
    base_ids = [
        idx["root_x_axis_joint"],
        idx["root_y_axis_joint"],
        idx["root_z_rotation_joint"],
    ]

    obj_name = be.object_actor_names[0]
    obj_actor = resolve_actor(env, obj_name)
    obj0 = actor_position(obj_actor).copy()
    push_z = float(obj0[2] + PUSH_Z_OFFSET)

    g = SlowGrasp(env)
    skill, lay = g.skill, g.lay
    op, cl = skill.open_gripper, skill.closed_gripper
    jaw_dir = np.array([1.0, 0.0, 0.0], np.float32)
    approach_dir = PUSH_DIR.copy()

    def qnow() -> np.ndarray:
        q = robot.get_qpos()
        return (q[0].cpu().numpy() if hasattr(q, "cpu") else np.asarray(q).reshape(-1)).astype(np.float64).copy()

    root_xy = vec3(robot.pose.p)[:2].astype(np.float64)

    def w2j(world_xy_yaw) -> np.ndarray:
        out = np.array(world_xy_yaw, np.float64).copy()
        out[0] -= root_xy[0]
        out[1] -= root_xy[1]
        return out

    def set_q(q: np.ndarray) -> None:
        robot.set_qpos(torch.as_tensor(q[None], dtype=torch.float32))

    def arm_base_xy() -> np.ndarray:
        for link in be.agent.robot.get_links():
            if link.name == "left_base":
                return vec3(link.pose.p)[:2].astype(np.float64)
        raise RuntimeError("left_base link not found")

    def act(base_t, arm_q, grip, lift):
        """base_t is WORLD (x, y) or (x, y, yaw); root joints are relative."""
        a = skill.current_action_template(env)
        yaw = float(base_t[2]) if len(base_t) > 2 else 0.0
        j = w2j((float(base_t[0]), float(base_t[1]), yaw))
        a[0], a[1], a[2] = float(j[0]), float(j[1]), float(j[2])
        a[3] = float(lift)
        a[lay["right_grip"]] = op
        skill.set_arm_action(a, "left", arm_q, grip)
        return a.astype(np.float32)

    def object_contact_pairs():
        scene = be.scene.sub_scenes[0]
        rows = []
        for c in scene.get_contacts():
            body_names = []
            for body in c.bodies:
                ent = getattr(body, "entity", body)
                body_names.append(getattr(ent, "name", str(ent)))
            lower = " ".join(body_names).lower()
            if "rubiks" not in lower and OBJ not in lower and obj_name.lower() not in lower:
                continue
            total = 0.0
            for p in c.points:
                total += float(np.linalg.norm(as_np(p.impulse)))
            if total > CONTACT_IMPULSE_EPS:
                rows.append((total, body_names))
        rows.sort(key=lambda r: -r[0])
        return rows

    def gripper_object_contacts(rows) -> list[tuple[float, list[str]]]:
        out = []
        for total, body_names in rows:
            lower = " ".join(body_names).lower()
            if "left" in lower or "fixed_jaw" in lower or "finger" in lower:
                out.append((total, body_names))
        return out

    def print_contacts(label: str, rows=None, limit: int = 8) -> None:
        rows = object_contact_pairs() if rows is None else rows
        if not rows:
            print(f"[CONTACT {label}] none involving object", flush=True)
            return
        pieces = []
        for total, body_names in rows[:limit]:
            pieces.append(f"{body_names[0]}<->{body_names[1]} imp={total:.4f}")
        print(f"[CONTACT {label}] " + " | ".join(pieces), flush=True)

    def run_actions(actions, phase: str, log_every: int = 20) -> int:
        grip_contacts = 0
        for i, a in enumerate(actions):
            env.step(torch.as_tensor(a)[None])
            objp = actor_position(obj_actor)
            rows = object_contact_pairs()
            g_rows = gripper_object_contacts(rows)
            grip_contacts += len(g_rows)
            if phase == "PUSH" and (i % log_every == 0 or i == len(actions) - 1):
                err = float(np.linalg.norm(objp[:2] - PUSH_TARGET_XY))
                print(
                    f"[TRAJ {phase} step={i:04d}] obj=({objp[0]:+.3f},{objp[1]:+.3f},{objp[2]:+.3f}) "
                    f"target=({PUSH_TARGET_XY[0]:+.3f},{PUSH_TARGET_XY[1]:+.3f}) xy_err={err:.3f}",
                    flush=True,
                )
                print_contacts(f"{phase} step={i:04d}", rows, limit=5)
            elif phase != "PUSH" and (i == 0 or i == len(actions) - 1):
                print(
                    f"[TRAJ {phase} step={i:04d}] obj=({objp[0]:+.3f},{objp[1]:+.3f},{objp[2]:+.3f})",
                    flush=True,
                )
        return grip_contacts

    def solve_path(points: list[np.ndarray], seed=None, label: str = "path") -> list[np.ndarray]:
        qs = []
        errs = []
        seed_q = seed
        for i, pt in enumerate(points):
            if seed_q is None:
                r = _best_full_pose(env, pt, approach_dir, jaw_dir, "left", 0.0)
            else:
                r = solve_arm_ik_full_pose(
                    env,
                    pt,
                    approach_dir,
                    jaw_dir,
                    arm="left",
                    lift_position=0.0,
                    seed=seed_q,
                    max_iters=250,
                )
                if r.error > 0.015:
                    alt = _best_full_pose(env, pt, approach_dir, jaw_dir, "left", 0.0)
                    if alt.error < r.error:
                        r = alt
            qs.append(r.arm_qpos.copy())
            errs.append(float(r.error))
            seed_q = r.arm_qpos
        print(f"[IK {label}] max_err={max(errs):.4f} errs={[round(e, 4) for e in errs]}", flush=True)
        if max(errs) > IK_ERR_LIMIT:
            raise RuntimeError(f"{label} IK max error {max(errs):.4f} exceeds {IK_ERR_LIMIT:.4f}")
        return qs

    def select_station(target_points: list[np.ndarray]):
        q0 = qnow()
        rest = skill.current_action_template(env)[lay["left_arm"]].astype(np.float32)
        best = None
        armoff = {
            # yaw=-90 turns the body toward the table. From a legal north-edge root
            # station, the left arm base sits south of the root and the gripper can
            # push farther south along the table plane.
            -np.pi / 2: (-0.041, -0.156),
        }
        for yaw, (ox, oy) in armoff.items():
            for dx in (-0.04, 0.0, 0.04):
                for by in (0.04, 0.01):
                    bx = float(target_points[0][0] - ox + dx)
                    j = w2j((bx, by, yaw))
                    q = q0.copy()
                    q[base_ids[0]], q[base_ids[1]], q[base_ids[2]] = j[0], j[1], j[2]
                    set_q(q)
                    hold = act((bx, by, yaw), rest, op, 0.0)
                    for _ in range(6):
                        env.step(torch.as_tensor(hold)[None])
                    qs = qnow()
                    base_err = float(np.hypot(qs[base_ids[0]] - j[0], qs[base_ids[1]] - j[1]))
                    if base_err > 0.02:
                        continue

                    ab = arm_base_xy()
                    _ = desired_approach_dir(target_points[0], 0.0, base_xy=tuple(ab))
                    seed_q = None
                    errs = []
                    qs_path = []
                    for pt in target_points:
                        kw = dict(arm="left", lift_position=0.0, max_iters=120)
                        if seed_q is None:
                            kw["shoulder_lift_seed"] = 1.0
                        else:
                            kw["seed"] = seed_q
                        r = solve_arm_ik_full_pose(env, pt, approach_dir, jaw_dir, **kw)
                        errs.append(float(r.error))
                        qs_path.append(r.arm_qpos.copy())
                        seed_q = r.arm_qpos
                    max_err = max(errs)
                    if max_err > IK_ERR_LIMIT:
                        continue
                    dist = float(np.linalg.norm(ab - target_points[0][:2]))
                    comfort = abs(dist - 0.20)
                    score = max_err + 0.02 * comfort
                    if best is None or score < best[0]:
                        best = (score, bx, by, yaw, qs_path[0], errs, ab.copy(), base_err)
        set_q(q0)
        if best is None:
            raise RuntimeError("no physically-valid push station found")
        print(
            f"[FEAS push] station=({best[1]:+.3f},{best[2]:+.3f},yaw={np.degrees(best[3]):+.0f}deg) "
            f"armbase=({best[6][0]:+.3f},{best[6][1]:+.3f}) ik_errs={[round(e, 4) for e in best[5]]} "
            f"base_err={best[7]:.4f}",
            flush=True,
        )
        return np.array([best[1], best[2], best[3]], np.float32), best[4]

    print(
        f"[SETUP] obj0={np.round(obj0, 4).tolist()} push_target_xy={np.round(PUSH_TARGET_XY, 4).tolist()} "
        f"push_dir={PUSH_DIR.tolist()} push_z={push_z:.3f}",
        flush=True,
    )

    start_pt = np.array([obj0[0], obj0[1], push_z], np.float32) - PUSH_DIR * START_BACK
    center_pt = np.array([obj0[0], obj0[1], push_z], np.float32)
    end_pt = np.array([PUSH_TARGET_XY[0], PUSH_TARGET_XY[1], push_z], np.float32) + PUSH_DIR * END_OVERDRIVE
    hover_pt = start_pt + np.array([0.0, 0.0, HOVER], np.float32)

    station_points = [start_pt, end_pt]
    station, station_seed = select_station(station_points)

    rest_q = skill.current_action_template(env)[lay["left_arm"]].astype(np.float32)
    base0 = qnow()[base_ids].copy()
    base0[0] += root_xy[0]
    base0[1] += root_xy[1]

    nav = [act(b, rest_q, cl, 0.0) for b in interp(base0, station, V_BASE)]
    nav += [nav[-1]] * 20
    run_actions(nav, "NAV", log_every=40)

    obj_pre = actor_position(obj_actor).copy()
    print(f"[NAVCHK] obj_pre={np.round(obj_pre, 4).tolist()} station={np.round(station, 4).tolist()}", flush=True)
    print_contacts("after_nav")

    start_pt = np.array([obj_pre[0], obj_pre[1], push_z], np.float32) - PUSH_DIR * START_BACK
    center_pt = np.array([obj_pre[0], obj_pre[1], push_z], np.float32)
    end_pt = np.array([PUSH_TARGET_XY[0], PUSH_TARGET_XY[1], push_z], np.float32) + PUSH_DIR * END_OVERDRIVE
    hover_pt = start_pt + np.array([0.0, 0.0, HOVER], np.float32)

    hover_q = solve_path([hover_pt], seed=station_seed, label="hover")[0]
    descend_points = cart_line(hover_pt, start_pt, WAYPOINT_SPACING)[1:]
    descend_qs = solve_path(descend_points, seed=hover_q, label="behind_descend")
    push_points = cart_line(start_pt, end_pt, WAYPOINT_SPACING)[1:]
    push_qs = solve_path(push_points, seed=descend_qs[-1], label="push_line")

    approach_actions = []
    for q in interp(rest_q, hover_q, V_ARM):
        approach_actions.append(act(station, q, cl, 0.0))
    prev = hover_q
    for q_next in descend_qs:
        for q in interp(prev, q_next, V_ARM_DESCEND):
            approach_actions.append(act(station, q, cl, 0.0))
        prev = q_next
    run_actions(approach_actions, "APPROACH", log_every=40)
    print_contacts("at_push_start")

    push_actions = []
    prev = descend_qs[-1]
    for q_next in push_qs:
        for q in interp(prev, q_next, V_ARM_DESCEND):
            push_actions.append(act(station, q, cl, 0.0))
        prev = q_next
    grip_contacts = run_actions(push_actions, "PUSH", log_every=15)
    print_contacts("after_push")

    for _ in range(20):
        env.step(torch.as_tensor(act(station, push_qs[-1], cl, 0.0))[None])
    objf = actor_position(obj_actor)
    xy_err = float(np.linalg.norm(objf[:2] - PUSH_TARGET_XY))
    on_table = (
        TABLE_X[0] <= float(objf[0]) <= TABLE_X[1]
        and TABLE_Y[0] <= float(objf[1]) <= TABLE_Y[1]
        and float(objf[2]) >= TABLE_Z - 0.005
    )
    success = bool(xy_err <= 0.05 and on_table)
    print(
        f"RESULT success={success} obj_final={np.round(objf, 4).tolist()} "
        f"push_target={np.round(PUSH_TARGET_XY, 4).tolist()} xy_err={xy_err:.4f} "
        f"on_table={on_table} gripper_object_contacts={grip_contacts}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
