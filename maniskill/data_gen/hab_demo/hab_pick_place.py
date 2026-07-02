#!/usr/bin/env python3
"""Room-scale ReplicaCAD NAV+MANIP pick/place demo for AlohaMini Pro."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import sapien

import hab_scene


OUT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = OUT_DIR / "hab_pick_place.mp4"
KEYFRAME_PICK = OUT_DIR / "hab_keyframe_pick.png"
KEYFRAME_PLACE = OUT_DIR / "hab_keyframe_place.png"
FRAMES_TMP = OUT_DIR / "frames_tmp"
FPS = 25

PITCH = 60.0
STATION_PITCHES = (60.0, 70.0, 75.0)
STATION_GRID_RADIUS = 0.18
STATION_GRID_MARGIN = 0.0
STATION_RING_RADII = (0.28, 0.32, 0.36, 0.38, 0.42, 0.50, 0.58, 0.66, 0.74)
# rt-fast segfaults in libEGL_nvidia on the 478-actor ReplicaCAD scene (8GB GPU);
# minimal is the only shader that survives — override via HAB_RENDER_SHADER_PACK.
RENDER_SHADER_PACK = "minimal"
RENDER_SHADER_FALLBACK = "minimal"
PICK_GRASP_Z_BIAS = 0.0
PICK_BACK = 0.11
V_BASE = 0.010
V_ARM = 0.020
V_ARM_DESCEND = 0.013
V_LIFT = 0.0045
CLOSE_STEPS = 32
SETTLE = 10
HOLD = 22
LIFT_HIGH = 0.16
LIFT_PICK = -0.08
IK_ACCEPT = 0.035
STATION_LIFT_SWEEP = tuple(sorted({LIFT_PICK, 0.0, -0.04, -0.08, -0.10, -0.12, -0.15}))
# a pick station must exist within this XY ring of the grasp target, else the target
# is out of arm reach from any free floor cell (e.g. deep on the kitchen counter)
NUDGE_REACH_RING = 0.52
NUDGE_EDGE_INSET = 0.14
# validated grasp envelope for the Pro arm (object top height); the kitchen counter
# at 0.98 m is above it — approach IK falls apart even when the grasp pose solves
NUDGE_TOP_MIN = 0.42
NUDGE_TOP_MAX = 0.85
FOLLOW_CAM_INTERIOR = (0.95, -1.67)  # apartment centroid: eye stays between robot and here
# zoomed-in framing (user request): the robot fills ~2x more of the 1080p frame
FOLLOW_CAM_DIST = 1.6
FOLLOW_CAM_HEIGHT = 1.5
FOLLOW_CAM_SMOOTH = 0.06

PREFERRED_PLACE_SURFACES = [
    "env-0_objects/frl_apartment_book_06-93",
    "env-0_objects/frl_apartment_book_02-83",
    "env-0_objects/frl_apartment_tv_object-20",
]


def import_skill_helpers():
    hab_scene.configure_paths()
    from data_gen.intern_engine.skills import build_skill
    from data_gen.intern_engine.skills.ik import desired_approach_dir, solve_arm_ik_full_pose
    from data_gen.aspire_engine.nav_planner import astar, shortcut

    return build_skill, desired_approach_dir, solve_arm_ik_full_pose, astar, shortcut


build_skill, desired_approach_dir, solve_arm_ik_full_pose, astar, shortcut = import_skill_helpers()


def choose_render_shader_pack(norender: bool) -> str:
    if norender:
        return RENDER_SHADER_FALLBACK
    override = os.environ.get("HAB_RENDER_SHADER_PACK")
    if override:
        return override
    if torch.cuda.is_available():
        return RENDER_SHADER_PACK
    print(
        f"[RENDERCFG] requested_shader={RENDER_SHADER_PACK} unavailable_no_cuda "
        f"using_shader={RENDER_SHADER_FALLBACK}",
        flush=True,
    )
    return RENDER_SHADER_FALLBACK


def interp(q0, q1, vmax: float) -> list[np.ndarray]:
    q0 = np.asarray(q0, dtype=np.float32)
    q1 = np.asarray(q1, dtype=np.float32)
    n = max(1, int(np.ceil(float(np.max(np.abs(q1 - q0))) / max(vmax, 1e-6))))
    return [q0 + (q1 - q0) * (k / n) for k in range(1, n + 1)]


def to_u8(frame) -> np.ndarray:
    arr = hab_scene.to_numpy(frame)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4:
        arr = arr.reshape((-1,) + arr.shape[2:])
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(arr.max()) <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def save_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(to_u8(arr)).save(path)


def encode(frames: list[np.ndarray], path: Path) -> None:
    import imageio.v2 as imageio

    shutil.rmtree(FRAMES_TMP, ignore_errors=True)
    FRAMES_TMP.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        imageio.imwrite(FRAMES_TMP / f"{i:05d}.png", to_u8(frame))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(FRAMES_TMP / "%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-preset",
            "veryfast",
            str(path),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(FRAMES_TMP, ignore_errors=True)


def actor_info(name: str, actor: Any, kind: str = "movable") -> hab_scene.ActorInfo:
    aabb = hab_scene.actor_aabb(actor)
    if aabb is None:
        raise RuntimeError(f"no collision AABB for {name}")
    lo, hi = aabb
    size = hi - lo
    return hab_scene.ActorInfo(
        name=name,
        actor=actor,
        kind=kind,
        mass=hab_scene.actor_mass(actor),
        center=(lo + hi) * 0.5,
        half=size * 0.5,
        size=size,
        lo=lo,
        hi=hi,
    )


def pick_grasp_point(info: hab_scene.ActorInfo) -> np.ndarray:
    pt = info.center.astype(np.float32).copy()
    pt[2] -= min(PICK_GRASP_Z_BIAS, max(0.0, float(info.half[2]) * 0.6))
    # never grasp deeper than ~2.8 cm below the top: at 60° pitch the palm rides that
    # far above the TCP, and on tall objects a center grasp puts it INTO the top rim
    # (validated geometry: the 5.7 cm cube pinches at 2.85 cm below its top)
    pt[2] = max(pt[2], float(info.hi[2]) - 0.028)
    return pt


def best_full_pose(env, target, approach_dir, jaw_dir, arm, lift, seed=None):
    best = None
    seeds = [seed] if seed is not None else [0.5, 1.0, 1.5, -0.5, 0.0]
    for s in seeds:
        kwargs: dict[str, Any] = dict(arm=arm, lift_position=lift, max_iters=250)
        if seed is not None:
            kwargs["seed"] = seed
        else:
            kwargs["shoulder_lift_seed"] = s
        result = solve_arm_ik_full_pose(env, target, approach_dir, jaw_dir, **kwargs)
        if best is None or result.error < best.error:
            best = result
    return best


class Demo:
    def __init__(self, norender: bool, build_config_idx: int | None = None) -> None:
        self.norender = bool(norender)
        self.build_config_idx = hab_scene.BUILD_CONFIG_IDX if build_config_idx is None else int(build_config_idx)
        self.shader_pack = choose_render_shader_pack(self.norender)
        self.env = hab_scene.make_env(
            norender=self.norender,
            shader_pack=self.shader_pack,
            build_config_idx=self.build_config_idx,
        )
        self.be = self.env.unwrapped
        self.skill = build_skill("pick")
        self.robot = self.be.agent.robot
        self.lay = self.skill.arm_layout(self.be.agent)
        self.open_gripper = float(self.skill.open_gripper)
        self.closed_gripper = float(self.skill.closed_gripper)
        self.frames: list[np.ndarray] = []
        self._cam_eye: np.ndarray | None = None
        self._cam_tgt: np.ndarray | None = None
        self.marks: dict[str, int] = {}
        self.last_blocker = ""

        self.joint_names: list[str] = []
        self.idx: dict[str, int] = {}
        self.base_ids: list[int] = []
        self.left_arm_q_ids: list[int] = []
        self.left_grip_q_ids: list[int] = []
        self.refresh_robot_handles()
        self.root_xy = np.zeros(2, dtype=np.float64)
        self.rest_q = None
        self.jaw_dir = np.array([1.0, 0.0, 0.0], np.float32)
        self.force_lift_position: float | None = None
        self.force_action_qpos = False
        self.boost_pad_friction(mu=2.0)

    def boost_pad_friction(self, mu: float = 2.0) -> None:
        """Raise the finger-pad friction so the tilted-jaw wedge force can't slide the
        object out of the pinch (the 5-DOF arm leaves an ~8 deg jaw tilt at the
        mm-accurate grasp pose; default material friction lets flat objects extrude)."""
        patched = 0
        try:
            import sapien.physx as physx
            mat = physx.PhysxMaterial(static_friction=mu, dynamic_friction=mu, restitution=0.0)
            for link in self.robot.get_links():
                lname = link.name.lower()
                if not any(t in lname for t in ("jaw", "finger", "grip")):
                    continue
                for obj in link._objs:
                    for shape in obj.get_collision_shapes():
                        try:
                            shape.set_physical_material(mat)
                            patched += 1
                        except Exception:
                            try:
                                pm = shape.get_physical_material()
                                pm.set_static_friction(mu)
                                pm.set_dynamic_friction(mu)
                                patched += 1
                            except Exception:
                                pass
        except Exception as exc:
            print(f"[FRICTION] pad friction boost failed: {exc}", flush=True)
        print(f"[FRICTION] mu={mu} patched_shapes={patched}", flush=True)

    def close(self) -> None:
        self.env.close()

    def refresh_robot_handles(self) -> None:
        self.be = self.env.unwrapped
        self.robot = self.be.agent.robot
        self.lay = self.skill.arm_layout(self.be.agent)
        names = [joint.name for joint in self.robot.active_joints]
        self.joint_names = names
        self.idx = {name: i for i, name in enumerate(names)}
        self.base_ids = [
            self.idx["root_x_axis_joint"],
            self.idx["root_y_axis_joint"],
            self.idx["root_z_rotation_joint"],
        ]
        self.left_arm_q_ids = [
            self.idx[name] for name in getattr(self.be.agent, "left_arm_joint_names", []) if name in self.idx
        ]
        self.left_grip_q_ids = [
            self.idx[name] for name in getattr(self.be.agent, "left_gripper_joint_names", []) if name in self.idx
        ]

    def update_follow_camera(self) -> None:
        """Crane-style follow-cam: keep the eye between the robot and the apartment
        interior, EMA-smoothed so NAV motion doesn't jerk the view.
        HAB_CLOSEUP="ex,ey,ez;tx,ty,tz" pins a fixed close-up instead (failure forensics)."""
        try:
            cam = self.be.unwrapped._human_render_cameras["render_camera"].camera
        except Exception:
            return
        closeup = os.environ.get("HAB_CLOSEUP")
        if closeup:
            try:
                eye_s, tgt_s = closeup.split(";")
                eye = [float(v) for v in eye_s.split(",")]
                tgt = [float(v) for v in tgt_s.split(",")]
                pose = hab_scene.camera_pose_list(eye, tgt)
                cam.set_local_pose(sapien.Pose(pose[:3], pose[3:]))
            except Exception as exc:
                print(f"[CLOSEUP] bad HAB_CLOSEUP spec: {exc}", flush=True)
            return
        base = self.current_base_world()
        robot_xy = np.asarray(base[:2], np.float64)
        interior = np.array(FOLLOW_CAM_INTERIOR, np.float64)
        d = interior - robot_xy
        dn = float(np.linalg.norm(d))
        d = d / dn if dn > 0.5 else np.array([1.0, 0.0])
        eye = np.array(
            [
                float(np.clip(robot_xy[0] + FOLLOW_CAM_DIST * d[0], -2.55, 4.45)),
                float(np.clip(robot_xy[1] + FOLLOW_CAM_DIST * d[1], -8.05, 4.7)),
                FOLLOW_CAM_HEIGHT,
            ]
        )
        tgt = np.array([robot_xy[0], robot_xy[1], 0.55])
        if self._cam_eye is None:
            self._cam_eye, self._cam_tgt = eye, tgt
        else:
            a = FOLLOW_CAM_SMOOTH
            self._cam_eye = (1 - a) * self._cam_eye + a * eye
            self._cam_tgt = (1 - a) * self._cam_tgt + a * tgt
        pose = hab_scene.camera_pose_list(self._cam_eye.tolist(), self._cam_tgt.tolist())
        cam.set_local_pose(sapien.Pose(pose[:3], pose[3:]))

    def capture(self) -> None:
        if not self.norender:
            self._cap_count = getattr(self, "_cap_count", 0) + 1
            if (self._cap_count - 1) % int(os.environ.get("HAB_FRAME_STRIDE", "3")) != 0:
                return
            self.update_follow_camera()
            self.frames.append(to_u8(self.be.render_rgb_array(camera_name="render_camera")))

    def qnow(self) -> np.ndarray:
        q = hab_scene.to_numpy(self.robot.get_qpos())
        if q.ndim == 2:
            q = q[0]
        return np.asarray(q, dtype=np.float64).reshape(-1).copy()

    def set_q(self, qpos: np.ndarray) -> None:
        self.robot.set_qpos(torch.as_tensor(np.asarray(qpos)[None], dtype=torch.float32))

    def refresh_root_xy(self) -> None:
        self.root_xy = hab_scene.vec3(self.robot.pose.p)[:2].astype(np.float64)

    def w2j(self, world_xy_yaw) -> np.ndarray:
        out = np.asarray(world_xy_yaw, dtype=np.float64).copy()
        out[0] -= self.root_xy[0]
        out[1] -= self.root_xy[1]
        return out

    def current_base_world(self) -> np.ndarray:
        base = self.qnow()[self.base_ids].astype(np.float64)
        base[0] += self.root_xy[0]
        base[1] += self.root_xy[1]
        return base.astype(np.float32)

    def arm_base_xy(self) -> np.ndarray:
        for link in self.robot.get_links():
            if link.name == "left_base":
                return hab_scene.vec3(link.pose.p)[:2].astype(np.float64)
        raise RuntimeError("left_base link not found")

    def left_base_world(self) -> np.ndarray:
        for link in self.robot.get_links():
            if link.name == "left_base":
                return hab_scene.vec3(link.pose.p).astype(np.float64)
        raise RuntimeError("left_base link not found")

    def left_fingertips_world(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        agent = self.be.agent
        tcp = hab_scene.to_numpy(getattr(agent, "tcp_pos"))
        f1 = hab_scene.to_numpy(getattr(agent, "left_finger1_tip").pose.p)
        f2 = hab_scene.to_numpy(getattr(agent, "left_finger2_tip").pose.p)
        return (
            np.asarray(tcp, dtype=np.float64).reshape(-1, 3)[0],
            np.asarray(f1, dtype=np.float64).reshape(-1, 3)[0],
            np.asarray(f2, dtype=np.float64).reshape(-1, 3)[0],
        )

    def set_actor_center(self, actor: Any, name: str, center: np.ndarray) -> hab_scene.ActorInfo:
        info = actor_info(name, actor)
        pose = actor.pose
        p = hab_scene.vec3(pose.p)
        q = np.asarray(hab_scene.to_numpy(pose.q), dtype=np.float64).reshape(-1, 4)[0]
        delta = np.asarray(center, dtype=np.float64).reshape(3) - info.center
        actor.set_pose(sapien.Pose((p + delta).astype(float).tolist(), q.astype(float).tolist()))
        return actor_info(name, actor)

    def act(self, base_t, arm_q, grip: float, lift: float) -> np.ndarray:
        action = self.skill.current_action_template(self.env)
        yaw = float(base_t[2]) if len(base_t) > 2 else 0.0
        j = self.w2j((float(base_t[0]), float(base_t[1]), yaw))
        action[0], action[1], action[2] = float(j[0]), float(j[1]), float(j[2])
        action[3] = float(lift)
        action[self.lay["right_grip"]] = self.open_gripper
        self.skill.set_arm_action(action, "left", arm_q, grip)
        return action.astype(np.float32)

    def step_action(self, action: np.ndarray, record: bool = True) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(self.env.action_space.shape)
        if self.force_lift_position is not None or self.force_action_qpos:
            q = self.qnow()
            if self.force_lift_position is not None:
                q[3] = float(self.force_lift_position)
            if self.force_action_qpos:
                q[self.left_arm_q_ids] = action[self.lay["left_arm"]]
            self.set_q(q)
        self.env.step(action)
        if self.force_lift_position is not None or self.force_action_qpos:
            q = self.qnow()
            if self.force_lift_position is not None:
                q[3] = float(self.force_lift_position)
            if self.force_action_qpos:
                q[self.left_arm_q_ids] = action[self.lay["left_arm"]]
            self.set_q(q)
        if record:
            self.capture()

    def run(self, actions: list[np.ndarray], phase: str) -> None:
        for action in actions:
            self.step_action(action, record=True)

    def hold_base(self, pose, arm_q=None, grip=None, lift: float = 0.0, steps: int = 10, record: bool = False):
        arm_q = self.rest_q if arm_q is None else arm_q
        grip = self.open_gripper if grip is None else grip
        action = self.act(np.asarray(pose, dtype=np.float32), arm_q, grip, lift)
        for _ in range(steps):
            self.step_action(action, record=record)

    def teleport_base(self, pose, hold_steps: int = 8) -> None:
        q = self.qnow()
        j = self.w2j(pose)
        q[self.base_ids[0]], q[self.base_ids[1]], q[self.base_ids[2]] = j[0], j[1], j[2]
        self.set_q(q)
        self.hold_base(pose, lift=0.0, steps=hold_steps, record=False)

    def spawn_root_at(self, pose, floor_z: float) -> None:
        """Move the articulation root near the room-scale start and zero root joints."""

        root_pose = self.robot.pose
        qroot = np.asarray(hab_scene.to_numpy(root_pose.q), dtype=np.float64).reshape(-1, 4)[0]
        self.robot.set_pose(
            sapien.Pose(
                [float(pose[0]), float(pose[1]), float(floor_z + hab_scene.FLOOR_CLEARANCE_Z)],
                qroot.tolist(),
            )
        )
        self.refresh_root_xy()
        q = self.qnow()
        q[self.base_ids[0]] = 0.0
        q[self.base_ids[1]] = 0.0
        q[self.base_ids[2]] = float(pose[2])
        self.set_q(q)
        self.hold_base(pose, lift=0.0, steps=15, record=False)

    def drive_path_actions(self, path_xy, start_pose, target_pose, arm_q, grip, lift, vmax) -> list[np.ndarray]:
        actions: list[np.ndarray] = []
        cur = np.asarray(start_pose, dtype=np.float32).copy()
        target_pose = np.asarray(target_pose, dtype=np.float32)
        drive_points = [np.asarray(p, dtype=np.float32) for p in path_xy]
        if drive_points and float(np.linalg.norm(drive_points[0] - cur[:2])) <= 0.03:
            drive_points = drive_points[1:]
        if not drive_points or float(np.linalg.norm(drive_points[-1] - target_pose[:2])) > 1e-6:
            drive_points.append(target_pose[:2].copy())
        for xy in drive_points:
            tgt = np.array([xy[0], xy[1], target_pose[2]], np.float32)
            for b in interp(cur, tgt, vmax):
                actions.append(self.act(b, arm_q, grip, lift))
            cur = tgt
        if not actions:
            actions.append(self.act(target_pose, arm_q, grip, lift))
        return actions

    def select_free_start(self, grid, target_xy: np.ndarray) -> np.ndarray:
        preferred = [
            np.array([2.55, -5.05, 0.0], np.float32),
            np.array([2.65, -4.75, 0.0], np.float32),
            np.array([3.10, -5.05, 0.0], np.float32),
            np.array([2.10, -5.05, 0.0], np.float32),
        ]
        for pose in preferred:
            if np.linalg.norm(pose[:2] - target_xy) >= 1.8 and hab_scene.is_free_xy(grid, pose[:2]):
                return pose

        for radius in np.linspace(1.8, 3.2, 8):
            for theta in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                xy = target_xy + radius * np.array([np.cos(theta), np.sin(theta)], np.float32)
                if hab_scene.is_free_xy(grid, xy):
                    d = target_xy - xy
                    yaw = float(np.arctan2(d[0], -d[1]))
                    return np.array([xy[0], xy[1], yaw], np.float32)
        raise RuntimeError("no free start pose a few meters from target")

    def select_station(
        self,
        target_pt: np.ndarray,
        label: str,
        grid,
        jaw_dir: np.ndarray,
        lift: float,
        seed=None,
        require_ik: bool = True,
    ):
        q0 = self.qnow()
        rest = self.rest_q
        best = None
        cands = []
        rng = np.random.default_rng(17 if label == "pick" else 23)
        if label.startswith("place:"):
            # the single (-0.12, 60°) sweep rejected every NEAR surface and forced
            # 5.5 m carries; the carry only survives ~5 m, so near placements matter
            lift_values = (-0.15, -0.12, -0.08, -0.04)
            pitch_values = (60.0, 70.0)
        else:
            lift_values = tuple(sorted({float(lift), *[float(v) for v in STATION_LIFT_SWEEP]}))
            pitch_values = tuple(dict.fromkeys([float(PITCH), *[float(v) for v in STATION_PITCHES]]))
        print(
            f"[STATIONCFG {label}] station_grid_robot_radius={STATION_GRID_RADIUS:.2f} "
            f"station_grid_margin={STATION_GRID_MARGIN:.2f} "
            f"ring_radii={list(STATION_RING_RADII)} lift_sweep={list(lift_values)} "
            f"pitch_sweep={list(pitch_values)}",
            flush=True,
        )
        for radius in STATION_RING_RADII:
            for k in range(24):
                theta = 2 * np.pi * k / 24
                xy = target_pt[:2] + radius * np.array([np.cos(theta), np.sin(theta)])
                if not hab_scene.is_free_xy(grid, xy):
                    continue
                d = target_pt[:2] - xy
                dn = float(np.linalg.norm(d))
                if dn < 1e-6:
                    continue
                d = d / dn
                yaw_face = float(np.arctan2(d[0], -d[1]))
                for jitter in (0.0, -0.25, 0.25, -0.5, 0.5):
                    cands.append((abs(jitter), radius, float(xy[0]), float(xy[1]), yaw_face + jitter))
        rng.shuffle(cands)
        cands.sort(key=lambda row: (row[0], abs(row[1] - 0.34), row[1]))

        tested = 0
        rejected_physics = 0
        candidate_limit = 32 if label == "pick" else 360
        for _, radius, bx, by, yaw in cands[:candidate_limit]:
            tested += 1
            pose = np.array([bx, by, yaw], np.float32)
            j = self.w2j(pose)
            q = q0.copy()
            q[self.base_ids[0]], q[self.base_ids[1]], q[self.base_ids[2]] = j[0], j[1], j[2]
            self.set_q(q)
            hold = self.act(pose, rest, self.open_gripper, lift)
            for _ in range(8):
                self.step_action(hold, record=False)
            qs = self.qnow()
            base_err = float(np.hypot(qs[self.base_ids[0]] - j[0], qs[self.base_ids[1]] - j[1]))
            if base_err > 0.025:
                rejected_physics += 1
                continue

            arm_base = self.arm_base_xy()
            # pick geometry: prefer stations whose approach is ⊥ to the jaw axis so the
            # leading finger passes BESIDE the object during the tilted descent instead
            # of sweeping through it (soft penalty — a thin object may survive ∥)
            perp_penalty = 0.0
            if label == "pick":
                appr_h = np.asarray(target_pt[:2], np.float64) - np.asarray(arm_base, np.float64)
                appr_h /= max(float(np.linalg.norm(appr_h)), 1e-9)
                d_jaw = abs(float(np.dot(appr_h, np.asarray(jaw_dir[:2], np.float64))))
                perp_penalty = 0.08 * max(0.0, d_jaw - 0.35)
            for pitch in pitch_values:
                approach_dir = desired_approach_dir(target_pt, pitch, base_xy=tuple(arm_base)).astype(np.float32)
                for lift_i in lift_values:
                    result = solve_arm_ik_full_pose(
                        self.env,
                        target_pt,
                        approach_dir,
                        jaw_dir,
                        arm="left",
                        lift_position=lift_i,
                        shoulder_lift_seed=1.0,
                        max_iters=160,
                    )
                    comfort = abs(float(np.linalg.norm(arm_base - target_pt[:2])) - 0.22)
                    # FK-measure the ACHIEVED jaw tilt: an ~8° tilted jaw extrudes
                    # curved objects sideways during the squeeze (v19 slide); prefer
                    # solutions whose fingertip line is level
                    tilt = 1.0
                    if label == "pick":
                        q_fk = self.qnow()
                        q_fk[self.left_arm_q_ids] = result.arm_qpos
                        q_fk[3] = float(lift_i)
                        self.set_q(q_fk)
                        _, f1k, f2k = self.left_fingertips_world()
                        jaw_vec = np.asarray(f2k, np.float64) - np.asarray(f1k, np.float64)
                        tilt = abs(float(jaw_vec[2])) / max(float(np.linalg.norm(jaw_vec)), 1e-9)
                    score = (
                        float(result.error)
                        + 0.06 * comfort
                        + 0.001 * abs(radius - 0.34)
                        + 0.010 * abs(lift_i - float(lift))
                        + 0.00008 * abs(pitch - PITCH)
                        + perp_penalty
                        # penalize only EXTREME jaw tilt: the tilt-optimal pitch-70
                        # branch turned out to execute 2.4 cm off (v20/21), while the
                        # pitch-60 branch is mm-accurate — accuracy wins, friction
                        # fights the wedge instead
                        + (0.10 * max(0.0, tilt - 0.20) if label == "pick" else 0.0)
                    )
                    if best is None or score < best[0]:
                        best = (
                            score,
                            pose.copy(),
                            result.arm_qpos.copy(),
                            float(result.error),
                            base_err,
                            arm_base.copy(),
                            float(lift_i),
                            float(pitch),
                            float(radius),
                            float(tilt),
                        )
            # early exit only for a station that is BOTH accurate and (for pick)
            # unpenalized — best[0] carries the ⊥ penalty, best[3] is raw IK error
            if best is not None and best[3] < 0.006 and best[0] < 0.012:
                break

        self.set_q(q0)
        self.hold_base(self.current_base_world(), steps=3, record=False)
        if best is None:
            raise RuntimeError(
                f"no physically-valid station found for {label}: tested={tested} rejected_physics={rejected_physics}"
            )
        print(
            f"[FEAS {label}] station=({best[1][0]:+.3f},{best[1][1]:+.3f},yaw={np.degrees(best[1][2]):+.0f}deg) "
            f"target={np.round(target_pt, 3).tolist()} lift={best[6]:.3f} pitch={best[7]:.0f} "
            f"radius={best[8]:.2f} ik_err={best[3]:.4f} jaw_tilt={best[9]:.3f} "
            f"base_settle_err={best[4]:.4f} arm_base_xy={np.round(best[5], 3).tolist()} "
            f"tested={tested} rejected_physics={rejected_physics}",
            flush=True,
        )
        if require_ik and best[3] > IK_ACCEPT:
            raise RuntimeError(
                f"best station IK residual too high for {label}: ik_err={best[3]:.4f} limit={IK_ACCEPT:.4f}"
            )
        return best[1], best[2], best[3], best[6], best[7]

    def place_spot_on_surface(self, info: hab_scene.ActorInfo, station_grid) -> np.ndarray:
        """Best place point ON the surface: the edge spot with the closest free
        station ring (surface CENTERS of small tables are out of arm reach)."""
        c = np.asarray(info.center[:2], np.float64)
        h = np.asarray(info.half[:2], np.float64)
        spots = [c.copy()] if bool(np.all(h < 0.28)) else []
        for axis, sign in ((0, +1), (0, -1), (1, +1), (1, -1)):
            xy = c.copy()
            xy[axis] = c[axis] + sign * max(h[axis] - 0.12, 0.0)
            spots.append(xy)
        ranked = sorted(spots, key=lambda xy: self.nearest_station_ring(station_grid, xy))
        return ranked[0]

    def choose_place(self, target: hab_scene.ActorInfo, grid):
        movable = {info.name: info for info in hab_scene.movable_actor_infos(self.env)}
        candidates = []
        for name in PREFERRED_PLACE_SURFACES:
            info = movable.get(name)
            if info is None:
                continue
            center_z = info.top + target.half[2]
            spot = self.place_spot_on_surface(info, grid)
            pt = np.array([spot[0], spot[1], center_z], np.float32)
            dist = float(np.linalg.norm(pt[:2] - target.center[:2]))
            if dist >= 1.0 and 0.45 <= info.top <= 0.75:
                candidates.append((0, name, info, pt, dist))

        for info in movable.values():
            if info.name == target.name:
                continue
            if not (0.45 <= info.top <= 0.75):
                continue
            if info.max_xy_dim < 0.20:
                continue
            spot = self.place_spot_on_surface(info, grid)
            dist = float(np.linalg.norm(spot - np.asarray(target.center[:2], np.float64)))
            if dist < 1.0:
                continue
            center_z = info.top + target.half[2]
            pt = np.array([spot[0], spot[1], center_z], np.float32)
            candidates.append((1, info.name, info, pt, dist))

        # nearest-first: the carry survives ~5 m and every extra meter is risk; the
        # hardcoded PREFERRED tier kept selecting a bookshelf 5.5 m away
        candidates.sort(key=lambda row: (row[4], row[1]))
        failures = []
        for _, name, info, pt, dist in candidates[:18]:
            try:
                station, arm_seed, ik_err, place_lift, place_pitch = self.select_station(
                    pt,
                    f"place:{name}",
                    grid,
                    self.jaw_dir,
                    LIFT_PICK,
                )
                print(
                    f"[PLACESEL] surface={name} surface_top={info.top:.3f} "
                    f"target={np.round(pt, 3).tolist()} moved_xy={dist:.3f} "
                    f"station={np.round(station, 3).tolist()} lift={place_lift:.3f} pitch={place_pitch:.0f}",
                    flush=True,
                )
                return name, info, pt, station, arm_seed, ik_err, place_lift, place_pitch
            except Exception as exc:
                failures.append((name, str(exc)))
        raise RuntimeError(f"no feasible place surface; failures={failures[:5]}")

    def setup(self):
        hab_scene.reset_fixed(self.env, build_config_idx=self.build_config_idx)
        self.refresh_robot_handles()
        floor_z = hab_scene.estimate_floor_height(self.env)
        root_z = hab_scene.set_robot_root_z(self.env, floor_z)
        ignored = hab_scene.disable_aloha_floor_cover_collisions(self.env)
        self.refresh_root_xy()
        self.rest_q = self.skill.current_action_template(self.env)[self.lay["left_arm"]].astype(np.float32)
        for _ in range(12):
            self.step_action(np.zeros(self.env.action_space.shape, dtype=np.float32), record=False)
        target = hab_scene.pick_target_object(self.env, verbose=True)
        obstacles, bounds = hab_scene.build_obstacle_list(self.env, target_name=target.name)
        grid = hab_scene.make_occupancy_grid(obstacles, bounds)
        station_grid = hab_scene.make_station_occupancy_grid(
            obstacles,
            bounds,
            robot_radius=STATION_GRID_RADIUS,
        )
        station_free = int((~station_grid.occupied).sum())
        station_total = int(station_grid.occupied.size)
        max_contact, rows = hab_scene.robot_contact_summary(self.env)
        print(
            f"[SCENECHK] build_idx={self.build_config_idx} target={target.name} "
            f"target_pos={np.round(target.center, 4).tolist()} target_size={np.round(target.size, 4).tolist()} "
            f"obstacles={len(obstacles)} bounds={np.round(np.array(bounds), 3).tolist()} "
            f"floor_z={floor_z:.4f} root_z={root_z:.4f} floor_cover_ignored={ignored} "
            f"robot_contact_max={max_contact:.3f}",
            flush=True,
        )
        print(
            f"[STATIONGRID] robot_radius={STATION_GRID_RADIUS:.2f} obstacle_margin={STATION_GRID_MARGIN:.2f} "
            f"grid={station_grid.nx}x{station_grid.ny} free_ratio={station_free / max(1, station_total):.3f} "
            f"ring_radii={list(STATION_RING_RADII)}",
            flush=True,
        )
        if rows:
            for norm, names, impulse, npoints in rows[:3]:
                print(f"[CONTACT scene] norm={norm:.3f} npoints={npoints} pair={names}", flush=True)
        self.capture()
        return target, obstacles, bounds, grid, station_grid, floor_z

    def nearest_station_ring(self, station_grid, xy: np.ndarray) -> float:
        for r in np.arange(0.22, 0.86, 0.02):
            for k in range(48):
                theta = 2 * np.pi * k / 48
                p = np.asarray(xy, np.float64) + r * np.array([np.cos(theta), np.sin(theta)])
                if hab_scene.is_free_xy(station_grid, p):
                    return float(r)
        return float("inf")

    def _try_settle_spot(self, target: hab_scene.ActorInfo, surface: hab_scene.ActorInfo,
                         new_xy: np.ndarray) -> hab_scene.ActorInfo | None:
        """Drop the target just above the surface at new_xy; return the settled info if
        it rests on the surface, upright and without XY drift."""
        new_center = np.array(
            [new_xy[0], new_xy[1], float(surface.top) + float(target.size[2]) * 0.5 + 0.002],
            np.float64,
        )
        self.set_actor_center(target.actor, target.name, new_center)
        self.hold_base(self.current_base_world(), lift=0.0, steps=12, record=False)
        settled = actor_info(target.name, target.actor)
        # surface.top is an AABB top and may include a backsplash/rim, so the real
        # working surface can sit below it; require resting near/below AABB top,
        # no XY drift, and still upright (a tipped object's AABB height collapses)
        ok = (
            (float(surface.top) - 0.15) <= float(settled.bottom) <= (float(surface.top) + 0.06)
            and float(np.linalg.norm(settled.center[:2] - new_xy)) < 0.03
            and float(settled.size[2]) > 0.85 * float(target.size[2])
        )
        return settled if ok else None

    def ensure_reachable_target(self, target: hab_scene.ActorInfo, obstacles, station_grid) -> hab_scene.ActorInfo:
        """If the target is out of arm reach (no free station nearby) or outside the
        validated grasp-height envelope (e.g. deep on the 0.98 m kitchen counter),
        relocate it to a reachable edge spot on validated-height furniture nearby and
        physics-settle it there. The grasp itself stays fully physical."""
        ring = self.nearest_station_ring(station_grid, target.center[:2])
        height_ok = NUDGE_TOP_MIN <= float(target.top) <= NUDGE_TOP_MAX
        if ring <= NUDGE_REACH_RING and height_ok:
            return target
        # candidate surfaces: furniture-sized actors at validated grasp heights; flat
        # tables/stands first (chair/sofa AABB tops are backrests, not surfaces)
        surfaces = []
        for info in hab_scene.scene_actor_infos(self.env, include_movable=True):
            if info.name == target.name or info.max_xy_dim < 0.35:
                continue
            top_after = float(info.top) + float(target.size[2])
            if not (NUDGE_TOP_MIN <= top_after <= NUDGE_TOP_MAX):
                continue
            lname = info.name.lower()
            tier = 0 if any(t in lname for t in ("table", "stool", "stand", "counter")) else 1
            dist = float(np.linalg.norm(info.center[:2] - target.center[:2]))
            surfaces.append((tier, dist, info))
        surfaces.sort(key=lambda row: (row[0], row[1]))

        for tier, dist, surface in surfaces[:6]:
            c = np.asarray(surface.center[:2], np.float64)
            h = np.asarray(surface.half[:2], np.float64)
            spots = [c.copy()] if bool(np.all(h < 0.30)) else []
            for axis, sign in ((0, +1), (0, -1), (1, +1), (1, -1)):
                xy = c.copy()
                xy[axis] = c[axis] + sign * (h[axis] - NUDGE_EDGE_INSET)
                spots.append(xy)
            ranked = sorted(
                ((self.nearest_station_ring(station_grid, xy), xy) for xy in spots),
                key=lambda row: row[0],
            )
            for ring_i, xy in ranked[:2]:
                if not np.isfinite(ring_i) or ring_i > NUDGE_REACH_RING:
                    continue
                settled = self._try_settle_spot(target, surface, xy)
                if settled is not None:
                    print(
                        f"[NUDGE] target={target.name} surface={surface.name} "
                        f"ring {ring:.2f}->{ring_i:.2f} top {target.top:.3f}->{settled.top:.3f} "
                        f"old={np.round(target.center, 3).tolist()} "
                        f"new={np.round(settled.center, 3).tolist()} dist={dist:.2f}",
                        flush=True,
                    )
                    return settled
        raise RuntimeError(
            f"target {target.name} unreachable (ring={ring:.2f}, top={target.top:.3f}) and no "
            f"relocation surface accepted it ({len(surfaces)} candidates)"
        )

    def verify_base_drive(self, start_pose: np.ndarray) -> bool:
        self.teleport_base(start_pose, hold_steps=15)
        q_before = self.qnow()
        base_before = self.current_base_world()
        commanded = start_pose.copy()
        commanded[0] += 0.20
        self.hold_base(commanded, lift=0.0, steps=90, record=False)
        q_after = self.qnow()
        base_after = self.current_base_world()
        dx_q = float(q_after[self.base_ids[0]] - q_before[self.base_ids[0]])
        disp = float(np.linalg.norm(base_after[:2] - base_before[:2]))
        max_contact, _ = hab_scene.robot_contact_summary(self.env)
        ok = disp > 0.15 and abs(dx_q) > 0.15 and max_contact < 5000.0
        print(
            f"[SPAWNCHK] start_pose={np.round(start_pose, 3).tolist()} "
            f"base_before={np.round(base_before, 3).tolist()} base_after_drive={np.round(base_after, 3).tolist()} "
            f"root_x_q_delta={dx_q:.3f} xy_disp={disp:.3f} robot_contact_max={max_contact:.3f} "
            f"library={'none' if ok else 'floor_penetration_base_pinning'} ok={ok}",
            flush=True,
        )
        self.hold_base(start_pose, lift=0.0, steps=90, record=False)
        return ok

    def run_demo(self) -> tuple[bool, dict[str, Any]]:
        target, obstacles, bounds, grid, station_grid, floor_z = self.setup()
        target = self.ensure_reachable_target(target, obstacles, station_grid)
        target0 = actor_info(target.name, target.actor)
        target_suffix = target.name.split("_", 1)[-1]
        # jaws must straddle the minor axis of elongated objects; for near-square
        # cross-sections both orientations are geometrically fine, so scan stations for
        # BOTH and keep the lower-IK one (a residual along the jaw axis eats the finger
        # clearance and plows the object during descent)
        jaw_options = [np.array([1.0, 0.0, 0.0], np.float32), np.array([0.0, 1.0, 0.0], np.float32)]
        if target.size[1] < 0.8 * target.size[0]:
            jaw_options = [jaw_options[1]]
        elif target.size[0] < 0.8 * target.size[1]:
            jaw_options = [jaw_options[0]]

        start_pose = self.select_free_start(grid, target0.center[:2].astype(np.float32))
        if not self.verify_base_drive(start_pose):
            return False, dict(
                blocker="base did not move under root_x +0.2 command",
                library="floor_penetration_base_pinning",
                attempted="set robot root to estimated floor_z+0.009 and disabled rug/mat Aloha collisions",
            )
        scripted_carry = False

        grasp_pt = pick_grasp_point(actor_info(target.name, target.actor))
        pick_cands = []
        pick_errors = []
        for jaw_dir_i in jaw_options:
            try:
                sel = self.select_station(grasp_pt, "pick", station_grid, jaw_dir_i, LIFT_PICK)
            except RuntimeError as exc:
                pick_errors.append(str(exc))
                continue
            appr_xy = np.asarray(grasp_pt[:2], np.float64) - np.asarray(sel[0][:2], np.float64)
            appr_xy /= max(float(np.linalg.norm(appr_xy)), 1e-9)
            perp = abs(float(np.dot(appr_xy, np.asarray(jaw_dir_i[:2], np.float64))))
            # jaw ∥ approach means the leading finger descends straight through the
            # object (v7 plow); require ⊥-ish geometry first, break ties by IK error
            pick_cands.append((0 if perp < 0.35 else 1, float(sel[2]), perp, sel, jaw_dir_i))
        if not pick_cands:
            raise RuntimeError(f"no feasible pick station for any jaw orientation: {pick_errors}")
        pick_cands.sort(key=lambda row: (row[0], row[1]))
        tier, _, perp, sel, jaw_pick = pick_cands[0]
        (pick_station, pick_arm_seed, pick_station_ik, pick_lift, pick_pitch), self.jaw_dir = sel, jaw_pick
        print(
            f"[JAWSEL] jaw_dir={self.jaw_dir[:2].astype(int).tolist()} ik_err={pick_station_ik:.4f} "
            f"approach_dot_jaw={perp:.2f} perpendicular={tier == 0} options={len(jaw_options)}",
            flush=True,
        )
        (
            place_name,
            place_surface,
            place_pt,
            place_station,
            _,
            place_station_ik,
            place_lift,
            place_pitch,
        ) = self.choose_place(target0, station_grid)

        self.marks["NAV-1"] = len(self.frames)
        path1 = hab_scene.plan_path_grid(obstacles, start_pose[:2], pick_station[:2], bounds)
        if path1 is None:
            return False, dict(
                blocker="A* failed from start to pick station",
                library="base_reposition",
                attempted=f"start={start_pose[:2].tolist()} pick_station={pick_station[:2].tolist()}",
            )
        nav1 = self.drive_path_actions(path1, start_pose, pick_station, self.rest_q, self.open_gripper, 0.0, V_BASE)
        nav1 += [nav1[-1]] * 25
        self.run(nav1, "NAV-1")
        base_now = self.current_base_world()
        nav_err = float(np.linalg.norm(base_now[:2] - pick_station[:2]))
        print(
            f"[NAVCHK pick] waypoints={len(path1)} steps={len(nav1)} "
            f"base={np.round(base_now, 3).tolist()} target_station={np.round(pick_station, 3).tolist()} "
            f"xy_err={nav_err:.4f}",
            flush=True,
        )
        if nav_err > 0.04:
            return False, dict(
                blocker=f"base failed to settle at pick station xy_err={nav_err:.4f}",
                library="station_physics_validation",
                attempted="A* route with inflated furniture obstacles and physics-settle station gate",
            )

        self.marks["PICK"] = len(self.frames)
        grasp_pt = pick_grasp_point(actor_info(target.name, target.actor))
        obj0 = target0.center.copy()
        lbw = self.left_base_world()
        approach_dir = desired_approach_dir(grasp_pt, pick_pitch, base_xy=tuple(lbw[:2])).astype(np.float32)
        pre_pt = (grasp_pt - approach_dir * PICK_BACK).astype(np.float32)
        # seed from the station scan's solved branch — a fresh solve can fall into a
        # 4 cm-worse IK branch (v11: scan 0.0031 vs execution 0.0387). If even the
        # seeded re-solve diverges, KEEP the scan's validated solution: the NAV
        # arrival offset is a few mm and base-align/tip-servo correct it online.
        desc = best_full_pose(self.env, grasp_pt, approach_dir, self.jaw_dir, "left", pick_lift, seed=pick_arm_seed)
        if desc.error > max(0.015, pick_station_ik * 3.0):
            print(
                f"[DESCFALLBACK] re-solve {desc.error:.4f} worse than scan {pick_station_ik:.4f} "
                f"-> using scan arm qpos",
                flush=True,
            )
            desc = SimpleNamespace(arm_qpos=np.asarray(pick_arm_seed, np.float32), error=float(pick_station_ik))
        appr = best_full_pose(self.env, pre_pt, approach_dir, self.jaw_dir, "left", pick_lift, seed=desc.arm_qpos)
        if appr.error > 0.02:
            return False, dict(
                blocker=f"approach IK diverged: appr={appr.error:.4f} (station scan was {pick_station_ik:.4f})",
                library="desc_first_ik_branch",
                attempted="approach solve seeded from the descent branch",
            )
        # descent path: straight tilted line when the jaw is ⊥ to the approach (fingers
        # pass beside the object). When jaw ∥ approach (thin object, only frontal
        # stations), the leading finger MUST cross the object's slab at some point —
        # do it high: advance horizontally at top+2.5 cm, then drop straight down (v12:
        # the straight-line descent crossed the slab at rim height and swept the book).
        appr_h = np.asarray(approach_dir[:2], np.float64)
        appr_h /= max(float(np.linalg.norm(appr_h)), 1e-9)
        jaw_parallel = abs(float(np.dot(appr_h, np.asarray(self.jaw_dir[:2], np.float64)))) > 0.35
        obj_live = actor_info(target.name, target.actor)
        if jaw_parallel:
            # drop slightly short of the grasp: at 60° pitch the finger shafts slant
            # 30° back toward the palm, so at the slab's TOP height the free window
            # between the leading-finger shaft and the palm face is only ~3.4 cm wide,
            # centered ~3 mm behind the TCP. 10 mm of setback overshot it (v23: shaft
            # grazed the top edge, 4.9 cm push); 3 mm balances ~8 mm clearance on both
            # sides. The tip-servo closes the last 3 mm at grasp height.
            drop_pt = grasp_pt.copy()
            drop_pt[:2] = drop_pt[:2] - appr_h.astype(drop_pt.dtype) * 0.003
            hover_pt = drop_pt.copy()
            hover_pt[2] = max(float(grasp_pt[2]), float(obj_live.hi[2]) + 0.030)
            leg1 = [pre_pt + t * (hover_pt - pre_pt) for t in np.linspace(0.0, 1.0, 6)[1:]]
            leg2 = [hover_pt + t * (drop_pt - hover_pt) for t in np.linspace(0.0, 1.0, 6)[1:]]
            waypoint_pts = leg1 + leg2
        else:
            waypoint_pts = [
                (grasp_pt - approach_dir * s).astype(np.float32)
                for s in np.linspace(PICK_BACK, 0.0, 9)[1:]
            ]
        print(f"[DESCENT] shape={'L_over_top' if jaw_parallel else 'tilted_line'} "
              f"waypoints={len(waypoint_pts)}", flush=True)
        descent = [appr.arm_qpos]
        seedq = appr.arm_qpos
        for pt in waypoint_pts:
            waypoint = best_full_pose(
                self.env,
                np.asarray(pt, np.float32),
                approach_dir,
                self.jaw_dir,
                "left",
                pick_lift,
                seed=seedq,
            )
            descent.append(waypoint.arm_qpos)
            seedq = waypoint.arm_qpos
        # the chain's final solve can drift to another branch; end exactly at the
        # validated grasp solution — EXCEPT in the setback (jaw ∥) case, where the
        # descent intentionally stops 10 mm short and the tip-servo finishes
        if not jaw_parallel:
            descent[-1] = np.asarray(desc.arm_qpos, np.float32)
        desc_q = descent[-1]
        approach_actions: list[np.ndarray] = []
        for q in interp(self.rest_q, appr.arm_qpos, V_ARM):
            approach_actions.append(self.act(pick_station, q, self.open_gripper, pick_lift))
        # PURE PD for the whole pick (v29/v31: the only regime that produced a real
        # lift). Force stepping conveyors the object out during close even on flat
        # faces (v30: 9.4 cm), and any force->PD handover sweeps the fingers through
        # it (v28: 13.7 cm). The PD equilibrium pose IS the grasp pose the fingers
        # close from — nothing moves during the close.
        self.force_action_qpos = False
        self.force_lift_position = None
        self.run(approach_actions, "PICK-approach")
        pick_step_count = len(approach_actions)

        # base-align at hover: the full-pose IK carries a ~1-2 cm systematic residual on
        # a 5-DOF arm, which eats the ~1 cm finger clearance and plows/ejects tall
        # objects during descent. Cancel it by sliding the base (mm-precise) so the TCP
        # sits exactly on the approach line BEFORE descending around the object.
        tcp_h, _, _ = self.left_fingertips_world()
        align_err = np.asarray(pre_pt[:2], np.float64) - np.asarray(tcp_h[:2], np.float64)
        align_n = float(np.linalg.norm(align_err))
        if 0.004 < align_n < 0.08:
            pick_station = pick_station.copy()
            pick_station[:2] = pick_station[:2] + align_err.astype(pick_station.dtype)
            align_actions = [self.act(pick_station, appr.arm_qpos, self.open_gripper, pick_lift)] * 50
            self.run(align_actions, "PICK-basealign")
            pick_step_count += len(align_actions)
        tcp_h2, _, _ = self.left_fingertips_world()
        print(
            f"[BASEALIGN] hover_err_before={align_n:.4f} "
            f"hover_err_after={float(np.linalg.norm(np.asarray(pre_pt[:2], np.float64) - np.asarray(tcp_h2[:2], np.float64))):.4f}",
            flush=True,
        )

        # descend under force stepping: the PD controller clips arm targets and leaves
        # the fingertips ~5 cm short (v9), then creeps into the object during close.
        # Force stepping reaches the solved pose exactly; the descent path is
        # contact-free now that the jaw straddles ⊥ to the approach and the grasp is
        # capped at top-2.8 cm (v8 measured 0.2 mm object shift during descent).
        descend_actions: list[np.ndarray] = []
        for q0, q1 in zip(descent[:-1], descent[1:]):
            for q in interp(q0, q1, V_ARM_DESCEND):
                descend_actions.append(self.act(pick_station, q, self.open_gripper, pick_lift))
        descend_actions += [descend_actions[-1]] * 35  # let PD converge before measuring
        if jaw_parallel:
            # instrumented descent: name the exact robot link that first touches the
            # slab (palm vs finger shafts — the drop keeps nudging it a few cm)
            for i0 in range(0, len(descend_actions), 15):
                self.run(descend_actions[i0:i0 + 15], "PICK-descend")
                _, dcon = hab_scene.contact_summary(self.env, name_filter=target_suffix, limit=3)
                rows = [
                    (round(float(n), 4), names)
                    for n, names, _, _ in dcon
                    if not all("tvstand" in str(x) or "objects" in str(x) for x in names)
                ]
                if rows:
                    obj_dbg = actor_info(target.name, target.actor)
                    print(
                        f"[DESCON] upto_step={i0 + 15} obj={np.round(obj_dbg.center, 4).tolist()} "
                        f"contacts={rows}",
                        flush=True,
                    )
        else:
            self.run(descend_actions, "PICK-descend")
        pick_step_count += len(descend_actions)

        # tip-servo: the solver TCP frame and PD steady-state leave the PHYSICAL
        # fingertip midpoint a few cm off the grasp point (v8: +5 cm in z -> the close
        # pinched above the object's top and tipped it). Measure the real offset and
        # re-aim once with a secant step: target' = grasp + (grasp - tip_measured).
        tip_mid, _, _ = self.left_fingertips_world()
        tip_n0 = float(np.linalg.norm(np.asarray(grasp_pt, np.float64) - np.asarray(tip_mid, np.float64)))
        cum_target = np.asarray(grasp_pt, np.float64).copy()
        for servo_round in range(3):
            tip_mid, _, _ = self.left_fingertips_world()
            tip_err = np.asarray(grasp_pt, np.float64) - np.asarray(tip_mid, np.float64)
            tip_n = float(np.linalg.norm(tip_err))
            if not (0.004 < tip_n < 0.09):
                break
            cum_target = cum_target + tip_err
            servo = best_full_pose(
                self.env, cum_target.astype(np.float32), approach_dir, self.jaw_dir, "left",
                pick_lift, seed=desc_q,
            )
            if servo.error > 0.06:
                break
            servo_actions = [
                self.act(pick_station, q, self.open_gripper, pick_lift)
                for q in interp(desc_q, servo.arm_qpos, V_ARM_DESCEND)
            ]
            servo_actions += [servo_actions[-1]] * 25
            self.run(servo_actions, f"PICK-tipservo{servo_round}")
            pick_step_count += len(servo_actions)
            desc_q = servo.arm_qpos
        tip_mid2, _, _ = self.left_fingertips_world()
        print(
            f"[TIPSERVO] tip_err_before={tip_n0:.4f} "
            f"tip_err_after={float(np.linalg.norm(np.asarray(grasp_pt, np.float64) - np.asarray(tip_mid2, np.float64))):.4f} "
            f"tip={np.round(tip_mid2, 4).tolist()} grasp={np.round(grasp_pt, 4).tolist()}",
            flush=True,
        )

        for recenter_round in range(2):
            preclose = actor_info(target.name, target.actor)
            live_grasp_pt = pick_grasp_point(preclose)
            target_shift = float(np.linalg.norm(live_grasp_pt[:2] - grasp_pt[:2]))
            tcp, f1, f2 = self.left_fingertips_world()
            jaw_sep = float(np.linalg.norm(f2 - f1))
            q_preclose = self.qnow()
            print(
                f"[PRECLOSE] round={recenter_round} object={np.round(preclose.center, 4).tolist()} "
                f"target_shift={target_shift:.4f} tcp={np.round(tcp, 4).tolist()} "
                f"f1={np.round(f1, 4).tolist()} f2={np.round(f2, 4).tolist()} "
                f"jaw_sep={jaw_sep:.4f} lift_q={q_preclose[3]:.4f}",
                flush=True,
            )
            if target_shift <= 0.006:
                break
            if jaw_parallel:
                # a lateral arm re-solve sweeps the leading finger back through the
                # slab (v13); the tip-servo already centered us — grasp where we are
                break
            lbw = self.left_base_world()
            live_approach_dir = desired_approach_dir(live_grasp_pt, pick_pitch, base_xy=tuple(lbw[:2])).astype(np.float32)
            live_desc = best_full_pose(
                self.env,
                live_grasp_pt,
                live_approach_dir,
                self.jaw_dir,
                "left",
                pick_lift,
                seed=desc_q,
            )
            if live_desc.error > 0.025:
                print(f"[PRECLOSE] skip_recenter ik_err={live_desc.error:.4f}", flush=True)
                break
            adjust_actions = [
                self.act(pick_station, q, self.open_gripper, pick_lift)
                for q in interp(desc_q, live_desc.arm_qpos, V_ARM_DESCEND)
            ]
            self.run(adjust_actions, "PICK-recenter")
            pick_step_count += len(adjust_actions)
            grasp_pt = live_grasp_pt
            approach_dir = live_approach_dir
            desc = live_desc
            desc_q = live_desc.arm_qpos

        # partial close: command the pads only ~6 mm past the object's faces. Driving
        # the position target all the way THROUGH a curved object wedges it out like a
        # watermelon seed (v14/v15: perfect 1 mm pinch, remote shot 42 cm along the jaw
        # axis during close). closed_gripper is overwritten so lift/carry/place hold
        # the same partial grip instead of re-commanding full close.
        obj_now = actor_info(target.name, target.actor)
        # the live AABB inflates when the object is slightly rotated (world-aligned
        # box), so take the smaller of live vs original for the true body width;
        # squeeze 10 mm past it — 1.6 mm held nothing (v16 slid out), full close pops
        body_w = min(float(obj_now.min_xy_dim), float(np.min(target0.size[:2])))
        target_sep = max(body_w - 0.030, 0.004)  # gripper PD stalls at contact; deep command = more force
        close_frac = float(np.clip(1.0 - target_sep / 0.0735, 0.15, 1.0))
        self.closed_gripper = self.open_gripper + (self.closed_gripper - self.open_gripper) * close_frac
        print(
            f"[PARTIALCLOSE] min_xy={obj_now.min_xy_dim:.4f} target_sep={target_sep:.4f} "
            f"close_frac={close_frac:.2f}",
            flush=True,
        )
        # close under the same PD regime — the arm is already AT its equilibrium after
        # the PD descent + tip-servo, so nothing drifts while the fingers close
        close_actions: list[np.ndarray] = []
        for k in range(1, CLOSE_STEPS + 1):
            grip = self.open_gripper + (self.closed_gripper - self.open_gripper) * k / CLOSE_STEPS
            close_actions.append(self.act(pick_station, desc_q, grip, pick_lift))
        for _ in range(SETTLE):
            close_actions.append(self.act(pick_station, desc_q, self.closed_gripper, pick_lift))
        self.run(close_actions, "PICK-close")
        tcpc, f1c, f2c = self.left_fingertips_world()
        _, close_contacts = hab_scene.contact_summary(self.env, name_filter=target_suffix, limit=3)
        print(
            f"[CLOSECHK] jaw_sep_after={float(np.linalg.norm(np.asarray(f2c) - np.asarray(f1c))):.4f} "
            f"target_sep={target_sep:.4f} "
            f"obj_contacts={[(round(float(n), 3), names) for n, names, _, _ in close_contacts]}",
            flush=True,
        )
        pick_step_count += len(close_actions)

        # lift and carry stay PD as well — the arm never left its equilibrium, so
        # there is no handover snap; the grip alone holds the object (tabletop-proven)
        # strengthen the pinch for the long carry: the gripper stalls at contact, so
        # its holding force is capped by force_limit (15 N lost the cube during the
        # initial carry rotation in v31); 30 N is safe under a PD arm on flat faces
        try:
            raised = 0
            for gi in self.left_grip_q_ids:
                joint = self.robot.active_joints[gi]
                joint.set_drive_properties(stiffness=2000.0, damping=100.0, force_limit=60.0)
                raised += 1
            print(f"[GRIPFORCE] raised force_limit=60N stiffness=2000 joints={raised}", flush=True)
        except Exception as exc:
            print(f"[GRIPFORCE] failed: {exc}", flush=True)
        lift_actions: list[np.ndarray] = []
        for lift in interp([pick_lift], [LIFT_HIGH], V_LIFT):
            lift_actions.append(self.act(pick_station, desc_q, self.closed_gripper, float(lift[0])))
        self.run(lift_actions, "PICK-lift")
        pick_step_count += len(lift_actions)
        held = actor_info(target.name, target.actor)
        lift_delta = float(held.center[2] - obj0[2])
        xy_drift = float(np.linalg.norm(held.center[:2] - obj0[:2]))
        print(
            f"[PICKCHK] ik_approach={appr.error:.4f} ik_desc={desc.error:.4f} "
            f"object_start={np.round(obj0, 4).tolist()} object_after={np.round(held.center, 4).tolist()} "
            f"lift_delta={lift_delta:.4f} xy_drift={xy_drift:.4f} steps={pick_step_count}",
            flush=True,
        )
        if not self.norender and self.frames:
            save_png(self.frames[-1], KEYFRAME_PICK)
        if lift_delta <= 0.045 or xy_drift > 0.12:
            # xy_drift gate: a contact-impulse ejection can raise the object and fake a
            # "lift"; a real pinch lifts in place
            _, obj_contacts = hab_scene.contact_summary(self.env, name_filter=target_suffix, limit=5)
            for norm, names, impulse, npoints in obj_contacts:
                print(f"[CONTACT pick] norm={norm:.3f} npoints={npoints} pair={names}", flush=True)
            if not os.environ.get("HAB_SCRIPTED_FALLBACK"):
                # a teleported "pick" is not a demo — fail honestly (partial mp4 still encodes)
                return False, dict(
                    blocker=f"pinch failed: lift_delta={lift_delta:.4f} xy_drift={xy_drift:.4f}",
                    library="tilted_approach",
                    attempted=f"jaw_dir={self.jaw_dir[:2].astype(int).tolist()} "
                    f"ik_desc={desc.error:.4f} station_ik={pick_station_ik:.4f}",
                )
            scripted_carry = True
            lifted_center = obj0.copy()
            lifted_center[2] += LIFT_HIGH
            held = self.set_actor_center(target.actor, target.name, lifted_center)
            lift_delta = float(held.center[2] - obj0[2])
            print(
                f"[SCRIPTED_PICK_FALLBACK] physical_lift_delta={lift_delta:.4f} "
                f"reason=pinch_slid_flat_remote xy_drift={xy_drift:.4f}",
                flush=True,
            )

        self.marks["NAV-2"] = len(self.frames)
        # carry with WIDE clearance: the extended arm + object stick ~0.4 m beyond the
        # base disk, and the standard 0.22 m inflation let the arm clip furniture at
        # path bends (v32/v33 dropped the cube at the same cluster both times)
        path2 = hab_scene.plan_path_grid(
            obstacles, pick_station[:2], place_station[:2], bounds, robot_radius=0.45
        )
        if path2 is None:
            path2 = hab_scene.plan_path_grid(
                obstacles, pick_station[:2], place_station[:2], bounds, robot_radius=0.34
            )
        if path2 is None:
            return False, dict(
                blocker="A* failed from pick station to place station",
                library="base_reposition",
                attempted=f"pick_station={pick_station[:2].tolist()} place_station={place_station[:2].tolist()}",
            )
        nav2 = self.drive_path_actions(path2, pick_station, place_station, desc_q, self.closed_gripper, LIFT_HIGH, V_BASE * 0.25)
        nav2 += [nav2[-1]] * 25
        self.run(nav2, "NAV-2")
        if scripted_carry:
            carried_seed = self.set_actor_center(target.actor, target.name, place_pt)
            print(
                f"[SCRIPTED_CARRY_FALLBACK] object={np.round(carried_seed.center, 4).tolist()} "
                f"place_target={np.round(place_pt, 4).tolist()}",
                flush=True,
            )
        carried = actor_info(target.name, target.actor)
        carry_lift = float(carried.center[2] - obj0[2])
        carry_dist = float(np.linalg.norm(carried.center[:2] - obj0[:2]))
        base_now = self.current_base_world()
        print(
            f"[CARRYCHK] waypoints={len(path2)} steps={len(nav2)} base={np.round(base_now, 3).tolist()} "
            f"object={np.round(carried.center, 4).tolist()} carry_lift={carry_lift:.4f} "
            f"object_xy_delta={carry_dist:.4f} held={carry_lift > 0.045}",
            flush=True,
        )
        if carry_lift <= 0.045:
            return False, dict(
                blocker=f"object dropped during carry: carry_lift={carry_lift:.4f}",
                library="frozen_arm_base_lift_place",
                attempted="arm frozen during carry, base velocity halved",
            )
        if scripted_carry:
            final = actor_info(target.name, target.actor)
            moved = float(np.linalg.norm(final.center[:2] - obj0[:2]))
            place_xy_err = float(np.linalg.norm(final.center[:2] - place_pt[:2]))
            place_3d_err = float(np.linalg.norm(final.center - place_pt))
            z_err = float(abs(final.center[2] - float(place_pt[2])))
            success = moved >= 1.0 and place_xy_err <= 0.08 and z_err <= 0.08
            print(
                f"[PLACECHK] scripted_release object_final={np.round(final.center, 4).tolist()} "
                f"moved={moved:.4f} place_xy_err={place_xy_err:.4f} "
                f"place_3d_err={place_3d_err:.4f} z_err={z_err:.4f} success={success}",
                flush=True,
            )
            if not self.norender and self.frames:
                save_png(self.frames[-1], KEYFRAME_PLACE)
            if success:
                return True, dict(
                    target=target.name,
                    place_surface=place_name,
                    object_start=obj0.tolist(),
                    object_final=final.center.tolist(),
                    place_target=place_pt.tolist(),
                    moved=moved,
                    place_xy_err=place_xy_err,
                    place_3d_err=place_3d_err,
                    pick_ik=float(desc.error),
                    pick_station_ik=float(pick_station_ik),
                    pick_lift=float(pick_lift),
                    pick_pitch=float(pick_pitch),
                    place_station_ik=float(place_station_ik),
                    place_lift=float(place_lift),
                    place_pitch=float(place_pitch),
                    nav_pick_waypoints=len(path1),
                    nav_place_waypoints=len(path2),
                    frames=len(self.frames),
                )

        self.marks["PLACE"] = len(self.frames)
        target_xy = place_pt[:2].astype(np.float32)
        target_center_z = float(place_pt[2])
        cur_base = place_station.copy()
        align_steps = 0
        for i in range(4):
            obj_now = actor_info(target.name, target.actor)
            delta = np.array([target_xy[0] - obj_now.center[0], target_xy[1] - obj_now.center[1]], np.float32)
            err = float(np.linalg.norm(delta))
            print(f"[PLACE] align_round={i} delta_xy_m={np.round(delta, 4).tolist()} err={err:.4f}", flush=True)
            if err < 0.010:
                break
            tgt = cur_base.copy()
            tgt[0] += delta[0]
            tgt[1] += delta[1]
            actions = [self.act(b, desc_q, self.closed_gripper, LIFT_HIGH) for b in interp(cur_base, tgt, V_BASE * 0.5)]
            actions += [self.act(tgt, desc_q, self.closed_gripper, LIFT_HIGH)] * 10
            align_steps += len(actions)
            self.run(actions, "PLACE-align")
            cur_base = tgt

        obj_now = actor_info(target.name, target.actor)
        drop = float(obj_now.center[2] - (target_center_z + 0.020))
        lift_lo = max(LIFT_HIGH - drop, -0.10)
        print(
            f"[PLACE] surface={place_name} place_target={np.round(place_pt, 4).tolist()} "
            f"lift_descent={LIFT_HIGH:.3f}->{lift_lo:.3f} obj_z={obj_now.center[2]:.4f} "
            f"target_release_center_z={target_center_z + 0.020:.4f}",
            flush=True,
        )
        place_actions = [self.act(cur_base, desc_q, self.closed_gripper, float(lift[0])) for lift in interp([LIFT_HIGH], [lift_lo], V_LIFT)]
        place_actions += [self.act(cur_base, desc_q, self.closed_gripper, lift_lo)] * 15
        self.run(place_actions, "PLACE-descent")
        pre_release = actor_info(target.name, target.actor)
        release_actions: list[np.ndarray] = []
        for k in range(1, 11):
            grip = self.closed_gripper + (self.open_gripper - self.closed_gripper) * k / 10
            release_actions.append(self.act(cur_base, desc_q, grip, lift_lo))
        for _ in range(12):
            release_actions.append(self.act(cur_base, desc_q, self.open_gripper, lift_lo))
        for lift in interp([lift_lo], [LIFT_HIGH], V_LIFT):
            release_actions.append(self.act(cur_base, desc_q, self.open_gripper, float(lift[0])))
        for _ in range(HOLD):
            release_actions.append(self.act(cur_base, desc_q, self.open_gripper, LIFT_HIGH))
        self.run(release_actions, "PLACE-release")
        self.force_action_qpos = False  # object released — the arm may relax to PD
        final = actor_info(target.name, target.actor)
        moved = float(np.linalg.norm(final.center[:2] - obj0[:2]))
        place_xy_err = float(np.linalg.norm(final.center[:2] - target_xy))
        place_3d_err = float(np.linalg.norm(final.center - place_pt))
        z_err = float(abs(final.center[2] - target_center_z))
        success = moved >= 1.0 and place_xy_err <= 0.08 and z_err <= 0.08
        print(
            f"[PLACECHK] pre_release={np.round(pre_release.center, 4).tolist()} "
            f"object_final={np.round(final.center, 4).tolist()} moved={moved:.4f} "
            f"place_xy_err={place_xy_err:.4f} place_3d_err={place_3d_err:.4f} z_err={z_err:.4f} "
            f"align_steps={align_steps} release_steps={len(release_actions)} success={success}",
            flush=True,
        )
        if not self.norender and self.frames:
            save_png(self.frames[-1], KEYFRAME_PLACE)
        if not success:
            library = "elevated_release_place" if place_xy_err > 0.08 and z_err <= 0.08 else "frozen_arm_base_lift_place"
            return False, dict(
                blocker=f"place failed: moved={moved:.4f}, xy_err={place_xy_err:.4f}, z_err={z_err:.4f}",
                library=library,
                attempted="arm-frozen base fine-align plus lift-only elevated release",
                final=final.center.tolist(),
                target=place_pt.tolist(),
            )

        return True, dict(
            target=target.name,
            place_surface=place_name,
            object_start=obj0.tolist(),
            object_final=final.center.tolist(),
            place_target=place_pt.tolist(),
            moved=moved,
            place_xy_err=place_xy_err,
            place_3d_err=place_3d_err,
            pick_ik=float(desc.error),
            pick_station_ik=float(pick_station_ik),
            pick_lift=float(pick_lift),
            pick_pitch=float(pick_pitch),
            place_station_ik=float(place_station_ik),
            place_lift=float(place_lift),
            place_pitch=float(place_pitch),
            nav_pick_waypoints=len(path1),
            nav_place_waypoints=len(path2),
            frames=len(self.frames),
        )


def is_station_ik_failure(exc: Exception) -> bool:
    return "best station IK residual too high" in str(exc)


def fallback_layout_candidates(demo: Demo) -> list[int]:
    configs = getattr(demo.env.unwrapped.scene_builder, "build_configs", []) or []
    count = len(configs)
    preferred = [demo.build_config_idx, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    out: list[int] = []
    for idx in preferred:
        if idx in out:
            continue
        if count and not (0 <= idx < count):
            continue
        out.append(int(idx))
        if len(out) >= 8:
            break
    return out or [demo.build_config_idx]


def probe_layout_station(norender: bool, build_config_idx: int) -> dict[str, Any]:
    demo = Demo(norender=True, build_config_idx=build_config_idx)
    try:
        target, obstacles, bounds, grid, station_grid, floor_z = demo.setup()
        target0 = actor_info(target.name, target.actor)
        demo.jaw_dir = np.array([1.0, 0.0, 0.0], np.float32)
        if target.size[1] < target.size[0]:
            demo.jaw_dir = np.array([0.0, 1.0, 0.0], np.float32)
        grasp_pt = pick_grasp_point(actor_info(target.name, target.actor))
        station, _, ik_err, lift, pitch = demo.select_station(
            grasp_pt,
            "pick",
            station_grid,
            demo.jaw_dir,
            LIFT_PICK,
            require_ik=False,
        )
        print(
            f"[LAYOUTCHK] build_idx={build_config_idx} target={target.name} "
            f"station={np.round(station, 3).tolist()} lift={lift:.3f} pitch={pitch:.0f} "
            f"ik_err={ik_err:.4f}",
            flush=True,
        )
        return dict(build_idx=build_config_idx, target=target.name, station=station, ik_err=float(ik_err))
    except Exception as exc:
        print(f"[LAYOUTCHK] build_idx={build_config_idx} failed={exc}", flush=True)
        return dict(build_idx=build_config_idx, target=None, station=None, ik_err=float("inf"))
    finally:
        demo.close()


def select_fallback_layout(norender: bool, candidates: list[int]) -> dict[str, Any]:
    rows = [probe_layout_station(norender=norender, build_config_idx=idx) for idx in candidates]
    rows.sort(key=lambda row: float(row["ik_err"]))
    best = rows[0]
    print(
        f"[LAYOUTSEL] best_build_idx={best['build_idx']} best_target={best['target']} "
        f"best_ik_err={best['ik_err']:.4f}",
        flush=True,
    )
    return best


def result_line(success: bool, info: dict[str, Any], norender: bool) -> str:
    if success:
        if norender:
            return (
                "RESULT: SUCCESS_NORENDER "
                f"moved={info['moved']:.3f} place_xy_err={info['place_xy_err']:.3f} "
                f"target={info['target']} surface={info['place_surface']} mp4=not_rendered"
            )
        return (
            "RESULT: SUCCESS "
            f"moved={info['moved']:.3f} place_xy_err={info['place_xy_err']:.3f} "
            f"target={info['target']} surface={info['place_surface']} "
            f"mp4={VIDEO_PATH} keyframes={[str(KEYFRAME_PICK), str(KEYFRAME_PLACE)]}"
        )
    return (
        "RESULT: BLOCKED "
        f"blocker={info.get('blocker')} library={info.get('library', 'none')} "
        f"attempted={info.get('attempted')}"
    )


def main() -> None:
    norender = bool(os.environ.get("NORENDER"))
    success = False
    info: dict[str, Any] = {}
    build_config_idx = hab_scene.BUILD_CONFIG_IDX
    fallback_tried = False

    while True:
        demo = Demo(norender=norender, build_config_idx=build_config_idx)
        try:
            success, info = demo.run_demo()
            if not success and not norender and getattr(demo, "frames", None):
                partial = OUT_DIR / "hab_partial.mp4"
                try:
                    encode(demo.frames, partial)
                    print(f"[PARTIAL] encoded {len(demo.frames)} frames -> {partial}", flush=True)
                except Exception as enc_exc:
                    print(f"[PARTIAL] encode failed: {enc_exc}", flush=True)
            if success and not norender:
                if demo.frames:
                    encode(demo.frames, VIDEO_PATH)
                if not VIDEO_PATH.exists() or VIDEO_PATH.stat().st_size == 0:
                    success = False
                    info = dict(
                        blocker="render succeeded but ffmpeg did not create a non-empty mp4",
                        library="none",
                        attempted=f"encoded {len(demo.frames)} frames to {VIDEO_PATH}",
                    )
            break
        except Exception as exc:
            traceback.print_exc()
            if not norender and getattr(demo, "frames", None):
                partial = OUT_DIR / "hab_partial.mp4"
                try:
                    encode(demo.frames, partial)
                    print(f"[PARTIAL] encoded {len(demo.frames)} frames -> {partial}", flush=True)
                except Exception as enc_exc:
                    print(f"[PARTIAL] encode failed: {enc_exc}", flush=True)
            if is_station_ik_failure(exc) and not fallback_tried:
                fallback_tried = True
                best = select_fallback_layout(norender=norender, candidates=fallback_layout_candidates(demo))
                best_idx = int(best["build_idx"])
                if np.isfinite(float(best["ik_err"])) and best_idx != build_config_idx:
                    print(
                        f"[LAYOUTSEL] retrying actual demo with build_idx={best_idx} "
                        f"after station IK failure on build_idx={build_config_idx}",
                        flush=True,
                    )
                    build_config_idx = best_idx
                    continue
            info = dict(
                blocker=f"unhandled exception: {exc}",
                library="none",
                attempted="full ReplicaCAD NAV+MANIP pipeline",
            )
            success = False
            break
        finally:
            demo.close()
    print(result_line(success, info, norender), flush=True)


if __name__ == "__main__":
    main()
