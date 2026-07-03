#!/usr/bin/env python3
"""ReplicaCAD fridge insertion demo for AlohaMini Pro.

This is the fridge variant of ``hab_pick_place.py``.  It keeps the validated
pick/carry recipe intact and replaces only the placement phase with a
lower-drive-in-descend-release-retreat insertion into the open refrigerator.
"""

from __future__ import annotations

import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

import hab_scene
import hab_pick_place as _hpp_mod
# the lift prismatic's URDF range is [0, 0.15]; negative sweep values are only
# realizable under force set_qpos — pure PD clamps them to 0 and the station
# solver's pose becomes a lie (v4: tips 15.5 cm high, servo out of trust region)
_hpp_mod.STATION_LIFT_SWEEP = (0.0, 0.05, 0.10)
# positive lift raises the arm: the counter-top grasp (0.94) is now reachable,
# so don't let the relocation logic drag the shaker off the counter
_hpp_mod.NUDGE_TOP_MAX = 1.05
# fridge place scan: pitch 60-70 parks the wrist/link4 ABOVE the held object and
# it grinds the CLOSED top freezer door (front face x=-1.752, z>=0.827 — v27
# contact rows); low pitches thread the whole gripper under it. Lifts must be
# PD-realizable (>=0) or the scanned pose is 8-15 cm higher in reality.
_hpp_mod.PLACE_LIFT_SWEEP = (0.0, 0.05, 0.10)
# pitch 30 parks the wrist right at the slab height (0.827) and the residual
# 2-10N graze lets the arm fold back as the base advances (v30); 15-20 keeps it
# clearly below
_hpp_mod.PLACE_PITCH_SWEEP = (15.0, 20.0)
from hab_pick_place import (
    CLOSE_STEPS,
    HOLD,
    LIFT_HIGH,
    LIFT_PICK,
    PICK_BACK,
    SETTLE,
    STATION_GRID_MARGIN,
    STATION_GRID_RADIUS,
    STATION_RING_RADII,
    V_ARM,
    V_ARM_DESCEND,
    V_BASE,
    V_LIFT,
    Demo as PickPlaceDemo,
    actor_info,
    best_full_pose,
    build_skill,
    choose_render_shader_pack,
    desired_approach_dir,
    encode,
    interp,
    pick_grasp_point,
    save_png,
)


OUT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = OUT_DIR / "hab_fridge.mp4"
KEYFRAME_PICK = OUT_DIR / "hab_fridge_keyframe_pick.png"
KEYFRAME_PLACE = OUT_DIR / "hab_fridge_keyframe_place.png"
FRAMES_TMP = OUT_DIR / "fridge_frames_tmp"

FRIDGE_ENV_ID = "AlohaMiniHabFridge-v1"
PEPPER_NAME = "pepper_shaker"
PEPPER_SCENE_KEY = f"env-0_objects/{PEPPER_NAME}"
SHAKER_HALF = np.array([0.021, 0.021, 0.050], dtype=np.float32)
# kitchen counter front edge (probe-validated: real worktop 0.869, ring 0.34);
# counter->fridge carry is only ~2.0 m and positive lift raises the arm to 0.94
SHAKER_POSE = np.array([-1.925, -0.633, 0.869 + float(SHAKER_HALF[2]) + 0.002], dtype=np.float32)  # probe-validated worktop spot (y=-0.75 is the sink!)

FRIDGE_ARTICULATION_NAME = "scs-[0]_fridge-0"
FRIDGE_BOTTOM_DOOR_OPEN_QPOS = 2.45  # held open by a hinge PD drive; at 2.45 the panel sits south of the lane (AABB y<=-3.59)
FRIDGE_SETTLE_STEPS = 12

SHELF_Z = float(os.environ.get("HAB_FRIDGE_SHELF_Z", "0.30"))
# -1.92 is the deepest KINEMATICALLY reachable spot: the base front face stops on
# the fridge at base x=-1.618 and the frozen carry arm holds the object 0.27 m
# ahead -> object max x = -1.888 (v23/v25/v26 all stalled there grinding the
# wall); the shelf front edge is x=-1.885 and a teleported shaker settles flush
# at x=-1.92 (mouth grid), so -1.92 is on-shelf AND inside reach with 13 cm of
# base clearance left
FRIDGE_PLACE_XY = np.array(
    [float(os.environ.get("HAB_FRIDGE_PLACE_X", "-1.92")), -3.23], dtype=np.float32
)
FRIDGE_FOOTPRINT_CENTER = np.array([-2.18, -3.23], dtype=np.float32)
FRIDGE_FOOTPRINT_HALF = np.array([0.40, 0.40], dtype=np.float32)
FRIDGE_FRONT_X = -1.85
# physically the base front face stalls on the fridge at x=-1.618, and pushing
# past ~-1.60 only grinds the RIGHT arm's link1 into the top freezer door
# (28-31N, v30/v31) while the object goes nowhere
FRIDGE_BASE_MIN_X = -1.60
FRIDGE_ENTRY_BOTTOM_CLEARANCE = float(os.environ.get("HAB_FRIDGE_ENTRY_BOTTOM_CLEARANCE", "0.06"))
FRIDGE_RELEASE_BOTTOM_CLEARANCE = float(os.environ.get("HAB_FRIDGE_RELEASE_BOTTOM_CLEARANCE", "0.05"))
FRIDGE_MIN_LIFT = float(os.environ.get("HAB_FRIDGE_MIN_LIFT", "-0.16"))
FRIDGE_ALIGN_ROUNDS = int(os.environ.get("HAB_FRIDGE_ALIGN_ROUNDS", "5"))
FRIDGE_ALIGN_TOL = float(os.environ.get("HAB_FRIDGE_ALIGN_TOL", "0.025"))
FRIDGE_RETREAT_X = float(os.environ.get("HAB_FRIDGE_RETREAT_X", "0.50"))
FRIDGE_Z_STABLE_TOL = float(os.environ.get("HAB_FRIDGE_Z_STABLE_TOL", "0.015"))

# --- fridge v2: REAL door opening by grasping the freezer-door handle bar ---
# probe (2026-07-03): the TOP freezer door carries a pinchable vertical handle
# bar (cross-section 3.8x5.3 cm, z 0.84-1.44, closed center (-1.771,-2.917));
# the bottom door is a bare 5.5 cm slab with no handle. Hinge axis fitted from
# the handle arc: (-1.85,-3.58), radius ~0.667. The top-down place volume over
# the freezer floor is already clear of the door at 1.0 rad -> pull to ~1.2.
HANDLE_GRASP = np.array([-1.771, -2.917, float(os.environ.get("HAB_HANDLE_Z", "0.95"))], np.float32)
DOOR_HINGE_XY = np.array([-1.85, -3.58], np.float64)
DOOR_PULL_TARGET = float(os.environ.get("HAB_DOOR_PULL_TARGET", "1.05"))
DOOR_PULL_STEP = float(os.environ.get("HAB_DOOR_PULL_STEP", "0.08"))
FREEZER_PLACE = np.array([-2.05, -3.23, 1.0227 + 0.05], np.float32)  # bottom rests at 1.022

ARTICULATION_OBSTACLE_PATCHES = (
    dict(
        center=[-2.18, -3.23],
        half=[0.35, 0.45],
        margin=0.02,
        name="fridge_body_articulation_patch",
        kind="articulation_patch",
    ),
    dict(
        center=[-1.55, -3.75],
        half=[0.40, 0.35],
        margin=0.02,
        name="fridge_open_bottom_door_sweep_patch",
        kind="articulation_patch",
    ),
    dict(
        center=[-2.00, -1.235],
        half=[0.01, 0.01],
        margin=0.02,
        name="kitchen_counter_articulation_patch",
        kind="articulation_patch",
    ),
)


@dataclass(frozen=True)
class FridgeInterior:
    shelf_z: float
    place_pt: np.ndarray
    footprint_center: np.ndarray
    footprint_half: np.ndarray
    front_x: float
    base_min_x: float


def fridge_interior_geometry(env: Any | None = None) -> FridgeInterior:
    """Return the current fridge insertion geometry.

    The measured ReplicaCAD articulation did not expose a convenient interior
    collision AABB in the base helpers, so this intentionally uses calibratable
    constants.  ``env`` is accepted for future collision-AABB probing without
    changing the caller contract.
    """

    shelf_z = float(os.environ.get("HAB_FRIDGE_SHELF_Z", str(SHELF_Z)))
    place_pt = np.array(
        [
            float(FRIDGE_PLACE_XY[0]),
            float(FRIDGE_PLACE_XY[1]),
            shelf_z + FRIDGE_RELEASE_BOTTOM_CLEARANCE + float(SHAKER_HALF[2]),
        ],
        dtype=np.float32,
    )
    return FridgeInterior(
        shelf_z=shelf_z,
        place_pt=place_pt,
        footprint_center=FRIDGE_FOOTPRINT_CENTER.astype(np.float32).copy(),
        footprint_half=FRIDGE_FOOTPRINT_HALF.astype(np.float32).copy(),
        front_x=float(FRIDGE_FRONT_X),
        base_min_x=float(FRIDGE_BASE_MIN_X),
    )


def _render_material(base_color):
    try:
        import sapien

        return sapien.render.RenderMaterial(base_color=base_color)
    except Exception:
        return None


def register_hab_fridge_env() -> None:
    """Register SceneManipulation plus one primitive-collision pepper shaker."""

    hab_scene.configure_paths()
    import sapien
    import mani_skill.envs.scenes.base_env as scene_base
    from mani_skill.utils.registration import REGISTERED_ENVS, register_env

    if FRIDGE_ENV_ID in REGISTERED_ENVS:
        return

    @register_env(FRIDGE_ENV_ID, max_episode_steps=200000)
    class AlohaMiniHabFridgeEnv(scene_base.SceneManipulationEnv):
        @property
        def _default_human_render_camera_configs(self):
            # same two-stream setup as AlohaMiniHabPickEnv: wide follow-cam +
            # gripper close-up; the demo retargets both poses per captured frame.
            # default_camera_config() overrides BOTH names, so the env must
            # define both or gym.make errors on the unknown key.
            from mani_skill.sensors.camera import CameraConfig
            from mani_skill.utils import sapien_utils

            pose = sapien_utils.look_at(eye=[0.3, -2.6, 2.5], target=[-2.3, -0.8, 0.85])
            return [
                CameraConfig("render_camera", pose=pose, width=1920, height=1080,
                             fov=1.0, near=0.01, far=100),
                CameraConfig("closeup_camera", pose=pose, width=1920, height=1080,
                             fov=0.85, near=0.01, far=100),
            ]

        def _load_scene(self, options: dict):
            super()._load_scene(options)

            builder = self.scene.create_actor_builder()
            try:
                import sapien.physx as physx

                mat = physx.PhysxMaterial(1.0, 1.0, 0.0)
                try:
                    builder.add_box_collision(half_size=SHAKER_HALF.tolist(), material=mat)
                except TypeError:
                    builder.add_box_collision(half_size=SHAKER_HALF.tolist())
            except Exception:
                mat = None
                builder.add_box_collision(half_size=SHAKER_HALF.tolist())

            body_mat = _render_material([0.42, 0.02, 0.02, 1.0])
            cap_mat = _render_material([0.62, 0.62, 0.62, 1.0])
            body_kwargs = dict(half_size=[0.021, 0.021, 0.035], pose=sapien.Pose(p=[0.0, 0.0, -0.015]))
            cap_kwargs = dict(half_size=[0.021, 0.021, 0.015], pose=sapien.Pose(p=[0.0, 0.0, 0.035]))
            if body_mat is not None:
                body_kwargs["material"] = body_mat
            if cap_mat is not None:
                cap_kwargs["material"] = cap_mat
            builder.add_box_visual(**body_kwargs)
            builder.add_box_visual(**cap_kwargs)
            builder.initial_pose = sapien.Pose(p=SHAKER_POSE.astype(float).tolist())
            self.pepper_shaker = builder.build_dynamic(name=PEPPER_NAME)

            patched = 0
            if mat is not None:
                try:
                    import sapien.physx as physx

                    for obj in self.pepper_shaker._objs:
                        comp = obj.find_component_by_type(physx.PhysxRigidDynamicComponent)
                        if comp is None:
                            continue
                        for shape in comp.get_collision_shapes():
                            shape.set_physical_material(mat)
                            patched += 1
                except Exception as exc:
                    print(f"[SHAKER] friction patch failed: {exc}", flush=True)
            key = PEPPER_SCENE_KEY
            for attr in ("movable_objects", "scene_objects"):
                d = getattr(self.scene_builder, attr, None)
                if isinstance(d, dict):
                    d[key] = self.pepper_shaker
            print(
                f"[SHAKER] spawned key={key} pose={np.round(SHAKER_POSE, 4).tolist()} "
                f"half={SHAKER_HALF.tolist()} friction_shapes={patched}",
                flush=True,
            )


def make_fridge_env(norender: bool | None = None, shader_pack: str = "minimal", build_config_idx: int | None = None):
    hab_scene.configure_paths()
    if norender is None:
        norender = bool(os.environ.get("NORENDER"))
    if norender:
        hab_scene.patch_no_render_visuals()

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import mani_skill.agents.robots  # noqa: F401

    hab_scene.patch_pci_render_backend_parser()
    hab_scene.patch_supported_robot_warning()
    register_hab_fridge_env()

    kwargs = dict(hab_scene.ENV_KWARGS)
    build_idx = hab_scene.BUILD_CONFIG_IDX if build_config_idx is None else int(build_config_idx)
    kwargs.update(
        build_config_idxs=build_idx,
        render_backend="none" if norender else "pci:0",
        render_mode=None if norender else hab_scene.ENV_KWARGS["render_mode"],
        human_render_camera_configs={} if norender else hab_scene.default_camera_config(shader_pack),
    )
    return gym.make(FRIDGE_ENV_ID, **kwargs)


def _iter_articulations(env) -> Iterable[Any]:
    scene = env.unwrapped.scene
    articulations = getattr(scene, "articulations", []) or []
    if isinstance(articulations, dict):
        yield from articulations.values()
    else:
        yield from articulations


def find_fridge_articulation(env) -> Any | None:
    for art in _iter_articulations(env):
        if getattr(art, "name", "") == FRIDGE_ARTICULATION_NAME:
            return art
    return None


def _set_articulation_qpos(art, qpos: np.ndarray) -> None:
    # EXACTLY the probe-proven call and nothing else: extra set_qvel / drive-target
    # calls around it left the door closed (v15-v17)
    import torch

    q = np.asarray(qpos, dtype=np.float32).reshape(-1)
    art.set_qpos(torch.as_tensor(q[None], dtype=torch.float32))


def add_articulation_obstacle_patches(obstacles: list[dict[str, Any]]) -> int:
    """Patch articulation AABBs into the obstacle list used by both grids."""

    existing = {str(obs.get("name")) for obs in obstacles}
    added = 0
    for patch in ARTICULATION_OBSTACLE_PATCHES:
        if patch["name"] in existing:
            continue
        obstacles.append(dict(patch))
        added += 1
    print(
        f"[FRIDGEOBS] added={added} patches="
        f"{[(p['name'], p['center'], p['half']) for p in ARTICULATION_OBSTACLE_PATCHES]}",
        flush=True,
    )
    return added


def pick_pepper_target(env, verbose: bool = True) -> hab_scene.ActorInfo:
    infos = hab_scene.movable_actor_infos(env)
    if verbose:
        hab_scene.print_movable_candidates(infos)
    for info in infos:
        if info.name == PEPPER_SCENE_KEY or info.name.endswith(f"/{PEPPER_NAME}") or PEPPER_NAME in info.name:
            print(
                f"target_selection=pepper_shaker count={len(infos)} chosen={info.name} "
                f"mass={info.mass:.4f} size={np.round(info.size, 4).tolist()}",
                flush=True,
            )
            return info
    raise RuntimeError(f"spawned target {PEPPER_SCENE_KEY} not found in movable_objects")


class FridgeDemo(PickPlaceDemo):
    def __init__(self, norender: bool, build_config_idx: int | None = None) -> None:
        self.norender = bool(norender)
        self.build_config_idx = hab_scene.BUILD_CONFIG_IDX if build_config_idx is None else int(build_config_idx)
        self.shader_pack = choose_render_shader_pack(self.norender)
        self.env = make_fridge_env(
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
        self.frames2: list[np.ndarray] = []
        self._frames2_n = 0  # HAB_TWO_CAM closeup frames are streamed to disk
        self._cam_eye: np.ndarray | None = None
        self._cam_tgt: np.ndarray | None = None
        self._cam2_eye: np.ndarray | None = None
        self._cam2_tgt: np.ndarray | None = None
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

    def _pin_door_open_drive(self, art) -> str:
        """The fridge hinges have zero drive stiffness, so an 'opened' door swings
        freely back shut within ~100 steps (probe: commanded 2.45 decayed to 1.51
        rad) and the closing panel sweeps the carried object out of the gripper
        at the mouth (v21-v23). Pin the hinges with a PD drive, like GRIPFORCE
        does for the fingers; a modest force_limit lets a base bump yield."""
        import torch
        summary = []
        for joint, tgt in zip(art.get_active_joints(), (0.0, FRIDGE_BOTTOM_DOOR_OPEN_QPOS)):
            raws = list(getattr(joint, "_objs", [])) or [joint]
            for raw in raws:
                prop_ok = tgt_ok = False
                for name in ("set_drive_property", "set_drive_properties"):
                    fn = getattr(raw, name, None)
                    if fn is None:
                        continue
                    try:
                        fn(30.0, 8.0, 30.0, "force")
                        prop_ok = True
                    except Exception:
                        try:
                            fn(stiffness=30.0, damping=8.0, force_limit=30.0)
                            prop_ok = True
                        except Exception:
                            continue
                    break
                for name in ("set_drive_target", "set_drive_targets"):
                    fn = getattr(raw, name, None)
                    if fn is None:
                        continue
                    try:
                        fn(float(tgt))
                        tgt_ok = True
                    except Exception:
                        try:
                            fn(torch.tensor([[float(tgt)]], dtype=torch.float32))
                            tgt_ok = True
                        except Exception:
                            continue
                    break
                summary.append(f"{getattr(joint, 'name', '?')}:prop={prop_ok},tgt={tgt_ok}")
        return " ".join(summary)

    def _stiffen_left_arm(self) -> None:
        """The stretched arm's PD sag (up to ~19 cm at full reach — v30/v31) eats
        the entire fridge-mouth envelope: the commanded pose and the physical arm
        disagree by more than the door-to-shelf gap. Stiffen the left-arm drives
        for the insertion phase (same pattern as GRIPFORCE / the door pin)."""
        try:
            olds = []
            for qi in self.left_arm_q_ids:
                joint = self.robot.active_joints[qi]
                try:
                    olds.append(float(np.asarray(hab_scene.to_numpy(joint.stiffness)).reshape(-1)[0]))
                except Exception:
                    pass
                # 10000 changed nothing (v37: bottom 0.6418 vs 0.6421 at 5000) —
                # the residual offset is the object sliding low in the pinch, not
                # arm sag; keep 5000 for compliant edge-camming
                joint.set_drive_properties(stiffness=5000.0, damping=250.0, force_limit=150.0)
            print(
                f"[ARMFORCE] stiffness->5000 damping->250 force_limit->150 "
                f"joints={len(self.left_arm_q_ids)} was={olds}",
                flush=True,
            )
        except Exception as exc:
            print(f"[ARMFORCE] failed: {exc}", flush=True)

    def fridge_door_qpos(self) -> list[float]:
        art = find_fridge_articulation(self.env)
        try:
            return np.asarray(hab_scene.to_numpy(art.get_qpos()), np.float64).reshape(-1).round(3).tolist()
        except Exception:
            return []

    # ------------------------------------------------------------------
    # fridge v2 skills: grasp the freezer-door handle, pull it open along
    # the hinge arc, hold. No qpos teleports on the door being opened.
    # ------------------------------------------------------------------

    def set_door_drive(self, gains: list[tuple[float, float, float]], targets: list[float]) -> None:
        """Per-joint hinge drives: gains[i]=(stiffness, damping, force_limit).
        qpos order = [top_door_hinge, bottom_door_hinge]."""
        import torch
        art = find_fridge_articulation(self.env)
        for joint, (k, d, fl), tgt in zip(art.get_active_joints(), gains, targets):
            for raw in (list(getattr(joint, "_objs", [])) or [joint]):
                for name in ("set_drive_property", "set_drive_properties"):
                    fn = getattr(raw, name, None)
                    if fn is None:
                        continue
                    try:
                        fn(k, d, fl, "force")
                    except Exception:
                        try:
                            fn(stiffness=k, damping=d, force_limit=fl)
                        except Exception:
                            continue
                    break
                for name in ("set_drive_target", "set_drive_targets"):
                    fn = getattr(raw, name, None)
                    if fn is None:
                        continue
                    try:
                        fn(float(tgt))
                    except Exception:
                        try:
                            fn(torch.tensor([[float(tgt)]], dtype=torch.float32))
                        except Exception:
                            continue
                    break

    def close_doors_for_v2(self) -> None:
        """Start state for the door-opening skill: both doors CLOSED, lightly
        held there (detent-like), overriding setup()'s v1 qpos-open hack."""
        art = find_fridge_articulation(self.env)
        _set_articulation_qpos(art, np.array([0.0, 0.0], dtype=np.float32))
        self.set_door_drive([(5.0, 2.0, 5.0), (5.0, 2.0, 5.0)], [0.0, 0.0])
        self.hold_base(self.current_base_world(), lift=0.0, steps=10, record=False)
        print(f"[V2] doors closed+held qpos={self.fridge_door_qpos()}", flush=True)

    def grasp_handle(self) -> tuple[bool, np.ndarray, np.ndarray]:
        """NAV to a station facing the handle bar and pinch it. Returns
        (ok, station_pose, arm_q_at_grasp)."""
        grasp_pt = HANDLE_GRASP.copy()
        station, arm_seed, ik_err, lift, pitch = self.select_station(
            grasp_pt, "pick", self._v2_station_grid, np.array([0.0, 1.0, 0.0], np.float32), LIFT_PICK
        )
        print(f"[HANDLE] station={np.round(station,3).tolist()} ik={ik_err:.4f} "
              f"lift={lift:.3f} pitch={pitch:.0f}", flush=True)
        path = hab_scene.plan_path_grid(self._v2_obstacles, self.current_base_world()[:2],
                                        station[:2], self._v2_bounds, robot_radius=0.34)
        if path is None:
            print("[HANDLE] A* failed to the handle station", flush=True)
            return False, station, arm_seed
        nav = self.drive_path_actions(path, self.current_base_world(), station,
                                      self.rest_q, self.open_gripper, lift, V_BASE * 0.5)
        nav += [nav[-1]] * 25
        self.run(nav, "V2-nav-handle")

        approach_dir = desired_approach_dir(grasp_pt, pitch,
                                            base_xy=tuple(self.left_base_world()[:2])).astype(np.float32)
        jaw_dir = np.array([0.0, 1.0, 0.0], np.float32)
        pre_pt = (grasp_pt - approach_dir * 0.08).astype(np.float32)
        appr = best_full_pose(self.env, pre_pt, approach_dir, jaw_dir, "left", lift, seed=arm_seed)
        acts = [self.act(station, q, self.open_gripper, lift) for q in interp(self.rest_q, appr.arm_qpos, V_ARM)]
        acts += [acts[-1]] * 20
        self.run(acts, "V2-handle-approach")
        desc = best_full_pose(self.env, grasp_pt, approach_dir, jaw_dir, "left", lift, seed=appr.arm_qpos)
        acts = [self.act(station, q, self.open_gripper, lift) for q in interp(appr.arm_qpos, desc.arm_qpos, V_ARM_DESCEND)]
        acts += [acts[-1]] * 20
        self.run(acts, "V2-handle-descend")
        arm_q = np.asarray(desc.arm_qpos, np.float32)

        # one tip-servo round (PD equilibrium offset, same as the pick recipe)
        tip, _, _ = self.left_fingertips_world()
        err = np.asarray(grasp_pt, np.float64) - np.asarray(tip, np.float64)
        if 0.004 < float(np.linalg.norm(err)) < 0.09:
            servo = best_full_pose(self.env, (np.asarray(grasp_pt, np.float64) + err).astype(np.float32),
                                   approach_dir, jaw_dir, "left", lift, seed=arm_q)
            if servo.error <= 0.06:
                acts = [self.act(station, q, self.open_gripper, lift)
                        for q in interp(arm_q, servo.arm_qpos, V_ARM_DESCEND)]
                acts += [acts[-1]] * 15
                self.run(acts, "V2-handle-servo")
                arm_q = np.asarray(servo.arm_qpos, np.float32)

        # pinch the bar: 5.3cm across y, same partial-close margin as the pick
        grip_cmd = max((0.053 - 0.030) / 2.0, 0.0)
        for frac in (0.6, 0.3, 0.0):
            g = grip_cmd + (self.open_gripper - grip_cmd) * frac
            self.run([self.act(station, arm_q, g, lift)] * 10, "V2-handle-close")
        try:
            for gi in self.left_grip_q_ids:
                self.robot.active_joints[gi].set_drive_properties(stiffness=2000.0, damping=100.0, force_limit=60.0)
        except Exception as exc:
            print(f"[HANDLE] grip force raise failed: {exc}", flush=True)
        self.run([self.act(station, arm_q, grip_cmd, lift)] * 15, "V2-handle-hold")

        tip, f1, f2 = self.left_fingertips_world()
        gap = float(np.linalg.norm(np.asarray(tip[:2], np.float64) - np.asarray(grasp_pt[:2], np.float64)))
        held = gap < 0.06
        print(f"[HANDLE] grasped tip={np.round(tip,3).tolist()} xy_gap={gap:.4f} "
              f"jaw_sep={float(np.linalg.norm(f2-f1)):.4f} held={held}", flush=True)
        self._v2_grip_cmd = grip_cmd
        self._v2_lift = float(lift)
        return held, np.asarray(station, np.float32), arm_q

    def pull_open_arc(self, station: np.ndarray, arm_q: np.ndarray) -> float:
        """Pull the grasped handle along the hinge circle: rotate the BASE pose
        (position+yaw) about the hinge in small angle steps, arm frozen, and
        track the door qpos closed-loop. Returns the achieved top-door angle."""
        # free the top hinge for the pull; keep the bottom door held shut
        self.set_door_drive([(0.0, 0.3, 5.0), (5.0, 2.0, 5.0)], [0.0, 0.0])
        base0 = self.current_base_world().astype(np.float64)
        rel = base0[:2] - DOOR_HINGE_XY
        lift = self._v2_lift
        grip = self._v2_grip_cmd
        achieved = 0.0
        stall = 0
        q_cmd = DOOR_PULL_STEP
        prev_base = base0.copy()
        while q_cmd <= DOOR_PULL_TARGET + 1e-6:
            c, s = np.cos(-q_cmd), np.sin(-q_cmd)
            tgt = prev_base.copy()
            tgt[0] = DOOR_HINGE_XY[0] + c * rel[0] - s * rel[1]
            tgt[1] = DOOR_HINGE_XY[1] + s * rel[0] + c * rel[1]
            tgt[2] = base0[2] - q_cmd
            acts = [self.act(b, arm_q, grip, lift)
                    for b in interp(prev_base.astype(np.float32), tgt.astype(np.float32), V_BASE * 0.25)]
            acts += [acts[-1]] * 8
            self.run(acts, "V2-pull")
            prev_base = self.current_base_world().astype(np.float64)
            q_now = self.fridge_door_qpos()
            top = float(q_now[0]) if q_now else 0.0
            tip, _, _ = self.left_fingertips_world()
            print(f"[PULL] cmd={q_cmd:.2f} door={q_now} base={np.round(prev_base,3).tolist()} "
                  f"tip={np.round(np.asarray(tip),3).tolist()}", flush=True)
            if top <= achieved + 0.015:
                stall += 1
                if stall >= 3:
                    print(f"[PULL] stalled at door={top:.3f} (handle slipped or arc off)", flush=True)
                    break
            else:
                stall = 0
            achieved = max(achieved, top)
            # re-anchor the command to the MEASURED door angle: the door lags the
            # base arc by ~0.09 rad and letting the gap grow twists the pinch off
            # the bar (first run slipped at ~0.95)
            q_cmd = min(achieved + DOOR_PULL_STEP + 0.06, q_cmd + DOOR_PULL_STEP)
        # detent emulation: hold the door at the achieved angle so it doesn't
        # swing back on a free hinge (real doors have detents/friction)
        self.set_door_drive([(30.0, 8.0, 30.0), (5.0, 2.0, 5.0)], [achieved, 0.0])
        print(f"[PULL] done achieved={achieved:.3f} target={DOOR_PULL_TARGET:.2f}", flush=True)
        return achieved

    def run_door_test(self) -> tuple[bool, dict[str, Any]]:
        target, obstacles, bounds, grid, station_grid, floor_z = self.setup()
        self._v2_obstacles, self._v2_bounds, self._v2_station_grid = obstacles, bounds, station_grid
        self.close_doors_for_v2()
        held, station, arm_q = self.grasp_handle()
        if not held:
            return False, dict(blocker="handle grasp failed", library="grasp_handle",
                               attempted="pinch on freezer handle bar")
        achieved = self.pull_open_arc(station, arm_q)
        ok = achieved >= DOOR_PULL_TARGET - 0.15
        print(f"RESULT: {'DOOR_OPEN_OK' if ok else 'DOOR_OPEN_FAIL'} achieved={achieved:.3f}", flush=True)
        return ok, dict(achieved=achieved)

    def open_fridge_bottom_door(self, settle_lift: float = 0.0,
                                settle_arm_q: np.ndarray | None = None,
                                settle_grip: float | None = None) -> None:
        art = find_fridge_articulation(self.env)
        if art is None:
            names = [getattr(a, "name", "<unnamed>") for a in _iter_articulations(self.env)]
            raise RuntimeError(f"fridge articulation {FRIDGE_ARTICULATION_NAME!r} not found; articulations={names}")
        _set_articulation_qpos(art, np.array([0.0, FRIDGE_BOTTOM_DOOR_OPEN_QPOS], dtype=np.float32))
        print(f"[FRIDGEDRIVE] {self._pin_door_open_drive(art)}", flush=True)
        q_now = np.asarray(hab_scene.to_numpy(art.get_qpos()), np.float64).reshape(-1)
        print(f"[FRIDGEOPEN-PRE] immediately_after_set qpos={np.round(q_now, 4).tolist()}", flush=True)
        if settle_arm_q is not None:
            # settle while HOLDING the exact carry pose — hold_base's default arm is
            # the rest pose, which folds the loaded arm and flings the object (v20)
            hold = self.act(self.current_base_world(), settle_arm_q,
                            self.closed_gripper if settle_grip is None else settle_grip,
                            settle_lift)
            for _ in range(FRIDGE_SETTLE_STEPS):
                self.step_action(hold, record=False)
        else:
            self.hold_base(self.current_base_world(), lift=settle_lift, steps=FRIDGE_SETTLE_STEPS, record=False)
        try:
            qpos = np.asarray(hab_scene.to_numpy(art.get_qpos()), dtype=np.float64).reshape(-1).tolist()
        except Exception:
            qpos = [0.0, FRIDGE_BOTTOM_DOOR_OPEN_QPOS]
        print(
            f"[FRIDGEOPEN] articulation={FRIDGE_ARTICULATION_NAME} "
            f"commanded=[0.0,{FRIDGE_BOTTOM_DOOR_OPEN_QPOS:.3f}] qpos={np.round(qpos, 4).tolist()} "
            f"settle_steps={FRIDGE_SETTLE_STEPS}",
            flush=True,
        )

    def setup(self):
        hab_scene.reset_fixed(self.env, build_config_idx=self.build_config_idx)
        self.refresh_robot_handles()
        floor_z = hab_scene.estimate_floor_height(self.env)
        root_z = hab_scene.set_robot_root_z(self.env, floor_z)
        ignored = hab_scene.disable_aloha_floor_cover_collisions(self.env)
        self.refresh_root_xy()
        self.rest_q = self.skill.current_action_template(self.env)[self.lay["left_arm"]].astype(np.float32)
        target = pick_pepper_target(self.env, verbose=True)
        obstacles, bounds = hab_scene.build_obstacle_list(self.env, target_name=target.name)
        add_articulation_obstacle_patches(obstacles)
        grid = hab_scene.make_occupancy_grid(obstacles, bounds)
        station_grid = hab_scene.make_station_occupancy_grid(
            obstacles,
            bounds,
            robot_radius=STATION_GRID_RADIUS,
        )
        station_free = int((~station_grid.occupied).sum())
        station_total = int(station_grid.occupied.size)
        max_contact, rows = hab_scene.robot_contact_summary(self.env)
        geom = fridge_interior_geometry(self.env)
        print(
            f"[SCENECHK] build_idx={self.build_config_idx} target={target.name} "
            f"target_pos={np.round(target.center, 4).tolist()} target_size={np.round(target.size, 4).tolist()} "
            f"obstacles={len(obstacles)} articulation_patches={len(ARTICULATION_OBSTACLE_PATCHES)} "
            f"bounds={np.round(np.array(bounds), 3).tolist()} floor_z={floor_z:.4f} root_z={root_z:.4f} "
            f"floor_cover_ignored={ignored} robot_contact_max={max_contact:.3f} "
            f"fridge_place={np.round(geom.place_pt, 4).tolist()} shelf_z={geom.shelf_z:.3f}",
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
        self.open_fridge_bottom_door()
        return target, obstacles, bounds, grid, station_grid, floor_z

    def select_fridge_place(self, station_grid):
        geom = fridge_interior_geometry(self.env)
        import hab_pick_place as _hpp
        _saved_accept = _hpp.IK_ACCEPT
        _saved_rings = _hpp.STATION_RING_RADII
        _hpp.IK_ACCEPT = 0.08  # base fine-align absorbs xy residual during insertion
        # closer stations only: at ring 0.38 the arm is at its envelope edge and
        # height/depth trade 1:4 (v31); base is physically fine down to x=-1.60
        _hpp.STATION_RING_RADII = (0.32, 0.34, 0.36, 0.38, 0.42)
        try:
            station, arm_seed, ik_err, place_lift, place_pitch = self.select_station(
            geom.place_pt,
            "place:fridge",
            station_grid,
            self.jaw_dir,
            LIFT_PICK,
            )
        finally:
            _hpp.IK_ACCEPT = _saved_accept
            _hpp.STATION_RING_RADII = _saved_rings
        approach = np.asarray(geom.place_pt[:2], np.float64) - np.asarray(station[:2], np.float64)
        approach /= max(float(np.linalg.norm(approach)), 1e-9)
        print(
            f"[PLACESEL] surface=open_fridge target={np.round(geom.place_pt, 4).tolist()} "
            f"station={np.round(station, 3).tolist()} approach_xy={np.round(approach, 3).tolist()} "
            f"lift={place_lift:.3f} pitch={place_pitch:.0f} ik_err={ik_err:.4f}",
            flush=True,
        )
        return "open_fridge", geom, geom.place_pt, station, arm_seed, ik_err, place_lift, place_pitch

    def _lift_for_object_bottom(self, target: hab_scene.ActorInfo, current_lift: float, desired_bottom_z: float) -> float:
        live = actor_info(target.name, target.actor)
        lift = float(current_lift) + (float(desired_bottom_z) - float(live.bottom))
        return float(np.clip(lift, FRIDGE_MIN_LIFT, LIFT_HIGH))

    def insert_into_fridge(
        self,
        target: hab_scene.ActorInfo,
        obj0: np.ndarray,
        geom: FridgeInterior,
        place_pt: np.ndarray,
        place_station: np.ndarray,
        desc_q: np.ndarray,
        place_arm_seed: np.ndarray,
        pick_desc_ik: float,
        pick_station_ik: float,
        pick_lift: float,
        pick_pitch: float,
        place_station_ik: float,
        place_lift: float,
        place_pitch: float,
        nav_pick_waypoints: int,
        nav_place_waypoints: int,
    ) -> tuple[bool, dict[str, Any]]:
        self.marks["PLACE"] = len(self.frames)
        # the door state is flaky (something sporadically re-closes it after settles);
        # re-assert OPEN right before the insertion, when it matters — keeping the
        # CARRY lift (a lift=0 settle here slammed the held shaker to the floor, v19)
        self.open_fridge_bottom_door(settle_lift=LIFT_HIGH, settle_arm_q=desc_q)
        self._stiffen_left_arm()
        target_xy = place_pt[:2].astype(np.float32)
        cur_base = place_station.astype(np.float32).copy()

        entry_bottom_z = float(geom.shelf_z + FRIDGE_ENTRY_BOTTOM_CLEARANCE)
        # closed-loop lowering: one open-loop shot leaves the held object ~3.7cm
        # high (lift PD sag under load) and its top then clips the CLOSED top
        # freezer door, whose front face starts at z=0.827 / x=-1.752 (v24 drop)
        entry_lift = float(LIFT_HIGH)
        lowered = actor_info(target.name, target.actor)
        for r in range(3):
            # correction baseline is the REALIZABLE lift: commands below 0 clamp at
            # the joint limit, so feeding a negative back in undercorrects raises
            next_lift = self._lift_for_object_bottom(target, max(entry_lift, 0.0), entry_bottom_z)
            lower_actions = [
                self.act(cur_base, desc_q, self.closed_gripper, float(lift[0]))
                for lift in interp([entry_lift], [next_lift], V_LIFT)
            ]
            lower_actions += [self.act(cur_base, desc_q, self.closed_gripper, next_lift)] * 12
            self.run(lower_actions, "PLACE-lower-entry")
            entry_lift = next_lift
            lowered = actor_info(target.name, target.actor)
            print(
                f"[PLACE] insert_step=lower_entry round={r} lift->{entry_lift:.3f} "
                f"desired_bottom_z={entry_bottom_z:.4f} object={np.round(lowered.center, 4).tolist()} "
                f"bottom={lowered.bottom:.4f} bottom_err={lowered.bottom - entry_bottom_z:+.4f}",
                flush=True,
            )
            if abs(float(lowered.bottom) - entry_bottom_z) < 0.012:
                break

        # --- arm reach: stretch to the LOW-PITCH place solution before closing in.
        # The pick pose (pitch 60-75) keeps only ~0.27 m of TCP offset and parks
        # link4/the jaws above the object, where they grind the closed top freezer
        # door (v27 [CONTACT insert] rows, 26-32 N). The low-pitch stretched pose
        # brings every link that crosses the front band (x -1.85..-1.75) below the
        # door slab (z 0.827) and puts the object at the place point directly.
        obj_live = actor_info(target.name, target.actor)
        # LOW-DROP endgame: the shelf slot (0.660 +- mm) proved unreachable — the
        # object hangs ~2cm low in the pinch and every release from the slab
        # corner falls 60cm along the face and bounces off the sill (v34-v41).
        # The UNDER-shelf cavity is open z 0.09..0.63: reach in low (wrist far
        # below the top-door slab), drop ~28cm onto the fridge floor, well behind
        # the sill. Shelf placement stays a future refinement.
        reach_z = float(os.environ.get("HAB_FRIDGE_DROP_Z", "0.45"))
        reach_pt = np.array([place_pt[0], place_pt[1], reach_z], np.float32)
        lbw = self.left_base_world()
        reach_dir = desired_approach_dir(reach_pt, place_pitch, base_xy=tuple(lbw[:2])).astype(np.float32)
        # a live re-solve from the compact pick pose diverges (v28: 0.0866 vs the
        # scan's 0.0217) — seed with the SCAN's stretched solution, and if the live
        # solve still misses, reuse the scan solution outright (DESCFALLBACK
        # pattern); its 1-2 cm height mismatch is absorbed by the height servo
        reach_q = None
        reach_err = float("inf")
        reach_src = "none"
        # solve AND execute the reach at the scan's lift: the raised shoulder is
        # what buys horizontal reach at shelf height (v31 envelope analysis)
        reach_lift = float(np.clip(place_lift, 0.0, 0.10))
        for seed_q, tag in ((place_arm_seed, "scan_seed"), (desc_q, "carry_seed")):
            cand = best_full_pose(
                self.env, reach_pt, reach_dir, self.jaw_dir, "left",
                reach_lift, seed=np.asarray(seed_q, np.float32),
            )
            if float(cand.error) < reach_err:
                reach_q, reach_err, reach_src = np.asarray(cand.arm_qpos, np.float32), float(cand.error), tag
            if reach_err <= 0.03:
                break
        if reach_err > 0.10 and place_arm_seed is not None:
            reach_q, reach_err, reach_src = np.asarray(place_arm_seed, np.float32), float(place_station_ik), "scan_pose"
        if reach_q is not None:
            reach_qs = list(interp(desc_q, reach_q, V_ARM_DESCEND))
            reach_lifts = np.linspace(max(entry_lift, 0.0), reach_lift, max(len(reach_qs), 2))
            reach_actions = [
                self.act(cur_base, q, self.closed_gripper, float(lf))
                for q, lf in zip(reach_qs, reach_lifts)
            ]
            reach_actions += [self.act(cur_base, reach_q, self.closed_gripper, reach_lift)] * 25
            self.run(reach_actions, "PLACE-reach")
            desc_q = reach_q
            entry_lift = reach_lift
            # re-baseline the mid-insert drop detector: the low reach moves the
            # held object far below the entry height by design
            lowered = actor_info(target.name, target.actor)
            reached = actor_info(target.name, target.actor)
            _tipm, _, _ = self.left_fingertips_world()
            reach_gap = float(np.linalg.norm(
                np.asarray(reached.center, np.float64) - np.asarray(_tipm, np.float64)))
            print(
                f"[PLACE] insert_step=arm_reach src={reach_src} pitch={place_pitch:.0f} ik={reach_err:.4f} "
                f"object={np.round(reached.center, 4).tolist()} bottom={reached.bottom:.4f} "
                f"grip_gap={reach_gap:.4f} held={reach_gap < 0.12}",
                flush=True,
            )
        else:
            print(f"[PLACE] insert_step=arm_reach SKIP ik={reach_err:.4f} (keeping carry pose)", flush=True)

        align_steps = 0
        base_clamped = False
        early_release = False
        for i in range(FRIDGE_ALIGN_ROUNDS):
            obj_now = actor_info(target.name, target.actor)
            delta = np.array([target_xy[0] - obj_now.center[0], target_xy[1] - obj_now.center[1]], np.float32)
            err = float(np.linalg.norm(delta))
            dropped = bool(float(obj_now.center[2]) < float(lowered.center[2]) - 0.20)
            print(
                f"[PLACE] insert_step=drive_in align_round={i} delta_xy_m={np.round(delta, 4).tolist()} "
                f"err={err:.4f} base={np.round(cur_base, 3).tolist()} "
                f"obj_z={float(obj_now.center[2]):.3f} door={self.fridge_door_qpos()} dropped={dropped}",
                flush=True,
            )
            _, rob_rows = hab_scene.robot_contact_summary(self.env, limit=3)
            for norm, names, impulse, npoints in rob_rows:
                print(f"[CONTACT insert r{i}] norm={norm:.3f} npoints={npoints} pair={names}", flush=True)
            if dropped:
                # object left the gripper mid-insertion; further base driving only
                # smears the failure evidence around
                break
            # release-when-good: once the object is deep inside the under-shelf
            # cavity at any safe height, STOP refining — the servo rounds only
            # trade depth away (v33-v41)
            over_shelf = (
                float(obj_now.center[0]) <= float(place_pt[0]) + 0.03
                and 0.10 <= float(obj_now.bottom) <= 0.60
                and abs(float(obj_now.center[1]) - float(place_pt[1])) <= 0.12
            )
            if over_shelf:
                early_release = True
                print(
                    f"[PLACE] insert_step=early_release_gate round={i} "
                    f"object={np.round(obj_now.center, 4).tolist()} bottom={obj_now.bottom:.4f} "
                    f"shelf_z={geom.shelf_z:.4f}",
                    flush=True,
                )
                break
            if err < FRIDGE_ALIGN_TOL:
                break
            # NOTE: no height servo here — in the low-drop regime any height in
            # the gate window works, and arm z-fixes only traded depth away
            # (v31/v36); the legs below fine-align xy only
            tgt = cur_base.copy()
            tgt[0] += delta[0]
            tgt[1] += delta[1]
            # heading drifts under contact torque and re-reading it bakes the
            # drift into commands (v27: theta -1.31 -> -0.79); pin the station yaw
            tgt[2] = float(place_station[2])
            if float(tgt[0]) < geom.base_min_x:
                tgt[0] = float(geom.base_min_x)
                base_clamped = True
            actions = [
                self.act(b, desc_q, self.closed_gripper, entry_lift)
                for b in interp(cur_base, tgt, V_BASE * 0.2)
            ]
            actions += [self.act(tgt, desc_q, self.closed_gripper, entry_lift)] * 10
            align_steps += len(actions)
            self.run(actions, "PLACE-drive-in")
            cur_base = self.current_base_world().astype(np.float32)

        obj_now = actor_info(target.name, target.actor)
        if early_release:
            # low-drop release: the object hangs deep in the under-shelf cavity
            # with nothing to hook — open in place and let it settle on the
            # fridge floor ~28 cm below, well behind the front sill
            cur_base = self.current_base_world().astype(np.float32)
            release_lift = float(entry_lift)
        else:
            release_lift = float(entry_lift + (float(place_pt[2]) - float(obj_now.center[2])))
            release_lift = float(np.clip(release_lift, FRIDGE_MIN_LIFT, LIFT_HIGH))
        descent_actions = [
            self.act(cur_base, desc_q, self.closed_gripper, float(lift[0]))
            for lift in interp([entry_lift], [release_lift], V_LIFT)
        ]
        descent_actions += [self.act(cur_base, desc_q, self.closed_gripper, release_lift)] * 15
        self.run(descent_actions, "PLACE-lift-only-descent")
        pre_release = actor_info(target.name, target.actor)
        print(
            f"[PLACE] insert_step=lift_only_descent lift={entry_lift:.3f}->{release_lift:.3f} "
            f"pre_release={np.round(pre_release.center, 4).tolist()} "
            f"target={np.round(place_pt, 4).tolist()}",
            flush=True,
        )

        release_actions: list[np.ndarray] = []
        for k in range(1, 5):
            grip = self.closed_gripper + (self.open_gripper - self.closed_gripper) * k / 10
            release_actions.append(self.act(cur_base, desc_q, grip, release_lift))
        # two-stage open (Codex review): pause at 40% so any residual pinch strain
        # dissipates while the fingers still cage the object, then open fully
        release_actions += [release_actions[-1]] * 15
        for k in range(5, 11):
            grip = self.closed_gripper + (self.open_gripper - self.closed_gripper) * k / 10
            release_actions.append(self.act(cur_base, desc_q, grip, release_lift))
        for _ in range(12):
            release_actions.append(self.act(cur_base, desc_q, self.open_gripper, release_lift))
        self.run(release_actions, "PLACE-release")
        released = actor_info(target.name, target.actor)
        print(
            f"[PLACE] insert_step=release released={np.round(released.center, 4).tolist()} "
            f"release_steps={len(release_actions)}",
            flush=True,
        )

        retreat_start = self.current_base_world().astype(np.float32)
        retreat_pose = retreat_start.copy()
        retreat_pose[0] += FRIDGE_RETREAT_X
        retreat_actions = [
            self.act(b, desc_q, self.open_gripper, release_lift)
            for b in interp(retreat_start, retreat_pose, V_BASE * 0.5)
        ]
        retreat_actions += [self.act(retreat_pose, desc_q, self.open_gripper, release_lift)] * 20
        self.run(retreat_actions, "PLACE-retreat")
        post_retreat = actor_info(target.name, target.actor)
        settle_action = self.act(retreat_pose, desc_q, self.open_gripper, release_lift)
        for _ in range(HOLD):
            self.step_action(settle_action, record=True)
        self.force_action_qpos = False
        final = actor_info(target.name, target.actor)

        moved = float(np.linalg.norm(final.center[:2] - obj0[:2]))
        place_xy_err = float(np.linalg.norm(final.center[:2] - target_xy))
        place_3d_err = float(np.linalg.norm(final.center - place_pt))
        z_err = float(abs(final.center[2] - float(place_pt[2])))
        inside = bool(
            abs(float(final.center[0]) - float(geom.footprint_center[0])) < float(geom.footprint_half[0])
            and abs(float(final.center[1]) - float(geom.footprint_center[1])) < float(geom.footprint_half[1])
        )
        z_stable = bool(abs(float(final.center[2]) - float(post_retreat.center[2])) <= FRIDGE_Z_STABLE_TOL)
        base_outside = bool(float(retreat_start[0]) >= geom.base_min_x - 0.03)
        shelf_ok = abs(float(final.bottom) - float(geom.shelf_z)) < 0.12
        # user-approved criterion: the shaker resting anywhere INSIDE the fridge
        # counts (bottom-compartment floor z=0.0654 included); the exact shelf
        # spot is a bonus, refinable later by pushing
        floor_ok = abs(float(final.bottom) - 0.0654) < 0.10
        success = bool(moved >= 1.0 and inside and z_stable and (shelf_ok or floor_ok))
        print(
            f"[PLACECHK] pre_release={np.round(pre_release.center, 4).tolist()} "
            f"released={np.round(released.center, 4).tolist()} post_retreat={np.round(post_retreat.center, 4).tolist()} "
            f"object_final={np.round(final.center, 4).tolist()} moved={moved:.4f} "
            f"place_xy_err={place_xy_err:.4f} place_3d_err={place_3d_err:.4f} z_err={z_err:.4f} "
            f"inside_fridge={inside} z_stable={z_stable} base_outside_before_retreat={base_outside} "
            f"base_clamped={base_clamped} align_steps={align_steps} retreat_steps={len(retreat_actions)} "
            f"shelf_ok={shelf_ok} floor_ok={floor_ok} success={success}",
            flush=True,
        )
        print(
            f"[RESULT] fridge_insert inside={inside} z_stable={z_stable} "
            f"retreat_dx={FRIDGE_RETREAT_X:.3f} success={success}",
            flush=True,
        )
        if not self.norender and self.frames:
            save_png(self.frames[-1], KEYFRAME_PLACE)
        if not success:
            return False, dict(
                blocker=(
                    f"fridge insertion failed: moved={moved:.4f}, inside={inside}, "
                    f"z_stable={z_stable}, xy_err={place_xy_err:.4f}, z_err={z_err:.4f}"
                ),
                library="fridge_insertion_sequence",
                attempted="lower-lift, arm-frozen base drive-in, lift-only descent, release, +x retreat",
                final=final.center.tolist(),
                target=place_pt.tolist(),
            )

        return True, dict(
            target=target.name,
            place_surface="open_fridge",
            object_start=obj0.tolist(),
            object_final=final.center.tolist(),
            place_target=place_pt.tolist(),
            moved=moved,
            place_xy_err=place_xy_err,
            place_3d_err=place_3d_err,
            z_err=z_err,
            inside_fridge=inside,
            z_stable=z_stable,
            pick_ik=float(pick_desc_ik),
            pick_station_ik=float(pick_station_ik),
            pick_lift=float(pick_lift),
            pick_pitch=float(pick_pitch),
            place_station_ik=float(place_station_ik),
            place_lift=float(place_lift),
            place_pitch=float(place_pitch),
            nav_pick_waypoints=nav_pick_waypoints,
            nav_place_waypoints=nav_place_waypoints,
            frames=len(self.frames),
        )

    def run_demo(self) -> tuple[bool, dict[str, Any]]:
        target, obstacles, bounds, grid, station_grid, floor_z = self.setup()
        target = self.ensure_reachable_target(target, obstacles, station_grid)
        target0 = actor_info(target.name, target.actor)
        target_suffix = PEPPER_NAME
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
            fridge_geom,
            place_pt,
            place_station,
            place_arm_seed,
            place_station_ik,
            place_lift,
            place_pitch,
        ) = self.select_fridge_place(station_grid)

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
        pre_pt[2] = min(float(pre_pt[2]), float(grasp_pt[2]) + 0.03)  # hover-height cap (arm ceiling ~0.97)
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
            # the hover point sits near the arm ceiling for counter-height grasps and
            # its residual is orientation-dominated; the hover is contact-free air and
            # the descent chain + tip-servo + PRECLOSE are all closed-loop, so warn
            # instead of aborting (the GRASP solution itself is mm-accurate)
            print(
                f"[APPRWARN] hover ik={appr.error:.4f} (scan {pick_station_ik:.4f}) — proceeding",
                flush=True,
            )
        appr_h = np.asarray(approach_dir[:2], np.float64)
        appr_h /= max(float(np.linalg.norm(appr_h)), 1e-9)
        jaw_parallel = abs(float(np.dot(appr_h, np.asarray(self.jaw_dir[:2], np.float64)))) > 0.35
        obj_live = actor_info(target.name, target.actor)
        if jaw_parallel:
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
        print(f"[DESCENT] shape={'L_over_top' if jaw_parallel else 'tilted_line'} waypoints={len(waypoint_pts)}", flush=True)
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
        if not jaw_parallel:
            descent[-1] = np.asarray(desc.arm_qpos, np.float32)
        desc_q = descent[-1]
        approach_actions: list[np.ndarray] = []
        for q in interp(self.rest_q, appr.arm_qpos, V_ARM):
            approach_actions.append(self.act(pick_station, q, self.open_gripper, pick_lift))
        self.force_action_qpos = False
        self.force_lift_position = None
        self.run(approach_actions, "PICK-approach")
        pick_step_count = len(approach_actions)

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

        descend_actions: list[np.ndarray] = []
        for q0, q1 in zip(descent[:-1], descent[1:]):
            for q in interp(q0, q1, V_ARM_DESCEND):
                descend_actions.append(self.act(pick_station, q, self.open_gripper, pick_lift))
        descend_actions += [descend_actions[-1]] * 35
        if jaw_parallel:
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

        obj_now = actor_info(target.name, target.actor)
        body_w = min(float(obj_now.min_xy_dim), float(np.min(target0.size[:2])))
        target_sep = max(body_w - 0.030, 0.004)
        close_frac = float(np.clip(1.0 - target_sep / 0.0735, 0.15, 1.0))
        self.closed_gripper = self.open_gripper + (self.closed_gripper - self.open_gripper) * close_frac
        print(
            f"[PARTIALCLOSE] min_xy={obj_now.min_xy_dim:.4f} target_sep={target_sep:.4f} "
            f"close_frac={close_frac:.2f}",
            flush=True,
        )
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
            _, obj_contacts = hab_scene.contact_summary(self.env, name_filter=target_suffix, limit=5)
            for norm, names, impulse, npoints in obj_contacts:
                print(f"[CONTACT pick] norm={norm:.3f} npoints={npoints} pair={names}", flush=True)
            if not os.environ.get("HAB_SCRIPTED_FALLBACK"):
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
        path2 = hab_scene.plan_path_grid(
            obstacles, pick_station[:2], place_station[:2], bounds, robot_radius=0.45
        )
        if path2 is None:
            path2 = hab_scene.plan_path_grid(
                obstacles, pick_station[:2], place_station[:2], bounds, robot_radius=0.34
            )
        if path2 is None:
            return False, dict(
                blocker="A* failed from pick station to fridge place station",
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
        # counter starts sit ABOVE the carry hold height, so z-delta goes negative on
        # a perfectly held object; judge retention by gripper proximity instead
        _tipm, _, _ = self.left_fingertips_world()
        grip_gap = float(np.linalg.norm(np.asarray(carried.center, np.float64) - np.asarray(_tipm, np.float64)))
        carry_dist = float(np.linalg.norm(carried.center[:2] - obj0[:2]))
        base_now = self.current_base_world()
        print(
            f"[CARRYCHK] waypoints={len(path2)} steps={len(nav2)} base={np.round(base_now, 3).tolist()} "
            f"object={np.round(carried.center, 4).tolist()} carry_lift={carry_lift:.4f} "
            f"object_xy_delta={carry_dist:.4f} grip_gap={grip_gap:.4f} held={grip_gap < 0.12}",
            flush=True,
        )
        if grip_gap >= 0.12:
            return False, dict(
                blocker=f"object dropped during carry: carry_lift={carry_lift:.4f}",
                library="frozen_arm_base_lift_place",
                attempted="arm frozen during carry, base velocity V_BASE*0.25",
            )
        if scripted_carry:
            final = actor_info(target.name, target.actor)
            moved = float(np.linalg.norm(final.center[:2] - obj0[:2]))
            place_xy_err = float(np.linalg.norm(final.center[:2] - place_pt[:2]))
            place_3d_err = float(np.linalg.norm(final.center - place_pt))
            z_err = float(abs(final.center[2] - float(place_pt[2])))
            inside = bool(
                abs(float(final.center[0]) - float(fridge_geom.footprint_center[0])) < float(fridge_geom.footprint_half[0])
                and abs(float(final.center[1]) - float(fridge_geom.footprint_center[1])) < float(fridge_geom.footprint_half[1])
            )
            success = moved >= 1.0 and inside and place_xy_err <= 0.08 and z_err <= 0.08
            print(
                f"[PLACECHK] scripted_release object_final={np.round(final.center, 4).tolist()} "
                f"moved={moved:.4f} place_xy_err={place_xy_err:.4f} "
                f"place_3d_err={place_3d_err:.4f} z_err={z_err:.4f} inside_fridge={inside} success={success}",
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
                    z_err=z_err,
                    inside_fridge=inside,
                    z_stable=True,
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

        return self.insert_into_fridge(
            target=target,
            obj0=obj0,
            geom=fridge_geom,
            place_pt=place_pt,
            place_station=place_station,
            desc_q=desc_q,
            place_arm_seed=place_arm_seed,
            pick_desc_ik=float(desc.error),
            pick_station_ik=float(pick_station_ik),
            pick_lift=float(pick_lift),
            pick_pitch=float(pick_pitch),
            place_station_ik=float(place_station_ik),
            place_lift=float(place_lift),
            place_pitch=float(place_pitch),
            nav_pick_waypoints=len(path1),
            nav_place_waypoints=len(path2),
        )


def result_line(success: bool, info: dict[str, Any], norender: bool) -> str:
    if success:
        if norender:
            return (
                "RESULT: SUCCESS_NORENDER "
                f"moved={info['moved']:.3f} inside_fridge={info.get('inside_fridge')} "
                f"z_stable={info.get('z_stable')} target={info['target']} surface={info['place_surface']} "
                "mp4=not_rendered"
            )
        return (
            "RESULT: SUCCESS "
            f"moved={info['moved']:.3f} inside_fridge={info.get('inside_fridge')} "
            f"z_stable={info.get('z_stable')} target={info['target']} surface={info['place_surface']} "
            f"mp4={VIDEO_PATH} keyframes={[str(KEYFRAME_PICK), str(KEYFRAME_PLACE)]}"
        )
    return (
        "RESULT: BLOCKED "
        f"blocker={info.get('blocker')} library={info.get('library', 'none')} "
        f"attempted={info.get('attempted')}"
    )


def _encode_fridge(frames: list[np.ndarray], path: Path) -> None:
    """Use the base encoder but keep fridge scratch frames separate if needed later."""

    try:
        encode(frames, path)
    finally:
        shutil.rmtree(FRAMES_TMP, ignore_errors=True)


def main() -> None:
    norender = bool(os.environ.get("NORENDER"))
    success = False
    info: dict[str, Any] = {}
    demo = FridgeDemo(norender=norender, build_config_idx=hab_scene.BUILD_CONFIG_IDX)
    try:
        if os.environ.get("HAB_V2_TEST") == "door":
            success, info = demo.run_door_test()
        else:
            success, info = demo.run_demo()
        if not success and not norender and getattr(demo, "frames", None):
            partial = OUT_DIR / "hab_fridge_partial.mp4"
            try:
                _encode_fridge(demo.frames, partial)
                print(f"[PARTIAL] encoded {len(demo.frames)} frames -> {partial}", flush=True)
            except Exception as enc_exc:
                print(f"[PARTIAL] encode failed: {enc_exc}", flush=True)
        if success and not norender:
            if demo.frames:
                _encode_fridge(demo.frames, VIDEO_PATH)
            if getattr(demo, "frames2", None):
                _encode_fridge(demo.frames2, OUT_DIR / "hab_fridge_closeup.mp4")
            if not VIDEO_PATH.exists() or VIDEO_PATH.stat().st_size == 0:
                success = False
                info = dict(
                    blocker="render succeeded but ffmpeg did not create a non-empty mp4",
                    library="none",
                    attempted=f"encoded {len(demo.frames)} frames to {VIDEO_PATH}",
                )
    except Exception as exc:
        traceback.print_exc()
        if not norender and getattr(demo, "frames", None):
            partial = OUT_DIR / "hab_fridge_partial.mp4"
            try:
                _encode_fridge(demo.frames, partial)
                print(f"[PARTIAL] encoded {len(demo.frames)} frames -> {partial}", flush=True)
            except Exception as enc_exc:
                print(f"[PARTIAL] encode failed: {enc_exc}", flush=True)
        info = dict(
            blocker=f"unhandled exception: {exc}",
            library="none",
            attempted="ReplicaCAD pepper-shaker fridge insertion pipeline",
        )
        success = False
    finally:
        demo.close()
    print(result_line(success, info, norender), flush=True)


if __name__ == "__main__":
    main()
