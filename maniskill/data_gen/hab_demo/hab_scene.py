#!/usr/bin/env python3
"""ReplicaCAD scene utilities for the ASPIRE TIER-2 AlohaMini Pro demo."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


OUT_DIR = Path(__file__).resolve().parent
REPO_MANISKILL = Path("/home/perelman/AlohaMini/maniskill")
ENV_ID = "ReplicaCAD_SceneManipulation-v1"
ROBOT_UID = "aloha_mini_2"
BUILD_CONFIG_IDX = 5
SCENE_SEED = 1005
GRID_RESOLUTION = 0.04
ROBOT_RADIUS = 0.24
FLOOR_CLEARANCE_Z = 0.009

ENV_KWARGS = dict(
    num_envs=1,
    obs_mode="state",
    control_mode="pd_joint_pos_fixed_base",
    render_mode="rgb_array",
    sim_backend="physx_cpu",
    robot_uids=ROBOT_UID,
)


@dataclass
class ActorInfo:
    name: str
    actor: Any
    kind: str
    mass: float
    center: np.ndarray
    half: np.ndarray
    size: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    @property
    def max_dim(self) -> float:
        return float(np.max(self.size))

    @property
    def max_xy_dim(self) -> float:
        return float(np.max(self.size[:2]))

    @property
    def min_xy_dim(self) -> float:
        return float(np.min(self.size[:2]))

    @property
    def top(self) -> float:
        return float(self.hi[2])

    @property
    def bottom(self) -> float:
        return float(self.lo[2])


def configure_paths() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(OUT_DIR / "xdg_cache"))
    os.environ.setdefault("MS_ASSET_DIR", str(Path.home() / ".maniskill"))
    if str(REPO_MANISKILL) not in sys.path:
        sys.path.insert(0, str(REPO_MANISKILL))
    (OUT_DIR / "mplconfig").mkdir(exist_ok=True)
    (OUT_DIR / "xdg_cache").mkdir(exist_ok=True)


def to_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def vec3(value: Any) -> np.ndarray:
    arr = np.asarray(to_numpy(value), dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.reshape(-1, arr.shape[-1])[0]
    return arr.reshape(-1)[:3].astype(np.float64)


def patch_pci_render_backend_parser() -> None:
    import mani_skill.envs.utils.system.backend as backend

    def parse_backend_device_id_fixed(name):
        if name is None:
            return name, None
        if isinstance(name, str) and name.startswith("pci:"):
            return name, None
        if isinstance(name, str) and ":" in name:
            return name.split(":", 1)
        return name, None

    backend.parse_backend_device_id = parse_backend_device_id_fixed


def patch_supported_robot_warning() -> None:
    import mani_skill.envs.scenes.base_env as scene_base

    cls = scene_base.SceneManipulationEnv
    current = list(getattr(cls, "SUPPORTED_ROBOTS", []) or [])
    if ROBOT_UID not in current:
        cls.SUPPORTED_ROBOTS = current + [ROBOT_UID]


def patch_no_render_visuals() -> None:
    """Allow render_backend='none' by suppressing render components only.

    ReplicaCAD still builds all collision geometry. Visual records and lighting calls
    are the parts that require a render system and crash SAPIEN during no-render
    teardown.
    """

    from sapien.wrapper.actor_builder import ActorBuilder

    def no_visual(self, *args, **kwargs):
        return self

    for method in (
        "add_visual_from_file",
        "add_box_visual",
        "add_capsule_visual",
        "add_cylinder_visual",
        "add_plane_visual",
        "add_sphere_visual",
    ):
        setattr(ActorBuilder, method, no_visual)

    import mani_skill.envs.scene as scene_mod

    scene_mod.ManiSkillScene.set_ambient_light = lambda self, color: None
    scene_mod.ManiSkillScene.add_point_light = lambda self, *args, **kwargs: None


PICK_CUBE_HALF = 0.0285
# dining table top at 0.765 -> grasp height ~0.79, the height the tabletop pick was
# validated at; the 0.62 m tvstand forced an arm posture whose jaw tilt (7-11 deg)
# wedge-ejected even a perfectly centered cube
PICK_CUBE_POSE = (3.48, -1.79, 0.765 + PICK_CUBE_HALF + 0.002)  # 4.5 cm in from the west edge
PICK_ENV_ID = "AlohaMiniHabPick-v1"


def register_hab_pick_env() -> None:
    """SceneManipulation + a spawned pinch-able target cube (true box collision).

    ReplicaCAD furniture objects are built from visual-mesh convex hulls / convex
    decompositions whose 'flat' faces tilt up to 32 deg (book_04 measured) — a
    parallel pinch cams them out regardless of alignment or friction. The
    ManiSkill-HAB benchmark itself manipulates SPAWNED YCB objects, so the demo
    does the same with the cube geometry our pick pipeline is validated on.
    """
    import sapien
    import mani_skill.envs.scenes.base_env as scene_base
    from mani_skill.utils.registration import REGISTERED_ENVS, register_env

    if PICK_ENV_ID in REGISTERED_ENVS:
        return

    @register_env(PICK_ENV_ID, max_episode_steps=200000)
    class AlohaMiniHabPickEnv(scene_base.SceneManipulationEnv):
        @property
        def _default_human_render_camera_configs(self):
            # two synchronized streams: wide follow-cam + gripper close-up (poses are
            # retargeted every captured frame by the demo)
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
            from mani_skill.utils.building import actors

            self.demo_pick_cube = actors.build_cube(
                self.scene,
                half_size=PICK_CUBE_HALF,
                color=[0.85, 0.12, 0.12, 1.0],
                name="demo_pick_cube",
                initial_pose=sapien.Pose(p=list(PICK_CUBE_POSE)),
            )
            try:
                import sapien.physx as physx

                mat = physx.PhysxMaterial(1.0, 1.0, 0.0)
                patched = 0
                for obj in self.demo_pick_cube._objs:
                    comp = obj.find_component_by_type(physx.PhysxRigidDynamicComponent)
                    for shape in comp.get_collision_shapes():
                        shape.set_physical_material(mat)
                        patched += 1
                print(f"[PICKCUBE] friction mu=1.0 patched_shapes={patched}", flush=True)
            except Exception as exc:
                print(f"[PICKCUBE] friction patch failed: {exc}", flush=True)
            # expose it to the demo's actor scans
            key = "env-0_objects/demo_pick_cube"
            for attr in ("movable_objects", "scene_objects"):
                d = getattr(self.scene_builder, attr, None)
                if isinstance(d, dict):
                    d[key] = self.demo_pick_cube


def camera_pose_list(eye, target) -> list[float]:
    from mani_skill.utils import sapien_utils

    pose = sapien_utils.look_at(eye=eye, target=target)
    p = np.asarray(pose.p, dtype=float).reshape(-1)[:3].tolist()
    q = np.asarray(pose.q, dtype=float).reshape(-1)[:4].tolist()
    return p + q


def default_camera_config(shader_pack: str = "minimal") -> dict[str, Any]:
    # both cameras are DEFINED by AlohaMiniHabPickEnv._default_human_render_camera_configs;
    # gym.make kwargs may only OVERRIDE attributes of existing cameras (shader etc.)
    return dict(
        render_camera=dict(shader_pack=shader_pack),
        closeup_camera=dict(shader_pack=shader_pack),
    )


def make_env(norender: bool | None = None, shader_pack: str = "minimal", build_config_idx: int | None = None):
    configure_paths()
    if norender is None:
        norender = bool(os.environ.get("NORENDER"))
    if norender:
        patch_no_render_visuals()

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import mani_skill.agents.robots  # noqa: F401

    patch_pci_render_backend_parser()
    patch_supported_robot_warning()
    register_hab_pick_env()

    kwargs = dict(ENV_KWARGS)
    build_idx = BUILD_CONFIG_IDX if build_config_idx is None else int(build_config_idx)
    kwargs.update(
        build_config_idxs=build_idx,
        render_backend="none" if norender else "pci:0",
        render_mode=None if norender else ENV_KWARGS["render_mode"],
        human_render_camera_configs={} if norender else default_camera_config(shader_pack),
    )
    return gym.make(PICK_ENV_ID, **kwargs)


def reset_fixed(env, seed: int = SCENE_SEED, build_config_idx: int | None = None) -> None:
    build_idx = BUILD_CONFIG_IDX if build_config_idx is None else int(build_config_idx)
    env.reset(seed=seed, options={"reconfigure": True, "build_config_idxs": build_idx})


def actor_aabb(actor: Any) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        meshes = actor.get_collision_meshes(to_world_frame=True)
    except TypeError:
        meshes = actor.get_collision_meshes()
    except Exception:
        return None
    bounds = []
    for mesh in meshes or []:
        vertices = getattr(mesh, "vertices", None)
        if vertices is None or np.asarray(vertices).size == 0:
            continue
        bounds.append(np.asarray(mesh.bounds, dtype=np.float64))
    if not bounds:
        return None
    b = np.asarray(bounds, dtype=np.float64)
    return b[:, 0, :].min(axis=0), b[:, 1, :].max(axis=0)


def actor_mass(actor: Any) -> float:
    try:
        return float(np.asarray(to_numpy(actor.get_mass()), dtype=np.float64).reshape(-1)[0])
    except Exception:
        return float("inf")


def movable_actor_infos(env) -> list[ActorInfo]:
    infos: list[ActorInfo] = []
    movable = getattr(env.unwrapped.scene_builder, "movable_objects", {}) or {}
    for name, actor in sorted(movable.items()):
        aabb = actor_aabb(actor)
        if aabb is None:
            continue
        lo, hi = aabb
        size = hi - lo
        infos.append(
            ActorInfo(
                name=name,
                actor=actor,
                kind="movable",
                mass=actor_mass(actor),
                center=(lo + hi) * 0.5,
                half=size * 0.5,
                size=size,
                lo=lo,
                hi=hi,
            )
        )
    return infos


def scene_actor_infos(env, include_movable: bool = True) -> list[ActorInfo]:
    sb = env.unwrapped.scene_builder
    movable = getattr(sb, "movable_objects", {}) or {}
    infos: list[ActorInfo] = []
    for name, actor in sorted((getattr(sb, "scene_objects", {}) or {}).items()):
        if not include_movable and name in movable:
            continue
        aabb = actor_aabb(actor)
        if aabb is None:
            continue
        lo, hi = aabb
        size = hi - lo
        infos.append(
            ActorInfo(
                name=name,
                actor=actor,
                kind="movable" if name in movable else "static",
                mass=actor_mass(actor) if name in movable else float("inf"),
                center=(lo + hi) * 0.5,
                half=size * 0.5,
                size=size,
                lo=lo,
                hi=hi,
            )
        )
    return infos


def scene_extent(env) -> tuple[float, float, float, float]:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for info in scene_actor_infos(env, include_movable=True):
        mins.append(info.lo)
        maxs.append(info.hi)
    bg = getattr(env.unwrapped.scene_builder, "bg", None)
    if bg is not None:
        aabb = actor_aabb(bg)
        if aabb is not None:
            mins.append(aabb[0])
            maxs.append(aabb[1])
    lo = np.min(np.asarray(mins), axis=0)
    hi = np.max(np.asarray(maxs), axis=0)
    pad = 0.25
    return float(lo[0] - pad), float(hi[0] + pad), float(lo[1] - pad), float(hi[1] + pad)


def estimate_floor_height(env) -> float:
    bg = getattr(env.unwrapped.scene_builder, "bg", None)
    samples: list[np.ndarray] = []
    if bg is not None:
        try:
            meshes = bg.get_collision_meshes(to_world_frame=True)
        except TypeError:
            meshes = bg.get_collision_meshes()
        for mesh in meshes or []:
            vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
            if vertices.size:
                samples.append(vertices[:, 2])
    if not samples:
        return 0.0
    z = np.concatenate(samples)
    near_zero = z[(z > -0.01) & (z < 0.01)]
    if near_zero.size >= 20:
        return float(np.round(np.median(near_zero), 3))
    z = z[(z > -0.10) & (z < 0.12)]
    if z.size == 0:
        return 0.0
    bins = np.arange(-0.10, 0.125, 0.005)
    hist, edges = np.histogram(z, bins=bins)
    idx = int(np.argmax(hist))
    return float((edges[idx] + edges[idx + 1]) * 0.5)


def set_robot_root_z(env, floor_z: float | None = None, clearance: float = FLOOR_CLEARANCE_Z) -> float:
    import sapien

    if floor_z is None:
        floor_z = estimate_floor_height(env)
    robot = env.unwrapped.agent.robot
    pose = robot.pose
    p = vec3(pose.p)
    q = np.asarray(to_numpy(pose.q), dtype=np.float64).reshape(-1, 4)[0]
    z = float(floor_z + clearance)
    robot.set_pose(sapien.Pose([float(p[0]), float(p[1]), z], q.tolist()))
    return z


def disable_aloha_floor_cover_collisions(env) -> int:
    try:
        from mani_skill.agents.robots.aloha_mini import (
            ALOHA_MINI_BASE_COLLISION_BIT,
            ALOHA_MINI_WHEELS_COLLISION_BIT,
        )
    except Exception:
        return 0

    for link in env.unwrapped.agent.robot.get_links():
        if getattr(link, "name", "").startswith("wheel"):
            try:
                link.set_collision_group_bit(group=2, bit_idx=ALOHA_MINI_WHEELS_COLLISION_BIT, bit=1)
            except Exception:
                pass

    count = 0
    for name, actor in (getattr(env.unwrapped.scene_builder, "scene_objects", {}) or {}).items():
        lname = name.lower()
        if not any(token in lname for token in ("rug", "mat", "carpet")):
            continue
        try:
            actor.set_collision_group_bit(group=2, bit_idx=ALOHA_MINI_WHEELS_COLLISION_BIT, bit=1)
            actor.set_collision_group_bit(group=2, bit_idx=ALOHA_MINI_BASE_COLLISION_BIT, bit=1)
            count += 1
        except Exception:
            pass
    return count


def contact_summary(env, name_filter: str | None = None, limit: int = 8) -> tuple[float, list[tuple]]:
    rows = []
    max_norm = 0.0
    for contact in env.unwrapped.scene.get_contacts():
        names = []
        for body in getattr(contact, "bodies", []):
            ent = getattr(body, "entity", None)
            names.append(getattr(ent, "name", repr(body)))
        if name_filter and not any(name_filter in name for name in names):
            continue
        impulses = []
        for point in getattr(contact, "points", []):
            impulses.append(to_numpy(getattr(point, "impulse", [0, 0, 0])))
        impulse = np.sum(np.asarray(impulses, dtype=np.float64), axis=0) if impulses else np.zeros(3)
        norm = float(np.linalg.norm(impulse))
        rows.append((norm, tuple(names), impulse.tolist(), len(impulses)))
        max_norm = max(max_norm, norm)
    rows.sort(key=lambda row: row[0], reverse=True)
    return max_norm, rows[:limit]


def robot_contact_summary(env, limit: int = 8) -> tuple[float, list[tuple]]:
    robot_names = {getattr(link, "name", "") for link in env.unwrapped.agent.robot.get_links()}
    rows = []
    max_norm = 0.0
    for contact in env.unwrapped.scene.get_contacts():
        names = []
        for body in getattr(contact, "bodies", []):
            ent = getattr(body, "entity", None)
            names.append(getattr(ent, "name", repr(body)))
        if not any(name in robot_names for name in names):
            continue
        impulses = []
        for point in getattr(contact, "points", []):
            impulses.append(to_numpy(getattr(point, "impulse", [0, 0, 0])))
        impulse = np.sum(np.asarray(impulses, dtype=np.float64), axis=0) if impulses else np.zeros(3)
        norm = float(np.linalg.norm(impulse))
        rows.append((norm, tuple(names), impulse.tolist(), len(impulses)))
        max_norm = max(max_norm, norm)
    rows.sort(key=lambda row: row[0], reverse=True)
    return max_norm, rows[:limit]


def print_movable_candidates(infos: list[ActorInfo], max_rows: int = 40) -> None:
    def sort_key(info: ActorInfo):
        strict_bad = not (info.mass < 1.0 and info.min_xy_dim <= 0.068 and info.max_dim < 0.40 and 0.30 <= info.top <= 1.10)
        fallback_bad = not (
            info.mass < 1.0 and info.min_xy_dim <= 0.075 and info.max_dim < 0.50 and 0.25 <= info.top <= 1.20
        )
        return (strict_bad, fallback_bad, abs(info.top - 0.64), info.max_dim, info.name)

    print("movable_candidates_begin", flush=True)
    for info in sorted(infos, key=sort_key)[:max_rows]:
        strict = info.mass < 1.0 and info.min_xy_dim <= 0.068 and info.max_dim < 0.40 and 0.30 <= info.top <= 1.10
        fallback = (
            info.mass < 1.0 and info.min_xy_dim <= 0.075 and info.max_dim < 0.50 and 0.25 <= info.top <= 1.20
        )
        print(
            f"candidate strict={strict} fallback={fallback} mass={info.mass:.4f} "
            f"maxdim={info.max_dim:.3f} min_xy={info.min_xy_dim:.3f} top={info.top:.3f} "
            f"name={info.name} pos={np.round(info.center, 4).tolist()} "
            f"size={np.round(info.size, 4).tolist()}",
            flush=True,
        )
    print("movable_candidates_end", flush=True)


def pick_target_object(env, verbose: bool = True) -> ActorInfo:
    infos = movable_actor_infos(env)
    if verbose:
        print_movable_candidates(infos)

    strict = [
        info
        for info in infos
        if info.mass < 1.0 and info.min_xy_dim <= 0.068 and info.max_dim < 0.40 and 0.30 <= info.top <= 1.10
    ]
    if strict:
        # demo preference: the SPAWNED cube first — it is the only object with true
        # parallel pinch faces. ReplicaCAD furniture is built from visual-mesh convex
        # hulls / decompositions whose "flat" faces tilt up to 32° (book_04 measured),
        # so a parallel pinch cams them out regardless of alignment or friction.
        strict.sort(key=lambda info: (0 if "demo_pick" in info.name else 1,
                                      info.min_xy_dim,
                                      -min(float(info.size[2]), 0.16), info.name))
        chosen = strict[0]
        print(f"target_selection=strict count={len(strict)}", flush=True)
        return chosen

    fallback = [
        info
        for info in infos
        if info.mass < 1.0
        and info.min_xy_dim <= 0.075
        and info.max_dim < 0.50
        and 0.25 <= info.top <= 1.20
    ]
    if not fallback:
        raise RuntimeError(
            "no strict target and no fallback target: need mass<0.5, reachable top z, and a grippable minor axis"
        )
    fallback.sort(key=lambda info: (abs(info.top - 0.64), info.min_xy_dim, info.max_dim, info.name))
    chosen = fallback[0]
    print(
        "target_selection=fallback_no_strict_candidates "
        f"strict_count=0 fallback_count={len(fallback)} chosen={chosen.name} "
        "library_match=none_scene_inventory_constraint "
        f"reason=maxdim={chosen.max_dim:.3f}>0.09 but mass={chosen.mass:.3f}, "
        f"minor_axis={chosen.min_xy_dim:.3f}, top={chosen.top:.3f}",
        flush=True,
    )
    return chosen


def build_obstacle_list(
    env,
    target_name: str | None = None,
    include_heavy_movable: bool = True,
) -> tuple[list[dict[str, Any]], tuple[float, float, float, float]]:
    bounds = scene_extent(env)
    x_min, x_max, y_min, y_max = bounds
    obstacles: list[dict[str, Any]] = []

    for info in scene_actor_infos(env, include_movable=True):
        if info.name == target_name:
            continue
        lname = info.name.lower()
        if any(token in lname for token in ("rug", "mat", "carpet", "floor", "ceiling")):
            continue
        if info.top < 0.08 or info.size[2] < 0.06:
            continue

        is_static = info.kind == "static"
        is_heavy_or_large = (
            include_heavy_movable
            and info.kind == "movable"
            and (info.mass > 2.0 or info.max_xy_dim > 0.24 or info.size[2] > 0.25)
        )
        if not (is_static or is_heavy_or_large):
            continue

        half_xy = np.maximum(info.half[:2], np.array([0.02, 0.02], dtype=np.float64))
        obstacles.append(
            dict(
                center=info.center[:2].astype(float).tolist(),
                half=half_xy.astype(float).tolist(),
                margin=0.02,
                name=info.name,
                kind=info.kind,
            )
        )

    wall_thick = 0.05
    obstacles.extend(
        [
            dict(center=[(x_min + x_max) * 0.5, y_min], half=[(x_max - x_min) * 0.5, wall_thick], margin=0.0, name="bounds_wall_south", kind="wall"),
            dict(center=[(x_min + x_max) * 0.5, y_max], half=[(x_max - x_min) * 0.5, wall_thick], margin=0.0, name="bounds_wall_north", kind="wall"),
            dict(center=[x_min, (y_min + y_max) * 0.5], half=[wall_thick, (y_max - y_min) * 0.5], margin=0.0, name="bounds_wall_west", kind="wall"),
            dict(center=[x_max, (y_min + y_max) * 0.5], half=[wall_thick, (y_max - y_min) * 0.5], margin=0.0, name="bounds_wall_east", kind="wall"),
        ]
    )
    return obstacles, bounds


def make_occupancy_grid(
    obstacles: list[dict[str, Any]],
    bounds,
    robot_radius: float = ROBOT_RADIUS,
    margin_override: float | None = None,
):
    from data_gen.aspire_engine.nav_planner import OccupancyGrid

    x_min, x_max, y_min, y_max = [float(v) for v in bounds]
    grid = OccupancyGrid(
        origin_xy=(x_min, y_min),
        size_xy=(x_max - x_min, y_max - y_min),
        resolution=GRID_RESOLUTION,
    )
    for obs in obstacles:
        margin = float(obs.get("margin", 0.0)) if margin_override is None else float(margin_override)
        grid.add_box(obs["center"], obs["half"], margin=margin)
    grid.inflate(robot_radius)
    return grid


def make_station_occupancy_grid(obstacles: list[dict[str, Any]], bounds, robot_radius: float = 0.18):
    return make_occupancy_grid(obstacles, bounds, robot_radius=robot_radius, margin_override=0.0)


def plan_path_grid(obstacles: list[dict[str, Any]], start_xy, goal_xy, bounds, robot_radius: float = ROBOT_RADIUS):
    from data_gen.aspire_engine.nav_planner import astar, shortcut

    grid = make_occupancy_grid(obstacles, bounds, robot_radius=robot_radius)
    path = astar(grid, start_xy, goal_xy)
    if path is None:
        return None
    return shortcut(path, grid)


def is_free_xy(grid, xy) -> bool:
    ij = grid.world_to_cell(xy)
    return grid.in_bounds(ij) and not grid.is_occupied(ij)


def main() -> None:
    norender = bool(os.environ.get("NORENDER"))
    env = make_env(norender=norender, shader_pack="minimal")
    try:
        reset_fixed(env)
        floor_z = estimate_floor_height(env)
        root_z = set_robot_root_z(env, floor_z)
        ignored = disable_aloha_floor_cover_collisions(env)
        for _ in range(10):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        target = pick_target_object(env, verbose=True)
        obstacles, bounds = build_obstacle_list(env, target_name=target.name)
        grid = make_occupancy_grid(obstacles, bounds)
        free_cells = int((~grid.occupied).sum())
        total_cells = int(grid.occupied.size)
        max_contact, rows = robot_contact_summary(env)
        print(
            f"[SCENECHK] build_config_idx={BUILD_CONFIG_IDX} "
            f"build_config={env.unwrapped.scene_builder.build_configs[BUILD_CONFIG_IDX]} "
            f"movable_count={len(movable_actor_infos(env))} obstacle_count={len(obstacles)} "
            f"bounds={np.round(np.array(bounds), 3).tolist()} grid={grid.nx}x{grid.ny} "
            f"free_ratio={free_cells / max(1, total_cells):.3f}",
            flush=True,
        )
        print(
            f"[FLOORCHK] estimated_floor_z={floor_z:.4f} robot_root_z={root_z:.4f} "
            f"ignored_floor_cover_collisions={ignored} robot_contact_max_impulse={max_contact:.3f}",
            flush=True,
        )
        if rows:
            for norm, names, impulse, npoints in rows[:4]:
                print(f"[CONTACT] norm={norm:.3f} npoints={npoints} pair={names} impulse={np.round(impulse, 3).tolist()}", flush=True)
        print(
            f"[TARGET] name={target.name} mass={target.mass:.4f} "
            f"pos={np.round(target.center, 4).tolist()} size={np.round(target.size, 4).tolist()} "
            f"top={target.top:.4f}",
            flush=True,
        )
        print("RESULT: SCENE_OK", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
