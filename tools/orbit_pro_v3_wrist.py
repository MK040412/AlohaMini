#!/usr/bin/env python3
"""360-degree orbit mp4 around the pro_v3 wrist/gripper junction, ray-traced
(rt shader, optix denoiser) so surface detail reads; falls back to raster if
rt is unavailable. Usage: orbit_pro_v3_wrist.py [outdir]"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import sapien

ROOT = Path(__file__).resolve().parent.parent
URDF = ROOT / os.environ.get("PRO_URDF", "maniskill/aloha_mini/aloha_mini_pro_v3.urdf")
OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/pro_v3_orbit")
OUTDIR.mkdir(parents=True, exist_ok=True)

shader = os.environ.get("SHADER", "rt")
if shader == "rt":
    try:
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(int(os.environ.get("SPP", "16")))
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("optix")
        print("shader: rt", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"rt unavailable ({exc}); raster fallback", flush=True)

scene = sapien.Scene()
scene.set_ambient_light([0.22, 0.22, 0.24])
scene.add_directional_light([0.4, -0.6, -1.0], [1.1, 1.05, 1.0], shadow=True)
scene.add_directional_light([-0.7, 0.5, -0.4], [0.4, 0.4, 0.45])

loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load(str(URDF))
q = np.zeros(robot.dof, dtype=np.float32)
names = [j.name for j in robot.get_active_joints()]
if "left_joint6" in names:
    q[names.index("left_joint6")] = float(os.environ.get("J6", "0"))
robot.set_qpos(q)
scene.update_render()

jaw = next(l for l in robot.get_links() if l.name == "left_Fixed_Jaw")
link4 = next(l for l in robot.get_links() if l.name == "left_link4")
focus = (np.asarray(jaw.pose.p) + np.asarray(link4.pose.p)) / 2.0  # the junction


def look_at(eye, target):
    f = np.asarray(target, np.float64) - np.asarray(eye, np.float64)
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4); m[:3, 0] = f; m[:3, 1] = -s; m[:3, 2] = u; m[:3, 3] = eye
    return sapien.Pose(m)


import imageio.v2 as imageio

cam = scene.add_camera("orbit", width=960, height=720, fovy=0.75, near=0.01, far=10)
N = int(os.environ.get("FRAMES", "120"))
R = float(os.environ.get("RADIUS", "0.35"))
frames = []
for i in range(N):
    th = 2 * math.pi * i / N
    eye = focus + np.array([R * math.cos(th), R * math.sin(th), 0.15])
    cam.entity.set_pose(look_at(eye, focus))
    scene.update_render()
    cam.take_picture()
    rgba = cam.get_picture("Color")
    frames.append((np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8))
    if i % 20 == 0:
        print(f"frame {i}/{N}", flush=True)

out = OUTDIR / "pro_v3_wrist_orbit.mp4"
imageio.mimwrite(out, frames, fps=25, codec="libx264", quality=8)
print("WROTE", out, flush=True)
