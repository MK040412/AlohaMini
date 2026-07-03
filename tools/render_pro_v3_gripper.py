#!/usr/bin/env python3
"""4-view close-up renders of the pro_v3 LEFT gripper region (front/top/side/
bottom) to visually verify the joint6 insertion and distal gripper mount."""
import os
import sys
from pathlib import Path

import numpy as np
import sapien
from sapien.utils import Viewer  # noqa: F401  (ensures renderer init on some builds)

ROOT = Path(__file__).resolve().parent.parent
URDF = ROOT / os.environ.get("PRO_URDF", "maniskill/aloha_mini/aloha_mini_pro_v3.urdf")
OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/pro_v3_views")
OUTDIR.mkdir(parents=True, exist_ok=True)

scene = sapien.Scene()
scene.set_ambient_light([0.55, 0.55, 0.55])
scene.add_directional_light([0.3, -0.5, -1.0], [1.6, 1.6, 1.6])
scene.add_directional_light([-0.6, 0.4, -0.6], [0.8, 0.8, 0.8])

loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load(str(URDF))
q = np.zeros(robot.dof, dtype=np.float32)
j6 = float(os.environ.get("J6", "0"))
names = [j.name for j in robot.get_active_joints()]
if "left_joint6" in names:
    q[names.index("left_joint6")] = j6
robot.set_qpos(q)
scene.update_render()

jaw = next(l for l in robot.get_links() if l.name == "left_Fixed_Jaw")
tip1 = next(l for l in robot.get_links() if l.name == "left_finger1_tip")
link5 = next((l for l in robot.get_links() if l.name == "left_link5"), jaw)
c = jaw.pose.p
print("frames: link5", np.round(link5.pose.p, 4), "jaw", np.round(c, 4),
      "tip1", np.round(tip1.pose.p, 4), flush=True)

# focus midway between wrist and fingertips
focus = (np.asarray(c) + np.asarray(tip1.pose.p)) / 2.0
D = 0.45
views = {
    "front": focus + np.array([0.0, -D, 0.0]),
    "side_right": focus + np.array([D, 0.0, 0.0]),
    "side_left": focus + np.array([-1.6*D, 0.05, 0.0]),
    "top": focus + np.array([0.0, 0.0, D]),
    "bottom": focus + np.array([0.0, 0.0, -D]),
}


def look_at(eye, target):
    f = np.asarray(target, np.float64) - np.asarray(eye, np.float64)
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4); m[:3, 0] = f; m[:3, 1] = -s; m[:3, 2] = u; m[:3, 3] = eye
    return sapien.Pose(m)


import imageio.v2 as imageio

cam = scene.add_camera("shot", width=960, height=720, fovy=0.9, near=0.01, far=10)
for name, eye in views.items():
    cam.entity.set_pose(look_at(eye, focus))
    scene.update_render()
    cam.take_picture()
    rgba = cam.get_picture("Color")
    img = (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)
    imageio.imwrite(OUTDIR / f"{name}.png", img)
    print("saved", name, flush=True)
print("DONE", OUTDIR, flush=True)
