#!/usr/bin/env python3
"""
Visual + functional verification of the AlohaMini parallel gripper.

For 5 consecutive rounds it: poses the left gripper in front of a close-up
camera, commands OPEN then CLOSED, saves a rendered image of each, and checks
that the two finger pads reach the commanded aperture symmetrically. The run is
"confirmed" only if all 5 rounds pass.

Run:
    /home/perelman/Basic_RL/.venv/bin/python maniskill/tools/verify_gripper_visual.py
Images are written to maniskill/data_gen/videos/gripper/.
"""

import os
import sys

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import data_gen  # noqa: F401  (registers AlohaMiniGripperView-v1)

OUT = os.path.join(os.path.dirname(__file__), "..", "data_gen", "videos", "gripper")
os.makedirs(OUT, exist_ok=True)
OPEN, CLOSED = 0.037, 0.0
ROUNDS = 5


def save_img(arr, path):
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, arr.astype(np.uint8))
    except Exception:
        from PIL import Image
        Image.fromarray(arr.astype(np.uint8)).save(path)


def main():
    env = gym.make("AlohaMiniGripperView-v1", num_envs=1, obs_mode="state",
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   reward_mode="none", sim_backend="physx_cpu")
    be = env.unwrapped
    names = [j.name for j in be.agent.robot.active_joints]
    LF = (names.index("left_finger_joint1"), names.index("left_finger_joint2"))
    VIEW_ARM = be.unwrapped.VIEW_ARM

    def action(grip):
        a = np.zeros(16, np.float32)
        a[4:9] = VIEW_ARM      # hold the viewing arm pose
        a[9] = grip            # left gripper aperture (meters)
        return torch.tensor(a[None])

    def settle_and_read(grip, steps=35):
        for _ in range(steps):
            env.step(action(grip))
        q = be.agent.robot.get_qpos()[0].cpu().numpy()
        return np.array([q[LF[0]], q[LF[1]]])

    results = []
    for r in range(ROUNDS):
        env.reset(seed=100 + r)
        fo = settle_and_read(OPEN)
        save_img(env.render(), os.path.join(OUT, f"open_{r}.png"))
        fc = settle_and_read(CLOSED)
        save_img(env.render(), os.path.join(OUT, f"closed_{r}.png"))

        open_ok = np.abs(fo - OPEN).max() < 0.006
        closed_ok = np.abs(fc - CLOSED).max() < 0.006
        sym_o = abs(fo[0] - fo[1]) < 0.004
        sym_c = abs(fc[0] - fc[1]) < 0.004
        travel = float(np.abs(fo - fc).mean())
        moved = travel > 0.02
        ok = open_ok and closed_ok and sym_o and sym_c and moved
        results.append(ok)
        print(f"round {r}: open={fo*1000} mm closed={fc*1000} mm travel={travel*1000:.1f}mm "
              f"| open_ok={open_ok} closed_ok={closed_ok} sym={sym_o and sym_c} moved={moved} "
              f"=> {'PASS' if ok else 'FAIL'}")

    env.close()
    n = sum(results)
    print("\n" + "=" * 56)
    print(f"GRIPPER VERIFICATION: {n}/{ROUNDS} rounds passed "
          f"-> {'CONFIRMED' if n == ROUNDS else 'NOT CONFIRMED'}")
    print(f"images in: {os.path.abspath(OUT)}")
    print("=" * 56)
    return 0 if n == ROUNDS else 1


if __name__ == "__main__":
    sys.exit(main())
