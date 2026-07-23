#!/usr/bin/env python3
"""GUI viewer for the two official AlohaMini robots.

Usage (from maniskill/, with mani-skill installed):
    python view_urdf.py            # list keys
    python view_urdf.py mini2      # open that robot in the SAPIEN viewer

Keys:
    mini1    AlohaMini1  aloha_mini_1.urdf  SO100 arms + parallel gripper (repo assets, install.py)
    mini2    AlohaMini2  aloha_mini_2.urdf  official AM2 Pro arms + parallel gripper (GitHub Releases)
"""
import sys

KEYS = {
    "mini1": "aloha_mini_1",
    "mini2": "aloha_mini_2",
}

if len(sys.argv) < 2 or sys.argv[1] not in KEYS:
    print(__doc__)
    sys.exit(0 if len(sys.argv) < 2 else 1)

uid = KEYS[sys.argv[1]]
headless = "--headless" in sys.argv  # load-check only, no window

import gymnasium as gym
import agents.aloha_mini  # noqa: registers aloha_mini_1 / aloha_mini_2
import mani_skill.envs  # noqa

env = gym.make(
    "Empty-v1",
    robot_uids=uid,
    obs_mode="state",
    sim_backend="physx_cpu",
    render_mode=None if headless else "human",
)
env.reset(seed=0)
agent = env.unwrapped.agent
kf = getattr(agent, "keyframes", None) or {}
if "rest" in kf:
    agent.robot.set_qpos(kf["rest"].qpos)

if headless:
    print(f"[VIEW] {uid}: URDF loaded OK ({agent.urdf_path})")
    sys.exit(0)

print(f"[VIEW] {uid} — {agent.urdf_path}\n[VIEW] close the SAPIEN window to exit", flush=True)
while True:
    env.render()
