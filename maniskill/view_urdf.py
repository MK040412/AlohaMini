#!/usr/bin/env python3
"""GUI viewer for every AlohaMini URDF variant — one key per robot.

Usage (from maniskill/, Basic_RL venv):
    python view_urdf.py            # list keys
    python view_urdf.py so100      # open that robot in the SAPIEN viewer

Keys:
    mini     aloha_mini.urdf              original AlohaMini Std (6-DOF arm + jaw gripper)
    so100    maniskill_so100_version.urdf SO100 arms + roboninecom parallel gripper  [ACTIVE: VLA/vec_datagen]
    pro_v2   aloha_mini_pro_v2.urdf       Pro arm 5-DOF + parallel gripper graft     [legacy]
    pro_v3   aloha_mini_pro_v3.urdf       pro_v2 + restored 6th wrist joint          [legacy]
    am2_pro  alohamini2pro_parallel.urdf  official AlohaMini2 Pro + parallel gripper [ACTIVE: data_gen/tasks + CuRobo IK]
"""
import sys

KEYS = {
    "mini": "aloha_mini",
    "so100": "aloha_mini_so100_v2",
    "pro_v2": "aloha_mini_pro_v2",
    "pro_v3": "aloha_mini_pro_v3",
    "am2_pro": "aloha_mini2_pro",
}

if len(sys.argv) < 2 or sys.argv[1] not in KEYS:
    print(__doc__)
    sys.exit(0 if len(sys.argv) < 2 else 1)

uid = KEYS[sys.argv[1]]
headless = "--headless" in sys.argv  # load-check only, no window

import gymnasium as gym
import aloha_mini  # noqa: registers aloha_mini (original)
import agents.aloha_mini  # noqa: registers so100_v2 / pro_v2 / pro_v3 / aloha_mini2_pro
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
