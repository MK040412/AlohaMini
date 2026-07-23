#!/usr/bin/env python3
"""
Headless validation for the AlohaMini 2 Pro ManiSkill agent (uid=aloha_mini_2).

Loads the real dual-arm 6-DOF AM2 Pro robot into an Empty-v1 scene on the CPU
physx backend (state obs, NO rendering — a photoreal GPU render crashed the
machine once, so this stays lightweight) and checks:

  * robot loads; active joint count + names + qpos shape;
  * action dim of the fixed-base control mode;
  * the parallel-gripper coupling: commanding one gripper action dim moves BOTH
    clamp joints together (leader -> +x, follower -> -x, mirrored);
  * TCP (left_Fixed_Jaw) world position.

Run:
    python maniskill/agents/aloha_mini/_validate_am2pro.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers Empty-v1)

# make `import agents.aloha_mini` importable when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import agents.aloha_mini  # noqa: F401  (registers aloha_mini_2)

ROBOT = "aloha_mini_2"
CONTROL_MODE = "pd_joint_pos_fixed_base"

# left_arm order = [pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll]
ARM_REST = [0.0, -1.5, 2.5, 0.0, 0.0, 0.0]

PASS, FAIL = [], []


def check(cond, msg):
    (PASS if cond else FAIL).append(msg)
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    return bool(cond)


def main():
    print(f"Creating Empty-v1 with robot={ROBOT}, control_mode={CONTROL_MODE}, "
          f"backend=physx_cpu, obs=state, render=None ...")
    env = gym.make(
        "Empty-v1",
        robot_uids=ROBOT,
        num_envs=1,
        control_mode=CONTROL_MODE,
        obs_mode="state",
        sim_backend="physx_cpu",
        render_mode=None,
    )
    env.reset(seed=0)
    base_env = env.unwrapped
    agent = base_env.agent
    robot = agent.robot

    # ---- structure ---------------------------------------------------------
    dof = int(robot.dof[0]) if hasattr(robot.dof, "__len__") else int(robot.dof)
    names = [j.name for j in robot.active_joints]
    print(f"\nrobot loaded: name={robot.name!r}  active DOF = {dof}")
    for i, n in enumerate(names):
        print(f"  {i:2d}: {n}")
    check(dof == 20, "active DOF == 20")

    qpos = robot.get_qpos()
    print(f"\nqpos shape = {tuple(qpos.shape)}")
    check(tuple(qpos.shape) == (1, 20), "qpos shape == (1, 20)")

    act_dim = int(env.action_space.shape[-1])
    print(f"action dim ({CONTROL_MODE}) = {act_dim}")
    # base3 + lift1 + (arm6 + gripper1) x2 = 18
    check(act_dim == 18, "action dim == 18  (base3 + lift1 + (arm6+grip1)x2)")

    # clamp joint qpos indices, by name (order is SAPIEN-decided)
    L_LEAD = names.index("left_right_clamp")     # leader,   [0, +0.037]
    L_FOLL = names.index("left_left_clamp")       # follower, [-0.038, 0]
    R_LEAD = names.index("right_right_clamp")
    R_FOLL = names.index("right_left_clamp")
    print(f"\nleft  gripper qpos idx: leader(left_right_clamp)={L_LEAD}, "
          f"follower(left_left_clamp)={L_FOLL}")
    print(f"right gripper qpos idx: leader(right_right_clamp)={R_LEAD}, "
          f"follower(right_left_clamp)={R_FOLL}")

    # action layout for fixed-base mode:
    #   base(3) lift(1) left_arm(6) left_grip(1) right_arm(6) right_grip(1)
    LG_ACT, RG_ACT = 10, 17

    def hold_action(gval):
        a = torch.zeros((1, act_dim), dtype=torch.float32)
        a[:, 4:10] = torch.tensor(ARM_REST)     # left arm hold
        a[:, 11:17] = torch.tensor(ARM_REST)    # right arm hold
        a[:, LG_ACT] = gval                      # left gripper
        a[:, RG_ACT] = gval                      # right gripper
        return a

    def settle(gval, steps=60):
        a = hold_action(gval)
        for _ in range(steps):
            env.step(a)
        return robot.get_qpos()[0].cpu().numpy()

    OPEN, CLOSED = 0.037, 0.0
    q_open = settle(OPEN)
    q_closed = settle(CLOSED)

    print("\n=== gripper coupling (one action dim drives both clamps) ===")
    for side, lead_i, foll_i in (("left", L_LEAD, L_FOLL), ("right", R_LEAD, R_FOLL)):
        lo, fo = q_open[lead_i], q_open[foll_i]
        lc, fc = q_closed[lead_i], q_closed[foll_i]
        print(f"  {side}: OPEN  cmd -> leader={lo:+.4f}  follower={fo:+.4f}  "
              f"(follower should mirror = -leader)")
        print(f"  {side}: CLOSE cmd -> leader={lc:+.4f}  follower={fc:+.4f}")
        # leader tracks the commanded open target
        check(abs(lo - OPEN) < 5e-3, f"{side} leader reaches OPEN target")
        # follower mirrors the leader (multiplier -1)
        check(abs(fo - (-lo)) < 5e-3, f"{side} follower mirrors leader when OPEN "
                                      f"(|follower-(-leader)|={abs(fo-(-lo)):.4f})")
        # both return near 0 when closed
        check(abs(lc) < 5e-3 and abs(fc) < 5e-3, f"{side} both clamps ~0 when CLOSED")
        # the two clamps actually MOVED together between the two commands
        moved_lead = abs(lo - lc)
        moved_foll = abs(fo - fc)
        check(moved_lead > 0.02 and moved_foll > 0.02,
              f"{side} both clamps moved (leader d={moved_lead:.3f}, follower d={moved_foll:.3f})")

    # ---- TCP ---------------------------------------------------------------
    # settle back to closed/open-neutral for a clean TCP read
    settle(OPEN, steps=30)
    tcp = agent.tcp_pos[0].cpu().numpy()
    tcp2 = agent.tcp_pos_2[0].cpu().numpy()
    palm = robot.links_map["left_Fixed_Jaw"].pose.p[0].cpu().numpy()
    print("\n=== TCP (palm links) ===")
    print(f"  left  TCP (left_Fixed_Jaw)  world pos = "
          f"[{tcp[0]:+.4f}, {tcp[1]:+.4f}, {tcp[2]:+.4f}]")
    print(f"  right TCP (right_Fixed_Jaw) world pos = "
          f"[{tcp2[0]:+.4f}, {tcp2[1]:+.4f}, {tcp2[2]:+.4f}]")
    check(np.allclose(tcp, palm, atol=1e-4), "agent.tcp_pos == left_Fixed_Jaw link pos")
    check(tcp[2] > 0.5, "left TCP is well above the floor (folded rest)")

    print(f"\n==== {len(PASS)} passed, {len(FAIL)} failed ====")
    env.close()
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
