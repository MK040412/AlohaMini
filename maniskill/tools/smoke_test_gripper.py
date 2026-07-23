#!/usr/bin/env python3
"""
End-to-end smoke test for the AlohaMini parallel gripper in ManiSkill.

Loads `aloha_mini_1` into an Empty-v1 scene on the GPU sim backend and
checks that:
  * the robot loads with 18 active DOF and the expected joint order,
  * the action space has the expected size (16: base3 + lift1 + arm5+grip1 x2),
  * commanding the gripper OPEN then CLOSED actually moves both fingers of each
    arm, symmetrically, to the commanded aperture.

Run (use the python that has mani_skill installed):
    /home/perelman/Basic_RL/.venv/bin/python maniskill/tools/smoke_test_gripper.py
"""

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers envs)

ROBOT = "aloha_mini_1"
NUM_ENVS = 4
OPEN = 0.042
CLOSED = 0.0


def settle(env, robot, action, steps=40):
    for _ in range(steps):
        env.step(action)
    return robot.get_qpos()


def main():
    print(f"Creating Empty-v1 with robot={ROBOT} on GPU ...")
    env = gym.make(
        "Empty-v1",
        robot_uids=ROBOT,
        num_envs=NUM_ENVS,
        control_mode="pd_joint_pos",
        obs_mode="state",
        sim_backend="physx_cuda",
    )
    env.reset(seed=0)
    base_env = env.unwrapped
    robot = base_env.agent.robot

    ok = True
    dof = robot.dof
    dof = int(dof[0]) if hasattr(dof, "__len__") else int(dof)
    names = [j.name for j in robot.active_joints]
    print(f"\nActive DOF = {dof}")
    for i, n in enumerate(names):
        print(f"  {i:2d}: {n}")
    ok &= (dof == 18)
    print(f"[{'PASS' if dof == 18 else 'FAIL'}] DOF == 18")

    # Resolve finger qpos indices BY NAME (SAPIEN's active-joint order is interleaved
    # across the two arms, so positional assumptions are unsafe).
    LEFT_FINGERS = (names.index("left_finger_joint1"), names.index("left_finger_joint2"))
    RIGHT_FINGERS = (names.index("right_finger_joint1"), names.index("right_finger_joint2"))
    print(f"left finger qpos idx = {LEFT_FINGERS}, right = {RIGHT_FINGERS}")

    act_dim = int(np.prod(env.action_space.shape[-1:]))
    print(f"\nAction dim = {act_dim}")
    ok &= (act_dim == 16)
    print(f"[{'PASS' if act_dim == 16 else 'FAIL'}] action dim == 16 "
          f"(base3 + lift1 + (arm5+grip1)x2)")

    for lk in ("left_finger1", "left_finger2", "right_finger1", "right_finger2",
               "left_Fixed_Jaw", "right_Fixed_Jaw"):
        present = lk in robot.links_map
        ok &= present
        print(f"[{'PASS' if present else 'FAIL'}] link present: {lk}")

    # Build actions: zeros everywhere (hold), gripper dims set to open/closed.
    # action layout = [base(3), lift(1), left_arm(5), left_grip(1), right_arm(5), right_grip(1)]
    LG, RG = 9, 15
    a_open = torch.zeros((NUM_ENVS, act_dim), dtype=torch.float32)
    a_open[:, LG] = OPEN
    a_open[:, RG] = OPEN
    a_close = torch.zeros((NUM_ENVS, act_dim), dtype=torch.float32)
    a_close[:, LG] = CLOSED
    a_close[:, RG] = CLOSED

    q_open = settle(env, robot, a_open).cpu().numpy()
    q_close = settle(env, robot, a_close).cpu().numpy()

    def finger_report(tag, q, target):
        lf = q[:, LEFT_FINGERS]
        rf = q[:, RIGHT_FINGERS]
        print(f"\n{tag} (target {target*1000:.0f} mm/finger):")
        print(f"  left  fingers (mean) = {lf.mean(0)*1000} mm")
        print(f"  right fingers (mean) = {rf.mean(0)*1000} mm")
        near = (np.abs(lf - target).max() < 0.006) and (np.abs(rf - target).max() < 0.006)
        sym = (np.abs(lf[:, 0] - lf[:, 1]).max() < 0.004) and \
              (np.abs(rf[:, 0] - rf[:, 1]).max() < 0.004)
        print(f"  [{'PASS' if near else 'FAIL'}] fingers reached target (<6 mm err)")
        print(f"  [{'PASS' if sym else 'FAIL'}] two fingers symmetric (<4 mm diff)")
        return near and sym

    ok &= finger_report("OPEN command", q_open, OPEN)
    ok &= finger_report("CLOSE command", q_close, CLOSED)

    # Travel: fingers must have actually moved between open and closed
    travel = float(np.abs(q_open[:, LEFT_FINGERS] - q_close[:, LEFT_FINGERS]).mean())
    print(f"\nMean left-finger travel open->closed = {travel*1000:.1f} mm")
    moved = travel > 0.02
    ok &= moved
    print(f"[{'PASS' if moved else 'FAIL'}] gripper actually actuates (>20 mm travel)")

    env.close()
    print("\n" + "=" * 56)
    print(f"SMOKE TEST {'PASSED' if ok else 'FAILED'}")
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
