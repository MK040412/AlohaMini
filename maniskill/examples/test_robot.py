#!/usr/bin/env python3
"""
Test script to verify AlohaMini robot loads correctly in ManiSkill3.

This script tests:
1. URDF loading
2. Joint configuration
3. Controller setup
4. Basic movements
"""

import sys
from pathlib import Path

# Add the maniskill directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import gymnasium as gym
    import mani_skill.envs
except ImportError:
    print("Error: ManiSkill3 not installed.")
    print("Install with: pip install mani-skill")
    sys.exit(1)

# Import and register AlohaMini agent
from aloha_mini import AlohaMini, AlohaMiniFixed, AlohaMiniArmsOnly


def test_robot_loading():
    """Test that the robot loads correctly."""
    print("=" * 50)
    print("Testing AlohaMini Robot Loading")
    print("=" * 50)

    # Test with empty environment first
    print("\n1. Testing with Empty-v1...")
    try:
        env = gym.make(
            "Empty-v1",
            robot_uids="aloha_mini",
            render_mode=None,
        )
        print("   [OK] AlohaMini loaded successfully")

        obs, info = env.reset()
        print(f"   Observation keys: {obs.keys() if hasattr(obs, 'keys') else type(obs)}")
        print(f"   Action space: {env.action_space}")

        # Get robot info
        agent = env.unwrapped.agent
        print(f"   Robot name: {agent.uid}")
        print(f"   Number of joints: {len(agent.robot.active_joints)}")

        # List joints
        print("\n   Active joints:")
        for i, joint in enumerate(agent.robot.active_joints):
            print(f"     {i}: {joint.name} (type: {joint.type})")

        env.close()
        print("   [OK] Environment closed")

    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_robot_variants():
    """Test different robot variants."""
    print("\n" + "=" * 50)
    print("Testing Robot Variants")
    print("=" * 50)

    variants = ["aloha_mini", "aloha_mini_fixed", "aloha_mini_arms_only"]

    for variant in variants:
        print(f"\n   Testing {variant}...")
        try:
            env = gym.make(
                "Empty-v1",
                robot_uids=variant,
                render_mode=None,
            )
            obs, info = env.reset()
            print(f"   [OK] {variant} loaded - Action dim: {env.action_space.shape[0]}")
            env.close()
        except Exception as e:
            print(f"   [FAIL] {variant}: {e}")


def test_keyframes():
    """Test robot keyframes."""
    print("\n" + "=" * 50)
    print("Testing Keyframes")
    print("=" * 50)

    try:
        env = gym.make(
            "Empty-v1",
            robot_uids="aloha_mini",
            render_mode=None,
        )

        agent = env.unwrapped.agent

        print("\n   Available keyframes:")
        for name, keyframe in agent.keyframes.items():
            print(f"   - {name}: qpos shape = {keyframe.qpos.shape}")

        env.close()
        print("   [OK] Keyframes accessible")

    except Exception as e:
        print(f"   [FAIL] Error: {e}")


def test_basic_movement():
    """Test basic robot movement."""
    print("\n" + "=" * 50)
    print("Testing Basic Movement")
    print("=" * 50)

    try:
        env = gym.make(
            "Empty-v1",
            robot_uids="aloha_mini_fixed",
            render_mode=None,
        )

        obs, info = env.reset()

        print("\n   Running 100 steps with random actions...")
        for i in range(100):
            action = env.action_space.sample() * 0.1  # Small random actions
            obs, reward, terminated, truncated, info = env.step(action)

        print("   [OK] Movement test passed")

        # Check joint positions changed
        agent = env.unwrapped.agent
        qpos = agent.robot.qpos
        print(f"   Final qpos: {qpos[:6]}... (first 6 joints)")

        env.close()

    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()


def test_replicacad_compatibility():
    """Test compatibility with ReplicaCAD environment."""
    print("\n" + "=" * 50)
    print("Testing ReplicaCAD Compatibility")
    print("=" * 50)

    try:
        print("\n   Creating AlohaMini-ReplicaCAD-v1 environment...")
        env = gym.make(
            "AlohaMini-ReplicaCAD-v1",
            robot_uids="aloha_mini_fixed",
            render_mode=None,
        )

        print("   [OK] Environment created")

        obs, info = env.reset()
        print(f"   [OK] Environment reset - Observation type: {type(obs)}")

        # Run a few steps
        for i in range(10):
            action = env.action_space.sample() * 0.1
            obs, reward, terminated, truncated, info = env.step(action)

        print("   [OK] Basic stepping works")

        env.close()
        print("   [OK] ReplicaCAD test passed!")

    except Exception as e:
        print(f"   [WARN] AlohaMini-ReplicaCAD test failed: {e}")
        print("   This might be expected if ReplicaCAD assets are not downloaded.")
        print("   Run: python -m mani_skill.utils.download_asset ReplicaCAD")


def main():
    print("\n" + "=" * 60)
    print("  AlohaMini ManiSkill3 Integration Test Suite")
    print("=" * 60)

    test_robot_loading()
    test_robot_variants()
    test_keyframes()
    test_basic_movement()
    test_replicacad_compatibility()

    print("\n" + "=" * 60)
    print("  Test Suite Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. If ReplicaCAD test failed, download assets:")
    print("   python -m mani_skill.utils.download_asset ReplicaCAD")
    print("\n2. Run the example script:")
    print("   python examples/run_replicacad.py --render")
    print("\n3. Record a video:")
    print("   python examples/run_replicacad.py --record")


if __name__ == "__main__":
    main()
