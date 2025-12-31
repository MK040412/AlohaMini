#!/usr/bin/env python3
"""
Test first-person camera view on CPU (no GPU required).
Works on laptops and machines without CUDA GPU.

Usage:
    python test_firstperson_cpu.py
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    output_dir = Path(__file__).parent / "camera_views"
    output_dir.mkdir(exist_ok=True)

    # Clear module caches to ensure fresh config
    for m in list(sys.modules.keys()):
        if 'aloha_mini' in m or 'mani_skill.agents' in m:
            del sys.modules[m]

    from mani_skill.agents.registration import REGISTERED_AGENTS
    REGISTERED_AGENTS.pop('aloha_mini_virtual', None)

    # Import agent
    sys.path.insert(0, str(Path(__file__).parent.parent / 'agents' / 'aloha_mini'))
    import aloha_mini_virtual

    import gymnasium as gym
    import mani_skill.envs

    print("Creating environment (CPU backend)...")
    env = gym.make(
        'ReplicaCAD_SceneManipulation-v1',
        robot_uids='aloha_mini_virtual',
        render_mode=None,
        obs_mode='rgbd',
        sim_backend='cpu',  # CPU backend - no GPU required
        control_mode='pd_joint_pos',
    )

    print("Resetting environment...")
    obs, _ = env.reset(options=dict(reconfigure=True))

    # Move arms to extended position for visibility
    action = np.zeros(16)
    action[3] = 0.3  # lift
    # Left arm extended forward
    action[4:10] = [0, np.radians(45), np.radians(-60), np.radians(15), 0, np.radians(30)]
    # Right arm extended forward
    action[10:16] = [0, np.radians(45), np.radians(-60), np.radians(15), 0, np.radians(30)]

    print("Running simulation steps...")
    for i in range(80):
        obs, _, _, _, _ = env.step(action)
        if (i + 1) % 20 == 0:
            print(f"  Step {i + 1}/80")

    # Save image
    if 'sensor_data' in obs and 'cam_main' in obs['sensor_data']:
        rgb = obs['sensor_data']['cam_main']['rgb']
        if hasattr(rgb, 'cpu'):
            rgb = rgb.cpu().numpy()
        if len(rgb.shape) == 4:
            rgb = rgb[0]

        img = Image.fromarray(rgb[:, :, :3].astype(np.uint8))
        img_path = output_dir / "firstperson_cpu_test.png"
        img.save(img_path)
        print(f"\nSaved: {img_path}")
        print("\nFirst-person view should show:")
        print("  - Both arms at bottom corners (left and right)")
        print("  - Looking down at workspace/floor")
        print("  - Room environment visible ahead")
    else:
        print("ERROR: Camera data not found in observation")

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
