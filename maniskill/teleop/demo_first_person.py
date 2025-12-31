#!/usr/bin/env python3
"""
First-Person View Demo for AlohaMini in ManiSkill3

Shows the robot's camera view (cam_main) on a pygame window
while allowing keyboard/VR teleoperation.
"""

import sys
import time
import argparse
import logging
import numpy as np
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame

try:
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.utils.wrappers import RecordEpisode
except ImportError:
    print("Error: ManiSkill3 not installed")
    sys.exit(1)

from teleop import TeleopController, TeleopConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="First-Person View Demo")
    parser.add_argument("--robot", default="aloha_mini_virtual", help="Robot UID")
    parser.add_argument("--sim-backend", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--no-render", action="store_true", help="Disable 3D viewer")
    args = parser.parse_args()

    # Initialize pygame
    pygame.init()

    # Screen size: camera view + control panel
    cam_width, cam_height = 640, 480  # Upscaled from 320x240
    panel_width = 300
    screen_width = cam_width + panel_width
    screen_height = cam_height

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("AlohaMini First-Person View")

    font = pygame.font.Font(None, 28)
    small_font = pygame.font.Font(None, 20)

    # Initialize teleop controller
    config = TeleopConfig()
    teleop = TeleopController(config)

    # Create environment with camera observation
    render_mode = None if args.no_render else "human"

    env = gym.make(
        "ReplicaCAD_SceneManipulation-v1",
        robot_uids=args.robot,
        render_mode=render_mode,
        obs_mode="rgbd",  # Get RGB-D observations including camera
        sim_backend=args.sim_backend,
        control_mode="pd_joint_pos",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=None,
    )

    obs, _ = env.reset(options=dict(reconfigure=True))

    if not args.no_render:
        env.render()

    # Get robot reference
    robot = env.unwrapped.agent.robot if hasattr(env.unwrapped, "agent") else None

    # Print help
    print("\n" + "=" * 60)
    print("AlohaMini First-Person View Demo")
    print("=" * 60)
    print(teleop.get_help_text())
    print("Press X or ESC to exit")
    print("=" * 60 + "\n")

    clock = pygame.time.Clock()
    step_counter = 0
    warmup_steps = 30
    fps_list = []

    # Initial key flush - ignore any keys held at startup
    pygame.event.pump()
    pygame.event.clear()

    while teleop.is_running:
        frame_start = time.time()

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                teleop.is_running = False

        # Get keyboard state
        keys = pygame.key.get_pressed()

        if step_counter >= warmup_steps:
            # Process keyboard input
            if not teleop.process_keyboard(keys):
                break

            # Compute action
            action = teleop.compute_action()
        else:
            action = teleop.compute_action()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        step_counter += 1

        if not args.no_render:
            env.render()

        # === Get First-Person Camera Image ===
        # The camera is named "cam_main" and mounted on vertical_link
        cam_image = None

        if 'sensor_data' in obs and 'cam_main' in obs['sensor_data']:
            # Get RGB image from camera
            rgb = obs['sensor_data']['cam_main']['rgb']
            if hasattr(rgb, 'cpu'):
                rgb = rgb.cpu().numpy()
            if len(rgb.shape) == 4:
                rgb = rgb[0]  # Remove batch dimension

            # Convert to pygame surface
            # ManiSkill returns HWC format, pygame needs HWC but with different axis order
            rgb = np.ascontiguousarray(rgb[:, :, :3])  # Remove alpha if present

            # Upscale to display size
            cam_image = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            cam_image = pygame.transform.scale(cam_image, (cam_width, cam_height))

        # === Draw Screen ===
        screen.fill((30, 30, 30))

        # Draw camera view
        if cam_image is not None:
            screen.blit(cam_image, (0, 0))
        else:
            # No camera image - show placeholder
            pygame.draw.rect(screen, (50, 50, 50), (0, 0, cam_width, cam_height))
            no_cam = font.render("Camera not available", True, (200, 200, 200))
            screen.blit(no_cam, (cam_width // 2 - 100, cam_height // 2))

        # Draw border around camera
        pygame.draw.rect(screen, (100, 100, 100), (0, 0, cam_width, cam_height), 2)

        # === Draw Control Panel ===
        panel_x = cam_width + 10
        y_pos = 10

        # Title
        if step_counter < warmup_steps:
            title = font.render(f"WARMUP: {step_counter}/{warmup_steps}", True, (255, 100, 100))
        else:
            title = font.render("First-Person View", True, (100, 255, 100))
        screen.blit(title, (panel_x, y_pos))
        y_pos += 30

        # FPS
        fps = clock.get_fps()
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)
        fps_text = small_font.render(f"FPS: {avg_fps:.1f}", True, (150, 150, 150))
        screen.blit(fps_text, (panel_x, y_pos))
        y_pos += 25

        # Arm states
        y_pos += 10
        state = teleop.get_state_info()

        # Left arm
        left_header = small_font.render("Left Arm:", True, (100, 200, 255))
        screen.blit(left_header, (panel_x, y_pos))
        y_pos += 20

        left_ee = f"  EE: ({state['left_arm']['ee_x']:.3f}, {state['left_arm']['ee_y']:.3f})"
        screen.blit(small_font.render(left_ee, True, (200, 200, 200)), (panel_x, y_pos))
        y_pos += 18

        left_joints = state['left_arm']['joints_deg']
        left_j = f"  J1-3: {left_joints[0]:.0f}, {left_joints[1]:.0f}, {left_joints[2]:.0f}"
        screen.blit(small_font.render(left_j, True, (200, 200, 200)), (panel_x, y_pos))
        y_pos += 18

        left_j2 = f"  J4-6: {left_joints[3]:.0f}, {left_joints[4]:.0f}, {left_joints[5]:.0f}"
        screen.blit(small_font.render(left_j2, True, (200, 200, 200)), (panel_x, y_pos))
        y_pos += 25

        # Right arm
        right_header = small_font.render("Right Arm:", True, (255, 200, 100))
        screen.blit(right_header, (panel_x, y_pos))
        y_pos += 20

        right_ee = f"  EE: ({state['right_arm']['ee_x']:.3f}, {state['right_arm']['ee_y']:.3f})"
        screen.blit(small_font.render(right_ee, True, (200, 200, 200)), (panel_x, y_pos))
        y_pos += 18

        right_joints = state['right_arm']['joints_deg']
        right_j = f"  J1-3: {right_joints[0]:.0f}, {right_joints[1]:.0f}, {right_joints[2]:.0f}"
        screen.blit(small_font.render(right_j, True, (200, 200, 200)), (panel_x, y_pos))
        y_pos += 18

        right_j2 = f"  J4-6: {right_joints[3]:.0f}, {right_joints[4]:.0f}, {right_joints[5]:.0f}"
        screen.blit(small_font.render(right_j2, True, (200, 200, 200)), (panel_x, y_pos))
        y_pos += 25

        # Controls help
        y_pos += 10
        help_header = small_font.render("Controls:", True, (200, 200, 200))
        screen.blit(help_header, (panel_x, y_pos))
        y_pos += 18

        controls = [
            "W/S: Forward/Back",
            "E/D: Up/Down",
            "Q/A: Rotate",
            "SPACE: Reset",
            "X/ESC: Exit",
        ]
        for ctrl in controls:
            screen.blit(small_font.render(ctrl, True, (150, 150, 150)), (panel_x + 10, y_pos))
            y_pos += 16

        pygame.display.flip()
        clock.tick(50)

    # Cleanup
    pygame.quit()
    env.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
