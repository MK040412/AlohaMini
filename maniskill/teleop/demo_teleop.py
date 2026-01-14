#!/usr/bin/env python3
"""
AlohaMini IK Teleoperation Demo for ManiSkill3

This demo uses XLeRobot-compatible keyboard controls for IK-based arm teleoperation.

Usage:
    python demo_teleop.py --render
    python demo_teleop.py --render --shader rt-fast

Keyboard Controls (XLeRobot Style):
    LEFT ARM:
        Q/A: Joint 1 (shoulder_pan) -/+
        W/S: End effector X (forward/back)
        E/D: End effector Y (down/up)
        R/F: Pitch adjustment +/-
        T/G: Wrist roll -/+
        Y/H: Gripper close/open

    RIGHT ARM:
        U/J: Joint 1 (shoulder_pan) -/+
        I/K: End effector X (forward/back)
        O/L: End effector Y (down/up)
        P/;: Pitch adjustment +/-
        [/': Wrist roll -/+
        ]/\\: Gripper close/open

    GENERAL:
        SPACE: Reset arms to initial position
        X/ESC: Exit
"""

import argparse
import math
import sys
import time

import numpy as np

try:
    import gymnasium as gym
    import mani_skill.envs
    import sapien
    # Import agents to ensure they are registered
    from mani_skill.agents.robots import aloha_mini
except ImportError:
    print("Error: ManiSkill3 not installed. Install with: pip install mani-skill")
    sys.exit(1)

try:
    import pygame
except ImportError:
    print("Error: pygame not installed. Install with: pip install pygame")
    sys.exit(1)

# Add parent directory to path for imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import agents to ensure they are registered
from agents.aloha_mini import AlohaMiniSO100V2

from teleop.controller import TeleopController
from teleop.config import TeleopConfig


def draw_arm_side_view(screen, arm_state, color, viz_rect, label, font):
    """
    Draw arm in side view (ee_x/ee_y plane).

    This shows the 2-link arm configuration with:
    - ee_x: forward distance from shoulder (horizontal)
    - ee_y: height from shoulder (vertical)
    """
    # Background
    pygame.draw.rect(screen, (40, 40, 40), viz_rect)
    pygame.draw.rect(screen, (80, 80, 80), viz_rect, 1)

    # Parameters
    center_x = viz_rect.left + 30  # Shoulder position (left side)
    base_y = viz_rect.bottom - 30  # Shoulder height
    pixels_per_meter = 500  # Scale factor

    # Arm lengths (SO101)
    l1 = 0.1159  # Upper arm
    l2 = 0.1350  # Lower arm

    # Draw workspace boundary (quarter arc)
    r_max = int(0.2509 * pixels_per_meter)
    r_min = int(0.0191 * pixels_per_meter)

    # Draw max reach arc
    arc_rect = pygame.Rect(
        center_x - r_max,
        base_y - r_max,
        r_max * 2,
        r_max * 2
    )
    pygame.draw.arc(screen, (60, 60, 60), arc_rect, 0, math.pi/2, 1)

    # Get current EE position
    ee_x = arm_state.ee_x
    ee_y = arm_state.ee_y

    # Convert EE to screen coordinates
    ee_screen_x = center_x + int(ee_x * pixels_per_meter)
    ee_screen_y = base_y - int(ee_y * pixels_per_meter)

    # Draw shoulder (origin point)
    pygame.draw.circle(screen, (150, 150, 150), (center_x, base_y), 4)

    # Draw a line from shoulder to EE (simplified - just shows the reach)
    pygame.draw.line(screen, (100, 100, 100), (center_x, base_y), (ee_screen_x, ee_screen_y), 1)

    # Draw EE point
    pygame.draw.circle(screen, color, (ee_screen_x, ee_screen_y), 6)

    # Draw axis labels
    small_font = pygame.font.SysFont(None, 16)

    # X axis arrow and label
    pygame.draw.line(screen, (100, 100, 100), (center_x, base_y), (center_x + 50, base_y), 1)
    pygame.draw.polygon(screen, (100, 100, 100), [
        (center_x + 50, base_y),
        (center_x + 45, base_y - 3),
        (center_x + 45, base_y + 3)
    ])
    x_label = small_font.render("X", True, (100, 100, 100))
    screen.blit(x_label, (center_x + 52, base_y - 6))

    # Y axis arrow and label
    pygame.draw.line(screen, (100, 100, 100), (center_x, base_y), (center_x, base_y - 50), 1)
    pygame.draw.polygon(screen, (100, 100, 100), [
        (center_x, base_y - 50),
        (center_x - 3, base_y - 45),
        (center_x + 3, base_y - 45)
    ])
    y_label = small_font.render("Y", True, (100, 100, 100))
    screen.blit(y_label, (center_x - 10, base_y - 60))

    # Draw label
    text = font.render(label, True, color)
    screen.blit(text, (viz_rect.x + 5, viz_rect.y + 3))

    # Draw EE coordinates
    coord_text = small_font.render(f"({ee_x:.3f}, {ee_y:.3f})", True, color)
    screen.blit(coord_text, (viz_rect.x + 5, viz_rect.y + 20))


def main():
    parser = argparse.ArgumentParser(description="AlohaMini IK Teleoperation Demo")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--shader", choices=["default", "rt", "rt-fast"], default="default")
    parser.add_argument("--sim-backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--robot", choices=["aloha_mini_so100_v2"],
                        default="aloha_mini_so100_v2", help="Robot variant")
    args = parser.parse_args()

    pygame.init()

    # Create control window (needed for keyboard focus)
    screen_width = 650
    screen_height = 580  # Increased height for arm visualization
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("AlohaMini IK Teleop - Keep this window focused!")
    font = pygame.font.SysFont(None, 22)

    np.set_printoptions(suppress=True, precision=3)

    # Create environment
    render_mode = "human" if args.render else None

    env = gym.make(
        "ReplicaCAD_SceneManipulation-v1",
        robot_uids=args.robot,
        render_mode=render_mode,
        obs_mode="state",
        sim_backend=args.sim_backend,
        control_mode="pd_joint_pos",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        enable_shadow=True,
        max_episode_steps=None,
    )

    obs, _ = env.reset(options=dict(reconfigure=True))

    if args.render:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = False
        env.render()

    # Get robot reference
    robot = None
    if hasattr(env.unwrapped, "agent"):
        robot = env.unwrapped.agent.robot
    print(f"Robot: {robot}")

    # Get action space info
    action = env.action_space.sample()
    action_dim = len(action)
    print(f"Action space dimension: {action_dim}")

    # Create teleop controller
    config = TeleopConfig()
    teleop = TeleopController(config, robot_variant=args.robot)

    # Print help
    print(teleop.get_help_text())

    clock = pygame.time.Clock()
    step_counter = 0
    warmup_steps = 30

    while teleop.is_running:
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

            # Compute action from teleop state
            action = teleop.compute_action()
        else:
            # Warmup - use initial IK positions
            action = teleop.compute_action()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        step_counter += 1

        if args.render:
            env.render()
            # Update camera to follow robot from above
            viewer = env.unwrapped._viewer
            if viewer is not None and robot is not None:
                try:
                    robot_pos = robot.pose.p
                    cam_x = float(robot_pos[0]) - 1.5
                    cam_y = float(robot_pos[1])
                    cam_z = float(robot_pos[2]) + 3.0
                    viewer.set_camera_xyz(cam_x, cam_y, cam_z)
                    viewer.set_camera_rpy(0, -1.0, 0)
                except:
                    pass

        # === Draw Control Panel ===
        screen.fill((30, 30, 30))
        y_pos = 10

        # Title
        if step_counter < warmup_steps:
            title = font.render(f"WARMUP: {step_counter}/{warmup_steps}", True, (255, 100, 100))
        else:
            title = font.render("AlohaMini IK Teleop (XLeRobot Style)", True, (100, 255, 100))
        screen.blit(title, (10, y_pos))
        y_pos += 28

        # Controls help (compact)
        controls = [
            "BASE: 8/5=Fwd/Back  4/6=Strafe  7/9=Rotate  PgUp/Dn=Lift",
            "LEFT: Q/A=J1  W/S=X  E/D=Y  R/F=pitch  T/G=roll  Y/H=grip",
            "RIGHT: U/J=J1  I/K=X  O/L=Y  P/;=pitch  [/'=roll  ]/\\=grip",
            "SPACE=Reset  X/ESC=Exit",
        ]
        for ctrl in controls:
            text = font.render(ctrl, True, (200, 200, 200))
            screen.blit(text, (10, y_pos))
            y_pos += 20

        y_pos += 10

        # Get state info
        state = teleop.get_state_info()

        # Left arm state
        text = font.render("LEFT ARM:", True, (100, 200, 255))
        screen.blit(text, (10, y_pos))
        y_pos += 22

        left = state['left_arm']
        text = font.render(f"  EE: x={left['ee_x']:.4f}  y={left['ee_y']:.4f}  pitch={left['pitch']:.1f}", True, (200, 200, 255))
        screen.blit(text, (10, y_pos))
        y_pos += 20

        joints = left['joints_deg']
        text = font.render(f"  J1-3: [{joints[0]:.1f}, {joints[1]:.1f}, {joints[2]:.1f}]", True, (200, 200, 255))
        screen.blit(text, (10, y_pos))
        y_pos += 20

        text = font.render(f"  J4-6: [{joints[3]:.1f}, {joints[4]:.1f}, {joints[5]:.1f}]", True, (200, 200, 255))
        screen.blit(text, (10, y_pos))
        y_pos += 25

        # Right arm state
        text = font.render("RIGHT ARM:", True, (255, 200, 100))
        screen.blit(text, (10, y_pos))
        y_pos += 22

        right = state['right_arm']
        text = font.render(f"  EE: x={right['ee_x']:.4f}  y={right['ee_y']:.4f}  pitch={right['pitch']:.1f}", True, (255, 200, 200))
        screen.blit(text, (10, y_pos))
        y_pos += 20

        joints = right['joints_deg']
        text = font.render(f"  J1-3: [{joints[0]:.1f}, {joints[1]:.1f}, {joints[2]:.1f}]", True, (255, 200, 200))
        screen.blit(text, (10, y_pos))
        y_pos += 20

        text = font.render(f"  J4-6: [{joints[3]:.1f}, {joints[4]:.1f}, {joints[5]:.1f}]", True, (255, 200, 200))
        screen.blit(text, (10, y_pos))
        y_pos += 25

        # Action output
        text = font.render("ACTION (rad):", True, (100, 255, 100))
        screen.blit(text, (10, y_pos))
        y_pos += 22

        text = font.render(f"  Base: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]  Lift: {action[3]:.3f}", True, (150, 255, 150))
        screen.blit(text, (10, y_pos))
        y_pos += 20

        text = font.render(f"  Left:  {np.array2string(action[4:10], precision=2, separator=', ')}", True, (150, 255, 150))
        screen.blit(text, (10, y_pos))
        y_pos += 20

        text = font.render(f"  Right: {np.array2string(action[10:16], precision=2, separator=', ')}", True, (150, 255, 150))
        screen.blit(text, (10, y_pos))
        y_pos += 25

        # Workspace info
        if hasattr(teleop.kinematics, 'workspace_limits'):
            limits = teleop.kinematics.workspace_limits
            r_min = limits['r_min']
            r_max = limits['r_max']
        else:
            # SO100Kinematics: compute from l1, l2
            r_min = abs(teleop.kinematics.l1 - teleop.kinematics.l2)
            r_max = teleop.kinematics.l1 + teleop.kinematics.l2
        text = font.render(f"Workspace: r_min={r_min:.4f}  r_max={r_max:.4f}", True, (150, 150, 150))
        screen.blit(text, (10, y_pos))
        y_pos += 25

        # === Arm Side-View Visualization ===
        viz_height = 140
        viz_width = 200

        # Left arm visualization
        left_viz_rect = pygame.Rect(10, y_pos, viz_width, viz_height)
        draw_arm_side_view(screen, teleop.left_arm, (100, 200, 255), left_viz_rect, "Left Arm", font)

        # Right arm visualization
        right_viz_rect = pygame.Rect(220, y_pos, viz_width, viz_height)
        draw_arm_side_view(screen, teleop.right_arm, (255, 200, 100), right_viz_rect, "Right Arm", font)

        pygame.display.flip()
        clock.tick(60)
        time.sleep(0.01)

    pygame.quit()
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
