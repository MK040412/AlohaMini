#!/usr/bin/env python3
"""
AlohaMini Virtual Base Demo

Test the AlohaMini robot with virtual mobile base (like XLeRobot).
Uses prismatic X/Y joints and rotation joint instead of actual wheels.

Usage:
    python demo_virtual_base.py --render
    python demo_virtual_base.py --render --shader rt-fast

Controls:
    A/D: Move forward/backward
    W/S: Rotate left/right
    Q/E: Strafe left/right
    R/F: Lift up/down

    Left Arm: 1-6 keys (decrease) / Shift+1-6 (increase)
    Right Arm: 7-0 keys

    X: Reset positions
    ESC: Quit
"""

import argparse
import sys
import time

import numpy as np

try:
    import gymnasium as gym
    import mani_skill.envs
    import sapien
except ImportError:
    print("Error: ManiSkill3 not installed. Install with: pip install mani-skill")
    sys.exit(1)

try:
    import pygame
except ImportError:
    print("Error: pygame not installed. Install with: pip install pygame")
    sys.exit(1)


def get_qpos(robot):
    """Get current joint positions from robot."""
    if robot is None:
        return np.zeros(16)

    qpos = robot.get_qpos()
    if hasattr(qpos, 'cpu'):
        qpos = qpos.cpu().numpy()
    elif hasattr(qpos, 'numpy'):
        qpos = qpos.numpy()

    if qpos.ndim > 1:
        qpos = qpos.squeeze()

    return qpos


def main():
    parser = argparse.ArgumentParser(description="AlohaMini Virtual Base Demo")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--shader", choices=["default", "rt", "rt-fast"], default="default")
    parser.add_argument("--sim-backend", choices=["cpu", "gpu"], default="gpu")
    args = parser.parse_args()

    pygame.init()

    # Create control window
    screen = pygame.display.set_mode((500, 400))
    pygame.display.set_caption("AlohaMini Virtual Base Control")
    font = pygame.font.SysFont(None, 24)

    np.set_printoptions(suppress=True, precision=3)

    # Create environment with virtual base robot
    render_mode = "human" if args.render else None

    env = gym.make(
        "ReplicaCAD_SceneManipulation-v1",
        robot_uids="aloha_mini_virtual",
        render_mode=render_mode,
        obs_mode="state",
        sim_backend=args.sim_backend,
        control_mode="pd_joint_pos",  # Use pd_joint_pos for virtual base
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
    action = np.zeros_like(action)
    action_dim = len(action)
    print(f"Action space dimension: {action_dim}")
    print(f"Expected: base(3) + lift(1) + left_arm(6) + right_arm(6) = 16")

    # Target positions for P-control
    # [base_vx, base_vy, base_omega, lift, left_arm(6), right_arm(6)]
    target = np.zeros(action_dim)

    # Control speeds
    move_speed = 0.5
    rotate_speed = 1.0
    lift_step = 0.01
    arm_step = 0.05

    print("\n" + "="*50)
    print("AlohaMini Virtual Base Control")
    print("="*50)
    print("A/D: Forward/Backward")
    print("W/S: Rotate Left/Right")
    print("Q/E: Strafe Left/Right")
    print("R/F: Lift Up/Down")
    print("X: Reset, ESC: Quit")
    print("="*50 + "\n")

    clock = pygame.time.Clock()
    running = True
    step_counter = 0
    warmup_steps = 30

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_x:
                    target = np.zeros(action_dim)
                    print("Reset all positions")

        keys = pygame.key.get_pressed()

        # Get current state
        current_qpos = get_qpos(robot)

        if step_counter >= warmup_steps:
            # === Base Control (velocity commands) ===
            # Action[0:3] = base velocities [vx, vy, omega]

            # Forward/backward (vx)
            if keys[pygame.K_a]:
                action[0] = move_speed
            elif keys[pygame.K_d]:
                action[0] = -move_speed
            else:
                action[0] = 0.0

            # Strafe left/right (vy)
            if keys[pygame.K_q]:
                action[1] = move_speed
            elif keys[pygame.K_e]:
                action[1] = -move_speed
            else:
                action[1] = 0.0

            # Rotation (omega)
            if keys[pygame.K_w]:
                action[2] = rotate_speed
            elif keys[pygame.K_s]:
                action[2] = -rotate_speed
            else:
                action[2] = 0.0

            # === Lift Control ===
            if keys[pygame.K_r]:
                target[3] += lift_step
                target[3] = min(0.15, target[3])
            if keys[pygame.K_f]:
                target[3] -= lift_step
                target[3] = max(0.0, target[3])

            # Apply target to lift (position control)
            action[3] = target[3]

            # Arms keep current position (no movement)
            for i in range(4, action_dim):
                action[i] = current_qpos[i] if i < len(current_qpos) else 0.0

        else:
            # Warmup - zero action
            action = np.zeros(action_dim)

        # Step environment
        obs, reward, _, _, info = env.step(action)
        step_counter += 1

        if args.render:
            env.render()

        # === Draw Control Panel ===
        screen.fill((30, 30, 30))
        y_pos = 10

        if step_counter < warmup_steps:
            title = font.render(f"WARMUP: {step_counter}/{warmup_steps}", True, (255, 100, 100))
        else:
            title = font.render("AlohaMini Virtual Base Control", True, (100, 255, 100))
        screen.blit(title, (10, y_pos))
        y_pos += 30

        # Controls help
        controls = [
            "A/D: Forward/Back    W/S: Rotate    Q/E: Strafe",
            "R/F: Lift Up/Down    X: Reset       ESC: Quit"
        ]
        for ctrl in controls:
            text = font.render(ctrl, True, (200, 200, 200))
            screen.blit(text, (10, y_pos))
            y_pos += 22

        y_pos += 10

        # Current action
        text = font.render("Action (velocities):", True, (255, 200, 100))
        screen.blit(text, (10, y_pos))
        y_pos += 22

        text = font.render(f"  Base: vx={action[0]:.2f} vy={action[1]:.2f} omega={action[2]:.2f}", True, (200, 200, 255))
        screen.blit(text, (10, y_pos))
        y_pos += 22

        text = font.render(f"  Lift: {action[3]:.3f}", True, (200, 200, 255))
        screen.blit(text, (10, y_pos))
        y_pos += 22

        y_pos += 10

        # Current qpos
        text = font.render("Current Joint Positions:", True, (255, 200, 100))
        screen.blit(text, (10, y_pos))
        y_pos += 22

        text = font.render(f"  Base [0-2]: {current_qpos[0:3].round(3)}", True, (200, 255, 200))
        screen.blit(text, (10, y_pos))
        y_pos += 22

        text = font.render(f"  Lift [3]: {current_qpos[3]:.3f}", True, (200, 255, 200))
        screen.blit(text, (10, y_pos))
        y_pos += 22

        if len(current_qpos) > 9:
            text = font.render(f"  Left Arm [4-9]: {current_qpos[4:10].round(2)}", True, (200, 255, 200))
            screen.blit(text, (10, y_pos))
            y_pos += 22

        if len(current_qpos) > 15:
            text = font.render(f"  Right Arm [10-15]: {current_qpos[10:16].round(2)}", True, (200, 255, 200))
            screen.blit(text, (10, y_pos))

        pygame.display.flip()
        clock.tick(60)
        time.sleep(0.01)

    pygame.quit()
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
