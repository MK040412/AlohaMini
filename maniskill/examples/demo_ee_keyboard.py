#!/usr/bin/env python3
"""
AlohaMini End-Effector Keyboard Control Demo

Based on XLeRobot's demo_ctrl_action_ee_keyboard.py
Control the AlohaMini dual-arm robot using end-effector positions and inverse kinematics.

Usage:
    python demo_ee_keyboard.py --render
    python demo_ee_keyboard.py --render --shader rt-fast

Controls:
    Base Movement (Omni Kinematics):
        W/S: Forward/Backward
        A/D: Strafe Left/Right
        Q/E: Rotate Left/Right

    Lift:
        R/F: Lift Up/Down

    Left Arm (End-Effector):
        Y/7: Joint 1 (base rotation)
        8/U: EE Y (up/down)
        9/I: EE X (forward/backward)
        0/O: Pitch adjust
        -/P: Wrist roll

    Right Arm (End-Effector):
        H/N: Joint 1 (base rotation)
        J/M: EE Y (up/down)
        K/,: EE X (forward/backward)
        L/.: Pitch adjust
        ;/?: Wrist roll

    X: Reset all positions
    ESC: Quit
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
except ImportError:
    print("Error: ManiSkill3 not installed. Install with: pip install mani-skill")
    sys.exit(1)

try:
    import pygame
except ImportError:
    print("Error: pygame not installed. Install with: pip install pygame")
    sys.exit(1)


def compute_omni_wheel_velocities(vx, vy, omega):
    """
    Compute wheel velocities for omnidirectional base movement.
    Based on hardware kinematics from lekiwi.py (4862 configuration)

    Hardware wheel mounting angles: [240, 0, 120] with -90 offset
    This gives [left, back, right] = [150, -90, 30] degrees

    URDF wheel order: [wheel1, wheel2, wheel3] = [left, right, back]

    Parameters:
        vx: Forward velocity (+forward)
        vy: Lateral velocity (+left strafe)
        omega: Angular velocity (+counter-clockwise)

    Returns:
        wheel_velocities: Array of [wheel1, wheel2, wheel3] velocities in URDF order
    """
    wheel_radius = 0.05   # 50mm
    base_radius = 0.125   # 125mm

    # Velocity vector: [vx, vy, omega] with hardware convention
    velocity_vector = np.array([-vx, -vy, omega])

    # Wheel angles from lekiwi.py: [240, 0, 120] - 90 degrees offset
    # Hardware order: [left, back, right]
    angles = np.radians(np.array([240, 0, 120]) - 90)

    # Build kinematic matrix (each row: [cos(angle), sin(angle), base_radius])
    m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

    # Compute wheel linear velocities
    wheel_linear_velocities = m @ velocity_vector

    # Convert to angular velocities
    wheel_angular_velocities = wheel_linear_velocities / wheel_radius

    # Hardware output order: [left, back, right]
    hw_left = wheel_angular_velocities[0]
    hw_back = wheel_angular_velocities[1]
    hw_right = wheel_angular_velocities[2]

    # Map to URDF order: [wheel1, wheel2, wheel3] = [left, right, back]
    urdf_w1 = hw_left   # wheel1 = left-back
    urdf_w2 = hw_right  # wheel2 = right-back
    urdf_w3 = hw_back   # wheel3 = front

    return np.array([urdf_w1, urdf_w2, urdf_w3])


def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """
    Calculate inverse kinematics for a 2-link robotic arm, considering joint offsets.
    From XLeRobot demo_ctrl_action_ee_keyboard.py

    Parameters:
        x: End effector x coordinate (forward distance)
        y: End effector y coordinate (height)
        l1: Upper arm length (default 0.1159 m)
        l2: Lower arm length (default 0.1350 m)

    Returns:
        joint2, joint3: Joint angles in radians
    """
    # Calculate joint2 and joint3 offsets
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

    # Calculate distance from origin to target point
    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2
    r_min = abs(l1 - l2)

    # Clamp to workspace boundaries
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max

    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min

    # Use law of cosines to calculate theta2
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = max(-1, min(1, cos_theta2))

    # Calculate theta2 (elbow angle)
    theta2 = math.pi - math.acos(cos_theta2)

    # Calculate theta1 (shoulder angle)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    # Convert to joint angles
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset

    # Apply joint limits
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))

    return joint2, joint3


def get_mapped_joints(robot):
    """
    Get current joint positions from the robot.
    AlohaMini joint order: [wheels(3), lift(1), left_arm(6), right_arm(6)] = 16 DOF
    """
    if robot is None:
        return np.zeros(16)

    full_joints = robot.get_qpos()

    if hasattr(full_joints, 'cpu'):
        full_joints = full_joints.cpu().numpy()
    elif hasattr(full_joints, 'numpy'):
        full_joints = full_joints.numpy()

    if full_joints.ndim > 1:
        full_joints = full_joints.squeeze()

    return full_joints


def main():
    parser = argparse.ArgumentParser(
        description="AlohaMini End-Effector Keyboard Control"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--shader", choices=["default", "rt", "rt-fast"], default="default")
    parser.add_argument("--sim-backend", choices=["cpu", "gpu"], default="gpu")
    args = parser.parse_args()

    pygame.init()

    # Create control panel window
    screen_width = 620
    screen_height = 750
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("AlohaMini EE Control - Keep this window focused!")
    font = pygame.font.SysFont(None, 24)

    np.set_printoptions(suppress=True, precision=3)

    # Create environment
    render_mode = "human" if args.render else None

    env = gym.make(
        "ReplicaCAD_SceneManipulation-v1",
        robot_uids="aloha_mini",
        render_mode=render_mode,
        obs_mode="state",
        sim_backend=args.sim_backend,
        control_mode="mobile_pd_joint_pos",
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

    # Initialize target joints
    # AlohaMini: [wheels(3), lift(1), left_arm(6), right_arm(6)] = 16 DOF
    target_joints = np.zeros(action_dim)

    # Initialize end effector positions (x=forward, y=height)
    initial_ee_pos_arm1 = np.array([0.162, 0.118])
    initial_ee_pos_arm2 = np.array([0.162, 0.118])
    ee_pos_arm1 = initial_ee_pos_arm1.copy()
    ee_pos_arm2 = initial_ee_pos_arm2.copy()

    # Pitch adjustments
    initial_pitch_1 = 0.0
    initial_pitch_2 = 0.0
    pitch_1 = initial_pitch_1
    pitch_2 = initial_pitch_2
    pitch_step = 0.02

    # Tip length for compensation
    tip_length = 0.108

    # Control parameters
    joint_step = 0.01
    ee_step = 0.005

    # P-gains for controller
    # [wheels(3), lift(1), left_arm(6), right_arm(6)]
    p_gain = np.ones(action_dim)
    p_gain[0:3] = 1.0    # Wheels - velocity control, not used for P-control
    p_gain[3] = 2.0      # Lift
    p_gain[4:10] = 1.0   # Left arm
    p_gain[10:16] = 1.0  # Right arm

    # Get initial joint positions
    current_joints = get_mapped_joints(robot)

    # Set initial target based on IK
    try:
        # Left arm joints: indices 4-9 (joint1, joint2, joint3, joint4, joint5, joint6)
        # IK gives joint2, joint3 values
        target_joints[5], target_joints[6] = inverse_kinematics(ee_pos_arm1[0], ee_pos_arm1[1])
        target_joints[8] = 1.57  # Wrist default

        # Right arm joints: indices 10-15
        target_joints[11], target_joints[12] = inverse_kinematics(ee_pos_arm2[0], ee_pos_arm2[1])
        target_joints[14] = 1.57  # Wrist default
    except Exception as e:
        print(f"Error calculating initial IK: {e}")

    # Warmup
    step_counter = 0
    warmup_steps = 50

    print("\n" + "="*50)
    print("AlohaMini End-Effector Keyboard Control")
    print("="*50)
    print("Base: W/S=forward/back, A/D=strafe, Q/E=rotate")
    print("Lift: R/F=up/down")
    print("Left Arm:  Y/7=joint1, 8/U=EE Y, 9/I=EE X, 0/O=pitch, -/P=wrist")
    print("Right Arm: H/N=joint1, J/M=EE Y, K/,=EE X, L/.=pitch, ;/?=wrist")
    print("X=Reset, ESC=Quit")
    print("="*50 + "\n")

    clock = pygame.time.Clock()
    running = True

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_x:
                    # Reset all positions
                    ee_pos_arm1 = initial_ee_pos_arm1.copy()
                    ee_pos_arm2 = initial_ee_pos_arm2.copy()
                    pitch_1 = initial_pitch_1
                    pitch_2 = initial_pitch_2
                    target_joints = np.zeros(action_dim)

                    try:
                        compensated_y1 = ee_pos_arm1[1] + tip_length * math.sin(pitch_1)
                        target_joints[5], target_joints[6] = inverse_kinematics(ee_pos_arm1[0], compensated_y1)
                        target_joints[7] = target_joints[5] - target_joints[6] + pitch_1
                        target_joints[8] = 1.57

                        compensated_y2 = ee_pos_arm2[1] + tip_length * math.sin(pitch_2)
                        target_joints[11], target_joints[12] = inverse_kinematics(ee_pos_arm2[0], compensated_y2)
                        target_joints[13] = target_joints[11] - target_joints[12] + pitch_2
                        target_joints[14] = 1.57
                    except Exception as e:
                        print(f"Error during reset IK: {e}")

                    print("All positions reset")

        keys = pygame.key.get_pressed()

        # Get current joints
        current_joints = get_mapped_joints(robot)

        # Update controls after warmup
        if step_counter >= warmup_steps:
            # === Base Control (Omni Wheel Kinematics) - FPS Style ===
            # Compute velocity commands
            # ============================================================
            # NOTE: DO NOT CHANGE THIS MAPPING!
            # W/S = 전진/후진 (forward/backward) -> vy
            # A/D = 좌/우 이동 (strafe left/right) -> vx
            # Q/E = 회전 (rotation) -> omega
            # ============================================================
            vx = 0.0  # Strafe left/right
            vy = 0.0  # Forward/backward
            omega = 0.0  # Rotation

            # Strafe left/right - A/D keys -> vx
            if keys[pygame.K_a]:
                vx = 0.3   # strafe left
            elif keys[pygame.K_d]:
                vx = -0.3  # strafe right

            # Forward/backward - W/S keys -> vy
            if keys[pygame.K_w]:
                vy = 0.3   # forward
            elif keys[pygame.K_s]:
                vy = -0.3  # backward

            # Rotation (Q/E for rotation)
            if keys[pygame.K_q]:
                omega = 0.5  # Rotate left
            elif keys[pygame.K_e]:
                omega = -0.5  # Rotate right

            # Compute wheel velocities using omni kinematics
            wheel_vels = compute_omni_wheel_velocities(vx, vy, omega)
            action[0] = wheel_vels[0]
            action[1] = wheel_vels[1]
            action[2] = wheel_vels[2]

            # === Lift Control ===
            if keys[pygame.K_r]:
                target_joints[3] += joint_step
                target_joints[3] = min(0.6, target_joints[3])  # Max lift 60cm
            if keys[pygame.K_f]:
                target_joints[3] -= joint_step
                target_joints[3] = max(0.0, target_joints[3])

            # === Left Arm Control ===
            # Joint 1 (base rotation) - index 4
            if keys[pygame.K_7]:
                target_joints[4] += joint_step
            if keys[pygame.K_y]:
                target_joints[4] -= joint_step

            # End effector Y (up/down)
            if keys[pygame.K_8]:
                ee_pos_arm1[1] += ee_step
            if keys[pygame.K_u]:
                ee_pos_arm1[1] -= ee_step

            # End effector X (forward/backward)
            if keys[pygame.K_9]:
                ee_pos_arm1[0] += ee_step
            if keys[pygame.K_i]:
                ee_pos_arm1[0] -= ee_step

            # Pitch control
            if keys[pygame.K_0]:
                pitch_1 += pitch_step
            if keys[pygame.K_o]:
                pitch_1 -= pitch_step

            # Wrist roll - index 9
            if keys[pygame.K_MINUS]:
                target_joints[9] += joint_step * 3
            if keys[pygame.K_p]:
                target_joints[9] -= joint_step * 3

            # Calculate IK for left arm
            try:
                compensated_y = ee_pos_arm1[1] + tip_length * math.sin(pitch_1)
                target_joints[5], target_joints[6] = inverse_kinematics(ee_pos_arm1[0], compensated_y)
                # Apply pitch to joint 4 (wrist pitch)
                target_joints[7] = target_joints[5] - target_joints[6] + pitch_1
            except Exception as e:
                pass

            # === Right Arm Control ===
            # Joint 1 (base rotation) - index 10
            if keys[pygame.K_h]:
                target_joints[10] += joint_step
            if keys[pygame.K_n]:
                target_joints[10] -= joint_step

            # End effector Y (up/down)
            if keys[pygame.K_j]:
                ee_pos_arm2[1] += ee_step
            if keys[pygame.K_m]:
                ee_pos_arm2[1] -= ee_step

            # End effector X (forward/backward)
            if keys[pygame.K_k]:
                ee_pos_arm2[0] += ee_step
            if keys[pygame.K_COMMA]:
                ee_pos_arm2[0] -= ee_step

            # Pitch control
            if keys[pygame.K_l]:
                pitch_2 += pitch_step
            if keys[pygame.K_PERIOD]:
                pitch_2 -= pitch_step

            # Wrist roll - index 15
            if keys[pygame.K_SEMICOLON]:
                target_joints[15] += joint_step * 3
            if keys[pygame.K_SLASH]:
                target_joints[15] -= joint_step * 3

            # Calculate IK for right arm
            try:
                compensated_y = ee_pos_arm2[1] + tip_length * math.sin(pitch_2)
                target_joints[11], target_joints[12] = inverse_kinematics(ee_pos_arm2[0], compensated_y)
                target_joints[13] = target_joints[11] - target_joints[12] + pitch_2
            except Exception as e:
                pass

            # === Apply P-control for lift and arms ===
            # Lift (index 3)
            action[3] = p_gain[3] * (target_joints[3] - current_joints[3])

            # Left arm (indices 4-9)
            for i in range(4, 10):
                action[i] = p_gain[i] * (target_joints[i] - current_joints[i])

            # Right arm (indices 10-15)
            for i in range(10, min(16, action_dim)):
                action[i] = p_gain[i] * (target_joints[i] - current_joints[i])

        else:
            # Warmup - zero action
            action = np.zeros(action_dim)

        # Step environment
        obs, reward, _, _, info = env.step(action)
        step_counter += 1

        if args.render:
            env.render()

        # === Draw Control Panel ===
        screen.fill((0, 0, 0))
        control_panel_x = 10
        y_pos = 10

        # Title
        if step_counter < warmup_steps:
            title = font.render(f"WARMUP: {step_counter}/{warmup_steps} steps", True, (255, 0, 0))
        else:
            title = font.render("AlohaMini EE Control (Keep this window focused!)", True, (100, 255, 100))
        screen.blit(title, (control_panel_x, y_pos))
        y_pos += 30

        # Controls
        control_texts = [
            "W/S: Forward/Backward    A/D: Strafe    Q/E: Rotate",
            "R/F: Lift Up/Down",
            "Y/7: Left Joint1    H/N: Right Joint1",
            "8/U: L-EE Y          J/M: R-EE Y",
            "9/I: L-EE X          K/,: R-EE X",
            "0/O: L-Pitch         L/.: R-Pitch",
            "-/P: L-Wrist         ;/?: R-Wrist",
            "X: Reset             ESC: Quit"
        ]
        for txt in control_texts:
            ctrl_text = font.render(txt, True, (200, 200, 200))
            screen.blit(ctrl_text, (control_panel_x, y_pos))
            y_pos += 22

        y_pos += 15

        # Current joints display
        text = font.render("Current Joints (qpos):", True, (255, 200, 100))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 25

        text = font.render(f"  Wheels [0-2]: {current_joints[0:3].round(3)}", True, (255, 255, 0))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 22

        text = font.render(f"  Lift [3]: {current_joints[3]:.3f}", True, (255, 255, 0))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 22

        if len(current_joints) > 9:
            text = font.render(f"  Left Arm [4-9]: {current_joints[4:10].round(2)}", True, (255, 255, 0))
            screen.blit(text, (control_panel_x, y_pos))
            y_pos += 22

        if len(current_joints) > 15:
            text = font.render(f"  Right Arm [10-15]: {current_joints[10:16].round(2)}", True, (255, 255, 0))
            screen.blit(text, (control_panel_x, y_pos))
            y_pos += 22

        y_pos += 15

        # Target joints display
        text = font.render("Target Joints:", True, (100, 255, 100))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 25

        text = font.render(f"  Lift [3]: {target_joints[3]:.3f}", True, (100, 255, 100))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 22

        text = font.render(f"  Left Arm [4-9]: {target_joints[4:10].round(2)}", True, (100, 255, 100))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 22

        text = font.render(f"  Right Arm [10-15]: {target_joints[10:16].round(2)}", True, (100, 255, 100))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 22

        y_pos += 15

        # End effector positions
        text = font.render("End Effector Positions:", True, (255, 100, 100))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 25

        comp_y1 = ee_pos_arm1[1] + tip_length * math.sin(pitch_1)
        text = font.render(f"  Left:  X={ee_pos_arm1[0]:.3f}, Y={ee_pos_arm1[1]:.3f} (comp={comp_y1:.3f})", True, (255, 150, 150))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 22

        comp_y2 = ee_pos_arm2[1] + tip_length * math.sin(pitch_2)
        text = font.render(f"  Right: X={ee_pos_arm2[0]:.3f}, Y={ee_pos_arm2[1]:.3f} (comp={comp_y2:.3f})", True, (255, 150, 150))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 22

        y_pos += 15

        # Pitch adjustments
        text = font.render(f"Pitch: Left={pitch_1:.3f}, Right={pitch_2:.3f}", True, (255, 100, 255))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 22

        y_pos += 15

        # Action values (velocities)
        text = font.render("Action (velocities):", True, (200, 200, 255))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 25

        text = font.render(f"  Wheels [0-2]: {action[0:3].round(3)}", True, (150, 150, 255))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 22

        text = font.render(f"  Lift [3]: {action[3]:.3f}", True, (150, 150, 255))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 22

        text = font.render(f"  Left Arm [4-9]: {action[4:10].round(2)}", True, (150, 150, 255))
        screen.blit(text, (control_panel_x, y_pos))
        y_pos += 22

        text = font.render(f"  Right Arm [10-15]: {action[10:16].round(2)}", True, (150, 150, 255))
        screen.blit(text, (control_panel_x, y_pos))

        pygame.display.flip()
        clock.tick(60)
        time.sleep(0.01)

    pygame.quit()
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
