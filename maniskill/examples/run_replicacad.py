#!/usr/bin/env python3
"""
AlohaMini ReplicaCAD Environment Demo

Run the AlohaMini dual-arm robot in ManiSkill3's ReplicaCAD indoor scene environment.

Usage:
    python run_replicacad.py --render              # Real-time visualization
    python run_replicacad.py --render --shader rt-fast  # Ray-traced rendering
    python run_replicacad.py --record              # Record video
    python run_replicacad.py --control keyboard    # Keyboard control
    python run_replicacad.py --show-camera         # Show robot camera views

Requirements:
    - ManiSkill3 with AlohaMini integration
    - ReplicaCAD dataset: python -m mani_skill.utils.download_asset ReplicaCAD
    - pygame (for camera view): pip install pygame
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.utils.wrappers import RecordEpisode
except ImportError:
    print("Error: ManiSkill3 not installed. Install with: pip install mani-skill")
    sys.exit(1)


class CameraViewer:
    """
    Pygame-based viewer for multiple robot camera feeds in 4x2 grid.
    """

    # Camera names to display (8 cameras)
    CAMERA_NAMES = [
        "cam1_back30", "cam2_back45", "cam3_back60", "cam4_overhead",
        "cam5_flip45", "cam6_flip30", "cam7_flip60", "cam8_flip50"
    ]
    CAMERA_LABELS = [
        "1: Back30", "2: Back45", "3: Back60", "4: Overhead",
        "5: Flip45", "6: Flip30", "7: Flip60", "8: Flip50"
    ]

    def __init__(self, width=320, height=240, scale=1.0):
        """
        Initialize the camera viewer for 4x2 grid display.

        Args:
            width: Single camera image width
            height: Single camera image height
            scale: Scale factor for display
        """
        import pygame
        self.pygame = pygame

        self.cam_width = int(width * scale)
        self.cam_height = int(height * scale)
        self.scale = scale
        # 4x2 grid with labels
        self.window_width = self.cam_width * 4 + 30  # 10px gaps
        self.window_height = self.cam_height * 2 + 60  # 30px label per row
        self.screen = None
        self.font = None
        self.running = True
        self.debug_printed = False

    def init_display(self):
        """Initialize pygame display."""
        self.pygame.init()
        self.screen = self.pygame.display.set_mode((self.window_width, self.window_height))
        self.pygame.display.set_caption("AlohaMini 8-Camera View")
        self.font = self.pygame.font.Font(None, 20)

    def update(self, obs):
        """Update the display with camera images in 2x2 grid."""
        if self.screen is None:
            self.init_display()

        # Handle pygame events
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.running = False
            elif event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_ESCAPE:
                    self.running = False

        # Clear screen
        self.screen.fill((30, 30, 30))

        # Debug: print observation structure once
        if not self.debug_printed:
            print(f"\n=== Observation Debug ===")
            print(f"Obs type: {type(obs)}")
            if isinstance(obs, dict):
                print(f"Obs keys: {obs.keys()}")
                if "sensor_data" in obs:
                    sensor_data = obs["sensor_data"]
                    print(f"  sensor_data keys: {sensor_data.keys()}")
                    for cam_name in sensor_data.keys():
                        cam_data = sensor_data[cam_name]
                        if isinstance(cam_data, dict):
                            print(f"    {cam_name} keys: {cam_data.keys()}")
                            if "rgb" in cam_data:
                                rgb = cam_data["rgb"]
                                if hasattr(rgb, 'shape'):
                                    print(f"      rgb: shape={rgb.shape}, dtype={rgb.dtype}")
            print("=========================\n")
            self.debug_printed = True

        # Get camera images from observation
        camera_images = self._extract_camera_images(obs)

        # Draw 4x2 grid (4 columns, 2 rows)
        positions = [
            (0, 0),                                    # Row 1, Col 1
            (self.cam_width + 10, 0),                  # Row 1, Col 2
            ((self.cam_width + 10) * 2, 0),            # Row 1, Col 3
            ((self.cam_width + 10) * 3, 0),            # Row 1, Col 4
            (0, self.cam_height + 30),                 # Row 2, Col 1
            (self.cam_width + 10, self.cam_height + 30),  # Row 2, Col 2
            ((self.cam_width + 10) * 2, self.cam_height + 30),  # Row 2, Col 3
            ((self.cam_width + 10) * 3, self.cam_height + 30),  # Row 2, Col 4
        ]

        for i, (cam_name, label) in enumerate(zip(self.CAMERA_NAMES, self.CAMERA_LABELS)):
            x, y = positions[i]

            # Draw label
            label_surface = self.font.render(label, True, (200, 200, 200))
            self.screen.blit(label_surface, (x + 5, y + 5))

            # Draw camera image or placeholder
            if cam_name in camera_images and camera_images[cam_name] is not None:
                self._draw_rgb_at(camera_images[cam_name], x, y + 25)
            else:
                # Draw placeholder
                rect = self.pygame.Rect(x, y + 25, self.cam_width, self.cam_height)
                self.pygame.draw.rect(self.screen, (50, 50, 50), rect)
                text = self.font.render(f"No {cam_name}", True, (100, 100, 100))
                text_rect = text.get_rect(center=(x + self.cam_width//2, y + 25 + self.cam_height//2))
                self.screen.blit(text, text_rect)

        self.pygame.display.flip()
        return self.running

    def _extract_camera_images(self, obs):
        """Extract RGB images from all cameras in observation."""
        images = {}

        if not isinstance(obs, dict):
            return images

        sensor_data = None
        if "sensor_data" in obs:
            sensor_data = obs["sensor_data"]
        elif "image" in obs:
            sensor_data = obs["image"]

        if sensor_data is None or not isinstance(sensor_data, dict):
            return images

        for cam_name in self.CAMERA_NAMES:
            if cam_name in sensor_data:
                cam_data = sensor_data[cam_name]
                if isinstance(cam_data, dict) and "rgb" in cam_data:
                    images[cam_name] = cam_data["rgb"]
                elif hasattr(cam_data, 'shape'):
                    images[cam_name] = cam_data

        return images

    def _draw_rgb_at(self, rgb, x, y):
        """Draw RGB image at specified position."""
        # Handle tensor/numpy conversion
        if hasattr(rgb, 'cpu'):
            rgb = rgb.cpu().numpy()

        # Handle batched observations (take first env)
        while rgb.ndim > 3:
            rgb = rgb[0]

        # Check if we have valid image data
        if rgb.ndim != 3 or rgb.shape[2] not in [3, 4]:
            return

        # Take only RGB channels if RGBA
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        # Ensure uint8
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        # Create pygame surface
        surface = self.pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

        # Scale to display size
        surface = self.pygame.transform.scale(surface, (self.cam_width, self.cam_height))

        self.screen.blit(surface, (x, y))

    def close(self):
        """Clean up pygame."""
        if self.screen is not None:
            self.pygame.quit()


def run_with_camera_view(env, num_steps=500, auto_reset=True, infinite=True,
                          render_mode=None, camera_scale=1.0):
    """
    Run environment with random actions and display head camera via pygame.

    Args:
        env: Gymnasium environment instance
        num_steps: Number of steps to run (ignored if infinite=True)
        auto_reset: Reset environment when episode ends
        infinite: Run until Ctrl+C or window closed
        render_mode: Render mode ('human', 'rgb_array', None)
        camera_scale: Scale factor for camera display
    """
    viewer = CameraViewer(width=640, height=480, scale=camera_scale)

    obs, info = env.reset(options=dict(reconfigure=True))
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    if render_mode == "human":
        env.render()

    if infinite:
        print("\nRunning with camera view. Close window or press ESC to stop.\n")

    total_reward = 0
    episode_count = 0
    step = 0

    try:
        while viewer.running:
            # Full random action for visible movement
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Update camera view
            if not viewer.update(obs):
                break

            if render_mode == "human":
                env.render()

            if step % 100 == 0:
                print(f"Step {step}, Reward: {reward:.4f}, Total: {total_reward:.4f}")

            # Auto reset on episode end (don't stop)
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count} ended at step {step}, resetting...")
                obs, info = env.reset()
                total_reward = 0

            step += 1

    except KeyboardInterrupt:
        print(f"\n\nStopped by user at step {step}")
    finally:
        viewer.close()

    return obs


def run_random_actions(env, num_steps=500, auto_reset=False, infinite=False, render_mode=None):
    """
    Run environment with random actions.

    Args:
        env: Gymnasium environment instance
        num_steps: Number of steps to run (ignored if infinite=True)
        auto_reset: Reset environment when episode ends
        infinite: Run until Ctrl+C
        render_mode: Render mode ('human', 'rgb_array', None)
    """
    obs, info = env.reset(options=dict(reconfigure=True))
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    if render_mode == "human":
        env.render()

    if infinite:
        print("\nRunning infinitely. Press Ctrl+C to stop.\n")

    total_reward = 0
    episode_count = 0
    step = 0

    try:
        while True:
            # Full random action for visible movement
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if render_mode == "human":
                env.render()

            if step % 100 == 0:
                print(f"Step {step}, Reward: {reward:.4f}, Total: {total_reward:.4f}")

            # Auto reset on episode end (don't stop when infinite)
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count} ended at step {step}, resetting...")
                obs, info = env.reset()
                total_reward = 0

            step += 1
            if not infinite and step >= num_steps:
                break

    except KeyboardInterrupt:
        print(f"\n\nStopped by user at step {step}")

    return obs


def compute_omni_wheel_velocities(vx, vy, omega):
    """
    Compute 3-wheel omnidirectional base velocities using hardware kinematics (4862).

    Hardware kinematics from lekiwi.py uses wheel order [left, back, right]:
    - base_left_wheel:  240-90 = 150 deg
    - base_back_wheel:  0-90 = -90 deg (270 deg)
    - base_right_wheel: 120-90 = 30 deg

    URDF wheel order is [wheel1, wheel2, wheel3]:
    - wheel1: left-back at angle -2.11 rad ≈ -121 deg (≈ 150 deg from +X)
    - wheel2: right-back at angle 2.09 rad ≈ 120 deg (≈ 30 deg from +X)
    - wheel3: front at angle 0 rad (≈ 270 deg from +X, facing back)

    Mapping: URDF[0]=hw[0](left), URDF[1]=hw[2](right), URDF[2]=hw[1](back)

    Args:
        vx: Forward velocity (positive = forward, robot +Y in world)
        vy: Sideways velocity (positive = left strafe)
        omega: Angular velocity (positive = CCW rotation)

    Returns:
        (w1, w2, w3): Wheel velocities for [wheel1, wheel2, wheel3] in URDF order
    """
    import numpy as np

    # Hardware parameters from lekiwi.py
    wheel_radius = 0.05   # 50mm wheel radius
    base_radius = 0.125   # 125mm from center to wheel

    # Create velocity vector [x, y, theta]
    # Hardware convention: [-x, -y, theta]
    velocity_vector = np.array([-vx, -vy, omega])

    # Hardware wheel mounting angles: [240, 0, 120] with -90 offset
    # This gives [left, back, right] = [150, -90, 30] degrees
    angles = np.radians(np.array([240, 0, 120]) - 90)

    # Build kinematic matrix
    m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

    # Compute wheel linear speeds and convert to angular speeds
    wheel_linear_speeds = m.dot(velocity_vector)
    wheel_angular_speeds = wheel_linear_speeds / wheel_radius

    # Hardware output order: [left, back, right]
    hw_left = wheel_angular_speeds[0]
    hw_back = wheel_angular_speeds[1]
    hw_right = wheel_angular_speeds[2]

    # Map to URDF order: [wheel1, wheel2, wheel3] = [left, right, back]
    urdf_w1 = hw_left   # wheel1 = left-back
    urdf_w2 = hw_right  # wheel2 = right-back
    urdf_w3 = hw_back   # wheel3 = front (but facing back, so same as hw_back)

    # Normalize to [-1, 1] range for controller
    speeds = np.array([urdf_w1, urdf_w2, urdf_w3])
    max_speed = max(np.abs(speeds).max(), 1.0)
    normalized = speeds / max_speed

    return float(normalized[0]), float(normalized[1]), float(normalized[2])


def run_keyboard_control(env, show_camera=False, camera_scale=1.0):
    """
    Run environment with keyboard control (FPS Style).

    Controls:
        W/S: Forward/Backward
        A/D: Strafe Left/Right
        Q/E: Rotate Left/Right
        R/F: Lift up/down
        Arrow keys + Z/X: Left arm
        IJKL + N/M: Right arm
        Space: Reset environment
        ESC: Quit
    """
    try:
        import pygame
    except ImportError:
        print("Keyboard control requires pygame: pip install pygame")
        return

    pygame.init()

    # Initialize camera viewer if requested
    viewer = CameraViewer(width=320, height=240, scale=camera_scale) if show_camera else None

    # IMPORTANT: pygame.key.get_pressed() requires a display window with focus
    # Create a small control window if camera viewer is not used
    control_window = None
    if viewer is None:
        control_window = pygame.display.set_mode((400, 100))
        pygame.display.set_caption("AlohaMini Keyboard Control - Keep this window focused!")

    obs, info = env.reset(options=dict(reconfigure=True))

    # Get robot reference for camera follow
    robot = None
    if hasattr(env.unwrapped, "agent"):
        robot = env.unwrapped.agent.robot

    # Get action space info
    action_dim = env.action_space.shape[0]
    print(f"\nAction space dimension: {action_dim}")

    print("\n=== Keyboard Controls ===")
    print("W/S: Move forward/backward")
    print("A/D: Strafe left/right")
    print("Q/E: Rotate left/right")
    print("R/F: Lift up/down")
    print("Arrow keys: Left arm joint 1-2")
    print("Z/X/C/V: Left arm joint 3-6")
    print("IJKL: Right arm joint 1-2")
    print("N/M/,/.: Right arm joint 3-6")
    print("Space: Reset | ESC: Quit")
    if show_camera:
        print("Camera view window is open")
    print("=========================\n")

    # Speed settings
    base_speed = 0.5
    arm_speed = 0.3
    lift_speed = 0.02

    running = True
    clock = pygame.time.Clock()

    while running:
        action = np.zeros(action_dim)
        keys = pygame.key.get_pressed()

        # ============================================================
        # NOTE: DO NOT CHANGE THIS MAPPING!
        # W/S = 전진/후진 (forward/backward) -> vy
        # A/D = 좌/우 이동 (strafe left/right) -> vx
        # Q/E = 회전 (rotation) -> omega
        # ============================================================
        vx = 0  # strafe left/right
        vy = 0  # forward/backward
        omega = 0  # rotation

        if keys[pygame.K_a]: vx += base_speed  # strafe left
        if keys[pygame.K_d]: vx -= base_speed  # strafe right
        if keys[pygame.K_w]: vy += base_speed  # forward
        if keys[pygame.K_s]: vy -= base_speed  # backward
        if keys[pygame.K_q]: omega += base_speed  # rotate CCW
        if keys[pygame.K_e]: omega -= base_speed  # rotate CW

        # Convert to wheel velocities
        w1, w2, w3 = compute_omni_wheel_velocities(vx, vy, omega)
        action[0] = w1
        action[1] = w2
        action[2] = w3

        # Lift (index 3)
        if keys[pygame.K_r]: action[3] = lift_speed
        if keys[pygame.K_f]: action[3] = -lift_speed

        # Left arm (indices 4-9)
        if keys[pygame.K_UP]: action[4] += arm_speed
        if keys[pygame.K_DOWN]: action[4] -= arm_speed
        if keys[pygame.K_LEFT]: action[5] += arm_speed
        if keys[pygame.K_RIGHT]: action[5] -= arm_speed
        if keys[pygame.K_z]: action[6] += arm_speed
        if keys[pygame.K_x]: action[6] -= arm_speed
        if keys[pygame.K_c]: action[7] += arm_speed
        if keys[pygame.K_v]: action[7] -= arm_speed
        if keys[pygame.K_b]: action[8] += arm_speed
        if keys[pygame.K_g]: action[8] -= arm_speed

        # Right arm (indices 10-15)
        if keys[pygame.K_i]: action[10] += arm_speed
        if keys[pygame.K_k]: action[10] -= arm_speed
        if keys[pygame.K_j]: action[11] += arm_speed
        if keys[pygame.K_l]: action[11] -= arm_speed
        if keys[pygame.K_n]: action[12] += arm_speed
        if keys[pygame.K_m]: action[12] -= arm_speed
        if keys[pygame.K_COMMA]: action[13] += arm_speed
        if keys[pygame.K_PERIOD]: action[13] -= arm_speed

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        # Camera follow robot from above
        render_viewer = env.unwrapped._viewer
        if render_viewer is not None and robot is not None:
            try:
                robot_pos = robot.pose.p
                cam_x = float(robot_pos[0]) - 1.5
                cam_y = float(robot_pos[1])
                cam_z = float(robot_pos[2]) + 3.0
                render_viewer.set_camera_xyz(cam_x, cam_y, cam_z)
                render_viewer.set_camera_rpy(0, -1.0, 0)
            except:
                pass

        # Update camera view
        if viewer is not None:
            if not viewer.update(obs):
                running = False

        # Update control window with current key state
        if control_window is not None:
            control_window.fill((40, 40, 40))
            font = pygame.font.Font(None, 24)

            # Show active keys (vx=strafe A/D, vy=forward/back W/S)
            active_keys = []
            if vx > 0: active_keys.append("A")   # strafe left
            if vx < 0: active_keys.append("D")   # strafe right
            if vy > 0: active_keys.append("W")   # forward
            if vy < 0: active_keys.append("S")   # backward
            if omega > 0: active_keys.append("Q")
            if omega < 0: active_keys.append("E")

            text1 = font.render(f"Base: {' '.join(active_keys) if active_keys else 'W/S=forward A/D=strafe Q/E=rotate'}", True, (200, 200, 200))
            text2 = font.render(f"Wheels: [{w1:.2f}, {w2:.2f}, {w3:.2f}]", True, (150, 200, 150))
            text3 = font.render("Keep this window focused for keyboard input!", True, (255, 200, 100))

            control_window.blit(text1, (10, 10))
            control_window.blit(text2, (10, 35))
            control_window.blit(text3, (10, 65))
            pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    obs, info = env.reset()
                    print("Environment reset")

        if terminated or truncated:
            obs, info = env.reset()

        clock.tick(60)  # 60 FPS

    if viewer is not None:
        viewer.close()
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Run AlohaMini in ReplicaCAD environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --render                    # Real-time visualization
  %(prog)s --render --shader rt-fast   # Ray-traced rendering
  %(prog)s --record --record-dir ./videos  # Record video
  %(prog)s --control keyboard --render # Keyboard control
  %(prog)s --show-camera               # Show robot camera views
  %(prog)s --show-camera --render      # Camera views + 3D visualization
  %(prog)s --control keyboard --show-camera --render  # Full control with cameras
        """
    )
    parser.add_argument("--render", action="store_true", help="Enable real-time rendering")
    parser.add_argument("--record", action="store_true", help="Record video")
    parser.add_argument("--record-dir", type=str, default="videos", help="Video output directory")
    parser.add_argument("--control", choices=["random", "keyboard"], default="random")
    parser.add_argument("--num-steps", type=int, default=500, help="Steps for random control")
    parser.add_argument("--robot", choices=["aloha_mini", "aloha_mini_fixed"],
                        default="aloha_mini_fixed", help="Robot variant")
    parser.add_argument("--sim-backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--shader", choices=["default", "rt", "rt-fast"], default="default",
                        help="Shader pack (rt-fast for ray tracing)")
    parser.add_argument("--auto-reset", action="store_true", help="Auto-reset on episode end")
    parser.add_argument("--max-episode-steps", type=int, default=100000,
                        help="Max steps per episode (default: 100000 to avoid early termination)")
    parser.add_argument("--infinite", action="store_true", help="Run until Ctrl+C")
    parser.add_argument("--show-camera", action="store_true",
                        help="Show robot camera views in pygame window")
    parser.add_argument("--camera-scale", type=float, default=1.0,
                        help="Scale factor for camera display (default: 1.0)")
    args = parser.parse_args()

    # Check pygame availability for camera view
    if args.show_camera:
        try:
            import pygame
        except ImportError:
            print("Error: Camera view requires pygame. Install with: pip install pygame")
            sys.exit(1)

    # Default to infinite mode when rendering or showing camera
    if (args.render or args.show_camera) and not args.record:
        args.infinite = True

    render_mode = "human" if args.render else ("rgb_array" if args.record else None)

    # Need rgbd observation mode for camera views (includes RGB images)
    obs_mode = "rgbd" if args.show_camera else "none"

    # Use mobile robot for keyboard control
    robot_uid = args.robot
    if args.control == "keyboard" and args.robot == "aloha_mini_fixed":
        robot_uid = "aloha_mini"
        print("Note: Switching to 'aloha_mini' (mobile) for keyboard control")

    print(f"Robot: {robot_uid} | Backend: {args.sim_backend} | Shader: {args.shader}")
    if args.show_camera:
        print(f"Camera view enabled (scale: {args.camera_scale})")

    env_kwargs = dict(
        robot_uids=robot_uid,
        render_mode=render_mode,
        obs_mode=obs_mode,
        sim_backend=args.sim_backend,
        max_episode_steps=args.max_episode_steps,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        enable_shadow=True,
    )

    # Use mobile controller for keyboard control with mobile robot
    if args.control == "keyboard" and robot_uid == "aloha_mini":
        env_kwargs["control_mode"] = "mobile_pd_joint_pos"
        print("Using mobile_pd_joint_pos controller for base movement")

    env = gym.make("ReplicaCAD_SceneManipulation-v1", **env_kwargs)

    if args.record:
        env = RecordEpisode(
            env, output_dir=args.record_dir, save_trajectory=False,
            save_video=True, max_steps_per_video=args.num_steps,
        )

    try:
        if args.control == "keyboard":
            if not args.render:
                print("Warning: Keyboard control works best with --render")
            run_keyboard_control(env, show_camera=args.show_camera, camera_scale=args.camera_scale)
        elif args.show_camera:
            run_with_camera_view(env, args.num_steps, args.auto_reset, args.infinite,
                                  render_mode, args.camera_scale)
        else:
            run_random_actions(env, args.num_steps, args.auto_reset, args.infinite, render_mode)
    finally:
        env.close()

    if args.record:
        print(f"\nVideos saved to: {args.record_dir}/")


if __name__ == "__main__":
    main()
