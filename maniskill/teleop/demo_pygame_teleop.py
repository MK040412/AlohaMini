#!/usr/bin/env python3
"""
Pygame-based Teleoperation for AlohaMini Robot

Controls:
    Movement (Base):
        W/S: Forward/Backward
        A/D: Strafe Left/Right
        Q/E: Rotate Left/Right
        Z/X: Lift Up/Down

    Left Arm:
        I/K: Shoulder Up/Down
        J/L: Elbow In/Out
        U/O: Wrist Up/Down

    Right Arm:
        Up/Down: Shoulder Up/Down
        Left/Right: Elbow In/Out
        PageUp/PageDown: Wrist Up/Down

    Gripper:
        G: Toggle Left Gripper
        H: Toggle Right Gripper

    Other:
        R: Reset environment
        ESC: Quit
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Clear module caches
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
import pygame


class AlohaMiniTeleop:
    def __init__(self, sim_backend='cpu'):
        # Initialize pygame
        pygame.init()
        self.screen_width = 640
        self.screen_height = 480
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("AlohaMini Teleoperation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        # Create environment
        print("Creating environment...")
        self.env = gym.make(
            'ReplicaCAD_SceneManipulation-v1',
            robot_uids='aloha_mini_virtual',
            render_mode=None,
            obs_mode='rgbd',
            sim_backend=sim_backend,
            control_mode='pd_joint_pos',
        )

        # Reset environment
        print("Resetting environment...")
        self.obs, _ = self.env.reset(options=dict(reconfigure=True))

        # Get current joint positions
        self.qpos = self.env.unwrapped.agent.robot.qpos[0].cpu().numpy().copy()

        # Control parameters
        self.base_speed = 0.02
        self.rotation_speed = 0.05
        self.lift_speed = 0.01
        self.arm_speed = 0.05

        # Gripper states (0=open, 0.04=closed for SO-101)
        self.left_gripper_closed = False
        self.right_gripper_closed = False

        # Joint limits
        self.lift_min = 0.0
        self.lift_max = 0.15

        print("Ready! Use keyboard to control the robot.")
        print("Press ESC to quit, R to reset.")

    def handle_input(self):
        """Handle keyboard input and update joint targets"""
        keys = pygame.key.get_pressed()

        # Base movement (indices 0, 1, 2)
        # Forward/Backward (Y axis)
        if keys[pygame.K_w]:
            self.qpos[1] += self.base_speed  # Forward
        if keys[pygame.K_s]:
            self.qpos[1] -= self.base_speed  # Backward

        # Strafe Left/Right (X axis)
        if keys[pygame.K_a]:
            self.qpos[0] -= self.base_speed  # Left
        if keys[pygame.K_d]:
            self.qpos[0] += self.base_speed  # Right

        # Rotation
        if keys[pygame.K_q]:
            self.qpos[2] += self.rotation_speed  # Rotate left
        if keys[pygame.K_e]:
            self.qpos[2] -= self.rotation_speed  # Rotate right

        # Lift (index 3)
        if keys[pygame.K_z]:
            self.qpos[3] = min(self.qpos[3] + self.lift_speed, self.lift_max)
        if keys[pygame.K_x]:
            self.qpos[3] = max(self.qpos[3] - self.lift_speed, self.lift_min)

        # Left arm (indices 4-9)
        # Joint 1 (shoulder rotation)
        if keys[pygame.K_i]:
            self.qpos[5] += self.arm_speed  # Shoulder up
        if keys[pygame.K_k]:
            self.qpos[5] -= self.arm_speed  # Shoulder down
        # Joint 2 (elbow)
        if keys[pygame.K_j]:
            self.qpos[6] -= self.arm_speed  # Elbow in
        if keys[pygame.K_l]:
            self.qpos[6] += self.arm_speed  # Elbow out
        # Joint 3 (wrist)
        if keys[pygame.K_u]:
            self.qpos[7] += self.arm_speed  # Wrist up
        if keys[pygame.K_o]:
            self.qpos[7] -= self.arm_speed  # Wrist down

        # Right arm (indices 10-15)
        # Joint 1 (shoulder rotation)
        if keys[pygame.K_UP]:
            self.qpos[11] += self.arm_speed
        if keys[pygame.K_DOWN]:
            self.qpos[11] -= self.arm_speed
        # Joint 2 (elbow)
        if keys[pygame.K_LEFT]:
            self.qpos[12] -= self.arm_speed
        if keys[pygame.K_RIGHT]:
            self.qpos[12] += self.arm_speed
        # Joint 3 (wrist)
        if keys[pygame.K_PAGEUP]:
            self.qpos[13] += self.arm_speed
        if keys[pygame.K_PAGEDOWN]:
            self.qpos[13] -= self.arm_speed

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    print("Resetting environment...")
                    self.obs, _ = self.env.reset()
                    self.qpos = self.env.unwrapped.agent.robot.qpos[0].cpu().numpy().copy()
                elif event.key == pygame.K_g:
                    # Toggle left gripper
                    self.left_gripper_closed = not self.left_gripper_closed
                    self.qpos[9] = 0.04 if self.left_gripper_closed else 0.0
                    print(f"Left gripper: {'closed' if self.left_gripper_closed else 'open'}")
                elif event.key == pygame.K_h:
                    # Toggle right gripper
                    self.right_gripper_closed = not self.right_gripper_closed
                    self.qpos[15] = 0.04 if self.right_gripper_closed else 0.0
                    print(f"Right gripper: {'closed' if self.right_gripper_closed else 'open'}")
        return True

    def render_frame(self):
        """Render camera view to pygame screen"""
        # Get RGB from camera
        if 'sensor_data' in self.obs and 'cam_main' in self.obs['sensor_data']:
            rgb = self.obs['sensor_data']['cam_main']['rgb']
            if hasattr(rgb, 'cpu'):
                rgb = rgb.cpu().numpy()
            if len(rgb.shape) == 4:
                rgb = rgb[0]
            rgb = rgb[:, :, :3].astype(np.uint8)

            # Resize to screen size
            rgb_resized = np.array(
                pygame.transform.scale(
                    pygame.surfarray.make_surface(rgb.swapaxes(0, 1)),
                    (self.screen_width, self.screen_height)
                ).convert()
            )

            # Create surface and blit
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            surface = pygame.transform.scale(surface, (self.screen_width, self.screen_height))
            self.screen.blit(surface, (0, 0))
        else:
            self.screen.fill((50, 50, 50))

        # Draw control hints
        hints = [
            "WASD: Move | QE: Rotate | ZX: Lift",
            "IJKL/UO: Left Arm | Arrows/PgUp/Dn: Right Arm",
            "G/H: Grippers | R: Reset | ESC: Quit"
        ]
        y = 10
        for hint in hints:
            text = self.font.render(hint, True, (255, 255, 255))
            # Draw shadow for better visibility
            shadow = self.font.render(hint, True, (0, 0, 0))
            self.screen.blit(shadow, (12, y + 2))
            self.screen.blit(text, (10, y))
            y += 20

        # Draw joint info
        info_text = f"Base: ({self.qpos[0]:.2f}, {self.qpos[1]:.2f}, {self.qpos[2]:.2f}) | Lift: {self.qpos[3]:.2f}"
        text = self.font.render(info_text, True, (255, 255, 0))
        shadow = self.font.render(info_text, True, (0, 0, 0))
        self.screen.blit(shadow, (12, self.screen_height - 28))
        self.screen.blit(text, (10, self.screen_height - 30))

        pygame.display.flip()

    def step(self):
        """Execute one simulation step"""
        # Create action from current qpos
        action = self.qpos.copy()

        # Step environment
        self.obs, _, _, _, _ = self.env.step(action)

        # Update qpos from actual robot state
        # self.qpos = self.env.unwrapped.agent.robot.qpos[0].cpu().numpy().copy()

    def run(self):
        """Main loop"""
        running = True

        while running:
            # Handle events
            running = self.handle_events()
            if not running:
                break

            # Handle continuous input
            self.handle_input()

            # Step simulation
            self.step()

            # Render
            self.render_frame()

            # Cap at 30 FPS
            self.clock.tick(30)

        # Cleanup
        self.env.close()
        pygame.quit()
        print("Teleoperation ended.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='AlohaMini Pygame Teleoperation')
    parser.add_argument('--gpu', action='store_true', help='Use GPU backend (requires CUDA)')
    args = parser.parse_args()

    sim_backend = 'gpu' if args.gpu else 'cpu'
    print(f"Using {sim_backend} backend")

    teleop = AlohaMiniTeleop(sim_backend=sim_backend)
    teleop.run()


if __name__ == "__main__":
    main()
