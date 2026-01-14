"""
Main Teleoperation Controller for AlohaMini in ManiSkill3

This controller bridges keyboard/VR inputs to ManiSkill3 environment actions.
"""

import asyncio
import numpy as np
from typing import Optional, Dict, Any

from .kinematics.aloha_mini_kinematics import AlohaMiniKinematics
from .kinematics.so100_kinematics import SO100Kinematics
from .kinematics.so100_kinematics_v2 import SO100KinematicsV2
from .inputs.base import ArmState, ControlGoal, ControlMode
from .inputs.keyboard_controller import KeyboardController, KeyboardConfig
from .config import TeleopConfig


class TeleopController:
    """
    Unified teleoperation controller for AlohaMini in ManiSkill3.

    Manages:
    - Keyboard input processing
    - VR WebSocket input processing (when VR server is running)
    - IK computation for both arms
    - Action generation for ManiSkill3 environment

    Action format for aloha_mini_so100_v2 (pd_joint_pos mode):
    [base_vx, base_vy, base_omega, lift,
     left_j1, left_j2, left_j3, left_j4, left_j5, left_j6,
     right_j1, right_j2, right_j3, right_j4, right_j5, right_j6]
    = 16 dimensions
    """

    # Joint indices in action array (SEQUENTIAL - controller order, not URDF order)
    # Controller defines: base(3) + lift(1) + left_arm(6) + right_arm(6) = 16
    BASE_VX_IDX = 0
    BASE_VY_IDX = 1
    BASE_OMEGA_IDX = 2
    LIFT_IDX = 3
    LEFT_ARM_START = 4
    LEFT_ARM_END = 10
    RIGHT_ARM_START = 10
    RIGHT_ARM_END = 16

    def __init__(self, config: TeleopConfig = None, robot_variant: str = "aloha_mini_so100_v2"):
        """
        Initialize the teleoperation controller.

        Args:
            config: Teleoperation configuration
            robot_variant: Robot variant name (e.g., "aloha_mini_so100")
        """
        self.config = config or TeleopConfig()
        self.robot_variant = robot_variant

        # Initialize kinematics based on robot variant
        if robot_variant in ["aloha_mini_so100", "aloha_mini_so100_v2"]:
            # Use SO100KinematicsV2 for SO100 variants
            self.kinematics = SO100KinematicsV2()
            # Use home position from kinematics
            initial_x = self.kinematics.initial_ee_x
            initial_y = self.kinematics.initial_ee_y
        else:
            # Default to AlohaMini kinematics for other variants
            self.kinematics = AlohaMiniKinematics()
            # AlohaMini uses URDF Y/Z convention
            initial_x = self.config.initial_ee_y  # URDF Y (forward)
            initial_y = self.config.initial_ee_z  # URDF Z (height)

        # Initialize arm states with variant-specific positions
        self.left_arm = ArmState(ee_x=initial_x, ee_y=initial_y)
        self.right_arm = ArmState(ee_x=initial_x, ee_y=initial_y)

        # Initialize IK for initial positions
        self._update_arm_ik(self.left_arm)
        self._update_arm_ik(self.right_arm)

        # Base and lift state
        self.base_velocity = np.zeros(3)  # [vx, vy, omega]
        self.lift_position = 0.3  # Initial lift position (middle of 0-0.6 range)

        # Keyboard controller
        keyboard_config = KeyboardConfig(
            ee_step=self.config.pos_step,
            joint_step=self.config.angle_step,
            pitch_step=self.config.pitch_step,
        )
        self.keyboard_controller = KeyboardController(keyboard_config)

        # Command queue for async VR inputs
        self.command_queue = asyncio.Queue()

        # VR server (initialized separately)
        self.vr_server = None

        # Control state
        self.is_running = True

    def _update_arm_ik(self, arm: ArmState):
        """
        Update arm IK (joints 2, 3, 4) based on current EE position and pitch.

        Args:
            arm: Arm state to update
        """
        if isinstance(self.kinematics, (SO100KinematicsV2, SO100Kinematics)):
            # SO100 variants: IK returns (j2, j3, j4)
            # j4 = pitch - (j2 + j3), same formula as AlohaMini
            joint2_rad, joint3_rad, joint4_rad = self.kinematics.inverse_kinematics(
                arm.ee_x, arm.ee_y, np.radians(arm.pitch)
            )
            arm.joint2_deg = np.degrees(joint2_rad)
            arm.joint3_deg = np.degrees(joint3_rad)
            arm.joint4_deg = np.degrees(joint4_rad)
        else:
            # AlohaMini: IK returns (j2, j3), compute j4 separately
            joint2_deg, joint3_deg = self.kinematics.inverse_kinematics(arm.ee_x, arm.ee_y)
            arm.joint2_deg = joint2_deg
            arm.joint3_deg = joint3_deg
            arm.joint4_deg = self.kinematics.compute_wrist_flex(
                arm.joint2_deg, arm.joint3_deg, arm.pitch
            )

    def process_keyboard(self, keys_pressed) -> bool:
        """
        Process pygame keyboard input.

        Args:
            keys_pressed: Result of pygame.key.get_pressed()

        Returns:
            True if should continue, False if exit requested
        """
        result = self.keyboard_controller.process_keys(keys_pressed)

        # Check for special commands
        if result['special'] == 'exit':
            self.is_running = False
            return False
        elif result['special'] == 'reset':
            self.reset_arms()
            return True

        # Update left arm
        left_delta = result['left']
        if self._has_delta(left_delta):
            self._apply_delta(self.left_arm, left_delta)

        # Update right arm
        right_delta = result['right']
        if self._has_delta(right_delta):
            self._apply_delta(self.right_arm, right_delta)

        # Update mobile base velocity from keyboard
        base_delta = result['base']
        self.base_velocity[0] = base_delta['base_vx']
        self.base_velocity[1] = base_delta['base_vy']
        self.base_velocity[2] = base_delta['base_omega']

        # Update lift position (accumulate delta)
        if abs(base_delta['lift_delta']) > 1e-9:
            self.lift_position += base_delta['lift_delta']
            self.lift_position = max(-0.15, min(0.6, self.lift_position))

        return True

    def _has_delta(self, delta: Dict[str, float]) -> bool:
        """Check if delta has any non-zero values."""
        return any(abs(v) > 1e-9 for v in delta.values())

    def _apply_delta(self, arm: ArmState, delta: Dict[str, float]):
        """
        Apply keyboard delta to arm state.

        - Joint 1, 5, 6: Direct control (no IK needed)
        - Position (ee_x, ee_y) and pitch: Update via IK (computes j2, j3, j4)
        """
        # Direct controls (no IK needed)
        arm.joint1_deg += delta['joint1_delta']
        arm.joint5_deg += delta['wrist_roll_delta']
        arm.joint6_deg += delta['gripper_delta']

        # Update position and pitch
        arm.ee_x += delta['ee_x_delta']
        arm.ee_y += delta['ee_y_delta']
        arm.pitch += delta['pitch_delta']

        # Recalculate IK for j2, j3, j4
        self._update_arm_ik(arm)

    def process_vr_goal(self, goal: ControlGoal):
        """
        Process VR controller goal.

        Args:
            goal: Control goal from VR input
        """
        # Determine which arm(s) to update
        if goal.arm == 'left' or goal.arm == 'both':
            self._apply_vr_goal_to_arm(self.left_arm, goal)
        if goal.arm == 'right' or goal.arm == 'both':
            self._apply_vr_goal_to_arm(self.right_arm, goal)

        # Handle base velocity from VR joysticks (NEW)
        if goal.base_vx is not None:
            self.base_velocity[0] = goal.base_vx
        if goal.base_vy is not None:
            self.base_velocity[1] = goal.base_vy
        if goal.base_omega is not None:
            self.base_velocity[2] = goal.base_omega

    def _apply_vr_goal_to_arm(self, arm: ArmState, goal: ControlGoal):
        """
        Apply VR goal to a single arm.

        - Joint 1, 5, 6: Direct control (no IK needed)
        - Position (ee_x, ee_y) and pitch: Update via IK (computes j2, j3, j4)
        """
        needs_ik_update = False

        # Update EE position if provided
        if goal.ee_x is not None:
            arm.ee_x = goal.ee_x
            needs_ik_update = True
        if goal.ee_y is not None:
            arm.ee_y = goal.ee_y
            needs_ik_update = True

        # Update pitch
        if goal.pitch_delta is not None:
            arm.pitch += goal.pitch_delta
            needs_ik_update = True
        if goal.wrist_flex_deg is not None:
            arm.pitch = goal.wrist_flex_deg
            needs_ik_update = True

        # Recalculate IK for j2, j3, j4 if needed
        if needs_ik_update:
            self._update_arm_ik(arm)

        # Direct joint overrides (no IK needed)
        if goal.joint1_deg is not None:
            arm.joint1_deg = goal.joint1_deg
        if goal.joint5_deg is not None:
            arm.joint5_deg = goal.joint5_deg
        if goal.wrist_roll_deg is not None:
            arm.joint5_deg = goal.wrist_roll_deg

        # Update gripper
        if goal.gripper_closed is not None:
            arm.gripper_closed = goal.gripper_closed
            # Open = 0° (home position), Close = -90° (original value)
            arm.joint6_deg = -90.0 if goal.gripper_closed else 0.0

    def reset_arms(self):
        """Reset both arms to initial positions."""
        if self.robot_variant in ["aloha_mini_so100", "aloha_mini_so100_v2"]:
            # SO100 variants use home position from kinematics
            initial_x = self.kinematics.initial_ee_x
            initial_y = self.kinematics.initial_ee_y
        else:
            # Default to AlohaMini kinematics
            initial_x = self.config.initial_ee_y
            initial_y = self.config.initial_ee_z

        self.left_arm = ArmState(ee_x=initial_x, ee_y=initial_y)
        self.right_arm = ArmState(ee_x=initial_x, ee_y=initial_y)

        # Update IK for j2, j3, j4
        self._update_arm_ik(self.left_arm)
        self._update_arm_ik(self.right_arm)

        print(f"Arms reset to initial position ({initial_x:.4f}, {initial_y:.4f})")

    def set_base_velocity(self, vx: float = 0.0, vy: float = 0.0, omega: float = 0.0):
        """
        Set base velocity.

        Args:
            vx: Forward velocity
            vy: Lateral velocity
            omega: Angular velocity
        """
        self.base_velocity = np.array([vx, vy, omega])

    def set_lift_position(self, position: float):
        """
        Set lift position.

        Args:
            position: Lift position in meters (-0.15 to 0.6)
        """
        self.lift_position = max(-0.15, min(0.6, position))  # Range: -0.15 to 0.6

    def compute_action(self) -> np.ndarray:
        """
        Generate ManiSkill3 action array from current state.

        Returns:
            16-dimensional action array

        Action format (SEQUENTIAL - controller order):
            [0:3]   = base velocity [vx, vy, omega]
            [3]     = lift position
            [4:10]  = left arm joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
            [10:16] = right arm joints (same order)
        """
        action = np.zeros(16)

        # Base velocity (indices 0-2)
        action[self.BASE_VX_IDX] = self.base_velocity[0]
        action[self.BASE_VY_IDX] = self.base_velocity[1]
        action[self.BASE_OMEGA_IDX] = self.base_velocity[2]

        # Lift position (index 3)
        action[self.LIFT_IDX] = self.lift_position

        # Get joint arrays [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        left_joints = self.left_arm.get_joint_positions_rad()
        right_joints = self.right_arm.get_joint_positions_rad()

        # Sequential format: all left arm joints, then all right arm joints
        action[self.LEFT_ARM_START:self.LEFT_ARM_END] = left_joints
        action[self.RIGHT_ARM_START:self.RIGHT_ARM_END] = right_joints

        return action

    def get_state_info(self) -> Dict[str, Any]:
        """
        Get current controller state for display.

        Returns:
            Dictionary with state information
        """
        return {
            'left_arm': {
                'ee_x': self.left_arm.ee_x,
                'ee_y': self.left_arm.ee_y,
                'joints_deg': self.left_arm.get_joint_positions_deg().tolist(),
                'pitch': self.left_arm.pitch,
            },
            'right_arm': {
                'ee_x': self.right_arm.ee_x,
                'ee_y': self.right_arm.ee_y,
                'joints_deg': self.right_arm.get_joint_positions_deg().tolist(),
                'pitch': self.right_arm.pitch,
            },
            'base': {
                'velocity': self.base_velocity.tolist(),
            },
            'lift': {
                'position': self.lift_position,
            },
        }

    async def process_vr_commands(self):
        """Process pending VR commands from the queue (non-blocking)."""
        while not self.command_queue.empty():
            try:
                goal = self.command_queue.get_nowait()
                self.process_vr_goal(goal)
            except asyncio.QueueEmpty:
                break

    @staticmethod
    def get_help_text() -> str:
        """Get help text for the controller."""
        return KeyboardController.get_help_text()
