"""
Keyboard Controller for AlohaMini Teleoperation

XLeRobot-compatible keyboard mapping for IK-based arm control.
Based on: /home/perelman/XLeRobot/software/examples/1_so100_keyboard_ee_control.py

Keyboard Mapping:
    Mobile Base (Numpad Style):
        8/5: Move forward/backward
        4/6: Strafe left/right
        7/9: Rotate left/right
        Page UP/DOWN: Lift up/down

    Left Arm:
        Q/A: Joint 1 (shoulder_pan) decrease/increase
        W/S: End effector X (forward/backward)
        E/D: End effector Y (up/down)
        R/F: Pitch adjustment increase/decrease
        T/G: Wrist roll (joint 5) decrease/increase
        Y/H: Gripper close/open

    Right Arm:
        U/J: Joint 1 (shoulder_pan) decrease/increase
        I/K: End effector X (forward/backward)
        O/L: End effector Y (up/down)
        P/;: Pitch adjustment increase/decrease
        [/': Wrist roll decrease/increase
        ]/\\: Gripper close/open

    General:
        X: Exit
        SPACE: Reset arms to initial position
"""

import pygame
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

from .base import ControlGoal, ArmState


@dataclass
class KeyboardConfig:
    """Configuration for keyboard controller."""
    # Movement step sizes
    ee_step: float = 0.004  # End-effector position step (meters)
    joint_step: float = 1.0  # Joint angle step (degrees)
    pitch_step: float = 1.0  # Pitch adjustment step (degrees)

    # Mobile base control
    base_vel_step: float = 0.3    # Base linear velocity (m/s)
    base_omega_step: float = 0.5  # Base angular velocity (rad/s)
    lift_step: float = 0.01       # Lift position step (meters)


# XLeRobot-compatible keyboard mapping for LEFT arm
LEFT_ARM_MAPPING = {
    # Key: (action_type, value)
    # action_type: 'joint1', 'ee_x', 'ee_y', 'pitch', 'wrist_roll', 'gripper'
    pygame.K_q: ('joint1', -1),      # Joint 1 decrease
    pygame.K_a: ('joint1', +1),      # Joint 1 increase
    pygame.K_w: ('ee_x', +1),        # EE forward (+x distance)
    pygame.K_s: ('ee_x', -1),        # EE backward (-x distance)
    pygame.K_e: ('ee_y', +1),        # EE up (+y height)
    pygame.K_d: ('ee_y', -1),        # EE down (-y height)
    pygame.K_r: ('pitch', +1),       # Pitch increase
    pygame.K_f: ('pitch', -1),       # Pitch decrease
    pygame.K_t: ('wrist_roll', -1),  # Wrist roll decrease
    pygame.K_g: ('wrist_roll', +1),  # Wrist roll increase
    pygame.K_y: ('gripper', -1),     # Gripper close
    pygame.K_h: ('gripper', +1),     # Gripper open
}

# Keyboard mapping for RIGHT arm (mirror layout)
RIGHT_ARM_MAPPING = {
    pygame.K_u: ('joint1', -1),      # Joint 1 decrease
    pygame.K_j: ('joint1', +1),      # Joint 1 increase
    pygame.K_i: ('ee_x', +1),        # EE forward
    pygame.K_k: ('ee_x', -1),        # EE backward
    pygame.K_o: ('ee_y', +1),        # EE up
    pygame.K_l: ('ee_y', -1),        # EE down
    pygame.K_p: ('pitch', +1),       # Pitch increase
    pygame.K_SEMICOLON: ('pitch', -1),  # Pitch decrease
    pygame.K_LEFTBRACKET: ('wrist_roll', -1),   # Wrist roll decrease
    pygame.K_QUOTE: ('wrist_roll', +1),         # Wrist roll increase
    pygame.K_RIGHTBRACKET: ('gripper', -1),     # Gripper close
    pygame.K_BACKSLASH: ('gripper', +1),        # Gripper open
}

# Special keys
SPECIAL_KEYS = {
    pygame.K_x: 'exit',
    pygame.K_SPACE: 'reset',
    pygame.K_ESCAPE: 'exit',
}

# Mobile base key mapping (numpad style)
BASE_MAPPING = {
    # Numpad-style controls
    pygame.K_8: ('base_vx', +1),        # Forward
    pygame.K_5: ('base_vx', -1),        # Backward
    pygame.K_4: ('base_vy', +1),        # Strafe left
    pygame.K_6: ('base_vy', -1),        # Strafe right
    pygame.K_7: ('base_omega', +1),     # Rotate left
    pygame.K_9: ('base_omega', -1),     # Rotate right
    pygame.K_PAGEUP: ('lift', +1),      # Lift up
    pygame.K_PAGEDOWN: ('lift', -1),    # Lift down
}


class KeyboardController:
    """
    Keyboard controller for AlohaMini dual-arm IK teleoperation.

    Uses XLeRobot-compatible key mapping for intuitive control.
    """

    def __init__(self, config: KeyboardConfig = None):
        """
        Initialize keyboard controller.

        Args:
            config: Keyboard configuration (optional)
        """
        self.config = config or KeyboardConfig()

        # Track which keys were pressed last frame (for edge detection)
        self._prev_keys = set()

    def process_keys(self, keys_pressed: pygame.key.ScancodeWrapper) -> Dict[str, Any]:
        """
        Process pygame key states and return control deltas.

        Args:
            keys_pressed: Result of pygame.key.get_pressed()

        Returns:
            Dictionary with control deltas for left/right arms and mobile base
        """
        result = {
            'left': self._create_empty_delta(),
            'right': self._create_empty_delta(),
            'base': self._create_base_delta(),
            'special': None,
        }

        # Check special keys first
        for key, action in SPECIAL_KEYS.items():
            if keys_pressed[key]:
                result['special'] = action
                return result

        # Process left arm keys
        for key, (action_type, direction) in LEFT_ARM_MAPPING.items():
            if keys_pressed[key]:
                self._apply_action(result['left'], action_type, direction)

        # Process right arm keys
        for key, (action_type, direction) in RIGHT_ARM_MAPPING.items():
            if keys_pressed[key]:
                self._apply_action(result['right'], action_type, direction)

        # Process mobile base keys
        for key, (action_type, direction) in BASE_MAPPING.items():
            if keys_pressed[key]:
                self._apply_base_action(result['base'], action_type, direction)

        return result

    def _create_empty_delta(self) -> Dict[str, float]:
        """Create empty delta dictionary for arm control."""
        return {
            'joint1_delta': 0.0,
            'ee_x_delta': 0.0,
            'ee_y_delta': 0.0,
            'pitch_delta': 0.0,
            'wrist_roll_delta': 0.0,
            'gripper_delta': 0.0,
        }

    def _create_base_delta(self) -> Dict[str, float]:
        """Create empty delta dictionary for mobile base control."""
        return {
            'base_vx': 0.0,
            'base_vy': 0.0,
            'base_omega': 0.0,
            'lift_delta': 0.0,
        }

    def _apply_action(self, delta: Dict[str, float], action_type: str, direction: int):
        """
        Apply keyboard action to arm delta dictionary.

        Args:
            delta: Delta dictionary to update
            action_type: Type of action ('joint1', 'ee_x', etc.)
            direction: Direction of change (+1 or -1)
        """
        if action_type == 'joint1':
            delta['joint1_delta'] += direction * self.config.joint_step
        elif action_type == 'ee_x':
            delta['ee_x_delta'] += direction * self.config.ee_step
        elif action_type == 'ee_y':
            delta['ee_y_delta'] += direction * self.config.ee_step
        elif action_type == 'pitch':
            delta['pitch_delta'] += direction * self.config.pitch_step
        elif action_type == 'wrist_roll':
            delta['wrist_roll_delta'] += direction * self.config.joint_step
        elif action_type == 'gripper':
            delta['gripper_delta'] += direction * self.config.joint_step

    def _apply_base_action(self, delta: Dict[str, float], action_type: str, direction: int):
        """
        Apply keyboard action to mobile base delta dictionary.

        Args:
            delta: Base delta dictionary to update
            action_type: Type of action ('base_vx', 'base_omega', 'lift')
            direction: Direction of change (+1 or -1)
        """
        if action_type == 'base_vx':
            delta['base_vx'] = direction * self.config.base_vel_step
        elif action_type == 'base_vy':
            delta['base_vy'] = direction * self.config.base_vel_step
        elif action_type == 'base_omega':
            delta['base_omega'] = direction * self.config.base_omega_step
        elif action_type == 'lift':
            delta['lift_delta'] = direction * self.config.lift_step

    def update_arm_state(self, arm_state: ArmState, delta: Dict[str, float], kinematics) -> ArmState:
        """
        Update arm state based on keyboard deltas.

        Args:
            arm_state: Current arm state
            delta: Delta dictionary from process_keys()
            kinematics: SO101Kinematics instance for IK computation

        Returns:
            Updated arm state
        """
        # Update joint 1 (shoulder pan)
        arm_state.joint1_deg += delta['joint1_delta']

        # Update end-effector position
        arm_state.ee_x += delta['ee_x_delta']
        arm_state.ee_y += delta['ee_y_delta']

        # Compute IK for joints 2 and 3
        joint2_deg, joint3_deg = kinematics.inverse_kinematics(arm_state.ee_x, arm_state.ee_y)
        arm_state.joint2_deg = joint2_deg
        arm_state.joint3_deg = joint3_deg

        # Update pitch
        arm_state.pitch += delta['pitch_delta']

        # Compute wrist flex (joint 4) to compensate for arm angles
        arm_state.joint4_deg = kinematics.compute_wrist_flex(
            arm_state.joint2_deg, arm_state.joint3_deg, arm_state.pitch
        )

        # Update wrist roll (joint 5)
        arm_state.joint5_deg += delta['wrist_roll_delta']

        # Update gripper (joint 6)
        arm_state.joint6_deg += delta['gripper_delta']

        return arm_state

    @staticmethod
    def get_help_text() -> str:
        """Get keyboard control help text."""
        return """
=== AlohaMini Keyboard Control (XLeRobot Style) ===

MOBILE BASE (Numpad Style):
  8/5: Move forward/backward
  4/6: Strafe left/right
  7/9: Rotate left/right
  Page UP/DOWN: Lift up/down

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
  SPACE: Reset arms
  X/ESC: Exit

================================================
"""
