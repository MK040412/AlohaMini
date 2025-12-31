"""
VR WebSocket Server with Mobile Base Control for AlohaMini Teleoperation

Extends VRWebSocketServer to add mobile base control via VR controller thumbsticks:
- Left controller thumbstick → Translation (base_vx, base_vy) using polar coordinates
- Right controller thumbstick → Rotation (base_omega)

Both controllers use polar coordinate system for smooth control.
"""

import asyncio
import json
import math
import logging
from typing import Dict, Optional
from dataclasses import dataclass

import numpy as np

from .vr_ws_server import VRWebSocketServer, VRControllerState
from .base import ControlGoal, ControlMode
from ..config import TeleopConfig

logger = logging.getLogger(__name__)


@dataclass
class MobileBaseConfig:
    """Configuration for mobile base control via VR joysticks."""
    max_linear_speed: float = 0.5      # Maximum linear velocity (m/s)
    max_angular_speed: float = 1.0     # Maximum angular velocity (rad/s)
    joystick_deadzone: float = 0.1     # Ignore joystick values below this
    velocity_smoothing: float = 0.3    # Low-pass filter alpha (0=no smoothing, 1=full smoothing)


class VRWebSocketServerWithMobile(VRWebSocketServer):
    """
    VR WebSocket server with mobile base control.

    Extends VRWebSocketServer to handle thumbstick input for mobile base:
    - Left controller thumbstick: Translation (polar system)
      - r (magnitude) → speed
      - theta (angle) → direction
      - Result: base_vx (forward/back), base_vy (strafe)
    - Right controller thumbstick: Rotation
      - x-axis → base_omega (angular velocity)

    Joystick Mapping:
    ```
    LEFT CONTROLLER (Translation)       RIGHT CONTROLLER (Rotation)
            ↑ Forward                          ↺ CCW
            │ (vx+)                            │ (omega+)
       ←────┼────→ Strafe              ←────┼────→
       (vy+)│(vy-)                     (omega+)│(omega-)
            │                                  │
            ↓ Backward                         ↻ CW
            (vx-)                              (omega-)
    ```
    """

    def __init__(
        self,
        command_queue: asyncio.Queue,
        config: TeleopConfig,
        mobile_config: MobileBaseConfig = None,
        teleop_controller=None
    ):
        """
        Initialize the VR server with mobile base control.

        Args:
            command_queue: Async queue for sending ControlGoal messages
            config: Teleoperation configuration
            mobile_config: Mobile base control configuration
            teleop_controller: Reference to TeleopController for getting current arm state
        """
        super().__init__(command_queue, config, teleop_controller)
        self.mobile_config = mobile_config or MobileBaseConfig()

        # Smoothed velocity state
        self._smoothed_vx = 0.0
        self._smoothed_vy = 0.0
        self._smoothed_omega = 0.0

    async def process_controller_data(self, data: Dict):
        """
        Process incoming VR controller data with mobile base control.

        Overrides parent to add thumbstick processing for mobile base.
        """
        # First, process arm control (parent behavior)
        await super().process_controller_data(data)

        # Then process thumbsticks for mobile base
        await self._process_mobile_base(data)

    async def _process_mobile_base(self, data: Dict):
        """
        Process thumbstick data for mobile base control.

        Left controller thumbstick → Translation (polar)
        Right controller thumbstick → Rotation
        """
        base_vx = 0.0
        base_vy = 0.0
        base_omega = 0.0

        # Left controller thumbstick → Translation
        if 'leftController' in data:
            left_data = data['leftController']
            thumbstick = left_data.get('thumbstick', {})
            tx = thumbstick.get('x', 0.0)
            ty = thumbstick.get('y', 0.0)

            # Apply polar transformation for translation
            vx, vy = self._polar_to_velocity(tx, ty, is_translation=True)
            base_vx = vx
            base_vy = vy

        # Right controller thumbstick → Rotation
        if 'rightController' in data:
            right_data = data['rightController']
            thumbstick = right_data.get('thumbstick', {})
            tx = thumbstick.get('x', 0.0)

            # Apply polar transformation for rotation (x-axis only)
            base_omega = self._polar_to_angular(tx)

        # Apply smoothing
        alpha = self.mobile_config.velocity_smoothing
        self._smoothed_vx = alpha * self._smoothed_vx + (1 - alpha) * base_vx
        self._smoothed_vy = alpha * self._smoothed_vy + (1 - alpha) * base_vy
        self._smoothed_omega = alpha * self._smoothed_omega + (1 - alpha) * base_omega

        # Send base velocity goal if any movement
        if abs(self._smoothed_vx) > 0.001 or abs(self._smoothed_vy) > 0.001 or abs(self._smoothed_omega) > 0.001:
            base_goal = ControlGoal(
                arm="both",  # Base control applies to whole robot
                base_vx=self._smoothed_vx,
                base_vy=self._smoothed_vy,
                base_omega=self._smoothed_omega,
                metadata={
                    "source": "vr_mobile",
                    "raw_vx": base_vx,
                    "raw_vy": base_vy,
                    "raw_omega": base_omega,
                }
            )
            await self.send_goal(base_goal)
        else:
            # Send zero velocity to stop
            stop_goal = ControlGoal(
                arm="both",
                base_vx=0.0,
                base_vy=0.0,
                base_omega=0.0,
                metadata={"source": "vr_mobile_stop"}
            )
            await self.send_goal(stop_goal)

    def _polar_to_velocity(self, x: float, y: float, is_translation: bool = True) -> tuple:
        """
        Convert joystick position to velocity using polar coordinates.

        Args:
            x: Joystick X axis (-1 to 1, left/right)
            y: Joystick Y axis (-1 to 1, up/down, inverted in WebXR)
            is_translation: True for translation, False for rotation

        Returns:
            Tuple of (vx, vy) velocities in m/s
        """
        # WebXR joystick Y-axis is typically inverted (up = negative)
        # Flip Y to make forward positive
        y = -y

        # Calculate polar coordinates
        r = math.sqrt(x * x + y * y)

        # Apply deadzone
        if r < self.mobile_config.joystick_deadzone:
            return (0.0, 0.0)

        # Normalize to remove deadzone
        r_normalized = (r - self.mobile_config.joystick_deadzone) / (1.0 - self.mobile_config.joystick_deadzone)
        r_normalized = min(r_normalized, 1.0)  # Clamp to 1.0

        # Calculate direction angle
        theta = math.atan2(y, x)

        # Convert back to Cartesian with scaled magnitude
        # Note: Robot coordinate system
        #   vx = forward velocity (joystick Y direction)
        #   vy = lateral velocity (joystick X direction)
        max_speed = self.mobile_config.max_linear_speed
        vx = r_normalized * math.sin(theta) * max_speed  # Forward from joystick Y
        vy = r_normalized * math.cos(theta) * max_speed  # Strafe from joystick X

        # ManiSkill3 AlohaMini coordinate system:
        #   +X = left, +Y = forward, +Z = up
        # So for intuitive control:
        #   Stick up (forward) → +Y → vy positive
        #   Stick right (strafe right) → -X → vx negative
        return (-r_normalized * x * max_speed, r_normalized * y * max_speed)

    def _polar_to_angular(self, x: float) -> float:
        """
        Convert joystick X position to angular velocity.

        Uses polar-style magnitude scaling from center.

        Args:
            x: Joystick X axis (-1 to 1)

        Returns:
            Angular velocity in rad/s (positive = CCW, negative = CW)
        """
        # Apply deadzone
        if abs(x) < self.mobile_config.joystick_deadzone:
            return 0.0

        # Normalize to remove deadzone
        sign = 1.0 if x > 0 else -1.0
        x_normalized = (abs(x) - self.mobile_config.joystick_deadzone) / (1.0 - self.mobile_config.joystick_deadzone)
        x_normalized = min(x_normalized, 1.0)  # Clamp

        # Scale to max angular speed
        # Positive X (right) = negative omega (CW rotation)
        # Negative X (left) = positive omega (CCW rotation)
        omega = -sign * x_normalized * self.mobile_config.max_angular_speed

        return omega

    def reset_mobile_state(self):
        """Reset smoothed velocity state."""
        self._smoothed_vx = 0.0
        self._smoothed_vy = 0.0
        self._smoothed_omega = 0.0

    async def stop(self):
        """Stop the server and reset mobile state."""
        self.reset_mobile_state()
        await super().stop()
