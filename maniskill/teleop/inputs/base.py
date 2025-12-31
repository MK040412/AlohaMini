"""
Base classes and data structures for input providers.

Ported from XLeRobot XLeVR: /home/perelman/XLeRobot/XLeVR/xlevr/inputs/base.py
"""

import asyncio
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any
from enum import Enum


class ControlMode(Enum):
    """Control modes for the teleoperation system."""
    POSITION_CONTROL = "position"
    VELOCITY_CONTROL = "velocity"
    IDLE = "idle"


@dataclass
class ControlGoal:
    """
    High-level control goal message sent from input providers.

    This dataclass represents a control command from either keyboard or VR input.

    Attributes:
        arm: Which arm this goal is for ("left", "right", or "both")
        mode: Control mode (position, velocity, or idle)
        target_position: 3D target position in robot coordinates [x, y, z]
        wrist_roll_deg: Wrist roll angle in degrees
        wrist_flex_deg: Wrist flex (pitch) angle in degrees
        gripper_closed: Gripper state (True=closed, False=open, None=no change)
        metadata: Additional data for debugging/monitoring
    """
    arm: Literal["left", "right", "both"] = "left"
    mode: Optional[ControlMode] = None  # Control mode (None = no mode change)
    target_position: Optional[np.ndarray] = None  # 3D position in robot coordinates
    wrist_roll_deg: Optional[float] = None  # Wrist roll angle in degrees
    wrist_flex_deg: Optional[float] = None  # Wrist flex (pitch) angle in degrees
    gripper_closed: Optional[bool] = None  # Gripper state (None = no change)

    # End-effector 2D position for IK (in arm plane)
    ee_x: Optional[float] = None  # Forward distance
    ee_y: Optional[float] = None  # Height

    # Joint angle overrides (degrees)
    joint1_deg: Optional[float] = None  # shoulder_pan
    joint5_deg: Optional[float] = None  # wrist_roll

    # Pitch adjustment
    pitch_delta: Optional[float] = None  # Change in pitch

    # Mobile base control (NEW)
    base_vx: Optional[float] = None      # Forward velocity (m/s)
    base_vy: Optional[float] = None      # Lateral velocity (m/s)
    base_omega: Optional[float] = None   # Angular velocity (rad/s)

    # Additional data for debugging/monitoring
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ArmState:
    """
    Internal state tracking for a single arm.

    Tracks end-effector position, joint angles, and pitch for IK computation.
    """
    # End-effector position in 2D IK plane (meters)
    ee_x: float = 0.1629  # Initial X (forward distance)
    ee_y: float = 0.1131  # Initial Y (height)

    # Joint angles in degrees
    joint1_deg: float = 0.0  # shoulder_pan
    joint2_deg: float = 0.0  # shoulder_lift (IK computed)
    joint3_deg: float = 0.0  # elbow_flex (IK computed)
    joint4_deg: float = 0.0  # wrist_flex (pitch compensation)
    joint5_deg: float = 0.0  # wrist_roll
    joint6_deg: float = 0.0  # gripper

    # Pitch adjustment (degrees)
    pitch: float = 0.0

    # Gripper state
    gripper_closed: bool = False

    def get_joint_positions_deg(self) -> np.ndarray:
        """Return joint positions in degrees."""
        return np.array([
            self.joint1_deg,
            self.joint2_deg,
            self.joint3_deg,
            self.joint4_deg,
            self.joint5_deg,
            self.joint6_deg,
        ])

    def get_joint_positions_rad(self) -> np.ndarray:
        """Return joint positions in radians for ManiSkill3."""
        return np.radians(self.get_joint_positions_deg())

    def copy(self) -> "ArmState":
        """Create a copy of this arm state."""
        return ArmState(
            ee_x=self.ee_x,
            ee_y=self.ee_y,
            joint1_deg=self.joint1_deg,
            joint2_deg=self.joint2_deg,
            joint3_deg=self.joint3_deg,
            joint4_deg=self.joint4_deg,
            joint5_deg=self.joint5_deg,
            joint6_deg=self.joint6_deg,
            pitch=self.pitch,
            gripper_closed=self.gripper_closed,
        )


class BaseInputProvider(ABC):
    """
    Abstract base class for input providers.

    Input providers (keyboard, VR, etc.) inherit from this class
    and implement the start/stop methods.
    """

    def __init__(self, command_queue: asyncio.Queue):
        """
        Initialize the input provider.

        Args:
            command_queue: Async queue for sending ControlGoal messages
        """
        self.command_queue = command_queue
        self.is_running = False

    @abstractmethod
    async def start(self):
        """Start the input provider."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the input provider."""
        pass

    async def send_goal(self, goal: ControlGoal):
        """
        Send a control goal to the command queue.

        Args:
            goal: ControlGoal to send
        """
        try:
            await self.command_queue.put(goal)
        except Exception as e:
            # Handle queue full or other errors
            pass
