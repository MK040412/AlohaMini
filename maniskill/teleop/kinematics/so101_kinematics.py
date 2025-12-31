"""
SO101 Decoupled Kinematics for AlohaMini Robot Arms

Ported directly from XLeRobot: /home/perelman/XLeRobot/software/src/model/SO101Robot.py

Uses a decoupled IK approach that separates the problem into:
1. Horizontal plane: joint1 (base rotation) - polar coordinate control
2. Vertical plane: joints 2,3 - law of cosines analytical solution
3. Tip orientation: joint4 (pitch), joint5 (roll) - direct control
"""

import math
from typing import Tuple


class SO101Kinematics:
    """
    Decoupled kinematics for SO101 5-DOF robot arm (+ gripper).
    All public methods use degrees for input/output.

    Ported from XLeRobot SO101Robot.py for exact compatibility.
    """

    def __init__(self, l1: float = 0.1159, l2: float = 0.1350):
        """
        Initialize SO101 kinematics.

        Args:
            l1: Length of the first link (upper arm) in meters. Default: 0.1159m
            l2: Length of the second link (lower arm) in meters. Default: 0.1350m
        """
        self.l1 = l1  # Length of the first link (upper arm)
        self.l2 = l2  # Length of the second link (lower arm)

    def inverse_kinematics(self, x: float, y: float, l1: float = None, l2: float = None, elbow_down: bool = False) -> Tuple[float, float]:
        """
        Calculate inverse kinematics for a 2-link robotic arm.

        Ported from XLeRobot SO101Robot.py inverse_kinematics()

        Parameters:
            x: End effector x coordinate
            y: End effector y coordinate
            l1: Upper arm length (default uses instance value)
            l2: Lower arm length (default uses instance value)
            elbow_down: If True, use elbow-down configuration (more stable in simulation)

        Returns:
            joint2_deg, joint3_deg: Joint angles in degrees (shoulder_lift, elbow_flex)
        """
        if l1 is None:
            l1 = self.l1
        if l2 is None:
            l2 = self.l2

        # Calculate joint2 and joint3 offsets in theta1 and theta2
        # These offsets account for URDF geometry
        theta1_offset = math.atan2(0.028, 0.11257)  # theta1 offset when joint2=0
        theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # theta2 offset when joint3=0

        # Calculate distance from origin to target point
        r = math.sqrt(x**2 + y**2)
        r_max = l1 + l2  # Maximum reachable distance

        # If target point is beyond maximum workspace, scale it to the boundary
        if r > r_max:
            scale_factor = r_max / r
            x *= scale_factor
            y *= scale_factor
            r = r_max

        # If target point is less than minimum workspace (|l1-l2|), scale it
        r_min = abs(l1 - l2)
        if r < r_min and r > 0:
            scale_factor = r_min / r
            x *= scale_factor
            y *= scale_factor
            r = r_min

        # Use law of cosines to calculate theta2
        cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)

        # Clamp cos_theta2 to valid range [-1, 1] to avoid domain errors
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))

        # Calculate theta2 (elbow angle)
        # Two solutions: elbow-up (default) or elbow-down
        if elbow_down:
            theta2 = -(math.pi - math.acos(cos_theta2))  # Elbow-down configuration
        else:
            theta2 = math.pi - math.acos(cos_theta2)     # Elbow-up configuration

        # Calculate theta1 (shoulder angle)
        beta = math.atan2(y, x)
        gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
        theta1 = beta + gamma

        # Convert theta1 and theta2 to joint2 and joint3 angles
        joint2 = theta1 + theta1_offset
        joint3 = theta2 + theta2_offset

        # Ensure angles are within URDF limits
        joint2 = max(-0.1, min(3.45, joint2))
        joint3 = max(-0.2, min(math.pi, joint3))

        # Convert from radians to degrees
        joint2_deg = math.degrees(joint2)
        joint3_deg = math.degrees(joint3)

        # Apply coordinate system transformation (critical for X/Y decoupling)
        joint2_deg = 90 - joint2_deg
        joint3_deg = joint3_deg - 90

        return joint2_deg, joint3_deg

    def forward_kinematics(self, joint2_deg: float, joint3_deg: float, l1: float = None, l2: float = None) -> Tuple[float, float]:
        """
        Calculate forward kinematics for a 2-link robotic arm.

        Ported from XLeRobot SO101Robot.py forward_kinematics()

        Parameters:
            joint2_deg: Shoulder lift joint angle in degrees
            joint3_deg: Elbow flex joint angle in degrees
            l1: Upper arm length (default uses instance value)
            l2: Lower arm length (default uses instance value)

        Returns:
            x, y: End effector coordinates in meters
        """
        if l1 is None:
            l1 = self.l1
        if l2 is None:
            l2 = self.l2

        # Convert degrees to radians and apply inverse transformation
        joint2_rad = math.radians(90 - joint2_deg)
        joint3_rad = math.radians(joint3_deg + 90)

        # Calculate joint2 and joint3 offsets (same as IK)
        theta1_offset = math.atan2(0.028, 0.11257)
        theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

        # Convert joint angles back to theta1 and theta2
        theta1 = joint2_rad - theta1_offset
        theta2 = joint3_rad - theta2_offset

        # Forward kinematics calculations
        # Link 2 angle = theta1 - theta2 (standard 2-link arm)
        # Note: XLeRobot uses theta1 + theta2 - pi which is NOT FK/IK consistent
        x = l1 * math.cos(theta1) + l2 * math.cos(theta1 - theta2)
        y = l1 * math.sin(theta1) + l2 * math.sin(theta1 - theta2)

        return x, y

    def compute_wrist_flex(self, joint2_deg: float, joint3_deg: float, pitch: float = 0.0) -> float:
        """
        Compute wrist flex angle to maintain end-effector pitch.

        The wrist flex compensates for shoulder and elbow angles to keep
        the end-effector at a constant pitch angle.

        Parameters:
            joint2_deg: Shoulder lift joint angle in degrees
            joint3_deg: Elbow flex joint angle in degrees
            pitch: Desired pitch adjustment in degrees (0 = level with ground)

        Returns:
            wrist_flex_deg: Wrist flex angle in degrees
        """
        # Wrist flex compensates for arm angles to maintain pitch
        wrist_flex_deg = -joint2_deg - joint3_deg + pitch
        return wrist_flex_deg

    @property
    def workspace_limits(self) -> dict:
        """
        Get workspace limits for the arm.

        Returns:
            Dictionary with r_min, r_max (reachable distance limits)
        """
        return {
            "r_min": abs(self.l1 - self.l2),
            "r_max": self.l1 + self.l2,
        }


# Convenience function for standalone use
def inverse_kinematics(x: float, y: float, l1: float = 0.1159, l2: float = 0.1350) -> Tuple[float, float]:
    """
    2-link planar IK using law of cosines (vertical plane only).

    Parameters:
        x: End effector x coordinate (forward distance)
        y: End effector y coordinate (height)
        l1: Upper arm length (default 0.1159 m)
        l2: Lower arm length (default 0.1350 m)

    Returns:
        joint2_deg, joint3_deg: Joint angles in degrees
    """
    kinematics = SO101Kinematics(l1, l2)
    return kinematics.inverse_kinematics(x, y)
