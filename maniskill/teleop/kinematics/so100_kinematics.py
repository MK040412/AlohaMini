"""
SO100 Arm Kinematics for AlohaMini

Based on XLeRobot's IK implementation (demo_ctrl_action_ee_keyboard.py).
Uses 2-link planar IK with URDF geometry offsets.

SO100 arm structure (from URDF):
- shoulder_pan: xyz=[0, -0.0452, 0.0165], axis=[0,-1,0], limits=[-2.0, 2.0]
- shoulder_lift: xyz=[0, 0.1025, 0.0306], axis=[1,0,0], limits=[-1.5708, 1.5708]
- elbow_flex: xyz=[0, 0.11257, 0.028], axis=[1,0,0], limits=[-1.5708, 1.5708]
- wrist_flex: xyz=[0, 0.0052, 0.1349], axis=[1,0,0], limits=[-1.8, 1.8]
- wrist_roll: xyz=[0, -0.0601, 0], axis=[0,1,0], limits=[-3.14159, 3.14159]
- gripper: axis=[0,0,1], limits=[-1.1, 1.1]

Coordinate system (XLeRobot convention):
    x = forward distance
    y = height (up)
"""

import math
from typing import Tuple, List
import numpy as np


class SO100Kinematics:
    """
    XLeRobot-style kinematics for SO100 robot arms.

    The IK uses 2-link planar kinematics with URDF geometry offsets.
    Pitch compensation and automatic wrist_flex calculation are supported.
    """

    def __init__(self):
        # Link lengths from URDF (same as XLeRobot)
        # l1: shoulder to elbow = sqrt(0.11257² + 0.028²) ≈ 0.1159m
        # l2: elbow to wrist = sqrt(0.0052² + 0.1349²) ≈ 0.1350m
        self.l1 = 0.1159
        self.l2 = 0.1350

        # URDF geometry offsets (joint angles when theta=0)
        self.theta1_offset = math.atan2(0.028, 0.11257)  # ~14° offset for shoulder
        self.theta2_offset = math.atan2(0.0052, 0.1349) + self.theta1_offset  # elbow offset

        # Tip length (wrist to gripper tip) for pitch compensation
        self.tip_length = 0.108

        # Joint limits (radians)
        self.joint_limits = {
            'shoulder_pan': (-2.0, 2.0),
            'shoulder_lift': (-0.1, 3.45),  # XLeRobot limits
            'elbow_flex': (-0.2, math.pi),  # XLeRobot limits
            'wrist_flex': (-1.8, 1.8),
            'wrist_roll': (-3.14159, 3.14159),
            'gripper': (-1.1, 1.1),
        }

        # Initial EE position (XLeRobot default)
        self.initial_ee_x = 0.162
        self.initial_ee_y = 0.118

        # Rest position (using IK for initial position)
        sh_lift, el_flex, wr_flex = self.inverse_kinematics(
            self.initial_ee_x, self.initial_ee_y, 0.0
        )
        self.rest_position = np.array([0, sh_lift, el_flex, wr_flex, 0, -1.1])

        # Ready position (arms slightly raised, gripper open)
        self.ready_position = np.array([0, sh_lift, el_flex, wr_flex, 0, 0.0])

    def get_joint_names(self, prefix: str = '') -> List[str]:
        """Get joint names with optional prefix (left_/right_)."""
        names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                 'wrist_flex', 'wrist_roll', 'gripper']
        if prefix:
            return [f"{prefix}{name}" for name in names]
        return names

    def clamp_joints(self, joints: np.ndarray) -> np.ndarray:
        """Clamp joint values to their limits."""
        clamped = np.copy(joints)
        limits = [
            self.joint_limits['shoulder_pan'],
            self.joint_limits['shoulder_lift'],
            self.joint_limits['elbow_flex'],
            self.joint_limits['wrist_flex'],
            self.joint_limits['wrist_roll'],
            self.joint_limits['gripper'],
        ]
        for i, (lower, upper) in enumerate(limits):
            clamped[i] = np.clip(joints[i], lower, upper)
        return clamped

    def inverse_kinematics(self, x: float, y: float, pitch: float = 0.0) -> Tuple[float, float, float]:
        """
        Compute shoulder_lift, elbow_flex, and wrist_flex for target position.

        This is the XLeRobot-style IK that accounts for URDF geometry offsets.

        Args:
            x: Forward distance (x in arm frame)
            y: Height (y in arm frame)
            pitch: Desired end-effector pitch angle (only used for wrist_flex, not IK)

        Returns:
            (shoulder_lift, elbow_flex, wrist_flex) in radians
        """
        # NOTE: Pitch affects wrist_flex only, not the 2-link IK target position
        # The target (x, y) is the gripper base position (before tip compensation)

        # Distance to target
        r = math.sqrt(x**2 + y**2)

        # Workspace limits
        r_max = self.l1 + self.l2
        r_min = abs(self.l1 - self.l2)

        # Scale to workspace if out of bounds
        if r > r_max:
            scale = r_max / r
            x *= scale
            y *= scale
            r = r_max
        elif r < r_min and r > 0:
            scale = r_min / r
            x *= scale
            y *= scale
            r = r_min

        # Law of cosines for elbow angle (NOTE: negative sign is critical!)
        cos_theta2 = -(r**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        theta2 = math.pi - math.acos(cos_theta2)

        # Shoulder angle using geometric approach
        beta = math.atan2(y, x)
        gamma = math.atan2(self.l2 * math.sin(theta2), self.l1 + self.l2 * math.cos(theta2))
        theta1 = beta + gamma

        # Convert to URDF joint angles (add offsets)
        joint2 = theta1 + self.theta1_offset  # shoulder_lift
        joint3 = theta2 + self.theta2_offset  # elbow_flex

        # Clamp to URDF joint limits
        joint2 = np.clip(joint2, -0.1, 3.45)
        joint3 = np.clip(joint3, -0.2, math.pi)

        # Compute wrist_flex to maintain end-effector pitch
        # wrist_flex = joint2 - joint3 + pitch
        # This maintains: EE_pitch = shoulder_lift - elbow_flex + wrist_flex
        wrist_flex = joint2 - joint3 + pitch
        wrist_flex = np.clip(wrist_flex, -1.8, 1.8)

        return joint2, joint3, wrist_flex

    def forward_kinematics(self, shoulder_lift: float, elbow_flex: float) -> Tuple[float, float]:
        """
        Compute end-effector X-Y from joint angles.
        
        This FK must match the inverse kinematics exactly (round-trip consistency).
        Uses geometric 2-link arm with URDF offsets.
        """
        # Convert from URDF joints to geometric angles (remove offsets)
        theta1 = shoulder_lift - self.theta1_offset
        theta2 = elbow_flex - self.theta2_offset

        # First link endpoint relative to shoulder
        x1 = self.l1 * math.cos(theta1)
        y1 = self.l1 * math.sin(theta1)

        # Second link: note theta2 = pi - elbow_internal_angle
        # So the second link is at angle: (theta1 - theta2) from horizontal
        theta12 = theta1 - theta2
        
        # Second link endpoint relative to first link endpoint
        x2 = x1 + self.l2 * math.cos(theta12)
        y2 = y1 + self.l2 * math.sin(theta12)

        return x2, y2

    @property
    def workspace_limits(self) -> dict:
        """Get workspace limits."""
        return {
            "r_min": abs(self.l1 - self.l2),
            "r_max": self.l1 + self.l2,
            "l1": self.l1,
            "l2": self.l2,
        }


def test_kinematics():
    """Test the SO100 kinematics (XLeRobot-style)."""
    kin = SO100Kinematics()

    print("SO100 Kinematics Test (XLeRobot-style)")
    print("=" * 60)
    print(f"Link lengths: l1={kin.l1:.4f}m, l2={kin.l2:.4f}m")
    print(f"Total reach: {kin.workspace_limits['r_max']:.4f}m")
    print(f"Theta1 offset: {math.degrees(kin.theta1_offset):.2f}°")
    print(f"Theta2 offset: {math.degrees(kin.theta2_offset):.2f}°")

    print("\nJoint limits:")
    for name, (lower, upper) in kin.joint_limits.items():
        print(f"  {name}: [{math.degrees(lower):.1f}°, {math.degrees(upper):.1f}°]")

    print("\nIK test with XLeRobot initial position (0.162, 0.118):")
    x, y = 0.162, 0.118
    sh_lift, el_flex, wr_flex = kin.inverse_kinematics(x, y, 0.0)
    print(f"  shoulder_lift: {math.degrees(sh_lift):.2f}°")
    print(f"  elbow_flex: {math.degrees(el_flex):.2f}°")
    print(f"  wrist_flex: {math.degrees(wr_flex):.2f}°")

    x_fk, y_fk = kin.forward_kinematics(sh_lift, el_flex)
    print(f"  FK result: ({x_fk:.4f}, {y_fk:.4f})")
    print(f"  Error: {math.sqrt((x-x_fk)**2 + (y-y_fk)**2)*1000:.3f}mm")

    print("\nWorkspace test (Target -> FK):")
    test_targets = [
        (0.05, 0.05),
        (0.10, 0.10),
        (0.15, 0.12),
        (0.20, 0.15),
        (0.22, 0.10),
    ]
    for x, y in test_targets:
        sh_lift, el_flex, wr_flex = kin.inverse_kinematics(x, y, 0.0)
        x_fk, y_fk = kin.forward_kinematics(sh_lift, el_flex)
        error = math.sqrt((x - x_fk)**2 + (y - y_fk)**2)
        print(f"  ({x:.2f}, {y:.2f}) -> ({x_fk:.3f}, {y_fk:.3f}) err={error*1000:.1f}mm")


if __name__ == "__main__":
    test_kinematics()
