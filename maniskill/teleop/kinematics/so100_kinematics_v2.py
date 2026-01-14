"""
SO100 Arm Kinematics V2 - XLeRobot-style Analytical IK using Law of Cosines

V2 URDF structure:
- base_joint: rpy="0 0 0" (no rotation)
- shoulder_pan: rpy="1.5708 0 0" (90° X rotation)
- shoulder_lift: rpy="0 0 0", axis=[1,0,0] (positive X axis)
- elbow_flex: rpy="0 0 0", axis=[1,0,0] (positive X axis)

The 90° X rotation in shoulder_pan transforms coordinates:
- Arm local Y → World Z (up)
- Arm local Z → World -Y (forward)

User coordinates (ee_x, ee_y):
- ee_x = forward distance (positive = forward = -World_Y)
- ee_y = height (positive = up = World_Z)

XLeRobot IK coordinate frame:
- x = forward (arm's Z axis)
- y = up (arm's Y axis)
- θ₁ = angle of first link from horizontal (x-axis)
- θ₂ = exterior elbow angle

At home (j2=0, j3=0):
- θ₁ ≈ 76° (first link points mostly up)
- θ₂ ≈ 106° (arm is folded)
"""

import math
from typing import Tuple
import numpy as np


class SO100KinematicsV2:
    """
    Kinematics for SO100 robot arms with V2 URDF structure.
    Uses XLeRobot-style analytical IK with law of cosines.

    The arm operates in a 2D vertical plane (Y-Z in arm frame).
    FK and IK are mathematically consistent for accurate round-trips.
    """

    def __init__(self):
        # Link lengths from URDF (measured from joint to joint)
        # L1: shoulder_lift to elbow_flex
        #     elbow_flex origin: (0, 0.11257, 0.028) from Upper_Arm
        # L2: elbow_flex to wrist_flex
        #     wrist_flex origin: (0, 0.0052, 0.1349) from Lower_Arm
        self.l1 = math.sqrt(0.11257**2 + 0.028**2)  # ~0.1159m
        self.l2 = math.sqrt(0.0052**2 + 0.1349**2)  # ~0.1350m

        # Geometry angles in arm's Y-Z plane (angle from +Y toward +Z)
        # These define the link directions at joint angle = 0
        self.alpha1 = math.atan2(0.028, 0.11257)    # ~14° (0.244 rad)
        self.alpha2 = math.atan2(0.1349, 0.0052)    # ~88° (1.536 rad)

        # Joint limits (radians)
        self.joint_limits = {
            'shoulder_pan': (-2.0, 2.0),
            'shoulder_lift': (-1.5708, 1.5708),  # ±90°
            'elbow_flex': (-1.5708, 1.5708),     # ±90°
            'wrist_flex': (-1.8, 1.8),
            'wrist_roll': (-3.14159, 3.14159),
            'gripper': (-1.1, 1.1),
        }

        # Compute wrist position at home (j2=0, j3=0) in arm's Y-Z plane
        theta1_home = self.alpha1  # ~14° from +Y
        theta2_home = self.alpha2  # ~88° from +Y
        self.wrist_y_home = self.l1 * math.cos(theta1_home) + self.l2 * math.cos(theta2_home)  # ~0.117m
        self.wrist_z_home = self.l1 * math.sin(theta1_home) + self.l2 * math.sin(theta2_home)  # ~0.163m

        # TCP position at home from simulation
        # World coordinates: TCP relative to base is (Y=-0.386, Z=0.267)
        # User coordinates: ee_x = -World_Y = 0.386, ee_y = World_Z = 0.267
        self.tcp_x_home = 0.386  # forward
        self.tcp_y_home = 0.267  # up

        # Constant offsets (accounts for base, shoulder, and gripper)
        # These transform between wrist position (from 2-link IK) and TCP position
        self.offset_x = self.tcp_x_home - self.wrist_z_home  # ~0.223m
        self.offset_y = self.tcp_y_home - self.wrist_y_home  # ~0.150m

        # Initial EE position in user coordinates
        self.initial_ee_x = self.tcp_x_home
        self.initial_ee_y = self.tcp_y_home

    def forward_kinematics(self, j2: float, j3: float) -> Tuple[float, float]:
        """
        Compute TCP position from joint angles using trigonometric FK.

        The arm operates in the Y-Z plane (arm frame) where:
        - Y axis = up direction (becomes World Z after 90° rotation)
        - Z axis = forward direction (becomes -World Y after 90° rotation)

        At joint angle = 0, each link points in the direction given by its
        geometry angle (alpha1, alpha2) from +Y toward +Z.

        Args:
            j2: shoulder_lift angle (radians)
            j3: elbow_flex angle (radians)

        Returns:
            (ee_x, ee_y): TCP position in user frame (forward, up)
        """
        # Absolute angles of links in arm's Y-Z plane (from +Y toward +Z)
        theta1 = j2 + self.alpha1  # Link 1 angle
        theta2 = j2 + j3 + self.alpha2  # Link 2 angle (j2 affects base, j3 is relative)

        # Wrist position in arm's Y-Z plane
        # Y component (up) uses cos, Z component (forward) uses sin
        wrist_y = self.l1 * math.cos(theta1) + self.l2 * math.cos(theta2)
        wrist_z = self.l1 * math.sin(theta1) + self.l2 * math.sin(theta2)

        # Transform to user coordinates and add constant offsets
        ee_x = wrist_z + self.offset_x  # forward
        ee_y = wrist_y + self.offset_y  # up

        return ee_x, ee_y

    def inverse_kinematics(self, x: float, y: float, pitch: float = 0.0) -> Tuple[float, float, float]:
        """
        Compute joint angles using XLeRobot-style analytical IK (law of cosines).

        This implements the IK formula:
        - θ₂ = π - arccos((l₁² + l₂² - r²) / (2l₁l₂))  [exterior elbow angle]
        - θ₁ = γ + ψ  where γ = atan2(wrist_y, wrist_z) and ψ from law of cosines

        The joint angles are computed by subtracting the geometry offsets.

        Args:
            x: Forward distance in user frame (positive = forward)
            y: Height in user frame (positive = up)
            pitch: Desired end-effector pitch angle (radians)

        Returns:
            (shoulder_lift, elbow_flex, wrist_flex) in radians
        """
        # Transform TCP position to wrist position in arm's Y-Z plane
        wrist_z = x - self.offset_x  # forward in arm frame
        wrist_y = y - self.offset_y  # up in arm frame

        # Distance from shoulder to wrist target
        r = math.sqrt(wrist_z**2 + wrist_y**2)

        # Workspace limits
        r_max = self.l1 + self.l2
        r_min = abs(self.l1 - self.l2)

        # Scale to workspace if out of bounds
        if r > r_max * 0.999:
            scale = r_max * 0.999 / r
            wrist_z *= scale
            wrist_y *= scale
            r = r_max * 0.999
        elif r > 0 and r < r_min * 1.001:
            scale = r_min * 1.001 / r
            wrist_z *= scale
            wrist_y *= scale
            r = r_min * 1.001

        # Angle to target from +Y axis (toward +Z)
        gamma = math.atan2(wrist_z, wrist_y)

        # Law of cosines for interior elbow angle (angle between link vectors)
        cos_phi = (self.l1**2 + self.l2**2 - r**2) / (2 * self.l1 * self.l2)
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        phi = math.acos(cos_phi)  # interior angle at elbow

        # Law of cosines for shoulder angle offset
        cos_psi = (r**2 + self.l1**2 - self.l2**2) / (2 * r * self.l1)
        cos_psi = np.clip(cos_psi, -1.0, 1.0)
        psi = math.acos(cos_psi)

        # theta1: absolute angle of link 1 from +Y toward +Z
        # For the "elbow-up" configuration (arm folded), use gamma - psi
        theta1 = gamma - psi

        # theta2: absolute angle of link 2 from +Y toward +Z
        # The exterior angle is (π - phi), and link 2 is at theta1 + (π - phi)
        theta2 = theta1 + (math.pi - phi)

        # Convert to joint angles by subtracting geometry offsets
        # From FK: theta1 = j2 + alpha1, theta2 = j2 + j3 + alpha2
        # So: j2 = theta1 - alpha1
        #     j3 = theta2 - j2 - alpha2 = theta2 - (theta1 - alpha1) - alpha2
        #        = theta2 - theta1 + alpha1 - alpha2
        j2 = theta1 - self.alpha1
        j3 = theta2 - theta1 + self.alpha1 - self.alpha2

        # Clamp to joint limits
        j2 = np.clip(j2, -1.5708, 1.5708)
        j3 = np.clip(j3, -1.5708, 1.5708)

        # Wrist flex for pitch compensation
        # Total arm pitch is approximately the angle of the EE from horizontal
        # wrist_flex adjusts to achieve desired pitch
        arm_pitch = j2 + j3  # simplified: sum of joint angles affects pitch
        wrist_flex = pitch - arm_pitch
        wrist_flex = np.clip(wrist_flex, -1.8, 1.8)

        return j2, j3, wrist_flex

    @property
    def workspace_limits(self) -> dict:
        """Get workspace limits."""
        return {
            "r_min": abs(self.l1 - self.l2),
            "r_max": self.l1 + self.l2,
            "l1": self.l1,
            "l2": self.l2,
        }


def test_kinematics_v2():
    """Test the V2 kinematics with comprehensive round-trip verification."""
    print("SO100 Kinematics V2 Test (XLeRobot-style Law of Cosines)")
    print("=" * 70)

    kin = SO100KinematicsV2()

    print(f"\nGeometry Parameters:")
    print(f"  Link lengths: l1={kin.l1:.4f}m, l2={kin.l2:.4f}m")
    print(f"  Geometry angles: α1={math.degrees(kin.alpha1):.2f}°, α2={math.degrees(kin.alpha2):.2f}°")
    print(f"  Workspace: r_min={kin.workspace_limits['r_min']:.4f}m, r_max={kin.workspace_limits['r_max']:.4f}m")
    print(f"\nOffset Parameters:")
    print(f"  Wrist at home: ({kin.wrist_z_home:.4f}, {kin.wrist_y_home:.4f}) [forward, up]")
    print(f"  TCP at home: ({kin.tcp_x_home:.4f}, {kin.tcp_y_home:.4f}) [forward, up]")
    print(f"  Offsets: ({kin.offset_x:.4f}, {kin.offset_y:.4f})")

    # Test FK at home position
    print("\n" + "-" * 70)
    print("FK at home (j2=0, j3=0):")
    x_home, y_home = kin.forward_kinematics(0, 0)
    print(f"  FK(0, 0) = ({x_home:.4f}, {y_home:.4f})")
    print(f"  Expected: ({kin.tcp_x_home:.4f}, {kin.tcp_y_home:.4f})")
    home_error = math.sqrt((x_home - kin.tcp_x_home)**2 + (y_home - kin.tcp_y_home)**2)
    print(f"  Error: {home_error*1000:.3f}mm {'✓' if home_error < 0.001 else '✗'}")

    # Test IK at home position
    print("\n" + "-" * 70)
    print("IK at home position (0.386, 0.267):")
    j2, j3, j4 = kin.inverse_kinematics(0.386, 0.267, 0.0)
    print(f"  IK(0.386, 0.267) = j2={math.degrees(j2):.2f}°, j3={math.degrees(j3):.2f}°, j4={math.degrees(j4):.2f}°")
    print(f"  Expected: j2≈0°, j3≈0°")
    joint_error = math.sqrt(j2**2 + j3**2)
    print(f"  Joint error: {math.degrees(joint_error):.2f}° {'✓' if joint_error < 0.05 else '✗'}")

    # FK/IK round-trip tests
    print("\n" + "-" * 70)
    print("FK/IK Round-trip tests:")

    # Test from various joint angles
    test_angles = [
        (0, 0),
        (0.2, 0),
        (0, 0.2),
        (-0.2, 0),
        (0, -0.2),
        (0.3, -0.2),
        (-0.2, 0.3),
        (0.4, -0.3),
        (-0.3, 0.4),
        (0.5, 0.5),
    ]

    print("\n  Joint → FK → IK → Joint:")
    all_pass = True
    for j2_in, j3_in in test_angles:
        x, y = kin.forward_kinematics(j2_in, j3_in)
        j2_out, j3_out, _ = kin.inverse_kinematics(x, y, 0.0)

        # Verify by FK again
        x_check, y_check = kin.forward_kinematics(j2_out, j3_out)
        pos_error = math.sqrt((x - x_check)**2 + (y - y_check)**2)

        status = "✓" if pos_error < 0.001 else "✗"
        if pos_error >= 0.001:
            all_pass = False

        print(f"    ({j2_in:+.2f}, {j3_in:+.2f}) → ({x:.3f}, {y:.3f}) → "
              f"({j2_out:+.2f}, {j3_out:+.2f}) → ({x_check:.3f}, {y_check:.3f}) "
              f"err={pos_error*1000:.2f}mm {status}")

    # Test from various positions
    print("\n  Position → IK → FK → Position:")
    test_positions = [
        (0.386, 0.267),  # home
        (0.35, 0.30),
        (0.40, 0.25),
        (0.38, 0.20),
        (0.42, 0.28),
        (0.33, 0.32),
        (0.45, 0.22),
        (0.30, 0.35),
    ]

    for x_in, y_in in test_positions:
        j2, j3, _ = kin.inverse_kinematics(x_in, y_in, 0.0)
        x_out, y_out = kin.forward_kinematics(j2, j3)

        pos_error = math.sqrt((x_in - x_out)**2 + (y_in - y_out)**2)
        status = "✓" if pos_error < 0.001 else "✗"
        if pos_error >= 0.001:
            all_pass = False

        print(f"    ({x_in:.3f}, {y_in:.3f}) → ({j2:+.2f}, {j3:+.2f}) → "
              f"({x_out:.3f}, {y_out:.3f}) err={pos_error*1000:.2f}mm {status}")

    print("\n" + "=" * 70)
    print(f"Overall: {'All tests passed! ✓' if all_pass else 'Some tests failed ✗'}")


if __name__ == "__main__":
    test_kinematics_v2()
