"""
AlohaMini Robot Agent with ManiSkill SO100 Arms V2

This variant uses the new maniskill_so100_version.urdf which is based on
the official ManiSkill SO100 arm structure (no rotation in base_joint,
rotation in shoulder_pan).

SO100 arms have proper grippers (Fixed_Jaw + Moving_Jaw) that can actually grasp objects.
"""

from copy import deepcopy
import math

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link

from pathlib import Path

from .base_agent import AlohaMiniBaseAgent, euler_to_quat_xyz


@register_agent()
class AlohaMiniSO100V2(AlohaMiniBaseAgent):
    """
    AlohaMini with official ManiSkill SO100 arms (V2).

    This robot uses virtual base joints and SO100 arm structure:
    - root_x_axis_joint: prismatic joint for X movement
    - root_y_axis_joint: prismatic joint for Y movement
    - root_z_rotation_joint: continuous joint for rotation
    - vertical_move: prismatic joint for lift
    - Left/Right arm: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

    Joint order: [base_x, base_y, base_rot, lift, left_arm(6), right_arm(6)] = 16 DOF

    This version uses maniskill_so100_version.urdf which has:
    - base_joint with NO rotation (rpy="0 0 0")
    - shoulder_pan with X rotation (rpy="1.5708 0 0") from official SO100
    """

    uid = "aloha_mini_so100_v2"
    urdf_path = str(Path(__file__).parent.parent.parent / "assets/robots/aloha_mini/maniskill_so100_version.urdf")

    urdf_config = dict(
        _materials=dict(
            gripper=dict(
                static_friction=2.0,
                dynamic_friction=2.0,
                restitution=0.0,
            ),
        ),
        link=dict(
            left_Fixed_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            left_Moving_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            right_Fixed_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            right_Moving_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    # Actual joint order from ManiSkill (URDF order, NOT interleaved):
    # 0: root_x_axis_joint, 1: root_y_axis_joint, 2: root_z_rotation_joint
    # 3: vertical_move
    # 4-9: left arm (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
    # 10-15: right arm (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                # Base (4): x, y, rotation, lift
                0.0, 0.0, 0.0, 0.0,
                # Left arm (6): all zeros like aloha_mini_virtual
                0.0, 0.0, 0.0, 0.0, 0.0, -1.1,  # gripper closed
                # Right arm (6)
                0.0, 0.0, 0.0, 0.0, 0.0, -1.1,  # gripper closed
            ]),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
        ready=Keyframe(
            qpos=np.array([
                # Base (4)
                0.0, 0.0, 0.0, 0.05,
                # Left arm (6) - slight pose with gripper open
                0.0, 0.3, -0.3, 0.0, 0.3, 0.0,
                # Right arm (6)
                0.0, 0.3, -0.3, 0.0, 0.3, 0.0,
            ]),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
        zero=Keyframe(
            qpos=np.array([0.0] * 16),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
    )

    @property
    def _sensor_configs(self):
        # Main camera: overhead view
        q_main = euler_to_quat_xyz(
            math.radians(90),
            math.radians(90),
            math.radians(0)
        )

        # Wrist cameras (on Fixed_Jaw, looking down at gripper)
        q_wrist = euler_to_quat_xyz(
            math.radians(0),
            math.radians(45),
            math.radians(0)
        )

        return [
            CameraConfig(
                uid="cam_main",
                pose=Pose.create_from_pq(
                    p=[0.0, 0.0, 1.45],
                    q=q_main,
                ),
                width=320,
                height=240,
                fov=1.5,
                near=0.01,
                far=100,
                entity_uid="base_link",
            ),
            CameraConfig(
                uid="cam_left_wrist",
                pose=Pose.create_from_pq(
                    p=[0.0, 0.0, 0.06],
                    q=q_wrist,
                ),
                width=128,
                height=128,
                fov=1.5,
                near=0.01,
                far=100,
                entity_uid="left_Fixed_Jaw",
            ),
            CameraConfig(
                uid="cam_right_wrist",
                pose=Pose.create_from_pq(
                    p=[0.0, 0.0, 0.06],
                    q=q_wrist,
                ),
                width=128,
                height=128,
                fov=1.5,
                near=0.01,
                far=100,
                entity_uid="right_Fixed_Jaw",
            ),
        ]

    def __init__(self, *args, **kwargs):
        # SO100 arm joints (XLeRobot-compatible naming)
        self.left_arm_joint_names = [
            "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
            "left_wrist_flex", "left_wrist_roll", "left_gripper"
        ]
        self.right_arm_joint_names = [
            "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
            "right_wrist_flex", "right_wrist_roll", "right_gripper"
        ]

        self.arm_joint_names = self.left_arm_joint_names + self.right_arm_joint_names

        # Controller parameters for arm joints (5 joints: shoulder_pan to wrist_roll)
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        # Controller parameters for gripper (stronger force for better grasping)
        self.gripper_stiffness = 100
        self.gripper_damping = 1e2
        self.gripper_force_limit = 5.0

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # Base controller (from base class)
        base_pd_joint_vel = self._create_base_controller()

        # Lift controllers (from base class)
        lift_pos = self._create_lift_pos_controller()
        lift_delta_pos = self._create_lift_delta_pos_controller()

        # Per-joint parameters: 5 arm joints + 1 gripper joint
        arm_stiffness_list = [self.arm_stiffness] * 5 + [self.gripper_stiffness]
        arm_damping_list = [self.arm_damping] * 5 + [self.gripper_damping]
        arm_force_limit_list = [self.arm_force_limit] * 5 + [self.gripper_force_limit]

        # Left arm controllers
        left_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=None,
            upper=None,
            stiffness=arm_stiffness_list,
            damping=arm_damping_list,
            force_limit=arm_force_limit_list,
            normalize_action=False,
        )

        left_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=[-0.05, -0.05, -0.05, -0.05, -0.05, -0.2],
            upper=[0.05, 0.05, 0.05, 0.05, 0.05, 0.2],
            stiffness=arm_stiffness_list,
            damping=arm_damping_list,
            force_limit=arm_force_limit_list,
            use_delta=True,
        )

        # Right arm controllers
        right_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.right_arm_joint_names,
            lower=None,
            upper=None,
            stiffness=arm_stiffness_list,
            damping=arm_damping_list,
            force_limit=arm_force_limit_list,
            normalize_action=False,
        )

        right_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.right_arm_joint_names,
            lower=[-0.05, -0.05, -0.05, -0.05, -0.05, -0.2],
            upper=[0.05, 0.05, 0.05, 0.05, 0.05, 0.2],
            stiffness=arm_stiffness_list,
            damping=arm_damping_list,
            force_limit=arm_force_limit_list,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_pos=dict(
                base=base_pd_joint_vel,
                lift=lift_pos,
                left_arm=left_arm_pd_joint_pos,
                right_arm=right_arm_pd_joint_pos,
            ),
            pd_joint_delta_pos=dict(
                base=base_pd_joint_vel,
                lift=lift_delta_pos,
                left_arm=left_arm_pd_joint_delta_pos,
                right_arm=right_arm_pd_joint_delta_pos,
            ),
        )

        return deepcopy(controller_configs)

    def _after_init(self):
        # Initialize base links and collision settings
        self._after_init_base()

        # Left arm gripper links
        self.left_finger1_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_Fixed_Jaw"
        )
        self.left_finger2_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_Moving_Jaw"
        )
        self.left_finger1_tip: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_Fixed_Jaw_tip"
        )
        self.left_finger2_tip: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_Moving_Jaw_tip"
        )

        # Right arm gripper links
        self.right_finger1_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_Fixed_Jaw"
        )
        self.right_finger2_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_Moving_Jaw"
        )
        self.right_finger1_tip: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_Fixed_Jaw_tip"
        )
        self.right_finger2_tip: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_Moving_Jaw_tip"
        )

    @property
    def tcp_pos(self):
        """Left arm TCP position (midpoint between finger tips)."""
        return (self.left_finger1_tip.pose.p + self.left_finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        """Left arm TCP pose."""
        return Pose.create_from_pq(self.tcp_pos, self.left_finger1_link.pose.q)

    @property
    def tcp_pos_2(self):
        """Right arm TCP position."""
        return (self.right_finger1_tip.pose.p + self.right_finger2_tip.pose.p) / 2

    @property
    def tcp_pose_2(self):
        """Right arm TCP pose."""
        return Pose.create_from_pq(self.tcp_pos_2, self.right_finger1_link.pose.q)

    def get_left_ee_pose(self):
        return self.tcp_pose

    def get_right_ee_pose(self):
        return self.tcp_pose_2

    def _check_single_arm_grasping(self, object: Actor, min_force=0.5, max_angle=110, arm_id=1):
        """Check if a single arm is grasping (SO100 style with two fingers)."""
        if arm_id == 1:
            finger1_link = self.left_finger1_link
            finger2_link = self.left_finger2_link
        elif arm_id == 2:
            finger1_link = self.right_finger1_link
            finger2_link = self.right_finger2_link
        else:
            raise ValueError(f"Invalid arm_id: {arm_id}. Must be 1 or 2.")

        l_contact_forces = self.scene.get_pairwise_contact_forces(
            finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)
