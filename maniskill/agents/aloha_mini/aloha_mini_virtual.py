"""
AlohaMini Robot Agent with Virtual Mobile Base for ManiSkill3

This variant uses prismatic X/Y joints and rotation joint for the base,
similar to XLeRobot, instead of actual wheel physics.
"""

from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link


ALOHA_MINI_WHEELS_COLLISION_BIT = 28
ALOHA_MINI_BASE_COLLISION_BIT = 29


@register_agent()
class AlohaMiniVirtual(BaseAgent):
    """
    AlohaMini with virtual mobile base (prismatic X/Y + rotation joints).

    This robot uses virtual base joints instead of physical wheels:
    - root_x_axis_joint: prismatic joint for X movement
    - root_y_axis_joint: prismatic joint for Y movement
    - root_z_rotation_joint: continuous joint for rotation

    Joint order: [base_x, base_y, base_rot, lift, left_arm(6), right_arm(6)] = 16 DOF
    """

    uid = "aloha_mini_virtual"
    urdf_path = f"{ASSET_DIR}/robots/aloha_mini/aloha_mini_virtual_base.urdf"

    urdf_config = dict(
        _materials=dict(
            gripper=dict(
                static_friction=2.0,
                dynamic_friction=2.0,
                restitution=0.0,
            ),
        ),
        link=dict(
            left_link6=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.05),
            right_link6=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.05),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                # Base (3): x, y, rotation
                0.0, 0.0, 0.0,
                # Lift (1)
                0.0,
                # Left arm (6)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # Right arm (6)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
        ready=Keyframe(
            qpos=np.array([
                # Base (3)
                0.0, 0.0, 0.0,
                # Lift (1)
                0.05,
                # Left arm (6)
                0.0, 0.3, -0.3, 0.0, 0.3, 0.0,
                # Right arm (6)
                0.0, 0.3, -0.3, 0.0, 0.3, 0.0,
            ]),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
    )

    @property
    def _sensor_configs(self):
        import math
        from scipy.spatial.transform import Rotation as R

        def euler_to_quat(roll, pitch, yaw):
            r = R.from_euler('xyz', [roll, pitch, yaw])
            q = r.as_quat()
            return [q[3], q[0], q[1], q[2]]

        q1 = euler_to_quat(math.radians(135), 0, 0)

        return [
            CameraConfig(
                uid="cam_main",
                pose=Pose.create_from_pq(
                    p=[0.0, -0.35, 0.45],
                    q=q1,
                ),
                width=320,
                height=240,
                fov=1.4,
                near=0.01,
                far=100,
                entity_uid="vertical_link",
            ),
        ]

    def __init__(self, *args, **kwargs):
        # Base joint names (virtual mobile base)
        self.base_joint_names = [
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_rotation_joint",
        ]

        # Lift joint
        self.lift_joint_names = ["vertical_move"]

        # Arm joints
        self.left_arm_joint_names = [
            "left_joint1", "left_joint2", "left_joint3",
            "left_joint4", "left_joint5", "left_joint6"
        ]
        self.right_arm_joint_names = [
            "right_joint1", "right_joint2", "right_joint3",
            "right_joint4", "right_joint5", "right_joint6"
        ]

        self.left_ee_link_name = "left_ee_link"
        self.right_ee_link_name = "right_ee_link"

        self.arm_joint_names = self.left_arm_joint_names + self.right_arm_joint_names

        # Controller parameters
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 50

        self.lift_stiffness = 1e3
        self.lift_damping = 1e2
        self.lift_force_limit = 100

        self.base_damping = 1000
        self.base_force_limit = 500

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # Base controller (virtual mobile base using PDBaseVelController)
        base_pd_joint_vel = PDBaseVelControllerConfig(
            self.base_joint_names,
            lower=[-1, -1, -3.14],
            upper=[1, 1, 3.14],
            damping=self.base_damping,
            force_limit=self.base_force_limit,
        )

        # Lift controller
        lift_pos = PDJointPosControllerConfig(
            self.lift_joint_names,
            lower=None,
            upper=None,
            stiffness=self.lift_stiffness,
            damping=self.lift_damping,
            force_limit=self.lift_force_limit,
            normalize_action=False,
        )

        lift_delta_pos = PDJointPosControllerConfig(
            self.lift_joint_names,
            lower=-0.05,
            upper=0.05,
            stiffness=self.lift_stiffness,
            damping=self.lift_damping,
            force_limit=self.lift_force_limit,
            use_delta=True,
        )

        # Left arm controllers
        left_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )

        left_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        # Right arm controllers
        right_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.right_arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )

        right_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.right_arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        controller_configs = dict(
            # Mobile mode with position control for arms
            pd_joint_pos=dict(
                base=base_pd_joint_vel,
                lift=lift_pos,
                left_arm=left_arm_pd_joint_pos,
                right_arm=right_arm_pd_joint_pos,
            ),
            # Mobile mode with delta position control
            pd_joint_delta_pos=dict(
                base=base_pd_joint_vel,
                lift=lift_delta_pos,
                left_arm=left_arm_pd_joint_delta_pos,
                right_arm=right_arm_pd_joint_delta_pos,
            ),
        )

        return deepcopy(controller_configs)

    def _after_init(self):
        self.left_ee_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.left_ee_link_name
        )
        self.right_ee_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.right_ee_link_name
        )
        self.left_link6: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_link6"
        )
        self.right_link6: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_link6"
        )

        self.base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "base_link"
        )
        self.vertical_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "vertical_link"
        )

        for link in [self.base_link]:
            link.set_collision_group_bit(
                group=2, bit_idx=ALOHA_MINI_BASE_COLLISION_BIT, bit=1
            )

        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    @property
    def tcp_pos(self):
        return self.left_ee_link.pose.p

    @property
    def tcp_pose(self):
        return self.left_ee_link.pose

    @property
    def tcp_pos_2(self):
        return self.right_ee_link.pose.p

    @property
    def tcp_pose_2(self):
        return self.right_ee_link.pose

    def get_left_ee_pose(self):
        return self.left_ee_link.pose

    def get_right_ee_pose(self):
        return self.right_ee_link.pose

    def get_ee_poses(self):
        return self.get_left_ee_pose(), self.get_right_ee_pose()

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110, arm_id=None):
        if arm_id is None:
            arm1_grasping = self._check_single_arm_grasping(object, min_force, max_angle, arm_id=1)
            arm2_grasping = self._check_single_arm_grasping(object, min_force, max_angle, arm_id=2)
            return torch.logical_or(arm1_grasping, arm2_grasping)
        else:
            return self._check_single_arm_grasping(object, min_force, max_angle, arm_id)

    def _check_single_arm_grasping(self, object: Actor, min_force=0.5, max_angle=110, arm_id=1):
        if arm_id == 1:
            finger_link = self.left_link6
        elif arm_id == 2:
            finger_link = self.right_link6
        else:
            raise ValueError(f"Invalid arm_id: {arm_id}. Must be 1 or 2.")

        contact_forces = self.scene.get_pairwise_contact_forces(
            finger_link, object
        )
        force = torch.linalg.norm(contact_forces, axis=1)

        direction = finger_link.pose.to_transformation_matrix()[..., :3, 2]
        angle = common.compute_angle_between(direction, contact_forces)
        flag = torch.logical_and(
            force >= min_force, torch.rad2deg(angle) <= max_angle
        )
        return flag

    def is_static(self, threshold=0.2):
        qvel = self.robot.get_qvel()[
            :, 3:
        ]  # exclude the base joints
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
