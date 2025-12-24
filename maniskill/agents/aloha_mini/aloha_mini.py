"""
AlohaMini Robot Agent for ManiSkill3

This module defines the AlohaMini dual-arm mobile robot for use in ManiSkill3 environments.
The robot features:
- A mobile base with 3 omnidirectional wheels
- A vertical lift mechanism
- Two 6-DOF manipulator arms (left and right)
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
"""Collision bit of the AlohaMini robot wheel links"""
ALOHA_MINI_BASE_COLLISION_BIT = 29
"""Collision bit of the AlohaMini base"""


@register_agent()
class AlohaMini(BaseAgent):
    """
    AlohaMini dual-arm mobile robot agent for ManiSkill3.

    This robot has:
    - 3 wheel joints (continuous) for mobile base
    - 1 prismatic joint for vertical lift
    - 6 revolute joints for left arm
    - 6 revolute joints for right arm

    Total: 16 actuated DOFs (13 if excluding wheels)
    """

    uid = "aloha_mini"
    urdf_path = f"{ASSET_DIR}/robots/aloha_mini/aloha_mini.urdf"

    # Physical configuration
    urdf_config = dict(
        _materials=dict(
            gripper=dict(
                static_friction=2.0,
                dynamic_friction=2.0,
                restitution=0.0,
            ),
            wheel=dict(
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.0,
            ),
        ),
        link=dict(
            left_link6=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.05),
            right_link6=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.05),
            wheel1=dict(material="wheel", patch_radius=0.04, min_patch_radius=0.02),
            wheel2=dict(material="wheel", patch_radius=0.04, min_patch_radius=0.02),
            wheel3=dict(material="wheel", patch_radius=0.04, min_patch_radius=0.02),
        ),
    )

    # Keyframes define preset robot configurations
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                # Wheels (3)
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
                # Wheels (3)
                0.0, 0.0, 0.0,
                # Lift (1)
                0.05,
                # Left arm (6) - slightly bent for manipulation
                0.0, 0.3, -0.3, 0.0, 0.3, 0.0,
                # Right arm (6) - slightly bent for manipulation
                0.0, 0.3, -0.3, 0.0, 0.3, 0.0,
            ]),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
        arms_up=Keyframe(
            qpos=np.array([
                # Wheels (3)
                0.0, 0.0, 0.0,
                # Lift (1)
                0.1,
                # Left arm (6) - arms up
                0.0, 0.8, -0.4, 0.0, 0.5, 0.0,
                # Right arm (6) - arms up
                0.0, 0.8, -0.4, 0.0, 0.5, 0.0,
            ]),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
    )

    @property
    def _sensor_configs(self):
        """
        Configure 8 camera views for AlohaMini robot.

        Target view: Behind and above robot, looking down at workspace with both arms visible.
        Reference image shows camera positioned behind robot (-Y), elevated, looking forward and down.
        """
        import math
        from scipy.spatial.transform import Rotation as R

        def euler_to_quat(roll, pitch, yaw):
            """Convert euler angles (XYZ order) to SAPIEN quaternion [w, x, y, z]."""
            r = R.from_euler('xyz', [roll, pitch, yaw])
            q = r.as_quat()  # [x, y, z, w]
            return [q[3], q[0], q[1], q[2]]  # SAPIEN format [w, x, y, z]

        # Target: Camera behind robot, looking forward and down at workspace
        # Robot +Y is forward, so camera should be at -Y position, looking toward +Y and down

        # Camera 1: Behind robot, looking forward-down (pitch ~60 deg from horizontal)
        # Roll=0, Pitch=60deg down from forward, Yaw=0
        q1 = euler_to_quat(math.radians(120), 0, 0)  # 90+30 = looking 30deg below horizontal

        # Camera 2: Behind robot, looking forward-down (pitch ~45 deg)
        q2 = euler_to_quat(math.radians(135), 0, 0)  # 90+45 = looking 45deg below horizontal

        # Camera 3: Behind robot, looking forward-down (pitch ~30 deg)
        q3 = euler_to_quat(math.radians(150), 0, 0)  # 90+60 = looking 60deg below horizontal

        # Camera 4: More overhead view
        q4 = euler_to_quat(math.radians(160), 0, 0)

        # Camera 5-8: Same angles but with different Y axis rotation to adjust orientation
        q5 = euler_to_quat(math.radians(135), 0, math.radians(180))  # flipped
        q6 = euler_to_quat(math.radians(120), 0, math.radians(180))
        q7 = euler_to_quat(math.radians(150), 0, math.radians(180))
        q8 = euler_to_quat(math.radians(140), 0, math.radians(180))

        return [
            CameraConfig(
                uid="cam1_back30",
                pose=Pose.create_from_pq(
                    p=[0.0, -0.4, 0.5],  # Behind and above
                    q=q1,
                ),
                width=320,
                height=240,
                fov=1.4,
                near=0.01,
                far=100,
                entity_uid="vertical_link",
            ),
            CameraConfig(
                uid="cam2_back45",
                pose=Pose.create_from_pq(
                    p=[0.0, -0.35, 0.45],
                    q=q2,
                ),
                width=320,
                height=240,
                fov=1.4,
                near=0.01,
                far=100,
                entity_uid="vertical_link",
            ),
            CameraConfig(
                uid="cam3_back60",
                pose=Pose.create_from_pq(
                    p=[0.0, -0.3, 0.4],
                    q=q3,
                ),
                width=320,
                height=240,
                fov=1.4,
                near=0.01,
                far=100,
                entity_uid="vertical_link",
            ),
            CameraConfig(
                uid="cam4_overhead",
                pose=Pose.create_from_pq(
                    p=[0.0, -0.25, 0.5],
                    q=q4,
                ),
                width=320,
                height=240,
                fov=1.5,
                near=0.01,
                far=100,
                entity_uid="vertical_link",
            ),
            CameraConfig(
                uid="cam5_flip45",
                pose=Pose.create_from_pq(
                    p=[0.0, -0.35, 0.45],
                    q=q5,
                ),
                width=320,
                height=240,
                fov=1.4,
                near=0.01,
                far=100,
                entity_uid="vertical_link",
            ),
            CameraConfig(
                uid="cam6_flip30",
                pose=Pose.create_from_pq(
                    p=[0.0, -0.4, 0.5],
                    q=q6,
                ),
                width=320,
                height=240,
                fov=1.4,
                near=0.01,
                far=100,
                entity_uid="vertical_link",
            ),
            CameraConfig(
                uid="cam7_flip60",
                pose=Pose.create_from_pq(
                    p=[0.0, -0.3, 0.4],
                    q=q7,
                ),
                width=320,
                height=240,
                fov=1.4,
                near=0.01,
                far=100,
                entity_uid="vertical_link",
            ),
            CameraConfig(
                uid="cam8_flip50",
                pose=Pose.create_from_pq(
                    p=[0.0, -0.32, 0.42],
                    q=q8,
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
        # Joint names for each component
        self.wheel_joint_names = ["wheel1_joint", "wheel2_joint", "wheel3_joint"]
        self.lift_joint_names = ["vertical_move"]
        self.left_arm_joint_names = [
            "left_joint1", "left_joint2", "left_joint3",
            "left_joint4", "left_joint5", "left_joint6"
        ]
        self.right_arm_joint_names = [
            "right_joint1", "right_joint2", "right_joint3",
            "right_joint4", "right_joint5", "right_joint6"
        ]

        # End-effector link names
        self.left_ee_link_name = "left_ee_link"
        self.right_ee_link_name = "right_ee_link"

        # All arm joint names combined
        self.arm_joint_names = self.left_arm_joint_names + self.right_arm_joint_names

        # Controller parameters
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 50

        self.lift_stiffness = 1e3
        self.lift_damping = 1e2
        self.lift_force_limit = 100

        self.wheel_damping = 50
        self.wheel_force_limit = 500

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        """
        Define controller configurations for different control modes.
        """
        # -------------------------------------------------------------------------- #
        # Wheel controllers
        # -------------------------------------------------------------------------- #
        wheel_vel = PDJointVelControllerConfig(
            self.wheel_joint_names,
            lower=-10.0,
            upper=10.0,
            damping=self.wheel_damping,
            force_limit=self.wheel_force_limit,
        )

        wheel_passive = PassiveControllerConfig(
            self.wheel_joint_names,
            damping=100,
        )

        # -------------------------------------------------------------------------- #
        # Lift controller
        # -------------------------------------------------------------------------- #
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

        # -------------------------------------------------------------------------- #
        # Arm controllers
        # -------------------------------------------------------------------------- #
        # Left arm position control
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

        # Right arm position control
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

        # Arm velocity control
        left_arm_pd_joint_vel = PDJointVelControllerConfig(
            self.left_arm_joint_names,
            lower=-1.0,
            upper=1.0,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
        )

        right_arm_pd_joint_vel = PDJointVelControllerConfig(
            self.right_arm_joint_names,
            lower=-1.0,
            upper=1.0,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
        )

        # -------------------------------------------------------------------------- #
        # Controller configurations
        # -------------------------------------------------------------------------- #
        controller_configs = dict(
            # Fixed base mode (for manipulation tasks in ReplicaCAD)
            pd_joint_pos=dict(
                wheels=wheel_passive,
                lift=lift_pos,
                left_arm=left_arm_pd_joint_pos,
                right_arm=right_arm_pd_joint_pos,
            ),
            # Fixed base with delta position control
            pd_joint_delta_pos=dict(
                wheels=wheel_passive,
                lift=lift_delta_pos,
                left_arm=left_arm_pd_joint_delta_pos,
                right_arm=right_arm_pd_joint_delta_pos,
            ),
            # Mobile mode (for navigation + manipulation)
            mobile_pd_joint_pos=dict(
                wheels=wheel_vel,
                lift=lift_pos,
                left_arm=left_arm_pd_joint_pos,
                right_arm=right_arm_pd_joint_pos,
            ),
            # Velocity control mode
            pd_joint_vel=dict(
                wheels=wheel_vel,
                lift=lift_pos,
                left_arm=left_arm_pd_joint_vel,
                right_arm=right_arm_pd_joint_vel,
            ),
        )

        return deepcopy(controller_configs)

    def _after_init(self):
        """Called after robot initialization."""
        # Get links for manipulation
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

        # Set collision groups
        for link in [self.base_link]:
            link.set_collision_group_bit(
                group=2, bit_idx=ALOHA_MINI_BASE_COLLISION_BIT, bit=1
            )

        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    @property
    def tcp_pos(self):
        """Get left end-effector position (tool center point)."""
        return self.left_ee_link.pose.p

    @property
    def tcp_pose(self):
        """Get left end-effector pose."""
        return self.left_ee_link.pose

    @property
    def tcp_pos_2(self):
        """Get right end-effector position."""
        return self.right_ee_link.pose.p

    @property
    def tcp_pose_2(self):
        """Get right end-effector pose."""
        return self.right_ee_link.pose

    def get_left_ee_pose(self):
        """Get the pose of the left end-effector."""
        return self.left_ee_link.pose

    def get_right_ee_pose(self):
        """Get the pose of the right end-effector."""
        return self.right_ee_link.pose

    def get_ee_poses(self):
        """Get poses of both end-effectors."""
        return self.get_left_ee_pose(), self.get_right_ee_pose()

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110, arm_id=None):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered
                to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping.
                Defaults to 110.
            arm_id (int, optional): Which arm to check (1 for left, 2 for right).
                If None, check both arms and return True if either is grasping.
        """
        if arm_id is None:
            arm1_grasping = self._check_single_arm_grasping(object, min_force, max_angle, arm_id=1)
            arm2_grasping = self._check_single_arm_grasping(object, min_force, max_angle, arm_id=2)
            return torch.logical_or(arm1_grasping, arm2_grasping)
        else:
            return self._check_single_arm_grasping(object, min_force, max_angle, arm_id)

    def _check_single_arm_grasping(self, object: Actor, min_force=0.5, max_angle=110, arm_id=1):
        """Internal method to check grasping for a specific arm"""
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
        ]  # exclude the wheel joints
        return torch.max(torch.abs(qvel), 1)[0] <= threshold


@register_agent()
class AlohaMiniFixed(AlohaMini):
    """
    AlohaMini with fixed base (no wheel movement).

    Use this variant for manipulation tasks in ReplicaCAD where
    the robot base should remain stationary.
    """

    uid = "aloha_mini_fixed"

    @property
    def _controller_configs(self):
        """Only expose fixed-base controllers."""
        wheel_passive = PassiveControllerConfig(
            self.wheel_joint_names,
            damping=100,
        )

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
            pd_joint_pos=dict(
                wheels=wheel_passive,
                lift=lift_pos,
                left_arm=left_arm_pd_joint_pos,
                right_arm=right_arm_pd_joint_pos,
            ),
            pd_joint_delta_pos=dict(
                wheels=wheel_passive,
                lift=lift_delta_pos,
                left_arm=left_arm_pd_joint_delta_pos,
                right_arm=right_arm_pd_joint_delta_pos,
            ),
        )

        return deepcopy(controller_configs)
