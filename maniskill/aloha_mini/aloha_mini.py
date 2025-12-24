"""
AlohaMini Robot Agent for ManiSkill3

This module defines the AlohaMini dual-arm mobile robot for use in ManiSkill3 environments.
The robot features:
- A mobile base with 3 omnidirectional wheels
- A vertical lift mechanism
- Two 6-DOF manipulator arms (left and right)
"""

from copy import deepcopy
from pathlib import Path


def deepcopy_dict(d):
    """Deep copy a dictionary containing controller configs."""
    return deepcopy(d)

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent


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
    urdf_path = str(Path(__file__).parent / "aloha_mini.urdf")

    # Physical configuration
    urdf_config = dict(
        _materials=dict(
            gripper=dict(
                static_friction=2.0,
                dynamic_friction=2.0,
                restitution=0.0,
            )
        ),
        link=dict(
            left_link6=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.05),
            right_link6=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.05),
        ),
    )

    # Joint names for each component
    wheel_joint_names = ["wheel1_joint", "wheel2_joint", "wheel3_joint"]
    lift_joint_names = ["vertical_move"]
    left_arm_joint_names = [
        "left_joint1", "left_joint2", "left_joint3",
        "left_joint4", "left_joint5", "left_joint6"
    ]
    right_arm_joint_names = [
        "right_joint1", "right_joint2", "right_joint3",
        "right_joint4", "right_joint5", "right_joint6"
    ]

    # End-effector link names
    left_ee_link_name = "left_ee_link"
    right_ee_link_name = "right_ee_link"

    # All arm joint names combined
    arm_joint_names = left_arm_joint_names + right_arm_joint_names

    # Controller parameters
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 50

    lift_stiffness = 1e3
    lift_damping = 1e2
    lift_force_limit = 100

    wheel_damping = 1e3
    wheel_force_limit = 100

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
    def _controller_configs(self):
        """
        Define controller configurations for different control modes.
        """
        # -------------------------------------------------------------------------- #
        # Wheel controllers
        # -------------------------------------------------------------------------- #
        wheel_vel = PDJointVelControllerConfig(
            self.wheel_joint_names,
            lower=-1.0,
            upper=1.0,
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

        return deepcopy_dict(controller_configs)

    def _after_init(self):
        """Called after robot initialization."""
        pass

    def get_left_ee_pose(self):
        """Get the pose of the left end-effector."""
        return self.robot.links_map[self.left_ee_link_name].pose

    def get_right_ee_pose(self):
        """Get the pose of the right end-effector."""
        return self.robot.links_map[self.right_ee_link_name].pose

    def get_ee_poses(self):
        """Get poses of both end-effectors."""
        return self.get_left_ee_pose(), self.get_right_ee_pose()


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

        return deepcopy_dict(controller_configs)


@register_agent()
class AlohaMiniArmsOnly(BaseAgent):
    """
    AlohaMini variant with only dual arms (no base, no wheels, no lift).

    This is useful for tasks where you want to focus on arm manipulation
    without the mobile base complexity.
    """

    uid = "aloha_mini_arms_only"
    urdf_path = str(Path(__file__).parent / "aloha_mini.urdf")

    disable_self_collisions = True

    urdf_config = dict(
        _materials=dict(
            gripper=dict(
                static_friction=2.0,
                dynamic_friction=2.0,
                restitution=0.0,
            )
        ),
        link=dict(
            left_link6=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.05),
            right_link6=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.05),
        ),
    )

    # Joint names
    base_joint_names = ["wheel1_joint", "wheel2_joint", "wheel3_joint", "vertical_move"]
    left_arm_joint_names = [
        "left_joint1", "left_joint2", "left_joint3",
        "left_joint4", "left_joint5", "left_joint6"
    ]
    right_arm_joint_names = [
        "right_joint1", "right_joint2", "right_joint3",
        "right_joint4", "right_joint5", "right_joint6"
    ]
    arm_joint_names = left_arm_joint_names + right_arm_joint_names

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 50

    keyframes = dict(
        rest=Keyframe(
            qpos=np.zeros(16),
            pose=sapien.Pose(p=[0, 0, 0.5]),
        ),
    )

    @property
    def _controller_configs(self):
        base_passive = PassiveControllerConfig(
            self.base_joint_names,
            damping=1e6,
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

        right_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.right_arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )

        controller_configs = dict(
            pd_joint_pos=dict(
                base=base_passive,
                left_arm=left_arm_pd_joint_pos,
                right_arm=right_arm_pd_joint_pos,
            ),
        )

        return deepcopy_dict(controller_configs)
