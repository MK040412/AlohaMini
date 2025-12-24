"""
Custom ManiSkill3 environments with AlohaMini support.

These environments extend the standard ManiSkill scene manipulation environments
to support the AlohaMini dual-arm robot.
"""

from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.registration import REGISTERED_SCENE_BUILDERS
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder

from .aloha_mini import AlohaMini, AlohaMiniFixed, AlohaMiniArmsOnly


class AlohaMiniReplicaCADSceneBuilder(ReplicaCADSceneBuilder):
    """
    Extended ReplicaCAD scene builder that supports AlohaMini robot.
    """

    def initialize(self, env_idx: torch.Tensor):
        """Initialize scene with AlohaMini robot support."""
        # teleport robot away for init
        self.env.agent.robot.set_pose(sapien.Pose([-10, 0, -100]))

        for obj, pose in self._default_object_poses:
            obj.set_pose(pose)
            from mani_skill.utils.structs.articulation import Articulation
            if isinstance(obj, Articulation):
                obj.set_qpos(obj.qpos[0] * 0)
                obj.set_qvel(obj.qvel[0] * 0)

        if self.scene.gpu_sim_enabled and len(env_idx) == self.env.num_envs:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene.px.step()
            self.scene._gpu_fetch_all()

        # teleport robot back to correct location based on robot type
        robot_uid = self.env.robot_uids
        if robot_uid == "fetch":
            self.env.agent.reset(self.env.agent.keyframes["rest"].qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1, 0, 0.02]))
        elif "aloha_mini" in robot_uid:
            # AlohaMini robot initialization
            if hasattr(self.env.agent, 'keyframes') and 'rest' in self.env.agent.keyframes:
                self.env.agent.reset(self.env.agent.keyframes["rest"].qpos)
            # Position AlohaMini in the scene
            # Slightly elevated to account for wheel radius
            self.env.agent.robot.set_pose(sapien.Pose([-1, 0, 0.05]))
        elif robot_uid == "panda":
            if hasattr(self.env.agent, 'keyframes') and 'rest' in self.env.agent.keyframes:
                self.env.agent.reset(self.env.agent.keyframes["rest"].qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1, 0, 0.0]))
        else:
            # Fallback for other robots
            if hasattr(self.env.agent, 'keyframes') and 'rest' in self.env.agent.keyframes:
                self.env.agent.reset(self.env.agent.keyframes["rest"].qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1, 0, 0.0]))


@register_env("AlohaMini-ReplicaCAD-v1", max_episode_steps=1000)
class AlohaMiniReplicaCADEnv(SceneManipulationEnv):
    """
    ReplicaCAD scene manipulation environment with AlohaMini robot.

    This environment allows the AlohaMini dual-arm robot to operate
    in ReplicaCAD indoor scenes.
    """

    SUPPORTED_ROBOTS = ["aloha_mini", "aloha_mini_fixed", "aloha_mini_arms_only", "panda", "fetch"]
    agent: Union[AlohaMini, AlohaMiniFixed, AlohaMiniArmsOnly]

    def __init__(
        self,
        *args,
        robot_uids="aloha_mini_fixed",
        **kwargs
    ):
        # Use our custom scene builder that supports AlohaMini
        super().__init__(
            *args,
            robot_uids=robot_uids,
            scene_builder_cls=AlohaMiniReplicaCADSceneBuilder,
            **kwargs
        )

    def _load_lighting(self, options: dict):
        """Load lighting - add extra lights for better visibility."""
        # Let the scene builder handle lighting first
        if self.scene_builder.builds_lighting:
            # Add additional ambient lighting for better visibility
            self.scene.set_ambient_light([0.5, 0.5, 0.5])
            return

        # Default lighting with extra brightness
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([1, 1, -1], [1.0, 1.0, 1.0], shadow=True)
        self.scene.add_directional_light([-1, -1, -1], [0.5, 0.5, 0.5])

    @property
    def _default_sensor_configs(self):
        """Default sensor configuration for AlohaMini."""
        if "aloha_mini" in self.robot_uids:
            # Use robot's built-in sensors
            return []
        return super()._default_sensor_configs

    @property
    def _default_human_render_camera_configs(self):
        """Camera configuration for human rendering."""
        if "aloha_mini" in self.robot_uids:
            # Room overview camera - positioned to see the robot well
            room_camera_pose = sapien_utils.look_at([2.0, -2.0, 2.0], [0.0, 0.0, 0.5])
            room_camera_config = CameraConfig(
                "render_camera",
                room_camera_pose,
                1280,
                720,
                1.0,
                0.01,
                100,
            )
            return [room_camera_config]

        return super()._default_human_render_camera_configs


@register_env("AlohaMini-Empty-v1", max_episode_steps=1000)
class AlohaMiniEmptyEnv(SceneManipulationEnv):
    """
    Empty environment with AlohaMini robot for testing and debugging.
    """

    SUPPORTED_ROBOTS = ["aloha_mini", "aloha_mini_fixed", "aloha_mini_arms_only"]

    def __init__(
        self,
        *args,
        robot_uids="aloha_mini_fixed",
        **kwargs
    ):
        # Use a minimal scene builder
        super().__init__(
            *args,
            robot_uids=robot_uids,
            scene_builder_cls="ReplicaCAD",  # Will be replaced
            **kwargs
        )

    def _load_scene(self, options: dict):
        """Load an empty scene with just a ground plane."""
        # Create simple ground plane
        builder = self.scene.create_actor_builder()
        builder.add_plane_collision(
            sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0, -0.7071068, 0]),
            material=sapien.physx.PhysxMaterial(
                static_friction=0.3,
                dynamic_friction=0.3,
                restitution=0.0
            )
        )
        builder.add_plane_visual(
            sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0, -0.7071068, 0]),
            scale=[100, 100, 1],
            material=sapien.render.RenderMaterial(base_color=[0.8, 0.8, 0.8, 1])
        )
        self.ground = builder.build_static(name="ground")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize robot in empty scene."""
        # Set robot to rest pose
        with torch.device(self.device):
            if hasattr(self.agent, 'keyframes') and 'rest' in self.agent.keyframes:
                qpos = torch.tensor(
                    self.agent.keyframes['rest'].qpos,
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0).expand(len(env_idx), -1)
                self.agent.robot.set_qpos(qpos)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.5, 1.5, 1.5], [0, 0, 0.5])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
