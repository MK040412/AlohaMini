"""AlohaMini Pro robot implementation - dual true 6-joint arms with lift + grippers."""

import numpy as np
from core.robots.base_robot import register_robot
from core.robots.template_robot import TemplateRobot


# pylint: disable=line-too-long,unused-argument
@register_robot
class AlohaMiniPro(TemplateRobot):
    """AlohaMini Pro v3 dual-arm robot with a lift joint + parallel grippers."""

    def _setup_joint_indices(self):
        self.left_joint_indices = self.cfg["left_joint_indices"]
        self.right_joint_indices = self.cfg["right_joint_indices"]
        self.left_gripper_indices = self.cfg["left_gripper_indices"]
        self.right_gripper_indices = self.cfg["right_gripper_indices"]
        self.body_indices = []
        self.head_indices = []
        self.lift_indices = self.cfg["lift_indices"]

    def _setup_paths(self):
        self.fl_ee_path = f"{self.robot_prim_path}/{self.cfg['fl_ee_path']}"
        self.fr_ee_path = f"{self.robot_prim_path}/{self.cfg['fr_ee_path']}"
        self.fl_base_path = f"{self.robot_prim_path}/{self.cfg['fl_base_path']}"
        self.fr_base_path = f"{self.robot_prim_path}/{self.cfg['fr_base_path']}"
        self.fl_hand_path = self.fl_ee_path
        self.fr_hand_path = self.fr_ee_path

    def _setup_gripper_keypoints(self):
        self.fl_gripper_keypoints = self.cfg["fl_gripper_keypoints"]
        self.fr_gripper_keypoints = self.cfg["fr_gripper_keypoints"]

    def _setup_collision_paths(self):
        self.fl_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_filter_paths"]]
        self.fr_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fr_filter_paths"]]
        self.fl_forbid_collision_paths = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_forbid_collision_paths"]]
        self.fr_forbid_collision_paths = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fr_forbid_collision_paths"]]

    def _calculate_ee_position(self, T_obj_tcp, depths, tcp_offset):
        # Pro v3 Fixed_Jaw frame (donor chain) has the opposite finger-axis
        # sense vs Std: the template's ee = tcp + axis*(depth - tcp_offset)
        # plants the wrist one tcp_offset PAST the grasp point (measured
        # |d|~0.18 m at close). Flip the compensation sign.
        tcp_center = T_obj_tcp[:, 0:3, 3]
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis = T_obj_tcp[:, 0:3, axis_map[self._get_ee_axis()]]
        ee_center = tcp_center + axis * (tcp_offset - depths)
        T_obj_ee = T_obj_tcp.copy()
        T_obj_ee[:, 0:3, 3] = ee_center
        return T_obj_ee

    def _get_gripper_state(self, gripper_home):
        return 1.0 if gripper_home and gripper_home[0] >= 0.02 else -1.0

    def _setup_joint_velocities(self):
        all_joint_indices = self.lift_indices + self.left_joint_indices + self.right_joint_indices
        if all_joint_indices:
            self._articulation_view.set_max_joint_velocities(
                np.array([500.0] * len(all_joint_indices)),
                joint_indices=np.array(all_joint_indices),
            )
        # The converted Pro USD ships with damping=0 on every arm/finger/lift
        # drive (Std has stiffness/10); an undamped position drive + the
        # hold-current command pattern is an energy pump and the arms thrash.
        # Match the Std gain profile at runtime.
        try:
            arm = np.array(self.left_joint_indices + self.right_joint_indices)
            self._articulation_view.set_gains(
                kps=np.full(len(arm), 35809.9), kds=np.full(len(arm), 3581.0),
                joint_indices=arm,
            )
            self._articulation_view.set_max_efforts(np.full(len(arm), 100.0), joint_indices=arm)
            fin = np.array(self.left_gripper_indices + self.right_gripper_indices)
            self._articulation_view.set_gains(
                kps=np.full(len(fin), 625.0), kds=np.full(len(fin), 62.5),
                joint_indices=fin,
            )
            lift = np.array(self.lift_indices)
            self._articulation_view.set_gains(
                kps=np.full(len(lift), 625.0), kds=np.full(len(lift), 62.5),
                joint_indices=lift,
            )
            print("[GAINS] Pro drives re-gained (arm kd=3581, finger/lift kd=62.5)", flush=True)
        except Exception as exc:
            print(f"[GAINS] failed: {exc}", flush=True)
