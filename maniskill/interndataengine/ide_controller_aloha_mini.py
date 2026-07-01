"""AlohaMini dual-arm (5-DOF SO100) controller - template-based.

Mirrors Lift2Controller but for the 5-DOF AlohaMini arms + parallel gripper. The CuRobo
config and the USD articulation both use the real joint names (left_shoulder_pan ...
left_wrist_roll), so raw_js_names == cmd_js_names (identity mapping, no rename needed).
"""

import numpy as np
from core.controllers.base_controller import register_controller
from core.controllers.template_controller import TemplateController

_LEFT_ARM = ["left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
             "left_wrist_flex", "left_wrist_roll"]
_RIGHT_ARM = ["right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
              "right_wrist_flex", "right_wrist_roll"]


# pylint: disable=unused-argument
@register_controller
class AlohaMiniController(TemplateController):
    def _get_default_ignore_substring(self):
        return ["material", "Plane", "conveyor", "scene", "table", "fluid"]

    def _configure_joint_indices(self, robot_file: str) -> None:
        if "left" in robot_file:
            self.raw_js_names = list(_LEFT_ARM)
            self.cmd_js_names = list(_LEFT_ARM)
            self.arm_indices = np.array(self.robot.cfg["left_joint_indices"])
            self.gripper_indices = np.array(self.robot.cfg["left_gripper_indices"])
            self.reference_prim_path = self.task.robots[self.name].fl_base_path
            self.lr_name = "left"
            self._gripper_state = 1.0 if self.robot.left_gripper_state == 1.0 else -1.0
        elif "right" in robot_file:
            self.raw_js_names = list(_RIGHT_ARM)
            self.cmd_js_names = list(_RIGHT_ARM)
            self.arm_indices = np.array(self.robot.cfg["right_joint_indices"])
            self.gripper_indices = np.array(self.robot.cfg["right_gripper_indices"])
            self.reference_prim_path = self.task.robots[self.name].fr_base_path
            self.lr_name = "right"
            self._gripper_state = 1.0 if self.robot.right_gripper_state == 1.0 else -1.0
        else:
            raise NotImplementedError("robot_file must contain 'left' or 'right'")
        self._gripper_joint_position = np.array([1.0])

    def get_gripper_action(self):
        # AlohaMini parallel gripper stroke per finger is 0..0.037 m (open at 0.037).
        return np.clip(self._gripper_state * self._gripper_joint_position, 0.0, 0.037)

    def forward(self, manip_cmd, eps=5e-3):
        ee_trans, ee_ori = manip_cmd[0:2]
        gripper_fn = manip_cmd[2]
        params = manip_cmd[3]
        assert hasattr(self, gripper_fn)
        method = getattr(self, gripper_fn)
        if gripper_fn in ["in_plane_rotation", "mobile_move", "dummy_forward", "joint_ctrl"]:
            return method(**params)
        elif gripper_fn in ["update_pose_cost_metric", "update_specific"]:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps=eps, skip_plan=True)
        else:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps=eps)
