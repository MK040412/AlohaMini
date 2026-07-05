"""AlohaMini Pro dual-arm (true 6-joint) controller - template-based."""

import numpy as np
from core.controllers.base_controller import register_controller
from core.controllers.template_controller import TemplateController

_LEFT_ARM = ["left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_joint6"]
_RIGHT_ARM = ["right_joint1", "right_joint2", "right_joint3", "right_joint4", "right_joint5", "right_joint6"]


# pylint: disable=unused-argument
@register_controller
class AlohaMiniProController(TemplateController):
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
        self._gripper_joint_position = np.array([1.0] * len(self.gripper_indices))

    def get_gripper_action(self):
        # keep 2mm at full close: the pads meet exactly at q=0 and with
        # articulation self-collision enabled a 0-target makes them fight
        return np.clip(self._gripper_state * self._gripper_joint_position, 0.002, 0.037)

    def lift_ctrl(self, target: float = 0.13):
        """Command the dedicated vertical_move joint (dof 3) directly; the arm
        holds its current commanded pose. Vertical descents/ascends through
        this joint need no CuRobo plan at all."""
        try:
            self.robot._articulation_view.set_joint_position_targets(
                np.array([float(target)]), joint_indices=np.array([3]))
        except Exception as exc:
            print(f"[LIFTCTRL] {exc}", flush=True)
        return None

    def forward(self, manip_cmd, eps=5e-3):
        ee_trans, ee_ori = manip_cmd[0:2]
        gripper_fn = manip_cmd[2]
        params = manip_cmd[3]
        assert hasattr(self, gripper_fn)
        method = getattr(self, gripper_fn)
        if gripper_fn == "lift_ctrl":
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps=eps, skip_plan=True)
        if gripper_fn in ["in_plane_rotation", "mobile_move", "dummy_forward", "joint_ctrl"]:
            return method(**params)
        elif gripper_fn in ["update_pose_cost_metric", "update_specific"]:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps=eps, skip_plan=True)
        else:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps=eps)
