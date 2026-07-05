import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.transformations import (
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)


# pylint: disable=unused-argument
@register_skill
class Heuristic_Skill(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg

        self.lr_hand = "right" if "right" in self.controller.robot_file else "left"
        if self.lr_hand == "left":
            self._joint_indices = self.robot.left_joint_indices
            self._joint_home = self.robot.left_joint_home
            self._joint_home = np.array(self._joint_home)
            if self.skill_cfg.get("gripper_state", None):
                self._gripper_state = self.skill_cfg["gripper_state"]
            else:
                self._gripper_state = self.robot.left_gripper_state
        else:
            self._joint_indices = self.robot.right_joint_indices
            self._joint_home = self.robot.right_joint_home
            self._joint_home = np.array(self._joint_home)
            if self.skill_cfg.get("gripper_state", None):
                self._gripper_state = self.skill_cfg["gripper_state"]
            else:
                self._gripper_state = self.robot.right_gripper_state

        ALLOWED_MODES = {"abs_qpos", "rel_qpos", "rel_ee", "home"}

        self.mode = self.skill_cfg.get("mode", "home")
        if self.mode not in ALLOWED_MODES:
            raise ValueError(
                f"Unsupported mode '{self.mode}' for JointMove. Allowed modes are: {sorted(ALLOWED_MODES)}"
            )
        self.move_steps = self.skill_cfg.get("move_steps", 50)
        self.t_eps = self.skill_cfg.get("t_eps", 0.088)

        # Keyposes should be generated after previous skill is done
        self.manip_list = []
        self._goal_joints = None

    def _compute_ee_goal(self, p_base_ee_cur, q_base_ee_cur, rel_ee):
        """
        rel_ee: (4,4) transformation matrix
        """
        T_base_ee = tf_matrix_from_pose(p_base_ee_cur, q_base_ee_cur)
        if isinstance(rel_ee, (list, tuple)):
            rel_ee = np.array(rel_ee)
        T_base_ee_tgt = rel_ee @ T_base_ee
        p_base_ee_tgt, q_base_ee_tgt = pose_from_tf_matrix(T_base_ee_tgt)
        return p_base_ee_tgt, q_base_ee_tgt

    def _solve_goal_joints_via_plan(self, ee_trans_goal, ee_ori_goal):
        """
        Use controller.plan to get a collision-free joint path,
        and take the last waypoint as goal arm joints.
        """
        if self.controller.use_batch:
            raise NotImplementedError

        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        result = self.controller.plan(ee_trans_goal, ee_ori_goal, sim_js, js_names)
        succ = result.success.item()
        if succ:
            cmd_plan = result.get_interpolated_plan()
            goal_arm_joints = cmd_plan[-1].position.cpu().numpy()  # replace by ik
            return goal_arm_joints
        else:
            return None

    def _build_joint_traj(self, curr_joints, goal_joints, p_base_ee_cur, q_base_ee_cur):
        """Build a list of dummy_forward commands interpolating in joint space."""
        manip_list = []
        for k in range(self.move_steps):
            alpha = float(k + 1) / float(self.move_steps) * 1.25
            arm_action = goal_joints * alpha + curr_joints * (1.0 - alpha)
            cmd = (
                p_base_ee_cur,
                q_base_ee_cur,
                "dummy_forward",
                {
                    "arm_action": arm_action,
                    "gripper_state": self._gripper_state,
                },
            )
            manip_list.append(cmd)
        return manip_list

    def simple_generate_manip_cmds(self):
        self.manip_list = []
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        curr_joints = self.robot.get_joint_positions()[self._joint_indices]

        if self.mode == "home":
            self._goal_joints = self._joint_home.copy()
        else:
            if self.mode == "abs_qpos":
                self._goal_joints = self.skill_cfg.get("value", self._joint_home)
            elif self.mode == "rel_qpos":
                self._goal_joints = self.skill_cfg.get("value", np.zeros(self._joint_home.shape))
            elif self.mode == "rel_ee":
                p_base_ee_tgt, q_base_ee_tgt = self._compute_ee_goal(
                    p_base_ee_cur, q_base_ee_cur, self.skill_cfg.get("value", np.eye(4))
                )
                self._goal_joints = self._solve_goal_joints_via_plan(p_base_ee_tgt, q_base_ee_tgt)
            else:
                raise NotImplementedError

        if self._goal_joints is None:
            self.manip_list = []
            cmd = (
                p_base_ee_cur,
                q_base_ee_cur,
                "update_specific",
                {
                    "ignore_substring": self.controller.ignore_substring,
                    "reference_prim_path": self.controller.reference_prim_path,
                },
            )
            self.manip_list.append(cmd)
            return

        self.manip_list = self._build_joint_traj(curr_joints, self._goal_joints, p_base_ee_cur, q_base_ee_cur)

    def is_feasible(self, th=5):
        return self.controller.num_plan_failed <= th

    def is_subtask_done(self, t_eps=0.088):
        if len(self.manip_list) == 0:
            return True
        if self._goal_joints is None:
            return True
        curr_joints = self.robot.get_joint_positions()[self._joint_indices]
        target_joints = self.manip_list[0][3]["arm_action"]
        diff_trans = np.linalg.norm(curr_joints - target_joints)
        pose_flag = diff_trans < t_eps
        self.plan_flag = self.controller.num_last_cmd > 10
        return np.logical_or(pose_flag, self.plan_flag)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done(t_eps=self.t_eps):
            self.manip_list.pop(0)
        if self.is_success(t_eps=self.t_eps):
            self.manip_list.clear()
            print("Heuristic Skill Done")
        return len(self.manip_list) == 0

    def is_success(self, t_eps=0.088):
        t_eps = float(self.skill_cfg.get("home_t_eps", t_eps))
        if self._goal_joints is None:
            print("cannot compute goal joints, skill failure")
            return False

        curr_joints = self.robot.get_joint_positions()[self._joint_indices]
        diff_trans = np.linalg.norm(curr_joints - self._goal_joints)
        return diff_trans < t_eps
