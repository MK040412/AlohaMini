import os
import random
from copy import deepcopy

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.plan_utils import (
    select_index_by_priority_dual,
    select_index_by_priority_single,
)
from core.utils.transformation_utils import poses_from_tf_matrices
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    tf_matrix_from_pose,
)


# pylint: disable=unused-argument
@register_skill
class Pick(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg
        object_name = self.skill_cfg["objects"][0]
        self.pick_obj = task.objects[object_name]

        # Get grasp annotation
        usd_path = [obj["path"] for obj in task.cfg["objects"] if obj["name"] == object_name][0]
        usd_path = os.path.join(self.task.asset_root, usd_path)
        grasp_pose_path = usd_path.replace(
            "Aligned_obj.usd", self.skill_cfg.get("npy_name", "Aligned_grasp_sparse.npy")
        )
        sparse_grasp_poses = np.load(grasp_pose_path)
        lr_arm = "right" if "right" in self.controller.robot_file else "left"
        self.T_obj_ee, self.scores = self.robot.pose_post_process_fn(
            sparse_grasp_poses,
            lr_arm=lr_arm,
            grasp_scale=self.skill_cfg.get("grasp_scale", 1),
            tcp_offset=self.skill_cfg.get("tcp_offset", self.robot.tcp_offset),
            constraints=self.skill_cfg.get("constraints", None),
        )

        # Keyposes should be generated after previous skill is done
        self.manip_list = []
        self.pickcontact_view = task.pickcontact_views[robot.name][lr_arm][object_name]
        self.process_valid = True
        self.obj_init_trans = deepcopy(self.pick_obj.get_local_pose()[0])
        final_gripper_state = self.skill_cfg.get("final_gripper_state", -1)
        if final_gripper_state == 1:
            self.gripper_cmd = "open_gripper"
        elif final_gripper_state == -1:
            self.gripper_cmd = "close_gripper"
        else:
            raise ValueError(f"final_gripper_state must be 1 or -1, got {final_gripper_state}")
        self.fixed_orientation = self.skill_cfg.get("fixed_orientation", None)
        if self.fixed_orientation is not None:
            self.fixed_orientation = np.array(self.fixed_orientation)

    def simple_generate_manip_cmds(self):
        manip_list = []

        # Update
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        cmd = (p_base_ee_cur, q_base_ee_cur, "update_pose_cost_metric", {"hold_vec_weight": None})
        manip_list.append(cmd)

        ignore_substring = deepcopy(self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []))
        ignore_substring.append(self.pick_obj.name)
        cmd = (
            p_base_ee_cur,
            q_base_ee_cur,
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)

        # Pre grasp
        print(f"[GENDBG] simple_generate_manip_cmds ENTER use_batch={self.controller.use_batch}", flush=True)
        T_base_ee_grasps = self.sample_ee_pose(
            max_length=self.skill_cfg.get("num_grasp_candidates", 20))  # (N, 4, 4)
        print(f"[GENDBG] T_base_ee_grasps shape={T_base_ee_grasps.shape} "
              f"|pos|max={np.abs(T_base_ee_grasps[:, :3, 3]).max():.3e} "
              f"pos0={T_base_ee_grasps[0, :3, 3].round(3).tolist()}", flush=True)
        T_base_ee_pregrasps = deepcopy(T_base_ee_grasps)
        self.controller.update_specific(
            ignore_substring=ignore_substring, reference_prim_path=self.controller.reference_prim_path
        )

        if "r5a" in self.controller.robot_file:
            T_base_ee_pregrasps[:, :3, 3] -= T_base_ee_pregrasps[:, :3, 0] * self.skill_cfg.get("pre_grasp_offset", 0.1)
        else:
            T_base_ee_pregrasps[:, :3, 3] -= T_base_ee_pregrasps[:, :3, 2] * self.skill_cfg.get("pre_grasp_offset", 0.1)

        p_base_ee_pregrasps, q_base_ee_pregrasps = poses_from_tf_matrices(T_base_ee_pregrasps)
        p_base_ee_grasps, q_base_ee_grasps = poses_from_tf_matrices(T_base_ee_grasps)

        if self.controller.use_batch:
            # Check if the input arrays are exactly the same
            if np.array_equal(p_base_ee_pregrasps, p_base_ee_grasps) and np.array_equal(
                q_base_ee_pregrasps, q_base_ee_grasps
            ):
                # Inputs are identical, compute only once to avoid redundant computation
                result = self.controller.test_batch_forward(p_base_ee_grasps, q_base_ee_grasps)
                index = select_index_by_priority_single(result)
            else:
                # Inputs are different, compute separately
                pre_result = self.controller.test_batch_forward(p_base_ee_pregrasps, q_base_ee_pregrasps)
                result = self.controller.test_batch_forward(p_base_ee_grasps, q_base_ee_grasps)
                index = select_index_by_priority_dual(pre_result, result)
        else:
            for index in range(T_base_ee_grasps.shape[0]):
                p_base_ee_pregrasp, q_base_ee_pregrasp = p_base_ee_pregrasps[index], q_base_ee_pregrasps[index]
                p_base_ee_grasp, q_base_ee_grasp = p_base_ee_grasps[index], q_base_ee_grasps[index]
                test_mode = self.skill_cfg.get("test_mode", "forward")
                if test_mode == "forward":
                    result_pre = self.controller.test_single_forward(p_base_ee_pregrasp, q_base_ee_pregrasp)
                elif test_mode == "ik":
                    result_pre = self.controller.test_single_ik(p_base_ee_pregrasp, q_base_ee_pregrasp)
                else:
                    raise NotImplementedError
                if self.skill_cfg.get("pre_grasp_offset", 0.1) > 0:
                    if test_mode == "forward":
                        result = self.controller.test_single_forward(p_base_ee_grasp, q_base_ee_grasp)
                    elif test_mode == "ik":
                        result = self.controller.test_single_ik(p_base_ee_grasp, q_base_ee_grasp)
                    else:
                        raise NotImplementedError
                    if result == 1 and result_pre == 1:
                        print("pick plan success")
                        break
                else:
                    if result_pre == 1:
                        print("pick plan success")
                        break

        if self.fixed_orientation is not None:
            q_base_ee_pregrasps[index] = self.fixed_orientation
            q_base_ee_grasps[index] = self.fixed_orientation

        # Pre-grasp
        cmd = (p_base_ee_pregrasps[index], q_base_ee_pregrasps[index], "open_gripper", {})
        manip_list.append(cmd)
        if self.skill_cfg.get("pre_grasp_hold_vec_weight", None) is not None:
            cmd = (
                p_base_ee_pregrasps[index],
                q_base_ee_pregrasps[index],
                "update_pose_cost_metric",
                {"hold_vec_weight": self.skill_cfg.get("pre_grasp_hold_vec_weight", None)},
            )
            manip_list.append(cmd)

        # Grasp
        cmd = (p_base_ee_grasps[index], q_base_ee_grasps[index], "open_gripper", {})
        manip_list.append(cmd)
        cmd = (p_base_ee_grasps[index], q_base_ee_grasps[index], self.gripper_cmd, {})
        manip_list.extend(
            [cmd] * self.skill_cfg.get("gripper_change_steps", 40)
        )  # Default we use 40 steps to make sure the gripper is fully closed
        ignore_substring = deepcopy(self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []))
        cmd = (
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)
        cmd = (
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            "attach_obj",
            {"obj_prim_path": self.pick_obj.mesh_prim_path},
        )
        manip_list.append(cmd)

        # Post-grasp
        post_grasp_offset = np.random.uniform(
            self.skill_cfg.get("post_grasp_offset_min", 0.05), self.skill_cfg.get("post_grasp_offset_max", 0.05)
        )
        if post_grasp_offset:
            p_base_ee_postgrasps = deepcopy(p_base_ee_grasps)
            p_base_ee_postgrasps[index][2] += post_grasp_offset
            cmd = (p_base_ee_postgrasps[index], q_base_ee_grasps[index], self.gripper_cmd, {})
            manip_list.append(cmd)

        # Whether return to pre-grasp
        if self.skill_cfg.get("return_to_pregrasp", False):
            cmd = (p_base_ee_pregrasps[index], q_base_ee_pregrasps[index], self.gripper_cmd, {})
            manip_list.append(cmd)

        print(f"[GENDBG] selected index={index} grasp_pos={p_base_ee_grasps[index].round(3).tolist()} "
              f"pregrasp_pos={p_base_ee_pregrasps[index].round(3).tolist()}", flush=True)
        self.manip_list = manip_list

    def sample_ee_pose(self, max_length=CUROBO_BATCH_SIZE):
        T_base_ee = self.get_ee_poses("armbase")

        num_pose = T_base_ee.shape[0]
        flags = {
            "x": np.ones(num_pose, dtype=bool),
            "y": np.ones(num_pose, dtype=bool),
            "z": np.ones(num_pose, dtype=bool),
            "direction_to_obj": np.ones(num_pose, dtype=bool),
        }
        filter_conditions = {
            "x": {
                "forward": (0, 0, 1),  # (row, col, direction)
                "backward": (0, 0, -1),
                "upward": (2, 0, 1),
                "downward": (2, 0, -1),
            },
            "y": {"forward": (0, 1, 1), "backward": (0, 1, -1), "downward": (2, 1, -1), "upward": (2, 1, 1)},
            "z": {"forward": (0, 2, 1), "backward": (0, 2, -1), "downward": (2, 2, -1), "upward": (2, 2, 1)},
        }
        for axis in ["x", "y", "z"]:
            filter_list = self.skill_cfg.get(f"filter_{axis}_dir", None)
            if filter_list is not None:
                # direction, value = filter_list
                direction = filter_list[0]
                row, col, sign = filter_conditions[axis][direction]
                if len(filter_list) == 2:
                    value = filter_list[1]
                    cos_val = np.cos(np.deg2rad(value))
                    flags[axis] = T_base_ee[:, row, col] >= cos_val if sign > 0 else T_base_ee[:, row, col] <= cos_val
                elif len(filter_list) == 3:
                    value1, value2 = filter_list[1:]
                    cos_val1 = np.cos(np.deg2rad(value1))
                    cos_val2 = np.cos(np.deg2rad(value2))
                    if sign > 0:
                        flags[axis] = np.logical_and(
                            T_base_ee[:, row, col] >= cos_val1, T_base_ee[:, row, col] <= cos_val2
                        )
                    else:
                        flags[axis] = np.logical_and(
                            T_base_ee[:, row, col] <= cos_val1, T_base_ee[:, row, col] >= cos_val2
                        )
        if self.skill_cfg.get("direction_to_obj", None) is not None:
            direction_to_obj = self.skill_cfg["direction_to_obj"]
            T_world_obj = tf_matrix_from_pose(*self.pick_obj.get_local_pose())
            T_base_world = get_relative_transform(
                get_prim_at_path(self.task.root_prim_path), get_prim_at_path(self.controller.reference_prim_path)
            )
            T_base_obj = T_base_world @ T_world_obj
            if direction_to_obj == "right":
                flags["direction_to_obj"] = T_base_ee[:, 1, 3] <= T_base_obj[1, 3]
            elif direction_to_obj == "left":
                flags["direction_to_obj"] = T_base_ee[:, 1, 3] > T_base_obj[1, 3]
            else:
                raise NotImplementedError

        combined_flag = np.logical_and.reduce(list(flags.values()))
        if sum(combined_flag) == 0:
            # idx_list = [i for i in range(max_length)]
            idx_list = list(range(max_length))
        else:
            tmp_scores = self.scores[combined_flag]
            tmp_idxs = np.arange(num_pose)[combined_flag]
            combined = list(zip(tmp_scores, tmp_idxs))
            combined.sort()
            idx_list = [idx for (score, idx) in combined[:max_length]]
            score_list = self.scores[idx_list]
            weights = 1.0 / (score_list + 1e-8)
            weights = weights / weights.sum()

            sampled_idx = random.choices(idx_list, weights=weights, k=max_length)
            sampled_scores = self.scores[sampled_idx]

            # Sort indices by their scores (ascending)
            sorted_pairs = sorted(zip(sampled_scores, sampled_idx))
            idx_list = [idx for _, idx in sorted_pairs]

        print(self.scores[idx_list])
        # print((T_base_ee[idx_list])[:, 0, 1])
        return T_base_ee[idx_list]

    def get_ee_poses(self, frame: str = "world"):
        # get grasp poses at specific frame
        if frame not in ["world", "body", "armbase"]:
            raise ValueError(
                f"poses in {frame} frame is not supported: accepted values are [world, body, armbase] only"
            )

        if frame == "body":
            return self.T_obj_ee

        # Frame consistency: the "armbase" branch below measures the arm base
        # relative to task.root_prim_path, so the object MUST be expressed in
        # the SAME frame. get_local_pose() is parent-relative (the object's
        # Aligned prim is one level below the object root, not the task root)
        # and get_world_pose() is /World-relative — both mix frames whenever
        # the hierarchy or task root carries a transform. Measure it directly.
        try:
            from omni.isaac.core.utils.transformations import get_relative_transform as _grt
            from omni.isaac.core.utils.prims import get_prim_at_path as _gpp
            T_world_obj = _grt(self.pick_obj.prim, _gpp(self.task.root_prim_path))
            print(f"[FRAMEAUDIT] root-rel obj_t={np.round(T_world_obj[:3,3],3).tolist()}", flush=True)
        except Exception as _e:
            print(f"[FRAMEAUDIT] fallback local: {_e}", flush=True)
            T_world_obj = tf_matrix_from_pose(*self.pick_obj.get_local_pose())
        try:
            print(f"[FLTDBG] obj_world_pose={self.pick_obj.get_world_pose()} "
                  f"|T_world_obj|max={np.abs(T_world_obj).max():.3e} "
                  f"|T_obj_ee|max={np.abs(self.T_obj_ee).max():.3e} "
                  f"T_obj_ee.shape={self.T_obj_ee.shape}", flush=True)
        except Exception as _e:  # pylint: disable=broad-except
            print("[FLTDBG] err", _e, flush=True)
        T_world_ee = T_world_obj[None] @ self.T_obj_ee
        try:
            _pad = T_world_ee[0] @ np.array([0.0, -0.091, 0.0, 1.0])
            print(f"[GENPAD] obj_w={np.round(T_world_obj[:3,3],3).tolist()} "
                  f"ee0_w={np.round(T_world_ee[0,:3,3],3).tolist()} "
                  f"pad0_w={np.round(_pad[:3],3).tolist()} "
                  f"pad-obj={np.round(_pad[:3]-T_world_obj[:3,3],3).tolist()}", flush=True)
        except Exception:
            pass

        if frame == "world":
            return T_world_ee

        if frame == "armbase":  # arm base frame
            T_world_base = get_relative_transform(
                get_prim_at_path(self.controller.reference_prim_path), get_prim_at_path(self.task.root_prim_path)
            )
            T_base_world = np.linalg.inv(T_world_base)
            try:
                # Compare query methods for the arm base world pose.
                bp = self.controller.reference_prim_path
                # (A) physics/fabric-backed query (same kind that works for objects)
                try:
                    from isaacsim.core.prims import SingleXFormPrim
                    xp = SingleXFormPrim(bp).get_world_pose()
                    a_str = f"pos={np.round(np.asarray(xp[0]),4).tolist()}"
                except Exception as _e2:  # pylint: disable=broad-except
                    a_str = f"ERR {_e2}"
                # (B) robot articulation root world pose
                try:
                    rp = self.robot.get_world_pose() if hasattr(self.robot, "get_world_pose") else None
                    b_str = f"pos={np.round(np.asarray(rp[0]),4).tolist()}" if rp is not None else "n/a"
                except Exception as _e3:  # pylint: disable=broad-except
                    b_str = f"ERR {_e3}"
                print(f"[BASEMAT] {np.round(T_world_base,5).tolist()}", flush=True)
                print(f"[OBJMAT] {np.round(tf_matrix_from_pose(*self.pick_obj.get_world_pose()),5).tolist()}", flush=True)
                print(f"[BASEDBG] USD_T_world_base Z={T_world_base[2,3]:.3e} X={T_world_base[0,3]:.3f} Y={T_world_base[1,3]:.3f}\n"
                      f"[BASEDBG]   XFormPrim.get_world_pose(left_Base): {a_str}\n"
                      f"[BASEDBG]   robot.get_world_pose(root): {b_str}", flush=True)
            except Exception as _e:  # pylint: disable=broad-except
                print("[BASEDBG] err", _e, flush=True)
            T_base_ee = T_base_world[None] @ T_world_ee
            return T_base_ee

    def get_contact(self, contact_threshold=0.0):
        contact = np.abs(self.pickcontact_view.get_contact_force_matrix()).squeeze()
        contact = np.sum(contact, axis=-1)
        indices = np.where(contact > contact_threshold)[0]
        return contact, indices

    def is_feasible(self, th=5):
        return self.controller.num_plan_failed <= th

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        p_base_ee, q_base_ee, *_ = self.manip_list[0]
        diff_trans = np.linalg.norm(p_base_ee_cur - p_base_ee)
        diff_ori = 2 * np.arccos(min(abs(np.dot(q_base_ee_cur, q_base_ee)), 1.0))
        pose_flag = np.logical_and(
            diff_trans < t_eps,
            diff_ori < o_eps,
        )
        self.plan_flag = self.controller.num_last_cmd > 10
        return np.logical_or(pose_flag, self.plan_flag)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done(t_eps=self.skill_cfg.get("t_eps", 1e-3), o_eps=self.skill_cfg.get("o_eps", 5e-3)):
            self.manip_list.pop(0)
            try:
                import numpy as _np
                from omni.isaac.core.utils.transformations import get_relative_transform
                from omni.isaac.core.utils.prims import get_prim_at_path
                op = _np.asarray(self.pick_obj.get_world_pose()[0]).reshape(-1)
                T_we = get_relative_transform(
                    get_prim_at_path(self.robot.fl_ee_path),
                    get_prim_at_path("/World"))
                tip = (T_we @ _np.array([0.0, -0.10, 0.0, 1.0]))[:3]
                T_wb = get_relative_transform(
                    get_prim_at_path(self.robot.fl_base_path),
                    get_prim_at_path("/World"))
                ee_prim_b = _np.linalg.inv(T_wb) @ T_we
                cur_p, cur_q = self.controller.get_ee_pose()
                nxt = self.manip_list[0] if self.manip_list else None
                tgt = _np.round(_np.asarray(nxt[0]).reshape(-1), 3).tolist() if nxt is not None else None
                print(f"[SUBPOP] remaining={len(self.manip_list)} "
                      f"cmd={nxt[2] if nxt else 'END'} "
                      f"obj={_np.round(op,3).tolist()} tip={_np.round(tip,3).tolist()} "
                      f"d={_np.round(tip-op,3).tolist()} | curobo_ee={_np.round(cur_p,3).tolist()} "
                      f"prim_ee_b={_np.round(ee_prim_b[:3,3],3).tolist()} tgt={tgt}", flush=True)
            except Exception as _e:
                print(f"[SUBPOP] dbg {_e}", flush=True)
        return len(self.manip_list) == 0

    def is_success(self):
        flag = True

        contact, indices = self.get_contact()
        if self.gripper_cmd == "close_gripper":
            flag = len(indices) >= 1
        try:
            obj_p = np.asarray(self.pick_obj.get_world_pose()[0]).reshape(-1)
            js = self.robot.get_joints_state()
            gq = np.asarray(js.positions)[self.controller.gripper_indices]
            print(f"[PICKSUCC2] cmd={self.gripper_cmd} contact={np.round(np.atleast_1d(np.asarray(contact,dtype=float)),3).tolist()} "
                  f"nidx={len(indices)} obj={np.round(obj_p,3).tolist()} grip_q={np.round(gq,4).tolist()} "
                  f"maxjv={float(np.max(np.abs(js.velocities))):.2f}", flush=True)
        except Exception as exc:
            print(f"[PICKSUCC2] debug failed: {exc}", flush=True)

        if self.skill_cfg.get("process_valid", True):
            self.process_valid = np.max(np.abs(self.robot.get_joints_state().velocities)) < 5 and (
                np.max(np.abs(self.pick_obj.get_linear_velocity())) < 5
            )
        flag = flag and self.process_valid

        if self.skill_cfg.get("lift_th", 0.0) > 0.0:
            p_world_obj = deepcopy(self.pick_obj.get_local_pose()[0])
            flag = flag and ((p_world_obj[2] - self.obj_init_trans[2]) > self.skill_cfg.get("lift_th", 0.0))

        return flag
