"""ManiSkill-style scripted pick: hover over the object center, descend,
close, lift. Bypasses graspnet candidates / R_ee conventions / direction
filters entirely — the grasp pose is computed directly from the object pose
(task-root frame, same frame as the armbase transform) with a fixed
reach-map-verified top-down orientation (identity quat in the arm base frame,
pads extending along -y_base = straight down)."""

from copy import deepcopy

import numpy as np
from core.skills.base_skill import register_skill
from core.skills.pick import Pick


# pylint: disable=too-many-locals
@register_skill
class ScriptedPick(Pick):
    def is_feasible(self, th=5):
        # the scripted flow advances on pose tolerance; transient short-segment
        # replan failures (the descend is only ~5cm) must not kill the episode
        return self.controller.num_plan_failed <= max(th, 60)

    def is_done(self):
        # update_* commands carry the generation-time EE pose as their target;
        # gravity settle can drift past the pose tolerance and (having no plan)
        # they never hit the idle-timeout fallback — pop them after 2 ticks.
        if self.manip_list and self.manip_list[0][2] in (
                "update_pose_cost_metric", "update_specific", "lift_ctrl"):
            self._upd_ticks = getattr(self, "_upd_ticks", 0) + 1
            if self._upd_ticks >= 2:
                self._upd_ticks = 0
                self.manip_list.pop(0)
                print(f"[SUBPOP] remaining={len(self.manip_list)} (update auto-pop)", flush=True)
            return len(self.manip_list) == 0
        done = super().is_done()
        if done:
            # don't let pick-phase counters leak into the next skill's gates:
            # num_plan_failed feeds is_feasible; num_last_cmd feeds the home
            # skill's plan_flag (stale >10 makes its joint traj pop instantly)
            self.controller.num_plan_failed = 0
            self.controller.num_last_cmd = 0
        return done

    def simple_generate_manip_cmds(self):
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.transformations import get_relative_transform

        root = get_prim_at_path(self.task.root_prim_path)
        T_root_obj = get_relative_transform(self.pick_obj.prim, root)
        T_root_base = get_relative_transform(
            get_prim_at_path(self.controller.reference_prim_path), root)
        T_base_obj = np.linalg.inv(T_root_base) @ T_root_obj
        obj_b = T_base_obj[:3, 3]

        # top-down: identity orientation in the base frame (verified solvable
        # family); EE origin sits pad_above over the grasp point so the pads
        # (at (0,-0.091,0) in the EE frame) land on the object
        q = np.array([1.0, 0.0, 0.0, 0.0])
        pad_above = float(self.skill_cfg.get("pad_above", 0.10))
        pre_extra = float(self.skill_cfg.get("pre_grasp_offset", 0.05))
        lift_up = float(self.skill_cfg.get("post_grasp_offset_min", 0.10))
        grasp_p = obj_b + np.array([0.0, pad_above, 0.0])
        pre_p = grasp_p + np.array([0.0, pre_extra, 0.0])
        lift_p = grasp_p + np.array([0.0, lift_up, 0.0])

        cur_p, cur_q = self.controller.get_ee_pose()
        ignore = deepcopy(self.controller.ignore_substring)
        ignore.append(self.pick_obj.name)

        manip_list = []
        manip_list.append((cur_p, cur_q, "update_pose_cost_metric",
                           {"hold_vec_weight": self.skill_cfg.get("hold_vec_weight", None)}))
        manip_list.append((cur_p, cur_q, "update_specific",
                           {"ignore_substring": ignore,
                            "reference_prim_path": self.controller.reference_prim_path}))
        lift_home = 0.13
        drop = float(self.skill_cfg.get("lift_drop", 0.06))
        # plan once to the hover pose, then descend/ascend with the dedicated
        # vertical_move joint — no short-segment CuRobo plans at all
        manip_list.append((pre_p, q, "open_gripper", {}))
        for _ in range(12):
            manip_list.append((pre_p, q, "lift_ctrl", {"target": lift_home - drop}))
        for _ in range(int(self.skill_cfg.get("gripper_change_steps", 15))):
            manip_list.append((pre_p, q, "close_gripper", {}))
        for i in range(12):
            ramp = lift_home - drop + drop * (i + 1) / 12.0
            manip_list.append((pre_p, q, "lift_ctrl", {"target": ramp}))
        for _ in range(6):
            manip_list.append((pre_p, q, "lift_ctrl", {"target": lift_home}))
        manip_list.append((lift_p, q, "close_gripper", {}))
        # hold at the lift target so the object reaches full height before the
        # final pop triggers is_success (pose tolerance pops ~2.5cm early)
        for _ in range(40):
            manip_list.append((lift_p, q, "close_gripper", {}))
        self.manip_list = manip_list
        self.gripper_cmd = "close_gripper"
        print(f"[SPBASE] mount_root_rel_t={np.round(T_root_base[:3,3],3).tolist()}", flush=True)
        print(f"[SCRIPTPICK] obj_b={np.round(obj_b, 3).tolist()} "
              f"grasp={np.round(grasp_p, 3).tolist()} pre={np.round(pre_p, 3).tolist()}",
              flush=True)
