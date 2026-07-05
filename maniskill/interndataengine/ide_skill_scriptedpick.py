"""ManiSkill-style scripted pick: hover over the object center, descend,
close, lift. Bypasses graspnet candidates / R_ee conventions / direction
filters entirely — the grasp pose is computed directly from the object pose
(task-root frame, same frame as the armbase transform) with a fixed
reach-map-verified top-down orientation (identity quat in the arm base frame,
pads extending along -y_base = straight down).

Nav pre-alignment: before manipulating, the chassis (root_z_rotation, dof 2)
is rotated so the target sits on the arm's IK-friendly centerline. Alignment
is a measured feedback loop (dBearing/dq was measured at ~-1.0, so each round
commands delta_q = +bearing and re-measures) — analytic pivot models
over-estimated the coupling and left ~50% residual. Manipulation commands are
generated only after alignment converges, from freshly measured transforms."""

from copy import deepcopy

import numpy as np
from core.skills.base_skill import register_skill
from core.skills.pick import Pick

ALIGN_TOL = 0.04       # rad; ~7mm lateral at 0.17m reach
ALIGN_MAX_ROUNDS = 8
ALIGN_MAX_STEP = 0.2   # rad per feedback round (settle between rounds re-seats the object)


# pylint: disable=too-many-locals
@register_skill
class ScriptedPick(Pick):
    def is_feasible(self, th=5):
        # the scripted flow advances on pose tolerance; transient short-segment
        # replan failures (the descend is only ~5cm) must not kill the episode
        return self.controller.num_plan_failed <= max(th, 60)

    # ---- nav alignment helpers -------------------------------------------

    def _measure_bearing(self, target_root):
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.transformations import get_relative_transform

        root = get_prim_at_path(self.task.root_prim_path)
        T_rb = get_relative_transform(
            get_prim_at_path(self.controller.reference_prim_path), root)
        tb = (np.linalg.inv(T_rb) @ np.append(np.asarray(target_root, dtype=float), 1.0))[:3]
        return float(np.arctan2(tb[0], tb[2]))

    def _align_ramp_cmds(self, bearing):
        """One feedback round: measured dBearing/dq ~ -1.0, so command
        delta_q = +bearing (clamped) as a ramp the chassis PD can track."""
        delta_q = float(np.clip(bearing, -ALIGN_MAX_STEP, ALIGN_MAX_STEP))
        q_now = float(self.robot._articulation_view.get_joint_positions()[0, 2])
        cur_p, cur_q = self.controller.get_ee_pose()
        cmds = []
        n = max(8, int(abs(delta_q) / getattr(self, "_align_step", 0.03)))
        for k in range(1, n + 1):
            cmds.append((cur_p, cur_q, "body_ctrl", {"target": q_now + delta_q * k / n}))
        # long settle: the bearing check must see a rested chassis, or the
        # fill-time transforms are stale while the joint keeps converging
        for _ in range(20):
            cmds.append((cur_p, cur_q, "body_ctrl", {"target": q_now + delta_q}))
        return cmds

    def _align_target(self):
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.transformations import get_relative_transform

        root = get_prim_at_path(self.task.root_prim_path)
        return get_relative_transform(self.pick_obj.prim, root)[:3, 3]

    # ---- skill lifecycle --------------------------------------------------

    def is_done(self):
        # update_* commands carry the generation-time EE pose as their target;
        # gravity settle can drift past the pose tolerance and (having no plan)
        # they never hit the idle-timeout fallback — pop them after 2 ticks.
        if self.manip_list and self.manip_list[0][2] in (
                "update_pose_cost_metric", "update_specific", "lift_ctrl",
                "body_ctrl", "wrist_ctrl"):
            self._upd_ticks = getattr(self, "_upd_ticks", 0) + 1
            if self._upd_ticks >= 2:
                self._upd_ticks = 0
                was_body = self.manip_list[0][2] == "body_ctrl"
                if getattr(self, "_carry_gripper", None) == "close_gripper":
                    from omni.isaac.core.utils.prims import get_prim_at_path
                    from omni.isaac.core.utils.transformations import get_relative_transform
                    _r = get_prim_at_path(self.task.root_prim_path)
                    _o = get_relative_transform(self.pick_obj.prim, _r)[:3, 3]
                    _g = self.robot._articulation_view.get_joint_positions()[
                        0, self.controller.gripper_indices]
                    print(f"[CARRYTRACE] cmd={self.manip_list[0][2]} "
                          f"tgt={round(self.manip_list[0][3].get('target', -1), 3)} "
                          f"obj_z={round(float(_o[2]), 3)} "
                          f"grip={np.round(np.asarray(_g), 4).tolist()}", flush=True)
                self.manip_list.pop(0)
                if (was_body and getattr(self, "_align_state", None) is not None
                        and (not self.manip_list or self.manip_list[0][2] != "body_ctrl")):
                    st = self._align_state
                    # re-read the live target: big swings can nudge the object
                    b = self._measure_bearing(self._align_target())
                    if abs(b) > ALIGN_TOL and st["rounds"] < ALIGN_MAX_ROUNDS:
                        st["rounds"] += 1
                        self.manip_list = self._align_ramp_cmds(b) + self.manip_list
                        print(f"[PREALIGN] round {st['rounds']}: bearing {b:+.3f}, "
                              f"correcting", flush=True)
                    else:
                        self._align_state = None
                        print(f"[PREALIGNDONE] bearing {b:+.3f} after "
                              f"{st['rounds']} round(s)", flush=True)
                        self.manip_list = self._post_align_cmds() + self.manip_list
                        if not self.manip_list:
                            self._fill_manip_cmds()
                            self._fill_done = True
                elif (not self.manip_list
                        and getattr(self, "_align_state", None) is None
                        and not getattr(self, "_fill_done", False)):
                    self._fill_manip_cmds()
                    self._fill_done = True
            return len(self.manip_list) == 0
        done = super().is_done()
        if done:
            # don't let pick-phase counters leak into the next skill's gates:
            # num_plan_failed feeds is_feasible; num_last_cmd feeds the home
            # skill's plan_flag (stale >10 makes its joint traj pop instantly)
            self.controller.num_plan_failed = 0
            self.controller.num_last_cmd = 0
        return done

    def _post_align_cmds(self):
        # settle the mast back to lift home before the grasp approach: the
        # approach-plan geometry was validated starting at 0.13 — from 0.15
        # the swoop brushed the object off its spot
        cur_p, cur_q = self.controller.get_ee_pose()
        cmds = []
        for k in range(1, 5):
            cmds.append((cur_p, cur_q, "lift_ctrl", {"target": 0.15 - 0.02 * k / 4}))
        # long settle: fill measures the mount pose right after this — an
        # unsettled lift (measured z 0.844 vs 0.83) skews the descend geometry
        for _ in range(10):
            cmds.append((cur_p, cur_q, "lift_ctrl", {"target": 0.13}))
        return cmds

    def _pre_align_cmds(self):
        # raise the mast before the chassis swing: at lift_home the hanging
        # fingers sweep at tabletop height and plow through the object band
        cur_p, cur_q = self.controller.get_ee_pose()
        cmds = []
        for k in range(1, 7):
            cmds.append((cur_p, cur_q, "lift_ctrl", {"target": 0.13 + 0.02 * k / 6}))
        for _ in range(4):
            cmds.append((cur_p, cur_q, "lift_ctrl", {"target": 0.15}))
        # re-home the chassis (body_home pi): episodes inherit the previous
        # episode's rotation, which otherwise forces -0.9 rad corrections
        q_now = float(self.robot._articulation_view.get_joint_positions()[0, 2])
        home = float(np.pi)
        delta = home - q_now
        if abs(delta) > 0.03:
            n = max(8, int(abs(delta) / 0.03))
            for k in range(1, n + 1):
                cmds.append((cur_p, cur_q, "body_ctrl", {"target": q_now + delta * k / n}))
            for _ in range(10):
                cmds.append((cur_p, cur_q, "body_ctrl", {"target": home}))
        return cmds

    def is_success(self, *args, **kwargs):
        ok = super().is_success(*args, **kwargs)
        if ok:
            # wedge rejection: a real grasp closes the fingers onto the object;
            # q stuck near fully-open (0.037) means the object is jammed, not
            # held, and will be flung during the nav swing
            q = self.robot._articulation_view.get_joint_positions()[
                0, self.controller.gripper_indices]
            closed = bool(np.max(q) < 0.030)
            print(f"[GRIPGATE] grip_q={np.round(np.asarray(q), 4).tolist()} "
                  f"closed={closed}", flush=True)
            ok = ok and closed
        return ok

    def simple_generate_manip_cmds(self):
        self._fill_done = False
        target = self._align_target()
        b0 = self._measure_bearing(target)
        pre = self._pre_align_cmds()
        if abs(b0) > ALIGN_TOL:
            self._align_state = {"target": target, "rounds": 1}
            self.manip_list = pre + self._align_ramp_cmds(b0)
            self.gripper_cmd = getattr(self, "_carry_gripper", "open_gripper")
            print(f"[PREALIGN] round 1: bearing {b0:+.3f}, correcting", flush=True)
        else:
            self._align_state = None
            print(f"[PREALIGN] bearing {b0:+.3f} already centered", flush=True)
            self._fill_manip_cmds()
            self.manip_list = pre + self.manip_list

    # ---- manipulation command generation (post-alignment geometry) --------

    def _fill_manip_cmds(self):
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
        try:
            self.controller.plan_config.time_dilation_factor = 1.0
        except AttributeError:
            pass

        manip_list = []
        manip_list.append((cur_p, cur_q, "update_pose_cost_metric",
                           {"hold_vec_weight": self.skill_cfg.get("hold_vec_weight", None)}))
        manip_list.append((cur_p, cur_q, "update_specific",
                           {"ignore_substring": ignore,
                            "reference_prim_path": self.controller.reference_prim_path}))
        lift_home = 0.13
        # 0.045 with a non-rolling object (obj_10): the crumpled-ball obj_8
        # rolled vertically out of the pinch at any depth/force (CARRYTRACE:
        # constant ~30mm/s roll-through at 137N) — the fix is the object, not
        # the squeeze
        drop = float(self.skill_cfg.get("lift_drop", 0.045))
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


@register_skill
class ScriptedPlace(ScriptedPick):
    """Drop the held object into a container (sort_the_rubbish-style place):
    align the chassis toward the container, plan — still holding — to a hover
    pose above its mouth, open, let the object fall in, then judge success by
    the object resting inside the container bbox. Reuses ScriptedPick's
    pop/counter/alignment machinery."""

    _carry_gripper = "close_gripper"  # keep holding during nav alignment

    def _post_align_cmds(self):
        # keep the mast raised: the drop needs the rim clearance
        return []

    # carried objects slipped out of the pads during the swing (obj z fell
    # 0.868->0.817 mid-rotation): halve the per-command step so chassis PD
    # acceleration stays below the pad friction budget
    _align_step = 0.006

    def _pre_align_cmds(self):
        # raise the shared mast to its cap before the chassis swings the held
        # object over the can (rim clearance + extra vertical IK margin);
        # ramped — a step command jerks the held object
        cur_p, cur_q = self.controller.get_ee_pose()
        cmds = []
        # (wrist roll removed: the "ejection" it worked around was really the
        # skip_plan gravity ratchet — with the hold fixed, rolling the pinch
        # only destabilizes the object)
        for k in range(1, 7):
            cmds.append((cur_p, cur_q, "lift_ctrl", {"target": 0.13 + 0.02 * k / 6}))
        for _ in range(4):
            cmds.append((cur_p, cur_q, "lift_ctrl", {"target": 0.15}))
        return cmds

    def __init__(self, robot, controller, task, cfg, *args, **kwargs):
        super().__init__(robot, controller, task, cfg, *args, **kwargs)
        self.place_obj = task.objects[self.skill_cfg["objects"][1]]

    def _mouth_root(self):
        from core.utils.usd_geom_utils import compute_bbox

        bbox = compute_bbox(self.place_obj.prim)
        return np.array([(bbox.min[0] + bbox.max[0]) / 2.0,
                         (bbox.min[1] + bbox.max[1]) / 2.0,
                         float(bbox.max[2])])

    def _align_target(self):
        pad_above = float(self.skill_cfg.get("pad_above", 0.14))
        return self._mouth_root() + np.array([0.0, 0.0, pad_above])

    def _fill_manip_cmds(self):
        from core.utils.usd_geom_utils import compute_bbox
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.transformations import get_relative_transform

        # nav+lift did the whole job: the held object rides at a fixed pose in
        # the base frame, and the chassis alignment has just put the can mouth
        # directly underneath it (band-vs-can annulus distances keep the
        # mismatch under the mouth half-width). No carry plan — just let go.
        root = get_prim_at_path(self.task.root_prim_path)
        T_root_base = get_relative_transform(
            get_prim_at_path(self.controller.reference_prim_path), root)
        obj_t = get_relative_transform(self.pick_obj.prim, root)[:3, 3]
        obj_b = (np.linalg.inv(T_root_base) @ np.append(obj_t, 1.0))[:3]
        bbox = compute_bbox(self.place_obj.prim)
        mouth = np.array([(bbox.min[0] + bbox.max[0]) / 2.0,
                          (bbox.min[1] + bbox.max[1]) / 2.0,
                          float(bbox.max[2])])

        cur_p, cur_q = self.controller.get_ee_pose()
        manip_list = []
        # adaptive radial trim: drop accuracy equals |held distance - can
        # aligned distance| (measured: 0.142 hold -> 1cm hit, 0.094 hold ->
        # 5.2cm rim miss). Nudge the EE by the measured horizontal delta
        # between the held object and the mouth before letting go.
        mouth_b = (np.linalg.inv(T_root_base) @ np.append(mouth, 1.0))[:3]
        delta = mouth_b - obj_b
        delta[1] = 0.0
        if 0.015 < float(np.linalg.norm(delta)) < 0.12:
            trim_p = np.asarray(cur_p, dtype=float) + delta
            manip_list.append((trim_p, cur_q, "close_gripper", {}))
            for _ in range(20):
                manip_list.append((trim_p, cur_q, "close_gripper", {}))
            cur_p = trim_p
            print(f"[DROPTRIM] delta={np.round(delta, 3).tolist()}", flush=True)
        for _ in range(int(self.skill_cfg.get("gripper_change_steps", 15))):
            manip_list.append((cur_p, cur_q, "open_gripper", {}))
        # fall + settle before the final pop triggers is_success
        for _ in range(30):
            manip_list.append((cur_p, cur_q, "open_gripper", {}))
        for _ in range(6):
            manip_list.append((cur_p, cur_q, "lift_ctrl", {"target": 0.13}))
        self.manip_list = manip_list
        self.gripper_cmd = "open_gripper"
        print(f"[SCRIPTPLACE] dropin obj_root={np.round(obj_t, 3).tolist()} "
              f"mouth={np.round(mouth, 3).tolist()} "
              f"xy_miss={round(float(np.linalg.norm(obj_t[:2] - mouth[:2])), 3)} "
              f"obj_b={np.round(obj_b, 3).tolist()}", flush=True)

    def is_success(self, th=0.0):
        from core.utils.usd_geom_utils import compute_bbox
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.transformations import get_relative_transform

        root = get_prim_at_path(self.task.root_prim_path)
        obj_t = get_relative_transform(self.pick_obj.prim, root)[:3, 3]
        bbox = compute_bbox(self.place_obj.prim)
        inside_xy = (bbox.min[0] + 0.005 < obj_t[0] < bbox.max[0] - 0.005) and (
            bbox.min[1] + 0.005 < obj_t[1] < bbox.max[1] - 0.005)
        below_rim = obj_t[2] < bbox.max[2] - 0.005
        above_bottom = obj_t[2] > bbox.min[2] - 0.03
        ok = bool(inside_xy and below_rim and above_bottom)
        print(f"[PLACESUCC] obj={np.round(obj_t, 3).tolist()} "
              f"can=[{np.round(np.asarray(bbox.min), 3).tolist()},"
              f"{np.round(np.asarray(bbox.max), 3).tolist()}] "
              f"inside_xy={inside_xy} below_rim={below_rim} -> {ok}", flush=True)
        return ok
