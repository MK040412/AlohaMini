"""
**AlohaMini 2** agent (uid ``aloha_mini_2``): the official AlohaMini2 Pro —
dual-arm, 6-DOF per arm, with the roboninecom parallel gripper on each wrist.
Converted from upstream ``alohamini2pro.urdf`` (liyiteng/alohamini); the same
robot drives the Isaac / video2sim pipeline. URDF + meshes ship as the
``aloha_mini_2_urdf.zip`` GitHub Release asset (see maniskill/README.md).

Robot structure (verified against the URDF):
  * virtual mobile base: root_x_axis_joint, root_y_axis_joint (prismatic),
    root_z_rotation_joint (revolute), vertical_move (prismatic lift, [-0.3, 0.3]).
  * LEFT arm, 6 revolute:  left_shoulder_pan, left_shoulder_lift,
    left_elbow_flex, left_wrist_flex, left_wrist_yaw_joint, left_wrist_roll.
    palm / TCP link: left_Fixed_Jaw.
  * RIGHT arm, 6 revolute: same names with the ``right_`` prefix.
  * grippers: each is a parallel jaw made of TWO prismatic clamps. The URDF has
    NO <mimic> tags, so the coupling is done in the controller here:
      left  gripper: left_right_clamp  (leader, [0, +0.037], opens +x)
                     left_left_clamp   (follower = leader * -1, [-0.038, 0])
      right gripper: right_right_clamp (leader), right_left_clamp (follower*-1)
    One action dim per gripper drives the leader; the follower mirrors it, so the
    two jaws translate symmetrically. Action convention (matches the Std agent):
    0.0 = closed, 0.037 = open.
  * chest/back/front/left/right camera links are FIXED frames (kept as-is so they
    can host sensor cameras later).

Active-joint (qpos) order, 20 DOF — SAPIEN interleaves the two arms (verified):
  0-2    base:  root_x_axis, root_y_axis, root_z_rotation
  3      lift:  vertical_move
  4-15   arms interleaved: L_pan,R_pan, L_lift,R_lift, L_elbow,R_elbow,
         L_wflex,R_wflex, L_wyaw,R_wyaw, L_wroll,R_wroll
  16-17  left gripper:  left_right_clamp, left_left_clamp
  18-19  right gripper: right_right_clamp, right_left_clamp
"""

from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
)
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link

from .aloha_mini_1 import AlohaMini1
from .base_agent import AlohaMiniBaseAgent


# ---------------------------------------------------------------- 5 real cameras
# The five physical AM2 Pro cameras (H65V1: HFOV 90 deg / VFOV 73.7 deg, 640x480,
# fx=fy=320). Each camera is SELF-MOUNTED on its own URDF camera link (front_camera
# / back_camera / chest_camera / left_camera / right_camera), which already sits at
# the true physical mount pose. The mount ORIENTATION is NOT a guessed look_at — it
# is derived from the camera geometry itself:
#   * The camera-module STL sits at the link origin ([0,0,0]); its thinnest bbox
#     axis is Z (19 mm vs 37 mm) and its centroid is offset toward +Z, i.e. the
#     LENS / OPTICAL AXIS points along the link's +Z (confirmed by the Robo
#     cameras_am2pro.yaml note "module +z optical axis").
#   * SAPIEN's camera convention is (forward,right,up) = (+X,-Y,+Z). So the single
#     fixed rotation q_opt maps  cam+X (view) <- link+Z  and  cam+Z (up) <- link-Y
#     (the USD image-up = module -Y). q_opt = [0.5, 0.5, -0.5, 0.5] for ALL five.
# This makes each rendered camera look exactly where the real one does: front =
# forward/45-down, chest = straight ahead, back = REARWARD/down (not the table!),
# wrists = down the jaw toward the grasp. Verified by PNG (probe_cams.py).
#   res : (w, h) 4:3 to keep HFOV 90 with fov=VFOV(1.286)   fov : vertical, radians
_Q_OPT = [0.5, 0.5, -0.5, 0.5]   # link frame -> SAPIEN camera frame (see above)
_VFOV = 1.286                    # 73.7 deg; with 4:3 aspect gives HFOV 90 deg
# The camera-module STL housing extends to ~+18-19 mm along +Z (the optical axis).
# Placing the pinhole at the link origin would sit it INSIDE the housing and render
# its own lens bezel as a ring across the frame — so nudge it just past the front
# face, to where the real lens aperture actually is.
_LENS_FWD = 0.020                # metres along link +Z (optical axis)
_CAM5 = {
    "front_camera": dict(parent="front_camera", res=(512, 384)),
    "back_camera":  dict(parent="back_camera",  res=(512, 384)),
    "chest_camera": dict(parent="chest_camera", res=(512, 384)),
    "left_camera":  dict(parent="left_camera",  res=(384, 288)),
    "right_camera": dict(parent="right_camera", res=(384, 288)),
}


# Collision bits. Reuse the same indices the other AlohaMini variants use — a
# data-gen scene has a single robot, so sharing the bit indices across variants
# is harmless (and consistent with aloha_mini_1).
_LEFT_CLAMP_BIT = 27       # mutual collision of the LEFT gripper's two clamps
_RIGHT_CLAMP_BIT = 28      # mutual collision of the RIGHT gripper's two clamps
_LEFT_ARM_GRIP_BIT = 25    # left arm <-> left gripper spurious self-collision
_RIGHT_ARM_GRIP_BIT = 26   # right arm <-> right gripper spurious self-collision


@register_agent()
class AlohaMini2(AlohaMini1):
    """AlohaMini 2 Pro: dual-arm 6-DOF + parallel grippers (real AM2 Pro model)."""

    uid = "aloha_mini_2"
    urdf_path = str(
        Path.home()
        / ".maniskill/data/robots/aloha_mini_2/aloha_mini_2.urdf"
    )

    # Gripper clamp stroke of the LEADER joint (left/right_right_clamp).
    # 0 = closed (jaws meet), 0.037 = open (~88 mm outer aperture).
    GRIPPER_OPEN = 0.037
    GRIPPER_CLOSED = 0.0

    # Friction material on the four clamp links (the object-contacting jaws).
    urdf_config = dict(
        _materials=dict(
            gripper=dict(
                static_friction=2.0,
                dynamic_friction=2.0,
                restitution=0.0,
            ),
        ),
        link={
            "left_right_clamp_link": dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            "left_left_clamp_link": dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            "right_right_clamp_link": dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            "right_left_clamp_link": dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        },
    )

    # Keyframe qpos arrays follow SAPIEN's interleaved active-joint order (see the
    # module docstring): base(3), lift(1), arms interleaved L,R,... (12), then
    # left gripper (2 clamps), then right gripper (2 clamps).
    #
    # REST is a FOLDED / tucked pose: shoulder lifted (-1.5) + elbow flexed (2.5)
    # so each forearm points straight UP and the palm sits ~1.1 m above the base,
    # directly over it (verified by FK). This keeps the arms clear of the table
    # and each other at spawn — the zero pose instead extends the arms forward and
    # low, which collides with a tabletop.
    _FOLD_LIFT = -1.5
    _FOLD_ELBOW = 2.5
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                0.0, 0.0, 0.0,                 # base: x, y, rot
                0.0,                           # lift
                0.0, 0.0,                      # L_pan,  R_pan
                _FOLD_LIFT, _FOLD_LIFT,        # L_lift, R_lift
                _FOLD_ELBOW, _FOLD_ELBOW,      # L_elbow, R_elbow
                0.0, 0.0,                      # L_wflex, R_wflex
                0.0, 0.0,                      # L_wyaw,  R_wyaw
                0.0, 0.0,                      # L_wroll, R_wroll
                GRIPPER_CLOSED, GRIPPER_CLOSED,  # left gripper: right_clamp, left_clamp (closed)
                GRIPPER_CLOSED, GRIPPER_CLOSED,  # right gripper: right_clamp, left_clamp (closed)
            ]),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
        ready=Keyframe(
            qpos=np.array([
                0.0, 0.0, 0.0,
                0.0,                           # lift
                0.0, 0.0,
                _FOLD_LIFT, _FOLD_LIFT,
                _FOLD_ELBOW, _FOLD_ELBOW,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                # grippers OPEN: leader +0.037, follower -0.037 (mirror)
                GRIPPER_OPEN, -GRIPPER_OPEN,   # left gripper
                GRIPPER_OPEN, -GRIPPER_OPEN,   # right gripper
            ]),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
        zero=Keyframe(
            qpos=np.zeros(20, dtype=np.float64),
            pose=sapien.Pose(p=[0, 0, 0]),
        ),
    )

    # ---------------------------------------------------------------- cameras
    @property
    def _sensor_configs(self):
        """The five REAL AM2 Pro cameras (front/back/chest + left/right wrist),
        mounted on their URDF link frames. Replaces the parent's AM1-class
        cam_main/cam_left_wrist/cam_right_wrist set. Orientation `fix` per camera
        is tuned empirically against rendered PNGs (probe_cams.py)."""
        cams = []
        for uid, c in _CAM5.items():
            w, h = c["res"]
            cams.append(
                CameraConfig(
                    uid=uid,
                    # self-mounted on the camera link; nudged +Z (optical axis) past
                    # the housing to the lens aperture. q_opt rotates the link frame
                    # into SAPIEN's camera frame (view axis = link +Z).
                    pose=Pose.create_from_pq(p=[0.0, 0.0, _LENS_FWD], q=_Q_OPT),
                    width=w,
                    height=h,
                    fov=_VFOV,
                    near=0.01,
                    far=100,
                    entity_uid=c["parent"],
                )
            )
        return cams

    def __init__(self, *args, **kwargs):
        # 6 revolute joints per arm (the real AM2 Pro wrist has yaw + roll).
        self.left_arm_joint_names = [
            "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
            "left_wrist_flex", "left_wrist_yaw_joint", "left_wrist_roll",
        ]
        self.right_arm_joint_names = [
            "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
            "right_wrist_flex", "right_wrist_yaw_joint", "right_wrist_roll",
        ]
        self.arm_joint_names = self.left_arm_joint_names + self.right_arm_joint_names

        # Parallel-gripper clamps. LEADER first, FOLLOWER second: the follower is
        # coupled as follower = leader * -1 (see _create_gripper_controller). The
        # leader is the +axis clamp (limits [0, +0.037]); the follower is the
        # -axis clamp (limits [-0.038, 0]).
        self.left_gripper_joint_names = ["left_right_clamp", "left_left_clamp"]
        self.right_gripper_joint_names = ["right_right_clamp", "right_left_clamp"]

        # Gains: heavier-arm Pro tuning. Stable at
        # 100 Hz; stiff enough that the 6-DOF arm does not sag/drift under gravity.
        self.arm_stiffness = 2e3
        self.arm_damping = 4e2
        self.arm_force_limit = 300

        # Gripper prismatic clamps (URDF effort = 30 N).
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 30.0

        # Firmer virtual base + lift so the mobile gantry holds station under the
        # arm's reaction forces (fixed-base data-gen).
        self.base_pos_stiffness = 2e4
        self.base_pos_damping = 2e3
        self.base_pos_force_limit = 4000
        self.lift_stiffness = 4e3
        self.lift_damping = 6e2
        self.lift_force_limit = 400

        # Skip AlohaMini1.__init__ (it hardcodes the 5-DOF arm names and the
        # finger-joint names); go straight to the shared base-agent initializer.
        AlohaMiniBaseAgent.__init__(self, *args, **kwargs)

    def _create_base_pos_controller(self):
        return PDJointPosControllerConfig(
            self.base_joint_names, lower=None, upper=None,
            stiffness=self.base_pos_stiffness, damping=self.base_pos_damping,
            force_limit=self.base_pos_force_limit, normalize_action=False,
        )

    def _create_lift_pos_controller(self):
        return PDJointPosControllerConfig(
            self.lift_joint_names, lower=self.LIFT_LOWER, upper=self.LIFT_UPPER,
            stiffness=self.lift_stiffness, damping=self.lift_damping,
            force_limit=self.lift_force_limit, normalize_action=False,
        )

    def _create_gripper_controller(self, gripper_joint_names):
        """Single-action parallel-clamp controller.

        ``gripper_joint_names`` is (leader, follower). The leader is commanded in
        [GRIPPER_CLOSED, GRIPPER_OPEN]; the follower mirrors it with multiplier
        -1 so the two jaws translate symmetrically (the URDF carries no <mimic>
        tag, so the coupling lives here).
        """
        leader, follower = gripper_joint_names
        return PDJointPosMimicControllerConfig(
            gripper_joint_names,
            lower=self.GRIPPER_CLOSED,
            upper=self.GRIPPER_OPEN,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=False,
            mimic={follower: {"joint": leader, "multiplier": -1.0, "offset": 0.0}},
        )

    # ------------------------------------------------------------------ links
    def _after_init(self):
        # Base links + base collision bit (from AlohaMiniBaseAgent). NOTE: we do
        # NOT call AlohaMini1._after_init — that one binds finger1/finger2
        # links that do not exist on this model.
        self._after_init_base()

        links = self.robot.get_links()
        self.left_palm_link: Link = sapien_utils.get_obj_by_name(links, "left_Fixed_Jaw")
        self.right_palm_link: Link = sapien_utils.get_obj_by_name(links, "right_Fixed_Jaw")

        # The four moving clamp links (the object-contacting jaws).
        self.left_right_clamp_link: Link = sapien_utils.get_obj_by_name(links, "left_right_clamp_link")
        self.left_left_clamp_link: Link = sapien_utils.get_obj_by_name(links, "left_left_clamp_link")
        self.right_right_clamp_link: Link = sapien_utils.get_obj_by_name(links, "right_right_clamp_link")
        self.right_left_clamp_link: Link = sapien_utils.get_obj_by_name(links, "right_left_clamp_link")

        # Disable mutual collision between the two clamps of each gripper (they are
        # siblings and overlap when closed, so SAPIEN would otherwise fight them).
        for clamp in (self.left_right_clamp_link, self.left_left_clamp_link):
            clamp.set_collision_group_bit(group=2, bit_idx=_LEFT_CLAMP_BIT, bit=1)
        for clamp in (self.right_right_clamp_link, self.right_left_clamp_link):
            clamp.set_collision_group_bit(group=2, bit_idx=_RIGHT_CLAMP_BIT, bit=1)

        # Disable spurious arm<->gripper self-collision on each side: the parallel
        # gripper meshes overlap the wrist links they are rigidly mounted to, and
        # the vertical column / mounting bracket graze the upper arm in low-reach
        # configs — such contacts are non-physical and destabilise the sim. The
        # jaws still collide with manipulated objects (objects lack these bits).
        robot_links = {l.name: l for l in links}
        for side, bit in (("left", _LEFT_ARM_GRIP_BIT), ("right", _RIGHT_ARM_GRIP_BIT)):
            group = [
                "vertical_link", f"{side}_Base", f"{side}_Rotation_Pitch",
                f"{side}_Upper_Arm", f"{side}_Lower_Arm", f"{side}_Wrist_Pitch_Roll",
                f"{side}_wrist_yaw", f"{side}_Fixed_Jaw",
                f"{side}_right_clamp_link", f"{side}_left_clamp_link",
                f"{side}_camera",  # wrist camera mount sits at the tip; its mesh
                                   # overlaps the jaw/clamps -> spurious contacts.
            ]
            for name in group:
                lk = robot_links.get(name)
                if lk is not None:
                    lk.set_collision_group_bit(group=2, bit_idx=bit, bit=1)

    # -------------------------------------------------------------------- TCP
    # TCP = the palm link (left_Fixed_Jaw / right_Fixed_Jaw). The grasp point is a
    # short offset forward of this along the palm's local +Y; callers add that
    # offset as needed (the clamp link ORIGINS are auto-generated frames far from
    # the actual jaw geometry, so they are unreliable as a TCP reference).
    @property
    def tcp_pos(self):
        return self.left_palm_link.pose.p

    @property
    def tcp_pose(self):
        return self.left_palm_link.pose

    @property
    def tcp_pos_2(self):
        return self.right_palm_link.pose.p

    @property
    def tcp_pose_2(self):
        return self.right_palm_link.pose

    def get_left_ee_pose(self):
        return self.tcp_pose

    def get_right_ee_pose(self):
        return self.tcp_pose_2

    # ------------------------------------------------------------ grasp check
    def _check_single_arm_grasping(self, object: Actor, min_force=0.5, max_angle=110, arm_id=1):
        """Parallel-clamp grasp check.

        The jaw-separation axis is the palm's local +X in world (verified to point
        from the ``right_clamp`` jaw toward the ``left_clamp`` jaw). When an object
        is clamped it pushes the right_clamp jaw along -sep and the left_clamp jaw
        along +sep; we require a real contact force on both, roughly along those
        outward directions.
        """
        if arm_id == 1:
            rclamp, lclamp, palm = (
                self.left_right_clamp_link, self.left_left_clamp_link, self.left_palm_link,
            )
        elif arm_id == 2:
            rclamp, lclamp, palm = (
                self.right_right_clamp_link, self.right_left_clamp_link, self.right_palm_link,
            )
        else:
            raise ValueError(f"Invalid arm_id: {arm_id}. Must be 1 or 2.")

        r_forces = self.scene.get_pairwise_contact_forces(rclamp, object)
        l_forces = self.scene.get_pairwise_contact_forces(lclamp, object)
        rforce = torch.linalg.norm(r_forces, axis=1)
        lforce = torch.linalg.norm(l_forces, axis=1)

        # Palm local +X axis in world (jaw-separation direction).
        sep = palm.pose.to_transformation_matrix()[..., :3, 0]

        # right_clamp reaction points along -sep; left_clamp along +sep.
        rangle = common.compute_angle_between(-sep, r_forces)
        langle = common.compute_angle_between(sep, l_forces)
        rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)
        lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
        return torch.logical_and(rflag, lflag)
