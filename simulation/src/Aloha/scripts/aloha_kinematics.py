#!/usr/bin/python3
"""
AlohaMini Full-Body Forward/Inverse Kinematics Library (pure numpy).

Parses the Aloha URDF to extract kinematic chains and provides:
  - AlohaMiniFK: Forward kinematics via homogeneous transformation matrices
  - AlohaMiniIK: Damped Least-Squares (DLS) numerical IK solver
    with null-space regularization to keep joints near a rest pose.
"""

import xml.etree.ElementTree as ET
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class JointInfo:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float = -np.pi
    upper: float = np.pi


def _rpy_to_rotation(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _axis_angle_to_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0],
    ])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def parse_urdf(urdf_path: str) -> Dict[str, JointInfo]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = {}
    for jelem in root.findall('joint'):
        name = jelem.get('name')
        jtype = jelem.get('type')
        parent = jelem.find('parent').get('link')
        child = jelem.find('child').get('link')

        origin = jelem.find('origin')
        if origin is not None:
            xyz = np.array([float(x) for x in origin.get('xyz', '0 0 0').split()])
            rpy = np.array([float(x) for x in origin.get('rpy', '0 0 0').split()])
        else:
            xyz = np.zeros(3)
            rpy = np.zeros(3)

        axis_elem = jelem.find('axis')
        if axis_elem is not None:
            axis = np.array([float(x) for x in axis_elem.get('xyz', '0 0 1').split()])
        else:
            axis = np.array([0.0, 0.0, 1.0])

        lower, upper = -np.pi, np.pi
        limit_elem = jelem.find('limit')
        if limit_elem is not None:
            lower = float(limit_elem.get('lower', str(-np.pi)))
            upper = float(limit_elem.get('upper', str(np.pi)))

        joints[name] = JointInfo(
            name=name, joint_type=jtype,
            parent_link=parent, child_link=child,
            origin_xyz=xyz, origin_rpy=rpy, axis=axis,
            lower=lower, upper=upper,
        )
    return joints


class AlohaMiniFK:
    LEFT_CHAIN = [
        'vertical_move', 'left_base',
        'left_joint1', 'left_joint2', 'left_joint3',
        'left_joint4', 'left_joint5', 'left_joint6',
    ]
    RIGHT_CHAIN = [
        'vertical_move', 'right_base',
        'right_joint1', 'right_joint2', 'right_joint3',
        'right_joint4', 'right_joint5', 'right_joint6',
    ]

    # Only the 6 revolute joints per arm (NOT vertical_move)
    LEFT_ARM_JOINTS = [
        'left_joint1', 'left_joint2', 'left_joint3',
        'left_joint4', 'left_joint5', 'left_joint6',
    ]
    RIGHT_ARM_JOINTS = [
        'right_joint1', 'right_joint2', 'right_joint3',
        'right_joint4', 'right_joint5', 'right_joint6',
    ]

    ALL_ACTIVE = [
        'vertical_move',
        'left_joint1', 'left_joint2', 'left_joint3',
        'left_joint4', 'left_joint5', 'left_joint6',
        'right_joint1', 'right_joint2', 'right_joint3',
        'right_joint4', 'right_joint5', 'right_joint6',
    ]

    REST_POSE = {
        'vertical_move': 0.05,
        'left_joint1': 0.0,
        'left_joint2': 0.3,
        'left_joint3': -0.3,
        'left_joint4': 0.0,
        'left_joint5': 0.0,
        'left_joint6': 0.0,
        'right_joint1': 0.0,
        'right_joint2': 0.3,
        'right_joint3': -0.3,
        'right_joint4': 0.0,
        'right_joint5': 0.0,
        'right_joint6': 0.0,
    }

    def __init__(self, urdf_path: str):
        self.joints = parse_urdf(urdf_path)
        self._limits = {}
        for name in self.ALL_ACTIVE:
            j = self.joints[name]
            self._limits[name] = (j.lower, j.upper)

    def joint_limits(self) -> Dict[str, Tuple[float, float]]:
        return dict(self._limits)

    def _joint_transform(self, joint_name: str, q: float = 0.0) -> np.ndarray:
        j = self.joints[joint_name]
        R_origin = _rpy_to_rotation(j.origin_rpy)
        T_origin = _make_transform(R_origin, j.origin_xyz)

        if j.joint_type == 'fixed':
            return T_origin
        elif j.joint_type in ('revolute', 'continuous'):
            R_joint = _axis_angle_to_rotation(j.axis, q)
            T_joint = _make_transform(R_joint, np.zeros(3))
            return T_origin @ T_joint
        elif j.joint_type == 'prismatic':
            axis_norm = j.axis / (np.linalg.norm(j.axis) + 1e-12)
            T_joint = _make_transform(np.eye(3), axis_norm * q)
            return T_origin @ T_joint
        else:
            return T_origin

    def fk_chain(self, chain: List[str], q_dict: Dict[str, float]) -> np.ndarray:
        T = np.eye(4)
        for jname in chain:
            q = q_dict.get(jname, 0.0)
            T = T @ self._joint_transform(jname, q)
        return T

    def fk_left(self, q_dict: Dict[str, float]) -> np.ndarray:
        return self.fk_chain(self.LEFT_CHAIN, q_dict)

    def fk_right(self, q_dict: Dict[str, float]) -> np.ndarray:
        return self.fk_chain(self.RIGHT_CHAIN, q_dict)

    def fk_both(self, q_dict: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        return self.fk_left(q_dict), self.fk_right(q_dict)


class AlohaMiniIK:
    """Damped Least-Squares IK with null-space regularization.

    Solves position-only IK for each 6-DOF arm independently.
    vertical_move is kept fixed (set externally) to avoid shared-joint conflict.
    """

    def __init__(self, fk: AlohaMiniFK,
                 damping: float = 0.01,
                 max_iter: int = 50,
                 tol: float = 5e-4,
                 alpha: float = 0.3,
                 null_space_gain: float = 0.5,
                 max_step_rad: float = 0.03,
                 delta_q: float = 1e-5):
        self.fk = fk
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.null_space_gain = null_space_gain
        self.max_step_rad = max_step_rad
        self.delta_q = delta_q

    def _numerical_jacobian(self, chain: List[str], active: List[str],
                            q_dict: Dict[str, float]) -> np.ndarray:
        T0 = self.fk.fk_chain(chain, q_dict)
        p0 = T0[:3, 3]
        n = len(active)
        J = np.zeros((3, n))
        for i, jname in enumerate(active):
            q_pert = dict(q_dict)
            q_pert[jname] = q_dict.get(jname, 0.0) + self.delta_q
            T_pert = self.fk.fk_chain(chain, q_pert)
            J[:, i] = (T_pert[:3, 3] - p0) / self.delta_q
        return J

    def solve_arm(self, chain: List[str], active: List[str],
                  target_pos: np.ndarray, q_init: Dict[str, float],
                  q_ref: Dict[str, float] = None,
                  ) -> Tuple[Dict[str, float], float, bool]:
        """Solve IK for a single arm (6-DOF revolute only, vertical_move fixed)."""
        if q_ref is None:
            q_ref = AlohaMiniFK.REST_POSE

        q = dict(q_init)
        limits = self.fk.joint_limits()
        lam2 = self.damping ** 2
        n = len(active)

        for _ in range(self.max_iter):
            T = self.fk.fk_chain(chain, q)
            p = T[:3, 3]
            err = target_pos - p
            dist = np.linalg.norm(err)
            if dist < self.tol:
                return q, dist, True

            J = self._numerical_jacobian(chain, active, q)

            JJT = J @ J.T + lam2 * np.eye(3)
            J_pinv = J.T @ np.linalg.inv(JJT)

            # Primary task
            dq_primary = J_pinv @ (self.alpha * err)

            # Null-space: push toward rest pose
            q_vec = np.array([q.get(jn, 0.0) for jn in active])
            q_ref_vec = np.array([q_ref.get(jn, 0.0) for jn in active])
            dq_null_raw = self.null_space_gain * (q_ref_vec - q_vec)
            N = np.eye(n) - J_pinv @ J
            dq_null = N @ dq_null_raw

            dq = dq_primary + dq_null

            # Clamp step
            max_abs = np.max(np.abs(dq))
            if max_abs > self.max_step_rad:
                dq *= self.max_step_rad / max_abs

            for i, jname in enumerate(active):
                q_new = q.get(jname, 0.0) + dq[i]
                lo, hi = limits.get(jname, (-np.pi, np.pi))
                q[jname] = float(np.clip(q_new, lo, hi))

        T = self.fk.fk_chain(chain, q)
        final_err = np.linalg.norm(target_pos - T[:3, 3])
        return q, final_err, final_err < self.tol

    def solve_left(self, target_pos, q_init, q_ref=None):
        """Solve left arm IK (6 revolute joints only)."""
        return self.solve_arm(
            AlohaMiniFK.LEFT_CHAIN, AlohaMiniFK.LEFT_ARM_JOINTS,
            target_pos, q_init, q_ref,
        )

    def solve_right(self, target_pos, q_init, q_ref=None):
        """Solve right arm IK (6 revolute joints only)."""
        return self.solve_arm(
            AlohaMiniFK.RIGHT_CHAIN, AlohaMiniFK.RIGHT_ARM_JOINTS,
            target_pos, q_init, q_ref,
        )

    def solve_both(self, left_target, right_target, q_init, q_ref=None):
        """Solve IK for both arms. vertical_move stays at q_init value."""
        q, l_err, l_ok = self.solve_left(left_target, q_init, q_ref)
        q, r_err, r_ok = self.solve_right(right_target, q, q_ref)
        return q, l_err, r_err, l_ok and r_ok


if __name__ == '__main__':
    import os
    import math
    urdf = os.path.join(os.path.dirname(__file__), '..', 'urdf', 'Aloha.urdf')
    urdf = os.path.abspath(urdf)
    print(f"Loading URDF: {urdf}")

    fk = AlohaMiniFK(urdf)
    q0 = dict(AlohaMiniFK.REST_POSE)

    T_left = fk.fk_left(q0)
    T_right = fk.fk_right(q0)
    print(f"Left  EE @ rest pose: {T_left[:3, 3]}")
    print(f"Right EE @ rest pose: {T_right[:3, 3]}")

    ik = AlohaMiniIK(fk)

    print("\nCircle trajectory test (20 points):")
    q = dict(q0)
    radius = 0.025
    for i in range(20):
        angle = 2.0 * math.pi * i / 20
        target_l = T_left[:3, 3] + np.array([
            radius * math.cos(angle), radius * math.sin(angle), 0.0])
        target_r = T_right[:3, 3] + np.array([
            radius * math.cos(-angle), radius * math.sin(-angle), 0.0])

        q, l_err, r_err, ok = ik.solve_both(target_l, target_r, q)
        print(f"  [{i:2d}] L_err={l_err:.5f} R_err={r_err:.5f} ok={ok}")

    print("\nFinal joint values:")
    for name in AlohaMiniFK.ALL_ACTIVE:
        lo, hi = fk.joint_limits().get(name, (-np.pi, np.pi))
        v = q.get(name, 0.0)
        at_limit = " [AT LIMIT!]" if abs(v - lo) < 0.01 or abs(v - hi) < 0.01 else ""
        print(f"  {name:20s}: {v:+.4f}  [{lo:+.2f}, {hi:+.2f}]{at_limit}")
