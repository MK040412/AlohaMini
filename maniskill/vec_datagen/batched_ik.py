"""Batched damped-least-squares IK for the AM2 Pro left arm on GPU.

Solves the 6 arm joints (shoulder_pan..wrist_roll) for a target TCP position +
optional approach/jaw orientation, holding the 4 base/lift joints fixed. All
N environments solved in parallel via pytorch_kinematics batched FK/Jacobian.
Replaces the single-env numpy DLS (solve_arm_ik_full_pose) for physx_gpu batch.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import pytorch_kinematics as pk

_URDF = str(Path.home() / ".maniskill/data/robots/aloha_mini_2/aloha_mini_2.urdf")
# chain order: root_x, root_y, root_z, vertical_move, then the 6 arm joints
_ARM = slice(4, 10)  # arm columns in the 10-DOF chain


class BatchedArmIK:
    def __init__(self, tcp_link="left_Fixed_Jaw", device="cuda", dtype=torch.float32):
        self.device, self.dtype = device, dtype
        chain = pk.build_serial_chain_from_urdf(open(_URDF, "rb").read(), tcp_link)
        self.chain = chain.to(dtype=dtype, device=device)
        self.jn = chain.get_joint_parameter_names()
        self.ndof = len(self.jn)               # 10
        # arm joint limits (columns 4:10)
        lo, hi = chain.get_joint_limits()
        self.lo = torch.as_tensor(lo, dtype=dtype, device=device)[_ARM]
        self.hi = torch.as_tensor(hi, dtype=dtype, device=device)[_ARM]

    def fk(self, q10: torch.Tensor) -> torch.Tensor:
        """q10: (N,10) full chain config -> (N,4,4) TCP world (base-frame) transform."""
        return self.chain.forward_kinematics(q10).get_matrix()

    def solve(self, base_lift: torch.Tensor, target_pos: torch.Tensor,
              approach_dir: torch.Tensor | None = None,
              jaw_dir: torch.Tensor | None = None,
              q_arm0: torch.Tensor | None = None,
              iters=60, damping=0.05, ori_w=0.3, max_step=0.2):
        """Batched DLS IK.
        base_lift: (N,4) fixed base(x,y,yaw)+lift.  target_pos: (N,3) world TCP target.
        approach_dir/jaw_dir: (N,3) optional desired palm axes (chain-frame).
        Returns (q_arm (N,6), pos_err (N,), converged mask (N,)).
        """
        N = base_lift.shape[0]
        dev, dt = self.device, self.dtype
        base_lift = base_lift.to(dev, dt)
        target_pos = target_pos.to(dev, dt)
        q_arm = (torch.zeros(N, 6, device=dev, dtype=dt) if q_arm0 is None
                 else q_arm0.to(dev, dt).clone())
        # seed shoulder_lift a bit up (col 1 of the arm) like the numpy version
        if q_arm0 is None:
            q_arm[:, 1] = 1.0
        ap = None if approach_dir is None else torch.nn.functional.normalize(
            approach_dir.to(dev, dt), dim=-1)
        jw = None if jaw_dir is None else torch.nn.functional.normalize(
            jaw_dir.to(dev, dt), dim=-1)
        for _ in range(iters):
            q10 = torch.cat([base_lift, q_arm], dim=1)
            J = self.chain.jacobian(q10)            # (N,6,10) spatial [v; w]
            Ja = J[:, :, _ARM]                       # (N,6,6) arm cols
            M = self.fk(q10)                         # (N,4,4)
            pos = M[:, :3, 3]
            e_pos = target_pos - pos                 # (N,3)
            res = [e_pos]
            if jw is not None:
                # palm x-axis = jaw separation axis; sign-free line
                jcur = M[:, :3, 0]
                flip = (jcur * jw).sum(-1, keepdim=True) < 0
                jcur = torch.where(flip, -jcur, jcur)
                res.append(ori_w * (jw - jcur))
            if ap is not None:
                acur = M[:, :3, 1]                   # palm y = approach axis
                res.append(ori_w * (ap - acur))
            e = torch.cat(res, dim=1)                # (N, 3 or 6 or 9)
            Juse = Ja[:, :e.shape[1], :]
            # DLS: dq = J^T (J J^T + l^2 I)^-1 e
            JJt = Juse @ Juse.transpose(1, 2)
            lam = damping ** 2 * torch.eye(JJt.shape[1], device=dev, dtype=dt)
            dq = Juse.transpose(1, 2) @ torch.linalg.solve(JJt + lam, e.unsqueeze(-1))
            dq = dq.squeeze(-1).clamp(-max_step, max_step)
            q_arm = torch.clamp(q_arm + dq, self.lo, self.hi)
        q10 = torch.cat([base_lift, q_arm], dim=1)
        pos_err = (target_pos - self.fk(q10)[:, :3, 3]).norm(dim=-1)
        return q_arm, pos_err, pos_err < 0.01


    def solve_wb(self, yaw_lift, target_pos, jaw_dir=None, iters=80,
                 damping=0.08, ori_w=0.0, max_step=0.15):
        """Whole-body position IK: solve root_x, root_y + 6 arm joints so the
        base repositions to bring target into reach. yaw_lift: (N,2) fixed
        [root_z_yaw, vertical_move]. Returns (base_xy (N,2), q_arm (N,6),
        err (N,), conv (N,))."""
        N = target_pos.shape[0]; dev, dt = self.device, self.dtype
        yaw_lift = yaw_lift.to(dev, dt); target_pos = target_pos.to(dev, dt)
        cols = [0, 1, 4, 5, 6, 7, 8, 9]  # root_x, root_y, arm6
        base_xy = torch.zeros(N, 2, device=dev, dtype=dt)
        q_arm = torch.zeros(N, 6, device=dev, dtype=dt); q_arm[:, 1] = 1.0
        lo = torch.cat([torch.full((2,), -0.6, device=dev, dtype=dt), self.lo])
        hi = torch.cat([torch.full((2,), 0.6, device=dev, dtype=dt), self.hi])
        def full(bx, qa):
            q = torch.zeros(N, 10, device=dev, dtype=dt)
            q[:, 0:2] = bx; q[:, 2] = yaw_lift[:, 0]; q[:, 3] = yaw_lift[:, 1]; q[:, 4:] = qa
            return q
        for _ in range(iters):
            q10 = full(base_xy, q_arm)
            J = self.chain.jacobian(q10)[:, :3, :][:, :, cols]  # (N,3,8) position rows
            e = (target_pos - self.fk(q10)[:, :3, 3]).unsqueeze(-1)
            JJt = J @ J.transpose(1, 2)
            lam = damping ** 2 * torch.eye(3, device=dev, dtype=dt)
            dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt + lam, e)).squeeze(-1).clamp(-max_step, max_step)
            x = torch.cat([base_xy, q_arm], 1) + dq
            x = torch.clamp(x, lo, hi)
            base_xy, q_arm = x[:, :2], x[:, 2:]
        q10 = full(base_xy, q_arm)
        err = (target_pos - self.fk(q10)[:, :3, 3]).norm(dim=-1)
        return base_xy, q_arm, err, err < 0.01


if __name__ == "__main__":
    # FK -> IK roundtrip: pick random reachable arm configs, FK to a target,
    # solve IK back from a different seed, check we recover the TCP.
    torch.manual_seed(0)
    ik = BatchedArmIK()
    N = 256
    base_lift = torch.zeros(N, 4, device="cuda")
    q_true = torch.rand(N, 6, device="cuda") * (ik.hi - ik.lo) + ik.lo
    q_true *= 0.4  # keep near home / reachable
    M = ik.fk(torch.cat([base_lift, q_true], 1))
    tgt = M[:, :3, 3]
    import time
    torch.cuda.synchronize(); t0 = time.time()
    q_sol, err, conv = ik.solve(base_lift, tgt, iters=60)
    torch.cuda.synchronize(); dt = time.time() - t0
    print(f"[IK] N={N} solved in {dt*1000:.0f} ms ({N/dt:.0f} solves/s)")
    print(f"[IK] pos err mm: median {err.median()*1000:.2f}  "
          f"p90 {err.quantile(0.9)*1000:.2f}  converged {conv.float().mean()*100:.0f}%")
