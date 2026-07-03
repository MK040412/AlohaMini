"""Which EE orientations are IK-feasible for the Pro arm at the pipeline's
failing goal position? Sweeps the 24 axis-aligned rotations; the feasible set
pins down the correct R_ee_graspnet for the Pro chain."""
import itertools
import sys

import numpy as np
import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

YML = sys.argv[1]
GOAL = [float(v) for v in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["-0.122", "0.028", "-0.298"])]

tensor_args = TensorDeviceType()
robot_cfg = RobotConfig.from_dict(load_yaml(YML)["robot_cfg"], tensor_args)
solver = IKSolver(IKSolverConfig.load_from_robot_config(robot_cfg, None, num_seeds=32, tensor_args=tensor_args))


def mat_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-6:
        # fallback for 180-degree rotations
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        q = np.zeros(4)
        q[i + 1] = np.sqrt(max(0.0, 1 + 2 * R[i, i] - (R[0, 0] + R[1, 1] + R[2, 2]))) / 2
        return np.array([0.0, *q[1:]])
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z])


axes = [np.array(v, float) for v in itertools.product([-1, 0, 1], repeat=3) if abs(sum(abs(np.array(v)))) == 1]
ok = []
for xa in axes:
    for ya in axes:
        za = np.cross(xa, ya)
        if abs(np.dot(xa, ya)) > 1e-6 or np.linalg.norm(za) < 0.5:
            continue
        R = np.stack([xa, ya, za], axis=1)
        q = mat_to_quat(R)
        pose = Pose(
            torch.tensor([GOAL], dtype=torch.float32, device=tensor_args.device),
            torch.tensor([q.tolist()], dtype=torch.float32, device=tensor_args.device),
        )
        res = solver.solve_batch(pose)
        if bool(res.success.reshape(-1)[0]):
            ok.append((np.round(R, 0).astype(int).tolist(), [round(v, 3) for v in q.tolist()]))
print(f"feasible orientations at {GOAL}: {len(ok)}", flush=True)
for R, q in ok:
    print("R=", R, "quat=", q, flush=True)
print("DONE", flush=True)
