"""FK-sweep candidate retract configs for the Pro v3 arm to find the
forward-facing pose (Std's retract points the EE to y~-0.33; the donor chain's
joint sign convention differs, so the copied retract points backward)."""
import sys

import torch

from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

YML = sys.argv[1]
tensor_args = TensorDeviceType()
robot_cfg = RobotConfig.from_dict(load_yaml(YML)["robot_cfg"], tensor_args)
ik_cfg = IKSolverConfig.load_from_robot_config(robot_cfg, None, num_seeds=8, tensor_args=tensor_args)
solver = IKSolver(ik_cfg)

cands = [
    [0.0, 0.3, -0.3, 0.0, 0.0, 0.0],
    [0.0, -0.3, 0.3, 0.0, 0.0, 0.0],
    [0.0, -0.3, -0.3, 0.0, 0.0, 0.0],
    [0.0, 0.3, 0.3, 0.0, 0.0, 0.0],
    [0.0, -0.5, 0.5, 0.3, 0.0, 0.0],
    [0.0, 0.5, -0.5, -0.3, 0.0, 0.0],
]
for c in cands:
    q = torch.tensor([c], dtype=torch.float32, device=tensor_args.device)
    fk = solver.fk(q)
    p = [round(v, 3) for v in fk.ee_position[0].tolist()]
    r = [round(v, 3) for v in fk.ee_quaternion[0].tolist()]
    print(f"q={c} -> pos={p} quat={r}", flush=True)
print("DONE", flush=True)
