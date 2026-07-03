"""Isolated CuRobo IK sanity for the Pro v3 arm config: solve IK back to the
retract pose's own FK — guaranteed reachable, so failure = broken config."""
import os
import sys

import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

YML = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__),
    "../../scratchpad/InternDataEngine/workflows/simbox/curobo/src/curobo/content/configs/robot/aloha_mini_pro_left_arm.yml",
)

tensor_args = TensorDeviceType()
data = load_yaml(YML)
robot_cfg = RobotConfig.from_dict(data["robot_cfg"], tensor_args)
ik_cfg = IKSolverConfig.load_from_robot_config(
    robot_cfg, None, num_seeds=32, self_collision_check=True,
    self_collision_opt=True, tensor_args=tensor_args,
)
solver = IKSolver(ik_cfg)
retract = solver.get_retract_config().unsqueeze(0)
print("retract:", retract.tolist(), flush=True)
fk = solver.fk(retract)
print("fk ee_pos:", fk.ee_position.tolist(), "quat:", fk.ee_quaternion.tolist(), flush=True)
goal = Pose(fk.ee_position, fk.ee_quaternion)
res = solver.solve_batch(goal)
print("IK success:", res.success.tolist(), "pos_err:", res.position_error.tolist(),
      "rot_err:", res.rotation_error.tolist(), flush=True)

# also a slightly offset goal (2cm forward) with same orientation
goal2 = Pose(fk.ee_position + torch.tensor([[0.02, 0.0, 0.0]], device=fk.ee_position.device), fk.ee_quaternion)
res2 = solver.solve_batch(goal2)
print("IK2 success:", res2.success.tolist(), "pos_err:", res2.position_error.tolist(), flush=True)
print("DONE", flush=True)
