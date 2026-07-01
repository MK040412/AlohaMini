"""
AlohaMini manual / scripted data-generation package for ManiSkill.

Goal (InternData-A1 style): generate high-fidelity synthetic demonstrations for
the AlohaMini parallel-gripper robot in SIMPLE, GPU-parallel tabletop tasks
(not the heavy ReplicaCAD scene), then export them as LeRobot datasets for
generalist-policy pre-training.

Importing this package registers the AlohaMini task environments:
    - AlohaMiniPickCube-v1
    - AlohaMiniStackCube-v1     (scaffold)
    - AlohaMiniHandover-v1      (scaffold)

Control is JOINT-space (the SO-100 5-DOF arms have no EE/IK controller in
ManiSkill, like koch/so100), driven by scripted waypoint policies.
"""

from . import tasks  # noqa: F401  (registers the envs)
