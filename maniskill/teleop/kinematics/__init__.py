"""
Kinematics module for AlohaMini robot.
"""

from .aloha_mini_kinematics import AlohaMiniKinematics
from .so101_kinematics import SO101Kinematics  # Legacy, for reference

__all__ = ["AlohaMiniKinematics", "SO101Kinematics"]
