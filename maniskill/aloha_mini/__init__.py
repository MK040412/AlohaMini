"""
AlohaMini Robot for ManiSkill3

This package provides the AlohaMini dual-arm mobile robot for use in ManiSkill3 environments.
"""

from .aloha_mini import AlohaMini, AlohaMiniFixed, AlohaMiniArmsOnly

# Import environments to register them
try:
    from .envs import AlohaMiniReplicaCADEnv, AlohaMiniEmptyEnv
    __all__ = [
        "AlohaMini",
        "AlohaMiniFixed",
        "AlohaMiniArmsOnly",
        "AlohaMiniReplicaCADEnv",
        "AlohaMiniEmptyEnv",
    ]
except ImportError:
    # Environment registration may fail if dependencies not available
    __all__ = ["AlohaMini", "AlohaMiniFixed", "AlohaMiniArmsOnly"]
