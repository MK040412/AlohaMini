"""
Input providers for teleoperation.
"""

from .base import ControlGoal, ControlMode, BaseInputProvider, ArmState
from .keyboard_controller import KeyboardController, KeyboardConfig

# VR server is optional (requires websockets)
try:
    from .vr_ws_server import VRWebSocketServer
    from .vr_ws_server_with_mobile import VRWebSocketServerWithMobile, MobileBaseConfig
    __all__ = [
        "ControlGoal",
        "ControlMode",
        "BaseInputProvider",
        "ArmState",
        "KeyboardController",
        "KeyboardConfig",
        "VRWebSocketServer",
        "VRWebSocketServerWithMobile",
        "MobileBaseConfig",
    ]
except ImportError:
    __all__ = [
        "ControlGoal",
        "ControlMode",
        "BaseInputProvider",
        "ArmState",
        "KeyboardController",
        "KeyboardConfig",
    ]
