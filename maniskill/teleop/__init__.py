"""
AlohaMini Teleoperation Module for ManiSkill3

This module provides IK-based teleoperation for the AlohaMini dual-arm robot,
supporting both keyboard and VR control interfaces.

Key Components:
- AlohaMiniKinematics: 2-link IK solver using actual URDF geometry
- TeleopController: Main controller bridging inputs to ManiSkill3
- KeyboardController: XLeRobot-compatible keyboard input
- VRWebSocketServer: WebXR VR controller input via WebSocket

Coordinate System (AlohaMini URDF):
- ee_y: Forward direction (usually negative, as arm folds backward at home)
- ee_z: Height (positive = up)

At home (j2=0, j3=0): ee_y=-0.0218m, ee_z=0.0324m

Usage:
    from teleop import TeleopController, TeleopConfig

    config = TeleopConfig()
    controller = TeleopController(config)

    # In your control loop:
    keys = pygame.key.get_pressed()
    controller.process_keyboard(keys)
    action = controller.compute_action()
    env.step(action)
"""

from .kinematics.aloha_mini_kinematics import AlohaMiniKinematics
from .inputs.base import ControlGoal, ControlMode, BaseInputProvider, ArmState
from .inputs.keyboard_controller import KeyboardController, KeyboardConfig
from .config import TeleopConfig
from .controller import TeleopController
from .utils import get_local_ip, generate_ssl_certificates, setup_ssl_context

# VR server is optional (requires websockets)
try:
    from .inputs.vr_ws_server import VRWebSocketServer
    _vr_available = True
except ImportError:
    VRWebSocketServer = None
    _vr_available = False

__all__ = [
    # Kinematics
    "AlohaMiniKinematics",
    # Input types
    "ControlGoal",
    "ControlMode",
    "BaseInputProvider",
    "ArmState",
    # Controllers
    "TeleopController",
    "KeyboardController",
    "KeyboardConfig",
    # Configuration
    "TeleopConfig",
    # Utilities
    "get_local_ip",
    "generate_ssl_certificates",
    "setup_ssl_context",
]

if _vr_available:
    __all__.append("VRWebSocketServer")
