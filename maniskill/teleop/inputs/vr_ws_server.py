"""
VR WebSocket Server for AlohaMini Teleoperation

Receives controller data from WebXR-based VR interfaces.
Ported from XLeRobot XLeVR: /home/perelman/XLeRobot/XLeVR/xlevr/inputs/vr_ws_server.py

The server receives VR controller positions and rotations via WebSocket
and converts them to ControlGoal messages for the teleoperation controller.
"""

import asyncio
import json
import ssl
import math
import logging
import http.server
import threading
import os
from typing import Dict, Optional, Set, Any

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import websockets
except ImportError:
    websockets = None
    print("Warning: websockets not installed. VR server will not be available.")
    print("Install with: pip install websockets")

from .base import BaseInputProvider, ControlGoal, ControlMode
from ..config import TeleopConfig
from ..utils import generate_ssl_certificates, setup_ssl_context

logger = logging.getLogger(__name__)


class SimpleHTTPSHandler(http.server.BaseHTTPRequestHandler):
    """Simple HTTP request handler for serving VR web UI files (XLeRobot style)."""

    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        try:
            super().end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, ssl.SSLError):
            pass

    def do_OPTIONS(self):
        """Handle preflight CORS requests."""
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress HTTP request logging."""
        pass  # Disable logging to reduce noise

    def do_GET(self):
        """Handle GET requests for static files."""
        if self.path == '/' or self.path == '/index.html':
            self.serve_file('index.html', 'text/html')
        elif self.path.endswith('.css'):
            filename = self.path.lstrip('/')
            self.serve_file(filename, 'text/css')
        elif self.path.endswith('.js'):
            filename = self.path.lstrip('/')
            self.serve_file(filename, 'application/javascript')
        elif self.path.endswith('.ico'):
            self.send_error(404, "Not found")
        else:
            self.send_error(404, "Not found")

    def serve_file(self, filename: str, content_type: str):
        """Serve a file with the given content type."""
        try:
            web_root = getattr(self.server, 'web_root_path', '.')
            file_path = os.path.join(web_root, filename)

            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    content = f.read()

                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.end_headers()
                self.wfile.write(content)
            else:
                logger.error(f"File not found: {file_path}")
                self.send_error(404, f"File not found: {filename}")
        except Exception as e:
            logger.error(f"Error serving file {filename}: {e}")
            self.send_error(500, "Internal server error")


class VRControllerState:
    """
    State tracking for a VR controller.

    Uses XLeRobot-style control:
    - Tracks VR origin position for relative control
    - Accumulates robot state (ee_x, ee_y, joint1) across frames
    - Delta limiting prevents sudden movements
    """

    def __init__(self, hand: str):
        self.hand = hand
        self.grip_active = False
        self.trigger_active = False

        # XLeRobot-style position tracking
        self.prev_vr_position: Optional[np.ndarray] = None
        self.vr_origin: Optional[np.ndarray] = None  # VR position when grip started

        # Accumulated robot state (persistent across frames)
        # NOTE: These are initial fallback values. They should be overwritten by
        # initialize_from_arm() when the TeleopController is available.
        # For AlohaMiniKinematics, ee_x (URDF Y) can be NEGATIVE at home position!
        self.current_ee_x: float = -0.0218  # URDF Y (forward, can be negative!)
        self.current_ee_y: float = 0.0324   # URDF Z (height)
        self.current_joint1_deg: float = 0.0
        self.current_pitch: float = 0.0
        self.current_wrist_roll: float = 0.0

        # Initial values (saved when grip started)
        self.initial_ee_x: float = -0.0218
        self.initial_ee_y: float = 0.0324

        # Delta-based wrist tracking
        self.prev_wrist_roll: Optional[float] = None
        self.prev_wrist_flex: Optional[float] = None

        # Quaternion-based rotation tracking
        self.prev_quaternion: Optional[np.ndarray] = None
        self.origin_quaternion: Optional[np.ndarray] = None  # Quaternion when grip started

    def reset(self):
        """Reset controller state."""
        self.grip_active = False
        self.prev_vr_position = None
        self.vr_origin = None
        # Don't reset current_ee_x/y/joint1 - keep accumulated state
        self.prev_wrist_roll = None
        self.prev_wrist_flex = None
        self.prev_quaternion = None
        self.origin_quaternion = None

    def initialize_from_arm(self, arm):
        """Initialize accumulated state from current arm state."""
        self.current_ee_x = arm.ee_x
        self.current_ee_y = arm.ee_y
        self.current_joint1_deg = arm.joint1_deg
        self.current_pitch = arm.pitch
        self.current_wrist_roll = arm.joint5_deg
        # Save initial values for relative control
        self.initial_ee_x = arm.ee_x
        self.initial_ee_y = arm.ee_y


class VRWebSocketServer(BaseInputProvider):
    """
    WebSocket server for VR controller input.

    Receives position and rotation data from WebXR controllers
    and generates ControlGoal messages for arm control.
    """

    def __init__(self, command_queue: asyncio.Queue, config: TeleopConfig, teleop_controller=None):
        super().__init__(command_queue)
        self.config = config
        self.clients: Set = set()
        self.server = None
        self.teleop_controller = teleop_controller  # Reference to get current arm state

        # HTTPS file server (XLeRobot style - integrated)
        self.https_server = None
        self.https_thread = None

        # Controller states
        self.left_controller = VRControllerState("left")
        self.right_controller = VRControllerState("right")

    async def start(self):
        """Start both HTTPS file server and WebSocket server (XLeRobot style)."""
        if websockets is None:
            logger.error("websockets package not installed")
            return

        if not self.config.enable_vr:
            logger.info("VR WebSocket server disabled in configuration")
            return

        # Ensure SSL certificates exist
        if not self.config.ssl_files_exist:
            logger.info("Generating SSL certificates...")
            success, message = generate_ssl_certificates(
                self.config.ssl_dir,
                "cert.pem",
                "key.pem",
            )
            if not success:
                logger.error(f"Failed to generate SSL certificates: {message}")
                return

        # Start HTTPS file server for VR web UI
        self._start_https_server()

        # Setup SSL context for WebSocket
        ssl_context = setup_ssl_context(self.config.certfile, self.config.keyfile)
        if ssl_context is None:
            logger.error("Failed to setup SSL context")
            return

        host = self.config.host_ip
        port = self.config.websocket_port

        try:
            self.server = await websockets.serve(
                self.websocket_handler,
                host,
                port,
                ssl=ssl_context,
            )
            self.is_running = True

            # Print connection info (XLeRobot style)
            from ..utils import get_local_ip
            local_ip = get_local_ip()
            logger.info(f"VR WebSocket server running on wss://{host}:{port}")
            print(f"\n{'='*60}")
            print("VR TELEOPERATION SERVER READY")
            print(f"{'='*60}")
            print(f"Open your VR headset browser and navigate to:")
            print(f"  https://{local_ip}:{self.config.https_port}")
            print(f"WebSocket: wss://{local_ip}:{port}")
            print(f"{'='*60}\n")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")

    def _start_https_server(self):
        """Start HTTPS file server for VR web UI (XLeRobot style)."""
        try:
            # Create HTTP server
            self.https_server = http.server.HTTPServer(
                (self.config.host_ip, self.config.https_port),
                SimpleHTTPSHandler
            )
            self.https_server.web_root_path = self.config.web_ui_path

            # Setup SSL context
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(self.config.certfile, self.config.keyfile)
            self.https_server.socket = context.wrap_socket(
                self.https_server.socket, server_side=True
            )

            # Start in background thread
            self.https_thread = threading.Thread(
                target=self.https_server.serve_forever,
                daemon=True
            )
            self.https_thread.start()

            logger.info(f"HTTPS file server started on https://{self.config.host_ip}:{self.config.https_port}")
        except Exception as e:
            logger.error(f"Failed to start HTTPS server: {e}")

    def _stop_https_server(self):
        """Stop the HTTPS file server."""
        if self.https_server:
            self.https_server.shutdown()
            if self.https_thread and self.https_thread.is_alive():
                self.https_thread.join(timeout=5)
            logger.info("HTTPS file server stopped")

    async def stop(self):
        """Stop both HTTPS file server and WebSocket server."""
        self.is_running = False

        # Stop HTTPS file server
        self._stop_https_server()

        # Stop WebSocket server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("VR WebSocket server stopped")

    async def websocket_handler(self, websocket, path=None):
        """Handle WebSocket connections from VR controllers."""
        client_address = websocket.remote_address
        logger.info(f"VR client connected: {client_address}")
        self.clients.add(websocket)

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_controller_data(data)
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message")
                except Exception as e:
                    logger.error(f"Error processing VR data: {e}")

        except Exception as e:
            logger.warning(f"VR client disconnected: {e}")
        finally:
            self.clients.discard(websocket)
            # Reset controller states
            self.left_controller.reset()
            self.right_controller.reset()
            logger.info(f"VR client {client_address} cleanup complete")

    async def process_controller_data(self, data: Dict):
        """Process incoming VR controller data (XLeRobot style)."""
        # Log thumbstick activity if any
        has_activity = False
        for hand_key in ['leftController', 'rightController']:
            if hand_key in data:
                controller_data = data[hand_key]
                thumbstick = controller_data.get('thumbstick', {})
                tx = thumbstick.get('x', 0)
                ty = thumbstick.get('y', 0)
                if abs(tx) > 0.1 or abs(ty) > 0.1:
                    has_activity = True
                    hand_name = hand_key.replace('Controller', '').upper()
                    logger.debug(f"[{hand_name}] Thumbstick: x={tx:.2f}, y={ty:.2f}")

        # Process left controller
        if 'leftController' in data:
            await self.process_single_controller('left', data['leftController'])

        # Process right controller
        if 'rightController' in data:
            await self.process_single_controller('right', data['rightController'])

    async def process_single_controller(self, hand: str, data: Dict):
        """
        Process data for a single controller using delta-based control (Vector Wang style).

        Delta control calculates frame-by-frame position changes and accumulates them,
        rather than using absolute position relative to an origin. This provides:
        - Robustness to VR tracking hiccups
        - Smooth control with delta limiting
        - No sudden jumps when VR origin resets
        """
        position = data.get('position', {})
        rotation = data.get('rotation', {})
        quaternion = data.get('quaternion', {})
        grip_active = data.get('gripActive', False)
        trigger = data.get('trigger', 0)
        thumbstick = data.get('thumbstick', {'x': 0, 'y': 0})

        controller = self.left_controller if hand == 'left' else self.right_controller

        # Handle trigger for gripper control
        trigger_active = trigger > 0.5
        if trigger_active != controller.trigger_active:
            controller.trigger_active = trigger_active
            gripper_goal = ControlGoal(
                arm=hand,
                gripper_closed=trigger_active,
                metadata={"source": "vr_trigger", "trigger": trigger}
            )
            await self.send_goal(gripper_goal)
            logger.info(f"Gripper {hand}: {'CLOSED' if trigger_active else 'OPEN'}")

        # Process position data with delta control
        if position and all(k in position for k in ['x', 'y', 'z']):
            position_array = np.array([
                position.get('x', 0),
                position.get('y', 0),
                position.get('z', 0)
            ])

            # Initialize on first frame
            if controller.prev_vr_position is None:
                controller.prev_vr_position = position_array.copy()

                # Initialize robot state from teleop_controller
                if self.teleop_controller is not None:
                    arm = self.teleop_controller.left_arm if hand == 'left' else self.teleop_controller.right_arm
                    controller.initialize_from_arm(arm)
                    logger.info(f"VR {hand} initialized: ee_x={controller.current_ee_x:.4f}, "
                               f"ee_y={controller.current_ee_y:.4f}, joint1={controller.current_joint1_deg:.1f}°")

                # Initialize quaternion tracking
                if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w']):
                    controller.prev_quaternion = np.array([
                        quaternion['x'], quaternion['y'],
                        quaternion['z'], quaternion['w']
                    ])
                return  # Skip first frame to establish baseline

            # Calculate delta from previous frame (Vector Wang style)
            delta_vr = position_array - controller.prev_vr_position
            controller.prev_vr_position = position_array.copy()  # Update for next frame

            # Per-axis scaling (Vector Wang style: 220, 70, 70 then * 0.01)
            # VR X (left/right) → shoulder pan: scale = 2.2 (220 * 0.01)
            # VR Y (up/down) → ee_y: scale = 0.7 (70 * 0.01)
            # VR Z (forward/back) → ee_x: scale = 0.7 (70 * 0.01)
            pan_scale = 2.2    # For shoulder pan (degrees per VR meter * factor)
            pos_scale = 0.7    # For ee_x, ee_y

            delta_pan = delta_vr[0] * pan_scale * 100.0  # Scale to degrees (100 deg/m)
            delta_y = delta_vr[1] * pos_scale
            delta_z = delta_vr[2] * pos_scale

            # Limit delta values to prevent sudden movements (Vector Wang: 0.01m, 8°)
            delta_limit = 0.01  # meters
            angle_limit = 8.0   # degrees
            delta_pan = np.clip(delta_pan, -angle_limit, angle_limit)
            delta_y = np.clip(delta_y, -delta_limit, delta_limit)
            delta_z = np.clip(delta_z, -delta_limit, delta_limit)

            # Apply delta to accumulated robot state
            # For AlohaMiniKinematics:
            #   ee_x = URDF Y (forward direction, NEGATIVE at home = -0.0218)
            #   ee_y = URDF Z (height, POSITIVE at home = 0.0324)
            #
            # WebXR coordinates: forward = -Z, up = +Y, right = +X
            # When VR moves forward: delta_z < 0
            # We want ee_x to DECREASE (more negative = extend arm forward)
            # So: ee_x += delta_z (NOT -delta_z!)
            controller.current_ee_x += delta_z  # Forward in VR → extend arm (more negative ee_x)
            controller.current_ee_y += delta_y  # Up in VR → raise arm (more positive ee_y)

            # VR X (right) → joint1 (shoulder pan)
            # Both arms: moving VR hand right should rotate arm in same visual direction
            # This provides intuitive mirrored control for both hands
            controller.current_joint1_deg += delta_pan

            # Clamp to workspace limits
            # NOTE: For AlohaMiniKinematics, ee_x (URDF Y) can be NEGATIVE!
            # The workspace is a circle in Y-Z plane with radius ~0.25m
            # Home position is ee_x=-0.0218, ee_y=0.0324
            controller.current_ee_x = np.clip(controller.current_ee_x, -0.25, 0.25)  # URDF Y (forward, can be negative!)
            controller.current_ee_y = np.clip(controller.current_ee_y, 0.0, 0.25)    # URDF Z (height, always positive)
            controller.current_joint1_deg = np.clip(controller.current_joint1_deg, -90.0, 90.0)

            # Handle wrist angles with delta control
            wrist_roll_deg = controller.current_wrist_roll
            pitch_deg = controller.current_pitch

            if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w']):
                current_quat = np.array([
                    quaternion['x'], quaternion['y'],
                    quaternion['z'], quaternion['w']
                ])

                if controller.prev_quaternion is not None:
                    # Extract relative rotation from previous frame
                    roll_delta = self._extract_roll(current_quat, controller.prev_quaternion)
                    pitch_delta = self._extract_pitch(current_quat, controller.prev_quaternion)

                    # Apply angle scaling and limiting
                    angle_scale = 4.0  # Vector Wang style
                    roll_delta = np.clip(roll_delta * angle_scale, -angle_limit, angle_limit)
                    pitch_delta = np.clip(pitch_delta * angle_scale, -angle_limit, angle_limit)

                    # Accumulate wrist angles
                    controller.current_wrist_roll += roll_delta
                    controller.current_pitch += pitch_delta

                    # No limits - allow full rotation
                    # controller.current_wrist_roll = np.clip(controller.current_wrist_roll, -180.0, 180.0)
                    # controller.current_pitch = np.clip(controller.current_pitch, -180.0, 180.0)

                    wrist_roll_deg = controller.current_wrist_roll
                    pitch_deg = controller.current_pitch

                controller.prev_quaternion = current_quat.copy()

            # Create control goal with accumulated values
            goal = ControlGoal(
                arm=hand,
                mode=ControlMode.POSITION_CONTROL,
                ee_x=controller.current_ee_x,
                ee_y=controller.current_ee_y,
                joint1_deg=controller.current_joint1_deg,
                wrist_roll_deg=wrist_roll_deg,
                wrist_flex_deg=pitch_deg,
                metadata={
                    "source": "vr_delta_control",
                    "delta_vr": delta_vr.tolist(),
                    "delta_applied": [delta_z, delta_y, delta_pan],  # Fixed: was [-delta_z, ...]
                    "trigger": trigger,
                    "thumbstick": thumbstick
                }
            )
            await self.send_goal(goal)

    def _euler_to_quaternion(self, euler_deg: Dict[str, float]) -> np.ndarray:
        """Convert Euler angles in degrees to quaternion [x, y, z, w]."""
        if not euler_deg:
            return np.array([0, 0, 0, 1])

        euler_rad = [
            math.radians(euler_deg.get('x', 0)),
            math.radians(euler_deg.get('y', 0)),
            math.radians(euler_deg.get('z', 0))
        ]
        rotation = R.from_euler('xyz', euler_rad)
        return rotation.as_quat()

    def _extract_roll(self, current_quat: np.ndarray, origin_quat: np.ndarray) -> float:
        """Extract roll rotation around Z-axis (wrist rotation).

        For AlohaMini:
        - Positive roll = rotate wrist clockwise (when viewed from above)
        - Maps to joint5 (wrist roll)
        """
        try:
            origin_rotation = R.from_quat(origin_quat)
            current_rotation = R.from_quat(current_quat)
            relative_rotation = current_rotation * origin_rotation.inv()

            rotvec = relative_rotation.as_rotvec()
            # WebXR: positive Z rotation = CCW when viewed from above
            # Robot: positive joint5 = CW when viewed from above (typically)
            # So we use positive sign (was negative before)
            z_rotation_deg = np.degrees(rotvec[2])
            return z_rotation_deg
        except Exception:
            return 0.0

    def _extract_pitch(self, current_quat: np.ndarray, origin_quat: np.ndarray) -> float:
        """Extract pitch rotation around X-axis (wrist flex).

        For AlohaMini:
        - Positive pitch = tilt wrist down
        - Affects joint4 calculation through pitch compensation
        """
        try:
            origin_rotation = R.from_quat(origin_quat)
            current_rotation = R.from_quat(current_quat)
            relative_rotation = current_rotation * origin_rotation.inv()

            rotvec = relative_rotation.as_rotvec()
            # WebXR: positive X rotation = tilt down (pitch down)
            # Robot: positive pitch = typically tilt down
            # Keep positive sign
            x_rotation_deg = np.degrees(rotvec[0])
            return x_rotation_deg
        except Exception:
            return 0.0
