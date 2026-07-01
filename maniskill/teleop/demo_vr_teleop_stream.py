#!/usr/bin/env python3
"""
AlohaMini VR Teleoperation with Camera Streaming

This demo streams the robot's first-person camera view to a web browser
while allowing VR teleoperation from WebXR controllers.

Uses TeleopController for proper end-effector control with IK.

Usage:
    python demo_vr_teleop_stream.py
    python demo_vr_teleop_stream.py --gpu

Then open in VR headset browser:
    https://<your-ip>:8443
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import ssl
import sys
import threading
import time
import http.server
from pathlib import Path
from typing import Set, Optional

import numpy as np
from PIL import Image

try:
    import websockets
except ImportError:
    print("Error: websockets not installed. Install with: pip install websockets")
    sys.exit(1)

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Clear module caches
for m in list(sys.modules.keys()):
    if 'aloha_mini' in m or 'mani_skill.agents' in m:
        del sys.modules[m]
# Import agents
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.aloha_mini import AlohaMiniSO100V2

import gymnasium as gym
import mani_skill.envs

# Import teleop controller
from teleop.controller import TeleopController
from teleop.config import TeleopConfig
from teleop.inputs.base import ControlGoal, ControlMode
from teleop.inputs.vr_ws_server import VRControllerState

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CameraStreamServer:
    """WebSocket server for streaming camera images and receiving VR control data."""

    def __init__(self, teleop_controller: TeleopController, host='0.0.0.0', ws_port=8442, https_port=8443):
        self.host = host
        self.ws_port = ws_port
        self.https_port = https_port
        self.clients: Set = set()
        self.current_frame: Optional[bytes] = None
        self.is_running = False

        # Teleop controller reference
        self.teleop = teleop_controller

        # VR controller states for arm tracking (delta-based control)
        self.left_controller = VRControllerState("left")
        self.right_controller = VRControllerState("right")

        # SSL paths
        self.ssl_dir = Path(__file__).parent
        self.certfile = self.ssl_dir / 'cert.pem'
        self.keyfile = self.ssl_dir / 'key.pem'

        # Get workspace limits from kinematics (SO100 vs AlohaMini)
        if hasattr(teleop_controller.kinematics, 'initial_ee_x'):
            # SO100 variant - different coordinate system
            self.ee_x_min = 0.20
            self.ee_x_max = 0.50
            self.ee_y_min = 0.10
            self.ee_y_max = 0.45
        else:
            # AlohaMini variant
            self.ee_x_min = -0.25
            self.ee_x_max = 0.25
            self.ee_y_min = 0.0
            self.ee_y_max = 0.25

    def update_frame(self, rgb_array: np.ndarray, camera_name: str = 'main'):
        """Update a camera frame to stream."""
        # Convert to JPEG for faster streaming
        img = Image.fromarray(rgb_array)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=70)
        frame_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        if camera_name == 'main':
            self.current_frame = frame_data
        elif not hasattr(self, 'wrist_frames'):
            self.wrist_frames = {}
            self.wrist_frames[camera_name] = frame_data
        else:
            self.wrist_frames[camera_name] = frame_data

    async def websocket_handler(self, websocket, path=None):
        """Handle WebSocket connections."""
        logger.info(f"Client connected: {websocket.remote_address}")
        self.clients.add(websocket)

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    self.process_vr_data(data)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected")

    def process_vr_data(self, data: dict):
        """Process incoming VR control data with arm tracking (delta-based control)."""
        # Process each controller
        if 'leftController' in data:
            self._process_single_controller('left', data['leftController'])

        if 'rightController' in data:
            self._process_single_controller('right', data['rightController'])

    def _process_single_controller(self, hand: str, data: dict):
        """Process single controller with delta-based arm tracking."""
        position = data.get('position', {})
        quaternion = data.get('quaternion', {})
        grip_active = data.get('gripActive', False)
        trigger = data.get('trigger', 0)
        thumbstick = data.get('thumbstick', {'x': 0, 'y': 0})

        controller = self.left_controller if hand == 'left' else self.right_controller

        # --- THUMBSTICK: Base/Lift control ---
        deadzone = 0.1
        tx = thumbstick.get('x', 0)
        ty = thumbstick.get('y', 0)

        if abs(tx) < deadzone:
            tx = 0
        if abs(ty) < deadzone:
            ty = 0

        if hand == 'left':
            # Left thumbstick: base translation
            max_speed = 0.3
            self.teleop.base_velocity[0] = -tx * max_speed
            self.teleop.base_velocity[1] = ty * max_speed  # Forward/back inverted
        else:
            # Right thumbstick: rotation only
            max_angular = 0.5
            self.teleop.base_velocity[2] = -tx * max_angular

        # --- BUTTONS: Lift control (A/B buttons on both controllers) ---
        buttons = data.get('buttons', {})
        # Debug: log button state if any pressed
        if buttons:
            logger.debug(f"Buttons {hand}: {buttons}")

        # A button = lift up, B button = lift down (works on both controllers)
        lift_step = 0.005  # Step size per frame
        lift_min = -0.15  # Min lift position (extended range)
        lift_max = 0.6    # Max lift position (URDF upper limit)

        if buttons.get('a', False):
            self.teleop.lift_position = np.clip(
                self.teleop.lift_position + lift_step, lift_min, lift_max
            )
            logger.info(f"Lift UP: {self.teleop.lift_position:.3f}")
        if buttons.get('b', False):
            self.teleop.lift_position = np.clip(
                self.teleop.lift_position - lift_step, lift_min, lift_max
            )
            logger.info(f"Lift DOWN: {self.teleop.lift_position:.3f}")

        # --- TRIGGER: Gripper aperture (analog squeeze: full trigger = closed) ---
        # Map the trigger value directly to the parallel-gripper aperture in meters
        # (0=closed, 0.042=open). Direct mapping avoids the chatter a hold-to-toggle
        # scheme produces near the closed limit.
        if trigger is not None:
            arm = self.teleop.left_arm if hand == 'left' else self.teleop.right_arm
            t = max(0.0, min(1.0, float(trigger)))
            arm.gripper_pos_m = 0.037 * (1.0 - t)

        # --- GRIP + POSITION: Arm tracking (delta-based control) ---
        if grip_active and position and all(k in position for k in ['x', 'y', 'z']):
            # VR position
            x_vr = position.get('x', 0)
            y_vr = position.get('y', 0)
            z_vr = position.get('z', 0)
            position_array = np.array([x_vr, y_vr, z_vr])

            # Initialize on first frame with grip active
            if controller.prev_vr_position is None:
                controller.prev_vr_position = position_array.copy()

                # Initialize from current arm state
                arm = self.teleop.left_arm if hand == 'left' else self.teleop.right_arm
                controller.initialize_from_arm(arm)
                logger.info(f"VR {hand} arm tracking started: ee_x={controller.current_ee_x:.4f}, "
                           f"ee_y={controller.current_ee_y:.4f}")

                # Initialize quaternion tracking
                if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w']):
                    controller.prev_quaternion = np.array([
                        quaternion['x'], quaternion['y'],
                        quaternion['z'], quaternion['w']
                    ])
                return

            # Calculate delta from previous frame
            delta_vr = position_array - controller.prev_vr_position
            controller.prev_vr_position = position_array.copy()

            # Scale factors for delta control
            pan_scale = 100.0   # VR X → shoulder pan (degrees per meter)
            pos_scale = 0.5     # VR Y/Z → ee position

            # Apply delta with limits
            delta_limit = 0.01  # meters
            angle_limit = 5.0   # degrees

            delta_pan = np.clip(delta_vr[0] * pan_scale, -angle_limit, angle_limit)
            delta_y = np.clip(delta_vr[1] * pos_scale, -delta_limit, delta_limit)
            delta_z = np.clip(delta_vr[2] * pos_scale, -delta_limit, delta_limit)

            # Accumulate to robot state
            # WebXR: Y+ = up, Z- = forward
            # Robot: ee_y+ = up, ee_x+ = forward
            controller.current_joint1_deg += delta_pan
            controller.current_ee_y += delta_y   # VR up (delta_y>0) → ee_y increase (up)
            controller.current_ee_x -= delta_z   # VR forward (delta_z<0) → ee_x increase (forward)

            # Clamp to workspace (uses dynamic limits based on robot variant)
            controller.current_ee_x = np.clip(controller.current_ee_x, self.ee_x_min, self.ee_x_max)
            controller.current_ee_y = np.clip(controller.current_ee_y, self.ee_y_min, self.ee_y_max)
            controller.current_joint1_deg = np.clip(controller.current_joint1_deg, -90.0, 90.0)

            # Handle wrist rotation from quaternion (delta-based)
            wrist_roll_deg = controller.current_wrist_roll
            pitch_deg = controller.current_pitch

            if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w']):
                from scipy.spatial.transform import Rotation as R
                current_quat = np.array([
                    quaternion['x'], quaternion['y'],
                    quaternion['z'], quaternion['w']
                ])

                if controller.prev_quaternion is not None:
                    try:
                        prev_rot = R.from_quat(controller.prev_quaternion)
                        current_rot = R.from_quat(current_quat)
                        relative_rot = current_rot * prev_rot.inv()
                        rotvec = relative_rot.as_rotvec()

                        # Delta rotation to wrist angles
                        roll_delta = np.clip(np.degrees(rotvec[2]) * 2.0, -angle_limit, angle_limit)
                        pitch_delta = np.clip(np.degrees(rotvec[0]) * 2.0, -angle_limit, angle_limit)

                        controller.current_wrist_roll += roll_delta
                        controller.current_pitch -= pitch_delta  # Inverted sign

                        # No limits - allow full rotation
                        # controller.current_wrist_roll = np.clip(controller.current_wrist_roll, -180.0, 180.0)
                        # controller.current_pitch = np.clip(controller.current_pitch, -180.0, 180.0)

                        wrist_roll_deg = controller.current_wrist_roll
                        pitch_deg = controller.current_pitch
                    except Exception as e:
                        logger.debug(f"Quaternion processing error: {e}")

                controller.prev_quaternion = current_quat.copy()

            # Send control goal
            goal = ControlGoal(
                arm=hand,
                mode=ControlMode.POSITION_CONTROL,
                ee_x=controller.current_ee_x,
                ee_y=controller.current_ee_y,
                joint1_deg=controller.current_joint1_deg,
                wrist_roll_deg=wrist_roll_deg,
                wrist_flex_deg=pitch_deg,
            )
            self.teleop.process_vr_goal(goal)

        elif not grip_active:
            # Reset tracking when grip released
            if controller.prev_vr_position is not None:
                controller.prev_vr_position = None
                controller.prev_quaternion = None
                logger.info(f"VR {hand} arm tracking stopped")

    async def broadcast_frame(self):
        """Broadcast all camera frames to connected clients."""
        if self.current_frame and self.clients:
            # Build message with all cameras
            message_data = {
                'type': 'frame',
                'data': self.current_frame,  # Main camera
            }

            # Add wrist cameras if available
            if hasattr(self, 'wrist_frames'):
                message_data['wrist_left'] = self.wrist_frames.get('left_wrist', '')
                message_data['wrist_right'] = self.wrist_frames.get('right_wrist', '')

            message = json.dumps(message_data)
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )

    async def run_server(self):
        """Run the WebSocket server."""
        # Setup SSL
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(str(self.certfile), str(self.keyfile))

        self.is_running = True
        async with websockets.serve(
            self.websocket_handler,
            self.host,
            self.ws_port,
            ssl=ssl_context
        ):
            logger.info(f"WebSocket server running on wss://{self.host}:{self.ws_port}")
            while self.is_running:
                await asyncio.sleep(0.033)  # ~30fps
                await self.broadcast_frame()

    def start_https_server(self):
        """Start HTTPS server for web UI."""
        web_root = Path(__file__).parent / 'web_ui_stream'

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(web_root), **kwargs)

            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                super().end_headers()

            def log_message(self, format, *args):
                pass  # Suppress logging

        server = http.server.HTTPServer((self.host, self.https_port), Handler)

        # Setup SSL
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(self.certfile), str(self.keyfile))
        server.socket = context.wrap_socket(server.socket, server_side=True)

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"HTTPS server running on https://{self.host}:{self.https_port}")


def get_local_ip():
    """Get local IP address."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def create_web_ui():
    """Create the web UI files for camera streaming with A-Frame VR."""
    web_dir = Path(__file__).parent / 'web_ui_stream'
    web_dir.mkdir(exist_ok=True)

    # Create index.html with A-Frame VR + multi-camera streaming
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AlohaMini VR Teleop + Camera</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://aframe.io/releases/1.7.1/aframe.min.js"></script>
    <style>
        body { margin: 0; background: #1a1a2e; color: #fff; font-family: sans-serif; }
        #desktop-ui {
            position: fixed; top: 10px; left: 10px; z-index: 100;
            background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px;
            max-width: 350px;
        }
        #desktop-ui h2 { color: #00d4ff; margin: 0 0 10px 0; }
        #status { padding: 5px 10px; border-radius: 4px; margin-bottom: 10px; display: inline-block; }
        .connected { background: #2ecc71; }
        .disconnected { background: #e74c3c; }
        .connecting { background: #f39c12; }
        .camera-row { display: flex; gap: 5px; margin-top: 10px; }
        #camera-preview {
            width: 200px; height: 150px; border: 2px solid #00d4ff;
            background: #000; display: block;
        }
        .wrist-preview {
            width: 80px; height: 60px; border: 1px solid #888;
            background: #000; display: block;
        }
        .wrist-container { display: flex; flex-direction: column; gap: 5px; }
        .wrist-label { font-size: 10px; color: #888; text-align: center; }
        .info { font-size: 12px; color: #aaa; margin-top: 10px; line-height: 1.6; }
        #enter-vr-btn {
            margin-top: 15px; padding: 15px 30px; font-size: 18px;
            background: #00d4ff; border: none; border-radius: 8px;
            cursor: pointer; color: #000; font-weight: bold; width: 100%;
        }
        #enter-vr-btn:hover { background: #00b8e6; }
        #enter-vr-btn:disabled { background: #666; cursor: not-allowed; }
        #debug-info { font-size: 11px; color: #888; margin-top: 8px; }
        /* Hidden canvases for textures */
        .hidden-canvas { display: none; }
    </style>
</head>
<body>
    <!-- Hidden canvases for camera textures -->
    <canvas id="texture-canvas-main" class="hidden-canvas" width="640" height="480"></canvas>
    <canvas id="texture-canvas-left" class="hidden-canvas" width="320" height="240"></canvas>
    <canvas id="texture-canvas-right" class="hidden-canvas" width="320" height="240"></canvas>

    <div id="desktop-ui">
        <h2>AlohaMini VR + Camera</h2>
        <div id="status" class="connecting">Connecting...</div>
        <div class="camera-row">
            <img id="camera-preview" alt="Main Camera" crossorigin="anonymous">
            <div class="wrist-container">
                <div>
                    <div class="wrist-label">Left Wrist</div>
                    <img id="wrist-left-preview" class="wrist-preview" alt="Left Wrist" crossorigin="anonymous">
                </div>
                <div>
                    <div class="wrist-label">Right Wrist</div>
                    <img id="wrist-right-preview" class="wrist-preview" alt="Right Wrist" crossorigin="anonymous">
                </div>
            </div>
        </div>
        <div class="info">
            <b>Controls:</b><br>
            Grip (hold): Track arm position<br>
            Trigger: Open/close gripper<br>
            Left Stick: Move base<br>
            A/B: Lift up/down
        </div>
        <button id="enter-vr-btn">Enter VR Mode</button>
        <div id="debug-info">Controllers: waiting...</div>
    </div>

    <a-scene vr-mode-ui="enabled: true" embedded
             style="position: fixed; top: 0; left: 0; width: 100%; height: 100%;">

        <!-- Main camera display fixed in front of headset (HUD style, center) -->
        <a-entity id="cameraRig" camera look-controls>
            <!-- Main camera screen (center, fixed to view) -->
            <a-entity id="camera-screen-main" position="0 0 -0.8" rotation="0 0 0">
                <a-plane id="camera-plane-main" width="0.5" height="0.375"
                         material="shader: flat; side: double; color: #fff">
                </a-plane>
            </a-entity>
        </a-entity>

        <!-- VR Controllers with wrist camera displays above each -->
        <a-entity id="leftHand" oculus-touch-controls="hand: left">
            <!-- Left wrist camera above left controller -->
            <a-entity id="camera-screen-left" position="0 0.12 -0.05" rotation="-30 0 0">
                <a-plane id="camera-plane-left" width="0.12" height="0.09"
                         material="shader: flat; side: double; color: #fff">
                </a-plane>
            </a-entity>
        </a-entity>
        <a-entity id="rightHand" oculus-touch-controls="hand: right">
            <!-- Right wrist camera above right controller -->
            <a-entity id="camera-screen-right" position="0 0.12 -0.05" rotation="-30 0 0">
                <a-plane id="camera-plane-right" width="0.12" height="0.09"
                         material="shader: flat; side: double; color: #fff">
                </a-plane>
            </a-entity>
        </a-entity>

        <!-- Simple environment -->
        <a-sky color="#1a1a2e"></a-sky>
        <a-plane rotation="-90 0 0" width="10" height="10" color="#222" position="0 0 0"></a-plane>
    </a-scene>

    <script>
    // =====================================================
    // A-Frame Component for VR Controller + Multi-Camera Streaming
    // =====================================================
    AFRAME.registerComponent('vr-teleop-stream', {
        init: function() {
            console.log('VR Teleop Stream component initialized');

            // Get controller elements
            this.leftHand = document.querySelector('#leftHand');
            this.rightHand = document.querySelector('#rightHand');

            // Get camera plane elements
            this.cameraPlaneMain = document.querySelector('#camera-plane-main');
            this.cameraPlaneLeft = document.querySelector('#camera-plane-left');
            this.cameraPlaneRight = document.querySelector('#camera-plane-right');

            // Get preview elements
            this.cameraPreview = document.querySelector('#camera-preview');
            this.wristLeftPreview = document.querySelector('#wrist-left-preview');
            this.wristRightPreview = document.querySelector('#wrist-right-preview');

            // Setup canvases for textures
            this.canvasMain = document.getElementById('texture-canvas-main');
            this.ctxMain = this.canvasMain.getContext('2d');
            this.canvasLeft = document.getElementById('texture-canvas-left');
            this.ctxLeft = this.canvasLeft.getContext('2d');
            this.canvasRight = document.getElementById('texture-canvas-right');
            this.ctxRight = this.canvasRight.getContext('2d');

            // Three.js textures
            this.textureMain = null;
            this.textureLeft = null;
            this.textureRight = null;

            // Temp images for loading frames
            this.tempImageMain = new Image();
            this.tempImageMain.crossOrigin = 'anonymous';
            this.tempImageLeft = new Image();
            this.tempImageLeft.crossOrigin = 'anonymous';
            this.tempImageRight = new Image();
            this.tempImageRight.crossOrigin = 'anonymous';

            // Image load handlers
            this.tempImageMain.onload = () => {
                this.ctxMain.drawImage(this.tempImageMain, 0, 0,
                    this.canvasMain.width, this.canvasMain.height);
                if (this.textureMain) this.textureMain.needsUpdate = true;
            };
            this.tempImageLeft.onload = () => {
                this.ctxLeft.drawImage(this.tempImageLeft, 0, 0,
                    this.canvasLeft.width, this.canvasLeft.height);
                if (this.textureLeft) this.textureLeft.needsUpdate = true;
            };
            this.tempImageRight.onload = () => {
                this.ctxRight.drawImage(this.tempImageRight, 0, 0,
                    this.canvasRight.width, this.canvasRight.height);
                if (this.textureRight) this.textureRight.needsUpdate = true;
            };

            // Controller state
            this.leftGripDown = false;
            this.rightGripDown = false;
            this.leftTriggerDown = false;
            this.rightTriggerDown = false;

            // WebSocket
            this.ws = null;
            this.connectWebSocket();

            // Setup controller events
            this.setupControllerEvents();

            // Setup textures after scene is ready
            this.setupTextures();

            // Frame counter for throttling
            this.frameCount = 0;
        },

        setupTextures: function() {
            // Setup texture for each camera plane
            const setupSingleTexture = (plane, canvas, ctx, name) => {
                return new Promise((resolve) => {
                    const checkMesh = () => {
                        if (!plane) { resolve(null); return; }
                        const mesh = plane.getObject3D('mesh');
                        if (mesh && mesh.material) {
                            const texture = new THREE.CanvasTexture(canvas);
                            texture.minFilter = THREE.LinearFilter;
                            texture.magFilter = THREE.LinearFilter;
                            mesh.material.map = texture;
                            mesh.material.needsUpdate = true;

                            // Draw placeholder
                            ctx.fillStyle = '#333';
                            ctx.fillRect(0, 0, canvas.width, canvas.height);
                            ctx.fillStyle = '#666';
                            ctx.font = '14px sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText(name, canvas.width/2, canvas.height/2);
                            texture.needsUpdate = true;

                            console.log('Texture applied:', name);
                            resolve(texture);
                        } else {
                            setTimeout(checkMesh, 100);
                        }
                    };
                    checkMesh();
                });
            };

            // Setup all textures
            setupSingleTexture(this.cameraPlaneMain, this.canvasMain, this.ctxMain, 'Main Camera')
                .then(tex => { this.textureMain = tex; });
            setupSingleTexture(this.cameraPlaneLeft, this.canvasLeft, this.ctxLeft, 'Left Wrist')
                .then(tex => { this.textureLeft = tex; });
            setupSingleTexture(this.cameraPlaneRight, this.canvasRight, this.ctxRight, 'Right Wrist')
                .then(tex => { this.textureRight = tex; });
        },

        connectWebSocket: function() {
            const host = window.location.hostname;
            const wsPort = 8442;
            const wsUrl = `wss://${host}:${wsPort}`;

            console.log('Connecting to WebSocket:', wsUrl);
            const statusEl = document.getElementById('status');
            statusEl.className = 'connecting';
            statusEl.textContent = 'Connecting...';

            try {
                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    statusEl.className = 'connected';
                    statusEl.textContent = 'Connected';
                };

                this.ws.onclose = () => {
                    console.log('WebSocket closed, reconnecting...');
                    statusEl.className = 'disconnected';
                    statusEl.textContent = 'Disconnected';
                    this.ws = null;
                    setTimeout(() => this.connectWebSocket(), 2000);
                };

                this.ws.onerror = (err) => {
                    console.error('WebSocket error:', err);
                    statusEl.className = 'disconnected';
                    statusEl.textContent = 'Error';
                };

                this.ws.onmessage = (event) => {
                    try {
                        const msg = JSON.parse(event.data);
                        if (msg.type === 'frame') {
                            // Main camera
                            if (msg.data) {
                                this.updateCameraFrame(msg.data, 'main');
                            }
                            // Wrist cameras
                            if (msg.wrist_left) {
                                this.updateCameraFrame(msg.wrist_left, 'left');
                            }
                            if (msg.wrist_right) {
                                this.updateCameraFrame(msg.wrist_right, 'right');
                            }
                        }
                    } catch (e) {
                        console.error('Message parse error:', e);
                    }
                };
            } catch (err) {
                console.error('WebSocket connection failed:', err);
                statusEl.className = 'disconnected';
                statusEl.textContent = 'Failed';
                setTimeout(() => this.connectWebSocket(), 3000);
            }
        },

        updateCameraFrame: function(base64Data, camera) {
            const dataUrl = 'data:image/jpeg;base64,' + base64Data;

            if (camera === 'main') {
                if (this.cameraPreview) this.cameraPreview.src = dataUrl;
                this.tempImageMain.src = dataUrl;
            } else if (camera === 'left') {
                if (this.wristLeftPreview) this.wristLeftPreview.src = dataUrl;
                this.tempImageLeft.src = dataUrl;
            } else if (camera === 'right') {
                if (this.wristRightPreview) this.wristRightPreview.src = dataUrl;
                this.tempImageRight.src = dataUrl;
            }
        },

        setupControllerEvents: function() {
            // Thumbstick state
            this.leftThumbstick = { x: 0, y: 0 };
            this.rightThumbstick = { x: 0, y: 0 };

            if (this.leftHand) {
                this.leftHand.addEventListener('gripdown', () => {
                    this.leftGripDown = true;
                    console.log('Left grip down');
                });
                this.leftHand.addEventListener('gripup', () => {
                    this.leftGripDown = false;
                    console.log('Left grip up');
                });
                this.leftHand.addEventListener('triggerdown', () => {
                    this.leftTriggerDown = true;
                    console.log('Left trigger down');
                });
                this.leftHand.addEventListener('triggerup', () => {
                    this.leftTriggerDown = false;
                    console.log('Left trigger up');
                });
                // Thumbstick event
                this.leftHand.addEventListener('thumbstickmoved', (evt) => {
                    this.leftThumbstick = { x: evt.detail.x, y: evt.detail.y };
                    console.log('Left thumbstick:', this.leftThumbstick);
                });
            }

            if (this.rightHand) {
                this.rightHand.addEventListener('gripdown', () => {
                    this.rightGripDown = true;
                    console.log('Right grip down');
                });
                this.rightHand.addEventListener('gripup', () => {
                    this.rightGripDown = false;
                    console.log('Right grip up');
                });
                this.rightHand.addEventListener('triggerdown', () => {
                    this.rightTriggerDown = true;
                    console.log('Right trigger down');
                });
                this.rightHand.addEventListener('triggerup', () => {
                    this.rightTriggerDown = false;
                    console.log('Right trigger up');
                });
                // Thumbstick event
                this.rightHand.addEventListener('thumbstickmoved', (evt) => {
                    this.rightThumbstick = { x: evt.detail.x, y: evt.detail.y };
                    console.log('Right thumbstick:', this.rightThumbstick);
                });
            }
        },

        getControllerData: function(hand, handEntity, gripDown, triggerDown) {
            const data = {
                hand: hand,
                position: null,
                quaternion: null,
                gripActive: gripDown,
                trigger: triggerDown ? 1 : 0,
                thumbstick: { x: 0, y: 0 }
            };

            if (!handEntity || !handEntity.object3D) return data;

            // Get position and quaternion from A-Frame object3D
            const pos = handEntity.object3D.position;
            const quat = handEntity.object3D.quaternion;

            data.position = { x: pos.x, y: pos.y, z: pos.z };
            data.quaternion = { x: quat.x, y: quat.y, z: quat.z, w: quat.w };

            // Get gamepad data via tracked-controls component
            const trackedControls = handEntity.components['tracked-controls'];

            // Debug: log once per hand when first accessing gamepad
            if (!this[`_${hand}GamepadLogged`]) {
                console.log(`${hand} trackedControls:`, !!trackedControls);
                if (trackedControls) {
                    console.log(`${hand} controller:`, !!trackedControls.controller);
                    if (trackedControls.controller) {
                        console.log(`${hand} gamepad:`, !!trackedControls.controller.gamepad);
                        if (trackedControls.controller.gamepad) {
                            const gp = trackedControls.controller.gamepad;
                            console.log(`${hand} gamepad.axes:`, gp.axes ? Array.from(gp.axes) : 'none');
                            console.log(`${hand} gamepad.buttons:`, gp.buttons ? gp.buttons.length : 'none');
                            this[`_${hand}GamepadLogged`] = true;
                        }
                    }
                }
            }

            if (trackedControls && trackedControls.controller) {
                const gamepad = trackedControls.controller.gamepad;
                if (gamepad) {
                    // Read thumbstick directly from gamepad.axes
                    // WebXR standard: axes[0] = X, axes[1] = Y (for each controller)
                    // Some controllers may use axes[2], axes[3] for thumbstick
                    if (gamepad.axes && gamepad.axes.length >= 2) {
                        // Try axes[0], axes[1] first (standard)
                        let tx = gamepad.axes[0] || 0;
                        let ty = gamepad.axes[1] || 0;

                        // If standard axes are zero, try axes[2], axes[3] (some controllers)
                        if (Math.abs(tx) < 0.01 && Math.abs(ty) < 0.01 && gamepad.axes.length >= 4) {
                            tx = gamepad.axes[2] || 0;
                            ty = gamepad.axes[3] || 0;
                        }

                        data.thumbstick = { x: tx, y: ty };

                        // Debug: log all axes when any movement detected
                        if (Math.abs(tx) > 0.1 || Math.abs(ty) > 0.1) {
                            console.log(`${hand} thumbstick:`, data.thumbstick, 'all axes:', Array.from(gamepad.axes));
                        }
                    } else if (gamepad.axes) {
                        // Debug: log axes info
                        console.log(`${hand} gamepad.axes.length:`, gamepad.axes.length);
                    }

                    // Quest 2 button mapping (WebXR Gamepad):
                    // buttons[0] = trigger, buttons[1] = grip
                    // buttons[3] = thumbstick press, buttons[4] = A/X, buttons[5] = B/Y
                    if (gamepad.buttons && gamepad.buttons.length > 5) {
                        const aPressed = gamepad.buttons[4]?.pressed || false;
                        const bPressed = gamepad.buttons[5]?.pressed || false;
                        data.buttons = {
                            a: aPressed,
                            b: bPressed
                        };
                    } else {
                        // Fallback: check all possible button indices
                        let foundA = false, foundB = false;
                        if (gamepad.buttons) {
                            for (let i = 0; i < gamepad.buttons.length; i++) {
                                if (gamepad.buttons[i]?.pressed) {
                                    if (i === 4) foundA = true;
                                    if (i === 5) foundB = true;
                                }
                            }
                        }
                        data.buttons = { a: foundA, b: foundB };
                    }
                }
            }

            return data;
        },

        updateGamepadsFromXR: function() {
            // Access XR session directly to get gamepads
            const scene = this.el.sceneEl;
            if (!scene || !scene.renderer || !scene.renderer.xr) return;

            const xrSession = scene.renderer.xr.getSession();
            if (!xrSession) return;

            const inputSources = xrSession.inputSources;
            if (!inputSources) return;

            for (const source of inputSources) {
                if (!source.gamepad) continue;

                const handedness = source.handedness;
                if (handedness === 'left') {
                    this.xrLeftGamepad = source.gamepad;
                    // Debug log once
                    if (!this._xrLeftLogged) {
                        console.log('XR Left gamepad found, axes:', Array.from(source.gamepad.axes));
                        this._xrLeftLogged = true;
                    }
                } else if (handedness === 'right') {
                    this.xrRightGamepad = source.gamepad;
                    if (!this._xrRightLogged) {
                        console.log('XR Right gamepad found, axes:', Array.from(source.gamepad.axes));
                        this._xrRightLogged = true;
                    }
                }
            }
        },

        tick: function() {
            // Send controller data every frame
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

            // Try to get gamepads from WebXR session directly
            this.updateGamepadsFromXR();

            const leftData = this.getControllerData('left', this.leftHand, this.leftGripDown, this.leftTriggerDown);
            const rightData = this.getControllerData('right', this.rightHand, this.rightGripDown, this.rightTriggerDown);

            // Override thumbstick from XR gamepads if available
            if (this.xrLeftGamepad) {
                const axes = this.xrLeftGamepad.axes;
                if (axes && axes.length >= 4) {
                    // Quest uses axes[2], axes[3] for thumbstick
                    leftData.thumbstick = { x: axes[2] || 0, y: axes[3] || 0 };
                } else if (axes && axes.length >= 2) {
                    leftData.thumbstick = { x: axes[0] || 0, y: axes[1] || 0 };
                }
                // Debug log when thumbstick moved
                if (Math.abs(leftData.thumbstick.x) > 0.1 || Math.abs(leftData.thumbstick.y) > 0.1) {
                    if (this.frameCount % 10 === 0) {
                        console.log('Left XR thumbstick:', leftData.thumbstick);
                    }
                }
            }
            if (this.xrRightGamepad) {
                const axes = this.xrRightGamepad.axes;
                if (axes && axes.length >= 4) {
                    rightData.thumbstick = { x: axes[2] || 0, y: axes[3] || 0 };
                } else if (axes && axes.length >= 2) {
                    rightData.thumbstick = { x: axes[0] || 0, y: axes[1] || 0 };
                }
                if (Math.abs(rightData.thumbstick.x) > 0.1 || Math.abs(rightData.thumbstick.y) > 0.1) {
                    if (this.frameCount % 10 === 0) {
                        console.log('Right XR thumbstick:', rightData.thumbstick);
                    }
                }
            }

            // Send to server
            const message = {
                timestamp: Date.now(),
                leftController: leftData,
                rightController: rightData
            };

            try {
                this.ws.send(JSON.stringify(message));
            } catch (e) {
                console.error('Send error:', e);
            }

            // Update debug info (throttled)
            this.frameCount++;
            if (this.frameCount % 30 === 0) {
                const debugEl = document.getElementById('debug-info');
                if (debugEl) {
                    const leftPos = leftData.position ? `(${leftData.position.x.toFixed(2)}, ${leftData.position.y.toFixed(2)}, ${leftData.position.z.toFixed(2)})` : 'N/A';
                    const rightPos = rightData.position ? `(${rightData.position.x.toFixed(2)}, ${rightData.position.y.toFixed(2)}, ${rightData.position.z.toFixed(2)})` : 'N/A';
                    debugEl.textContent = `L: ${leftPos} | R: ${rightPos}`;
                }
            }
        }
    });

    // Initialize when DOM ready
    document.addEventListener('DOMContentLoaded', () => {
        const scene = document.querySelector('a-scene');
        const vrButton = document.getElementById('enter-vr-btn');

        // Add component to scene
        if (scene) {
            if (scene.hasLoaded) {
                scene.setAttribute('vr-teleop-stream', '');
            } else {
                scene.addEventListener('loaded', () => {
                    scene.setAttribute('vr-teleop-stream', '');
                });
            }

            // VR button handler
            vrButton.addEventListener('click', () => {
                scene.enterVR().catch(err => {
                    console.error('Failed to enter VR:', err);
                    alert('Failed to enter VR: ' + err.message);
                });
            });

            // Update button state
            scene.addEventListener('enter-vr', () => {
                vrButton.textContent = 'In VR Mode';
                vrButton.disabled = true;

                // Keep session alive even when headset is removed
                const xrSession = scene.renderer.xr.getSession();
                if (xrSession) {
                    xrSession.addEventListener('visibilitychange', (event) => {
                        console.log('XR Session visibility:', event.session.visibilityState);
                        // Don't end session - just log the visibility change
                        // Controllers should still work even when 'hidden'
                    });
                    console.log('XR Session started - controllers will stay active even if headset removed');
                }
            });
            scene.addEventListener('exit-vr', () => {
                vrButton.textContent = 'Enter VR Mode';
                vrButton.disabled = false;
            });
        }

        // Check VR support
        if (navigator.xr) {
            navigator.xr.isSessionSupported('immersive-vr').then(supported => {
                if (!supported) {
                    vrButton.textContent = 'VR Not Supported';
                    vrButton.disabled = true;
                }
            });
        } else {
            vrButton.textContent = 'WebXR Not Available';
            vrButton.disabled = true;
        }
    });
    </script>
</body>
</html>'''

    (web_dir / 'index.html').write_text(html_content)
    logger.info(f"Created web UI at {web_dir}")


async def main_async(args):
    """Main async function."""
    # Create web UI
    create_web_ui()

    # Create teleop controller (handles IK and arm control)
    config = TeleopConfig()
    teleop = TeleopController(config, robot_variant=args.robot)

    # Create camera stream server with teleop controller
    server = CameraStreamServer(teleop)
    server.start_https_server()

    # Start WebSocket server in background
    ws_task = asyncio.create_task(server.run_server())

    # Create environment
    print(f"Creating environment with robot: {args.robot}...")
    render_mode = "human" if args.render else None
    env = gym.make(
        'ReplicaCAD_SceneManipulation-v1',
        robot_uids=args.robot,
        render_mode=render_mode,
        obs_mode='rgbd',
        sim_backend=args.backend,
        control_mode='pd_joint_pos',
    )

    print("Resetting environment...")
    obs, _ = env.reset(options=dict(reconfigure=True))

    # Print connection info
    local_ip = get_local_ip()
    print(f"\n{'='*60}")
    print("VR TELEOPERATION WITH CAMERA STREAMING")
    print(f"{'='*60}")
    print(f"Open in VR headset browser:")
    print(f"  https://{local_ip}:8443")
    print(f"\nWebSocket: wss://{local_ip}:8442")
    print(f"{'='*60}")
    print("\nControls:")
    print("  Left Thumbstick: Move Forward/Back, Strafe Left/Right")
    print("  Right Thumbstick: Rotate Left/Right, Lift Up/Down")
    print("  Triggers: Close Grippers")
    print("\nPress Ctrl+C to quit")
    print(f"{'='*60}\n")

    try:
        frame_count = 0
        while True:
            # Compute action from teleop controller (uses IK for arms)
            action = teleop.compute_action()

            # Step environment
            obs, _, _, _, _ = env.step(action)

            # Render to window if human mode enabled
            if render_mode == "human":
                env.render()

            # Update camera frames (main + wrist cameras)
            if 'sensor_data' in obs:
                # Main camera
                if 'cam_main' in obs['sensor_data']:
                    rgb = obs['sensor_data']['cam_main']['rgb']
                    if hasattr(rgb, 'cpu'):
                        rgb = rgb.cpu().numpy()
                    if len(rgb.shape) == 4:
                        rgb = rgb[0]
                    server.update_frame(rgb[:, :, :3].astype(np.uint8), 'main')

                # Left wrist camera
                if 'cam_left_wrist' in obs['sensor_data']:
                    rgb = obs['sensor_data']['cam_left_wrist']['rgb']
                    if hasattr(rgb, 'cpu'):
                        rgb = rgb.cpu().numpy()
                    if len(rgb.shape) == 4:
                        rgb = rgb[0]
                    server.update_frame(rgb[:, :, :3].astype(np.uint8), 'left_wrist')

                # Right wrist camera
                if 'cam_right_wrist' in obs['sensor_data']:
                    rgb = obs['sensor_data']['cam_right_wrist']['rgb']
                    if hasattr(rgb, 'cpu'):
                        rgb = rgb.cpu().numpy()
                    if len(rgb.shape) == 4:
                        rgb = rgb[0]
                    server.update_frame(rgb[:, :, :3].astype(np.uint8), 'right_wrist')

            frame_count += 1
            if frame_count % 100 == 0:
                state = teleop.get_state_info()
                print(f"Frame {frame_count}, Clients: {len(server.clients)}, "
                      f"Base: [{teleop.base_velocity[0]:.2f}, {teleop.base_velocity[1]:.2f}], "
                      f"Lift: {teleop.lift_position:.3f}")

            await asyncio.sleep(0.033)  # ~30 FPS

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.is_running = False
        env.close()


def main():
    parser = argparse.ArgumentParser(description='AlohaMini VR Teleop with Camera Streaming')
    parser.add_argument('--gpu', action='store_true', help='Use GPU backend')
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--render', action='store_true', help='Enable human render mode (display window)')
    parser.add_argument('--robot', default='aloha_mini_so100_v2',
                        choices=['aloha_mini_so100_v2'],
                        help='Robot variant to use')
    args = parser.parse_args()

    if args.gpu:
        args.backend = 'gpu'

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
