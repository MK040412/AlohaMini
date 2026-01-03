#!/usr/bin/env python3
"""
AlohaMini VR Teleoperation with Camera Streaming

This demo streams the robot's first-person camera view to a web browser
while allowing VR teleoperation from WebXR controllers.

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
from mani_skill.agents.registration import REGISTERED_AGENTS
REGISTERED_AGENTS.pop('aloha_mini_virtual', None)

# Import agent
sys.path.insert(0, str(Path(__file__).parent.parent / 'agents' / 'aloha_mini'))
import aloha_mini_virtual

import gymnasium as gym
import mani_skill.envs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CameraStreamServer:
    """WebSocket server for streaming camera images and receiving VR control data."""

    def __init__(self, host='0.0.0.0', ws_port=8442, https_port=8443):
        self.host = host
        self.ws_port = ws_port
        self.https_port = https_port
        self.clients: Set = set()
        self.current_frame: Optional[bytes] = None
        self.is_running = False

        # VR control state
        self.vr_control = {
            'base_x': 0.0,
            'base_y': 0.0,
            'base_rot': 0.0,
            'lift': 0.3,
            'left_arm': [0.0, 0.3, -0.5, 0.0, 0.0, 0.0],
            'right_arm': [0.0, 0.3, -0.5, 0.0, 0.0, 0.0],
            'left_gripper': False,
            'right_gripper': False,
        }

        # SSL paths
        self.ssl_dir = Path(__file__).parent
        self.certfile = self.ssl_dir / 'cert.pem'
        self.keyfile = self.ssl_dir / 'key.pem'

    def update_frame(self, rgb_array: np.ndarray):
        """Update the current frame to stream."""
        # Convert to JPEG for faster streaming
        img = Image.fromarray(rgb_array)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=70)
        self.current_frame = base64.b64encode(buffer.getvalue()).decode('utf-8')

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
        """Process incoming VR control data."""
        # Thumbstick for base movement
        if 'leftController' in data:
            left = data['leftController']
            thumbstick = left.get('thumbstick', {})
            # Left thumbstick: forward/back and strafe
            self.vr_control['base_y'] += thumbstick.get('y', 0) * 0.02
            self.vr_control['base_x'] += thumbstick.get('x', 0) * 0.02

            # Trigger for gripper
            if left.get('trigger', 0) > 0.5:
                self.vr_control['left_gripper'] = True
            else:
                self.vr_control['left_gripper'] = False

        if 'rightController' in data:
            right = data['rightController']
            thumbstick = right.get('thumbstick', {})
            # Right thumbstick: rotation and lift
            self.vr_control['base_rot'] += thumbstick.get('x', 0) * 0.05
            self.vr_control['lift'] += thumbstick.get('y', 0) * 0.01
            self.vr_control['lift'] = np.clip(self.vr_control['lift'], 0.0, 0.15)

            # Trigger for gripper
            if right.get('trigger', 0) > 0.5:
                self.vr_control['right_gripper'] = True
            else:
                self.vr_control['right_gripper'] = False

    async def broadcast_frame(self):
        """Broadcast current frame to all connected clients."""
        if self.current_frame and self.clients:
            message = json.dumps({
                'type': 'frame',
                'data': self.current_frame
            })
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
    """Create the web UI files for camera streaming."""
    web_dir = Path(__file__).parent / 'web_ui_stream'
    web_dir.mkdir(exist_ok=True)

    # Create index.html
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlohaMini VR Teleop</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #1a1a2e;
            color: #eee;
            font-family: 'Segoe UI', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 { color: #00d4ff; margin-bottom: 20px; }
        #status {
            padding: 10px 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .connected { background: #2ecc71; color: #fff; }
        .disconnected { background: #e74c3c; color: #fff; }
        .connecting { background: #f39c12; color: #fff; }
        #camera-view {
            border: 3px solid #00d4ff;
            border-radius: 10px;
            max-width: 100%;
            background: #000;
        }
        #controls {
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            max-width: 600px;
        }
        .control-group {
            background: #16213e;
            padding: 15px;
            border-radius: 10px;
        }
        .control-group h3 {
            color: #00d4ff;
            margin-bottom: 10px;
        }
        .control-item {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }
        #vr-button {
            margin-top: 20px;
            padding: 15px 40px;
            font-size: 18px;
            background: #00d4ff;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            color: #1a1a2e;
            font-weight: bold;
        }
        #vr-button:hover { background: #00b4d8; }
        #vr-button:disabled { background: #666; cursor: not-allowed; }
        .info { color: #888; font-size: 14px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>AlohaMini VR Teleoperation</h1>
    <div id="status" class="connecting">Connecting...</div>

    <img id="camera-view" width="640" height="480" alt="Camera View">

    <div id="controls">
        <div class="control-group">
            <h3>Left Controller</h3>
            <div class="control-item"><span>Thumbstick:</span><span id="left-thumb">0.00, 0.00</span></div>
            <div class="control-item"><span>Trigger:</span><span id="left-trigger">0.00</span></div>
            <div class="control-item"><span>Grip:</span><span id="left-grip">Released</span></div>
        </div>
        <div class="control-group">
            <h3>Right Controller</h3>
            <div class="control-item"><span>Thumbstick:</span><span id="right-thumb">0.00, 0.00</span></div>
            <div class="control-item"><span>Trigger:</span><span id="right-trigger">0.00</span></div>
            <div class="control-item"><span>Grip:</span><span id="right-grip">Released</span></div>
        </div>
    </div>

    <button id="vr-button" onclick="startVR()">Enter VR Mode</button>

    <p class="info">
        Controls: Left thumbstick = Move | Right thumbstick = Rotate/Lift | Triggers = Grippers
    </p>

    <script>
        const wsPort = 8442;
        const host = window.location.hostname;
        let ws = null;
        let xrSession = null;

        function connect() {
            const statusEl = document.getElementById('status');
            statusEl.className = 'connecting';
            statusEl.textContent = 'Connecting...';

            ws = new WebSocket(`wss://${host}:${wsPort}`);

            ws.onopen = () => {
                statusEl.className = 'connected';
                statusEl.textContent = 'Connected';
                console.log('WebSocket connected');
            };

            ws.onclose = () => {
                statusEl.className = 'disconnected';
                statusEl.textContent = 'Disconnected - Reconnecting...';
                setTimeout(connect, 2000);
            };

            ws.onerror = (err) => {
                console.error('WebSocket error:', err);
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'frame') {
                        document.getElementById('camera-view').src =
                            'data:image/jpeg;base64,' + msg.data;
                    }
                } catch (e) {
                    console.error('Parse error:', e);
                }
            };
        }

        function sendControllerData(leftController, rightController) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    leftController: leftController,
                    rightController: rightController
                }));
            }
        }

        async function startVR() {
            if (!navigator.xr) {
                alert('WebXR not supported');
                return;
            }

            try {
                xrSession = await navigator.xr.requestSession('immersive-vr', {
                    requiredFeatures: ['local-floor']
                });

                document.getElementById('vr-button').textContent = 'VR Active';
                document.getElementById('vr-button').disabled = true;

                xrSession.addEventListener('end', () => {
                    document.getElementById('vr-button').textContent = 'Enter VR Mode';
                    document.getElementById('vr-button').disabled = false;
                    xrSession = null;
                });

                const gl = document.createElement('canvas').getContext('webgl2');
                await xrSession.updateRenderState({
                    baseLayer: new XRWebGLLayer(xrSession, gl)
                });

                const refSpace = await xrSession.requestReferenceSpace('local-floor');

                function onXRFrame(time, frame) {
                    const session = frame.session;
                    session.requestAnimationFrame(onXRFrame);

                    const inputSources = session.inputSources;
                    let leftData = null;
                    let rightData = null;

                    for (const source of inputSources) {
                        const gamepad = source.gamepad;
                        if (!gamepad) continue;

                        const data = {
                            thumbstick: {
                                x: gamepad.axes[2] || 0,
                                y: gamepad.axes[3] || 0
                            },
                            trigger: gamepad.buttons[0]?.value || 0,
                            grip: gamepad.buttons[1]?.pressed || false
                        };

                        if (source.handedness === 'left') {
                            leftData = data;
                            document.getElementById('left-thumb').textContent =
                                `${data.thumbstick.x.toFixed(2)}, ${data.thumbstick.y.toFixed(2)}`;
                            document.getElementById('left-trigger').textContent =
                                data.trigger.toFixed(2);
                            document.getElementById('left-grip').textContent =
                                data.grip ? 'Pressed' : 'Released';
                        } else if (source.handedness === 'right') {
                            rightData = data;
                            document.getElementById('right-thumb').textContent =
                                `${data.thumbstick.x.toFixed(2)}, ${data.thumbstick.y.toFixed(2)}`;
                            document.getElementById('right-trigger').textContent =
                                data.trigger.toFixed(2);
                            document.getElementById('right-grip').textContent =
                                data.grip ? 'Pressed' : 'Released';
                        }
                    }

                    if (leftData || rightData) {
                        sendControllerData(leftData || {}, rightData || {});
                    }
                }

                xrSession.requestAnimationFrame(onXRFrame);

            } catch (err) {
                console.error('VR Error:', err);
                alert('Failed to start VR: ' + err.message);
            }
        }

        // Check VR support
        if (navigator.xr) {
            navigator.xr.isSessionSupported('immersive-vr').then(supported => {
                if (!supported) {
                    document.getElementById('vr-button').disabled = true;
                    document.getElementById('vr-button').textContent = 'VR Not Supported';
                }
            });
        } else {
            document.getElementById('vr-button').disabled = true;
            document.getElementById('vr-button').textContent = 'WebXR Not Available';
        }

        // Start connection
        connect();
    </script>
</body>
</html>'''

    (web_dir / 'index.html').write_text(html_content)
    logger.info(f"Created web UI at {web_dir}")


async def main_async(args):
    """Main async function."""
    # Create web UI
    create_web_ui()

    # Create camera stream server
    server = CameraStreamServer()
    server.start_https_server()

    # Start WebSocket server in background
    ws_task = asyncio.create_task(server.run_server())

    # Create environment
    print("Creating environment...")
    env = gym.make(
        'ReplicaCAD_SceneManipulation-v1',
        robot_uids='aloha_mini_virtual',
        render_mode=None,
        obs_mode='rgbd',
        sim_backend=args.backend,
        control_mode='pd_joint_pos',
    )

    print("Resetting environment...")
    obs, _ = env.reset(options=dict(reconfigure=True))

    # Get initial qpos
    qpos = env.unwrapped.agent.robot.qpos[0].cpu().numpy().copy()

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
            # Apply VR control
            qpos[0] = server.vr_control['base_x']
            qpos[1] = server.vr_control['base_y']
            qpos[2] = server.vr_control['base_rot']
            qpos[3] = server.vr_control['lift']

            # Set arm positions (extended forward)
            qpos[4:10] = [0, 0.5, -0.8, 0.0, 0.0, 0.04 if server.vr_control['left_gripper'] else 0.0]
            qpos[10:16] = [0, 0.5, -0.8, 0.0, 0.0, 0.04 if server.vr_control['right_gripper'] else 0.0]

            # Step environment
            action = qpos.copy()
            obs, _, _, _, _ = env.step(action)

            # Update camera frame
            if 'sensor_data' in obs and 'cam_main' in obs['sensor_data']:
                rgb = obs['sensor_data']['cam_main']['rgb']
                if hasattr(rgb, 'cpu'):
                    rgb = rgb.cpu().numpy()
                if len(rgb.shape) == 4:
                    rgb = rgb[0]
                server.update_frame(rgb[:, :, :3].astype(np.uint8))

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Frame {frame_count}, Clients: {len(server.clients)}")

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
    args = parser.parse_args()

    if args.gpu:
        args.backend = 'gpu'

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
