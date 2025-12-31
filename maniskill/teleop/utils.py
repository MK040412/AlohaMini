"""
Utility functions for AlohaMini Teleoperation

Includes SSL certificate generation for WebXR compatibility.
"""

import os
import ssl
import subprocess
import http.server
import threading
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    """
    Get the local IP address of the machine.

    Returns:
        Local IP address as string
    """
    import socket
    try:
        # Create a socket and connect to an external address
        # This doesn't actually send data, just determines the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generate_ssl_certificates(
    ssl_dir: str,
    certfile: str = "cert.pem",
    keyfile: str = "key.pem",
    days: int = 365,
    common_name: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Generate self-signed SSL certificates for WebXR.

    WebXR requires HTTPS, so we need SSL certificates even for local development.

    Args:
        ssl_dir: Directory to store certificates
        certfile: Certificate file name
        keyfile: Key file name
        days: Certificate validity in days
        common_name: Common name for certificate (default: local IP)

    Returns:
        Tuple of (success, message)
    """
    ssl_path = Path(ssl_dir)

    # Create directory if needed
    try:
        ssl_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Failed to create SSL directory: {e}"

    cert_path = ssl_path / certfile
    key_path = ssl_path / keyfile

    # Check if certificates already exist
    if cert_path.exists() and key_path.exists():
        return True, "SSL certificates already exist"

    # Get common name (local IP if not specified)
    if common_name is None:
        common_name = get_local_ip()

    # Try to generate using openssl command
    try:
        # Generate private key and certificate
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:4096",
            "-keyout", str(key_path),
            "-out", str(cert_path),
            "-days", str(days),
            "-nodes",  # No password
            "-subj", f"/CN={common_name}",
            "-addext", f"subjectAltName=IP:{common_name},DNS:localhost",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info(f"SSL certificates generated successfully at {ssl_dir}")
            return True, f"SSL certificates generated at {ssl_dir}"
        else:
            logger.error(f"openssl failed: {result.stderr}")
            return False, f"openssl failed: {result.stderr}"

    except FileNotFoundError:
        return False, "openssl command not found. Please install OpenSSL."
    except Exception as e:
        return False, f"Failed to generate SSL certificates: {e}"


def setup_ssl_context(certfile: str, keyfile: str) -> Optional[ssl.SSLContext]:
    """
    Create an SSL context for the WebSocket server.

    Args:
        certfile: Path to certificate file
        keyfile: Path to key file

    Returns:
        SSL context or None if setup fails
    """
    if not os.path.exists(certfile) or not os.path.exists(keyfile):
        logger.error(f"SSL files not found: {certfile}, {keyfile}")
        return None

    try:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=certfile, keyfile=keyfile)
        logger.info("SSL context created successfully")
        return ssl_context
    except ssl.SSLError as e:
        logger.error(f"Error loading SSL cert/key: {e}")
        return None


def print_network_info(https_port: int, ws_port: int):
    """
    Print network information for VR connection.

    Args:
        https_port: HTTPS server port
        ws_port: WebSocket server port
    """
    local_ip = get_local_ip()

    print("\n" + "=" * 60)
    print("VR TELEOPERATION SERVER")
    print("=" * 60)
    print(f"\nLocal IP: {local_ip}")
    print(f"\nVR Web Interface:")
    print(f"  https://{local_ip}:{https_port}")
    print(f"\nWebSocket Server:")
    print(f"  wss://{local_ip}:{ws_port}")
    print("\nNOTE: Accept the self-signed certificate warning in your browser")
    print("=" * 60 + "\n")


class SimpleHTTPSHandler(http.server.BaseHTTPRequestHandler):
    """Simple HTTP request handler for serving VR web UI files."""

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


class HTTPSFileServer:
    """
    HTTPS file server for serving VR web UI.

    Required for WebXR as browsers require HTTPS for WebXR API access.
    """

    def __init__(self, web_ui_path: str, port: int, certfile: str, keyfile: str, host: str = "0.0.0.0"):
        """
        Initialize HTTPS file server.

        Args:
            web_ui_path: Path to directory containing web UI files
            port: HTTPS server port
            certfile: Path to SSL certificate file
            keyfile: Path to SSL key file
            host: Host address to bind to
        """
        self.web_ui_path = web_ui_path
        self.port = port
        self.certfile = certfile
        self.keyfile = keyfile
        self.host = host
        self.httpd = None
        self.server_thread = None
        self.is_running = False

    def start(self):
        """Start the HTTPS file server in a background thread."""
        try:
            # Create HTTP server
            self.httpd = http.server.HTTPServer((self.host, self.port), SimpleHTTPSHandler)
            self.httpd.web_root_path = self.web_ui_path

            # Setup SSL context
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(self.certfile, self.keyfile)
            self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)

            # Start in background thread
            self.server_thread = threading.Thread(target=self._serve_forever, daemon=True)
            self.server_thread.start()
            self.is_running = True

            logger.info(f"HTTPS file server started on https://{self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start HTTPS server: {e}")
            return False

    def _serve_forever(self):
        """Serve requests until stopped."""
        try:
            self.httpd.serve_forever()
        except Exception as e:
            logger.error(f"HTTPS server error: {e}")
        finally:
            self.is_running = False

    def stop(self):
        """Stop the HTTPS file server."""
        if self.httpd:
            self.httpd.shutdown()
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=5)
            logger.info("HTTPS file server stopped")
        self.is_running = False
