"""
Configuration for AlohaMini Teleoperation System

Based on XLeRobot XLeVR config pattern.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class TeleopConfig:
    """
    Configuration for the AlohaMini teleoperation system.

    Attributes:
        # Network settings
        host_ip: IP address to bind servers to
        https_port: Port for HTTPS server (VR web UI)
        websocket_port: Port for WebSocket server (VR data)

        # SSL settings
        ssl_dir: Directory for SSL certificates
        certfile: SSL certificate file path
        keyfile: SSL key file path

        # Feature flags
        enable_vr: Enable VR WebSocket server
        enable_keyboard: Enable keyboard control

        # VR settings
        vr_to_robot_scale: Scale factor from VR space to robot space

        # Keyboard settings
        pos_step: Position step per keypress (meters)
        angle_step: Angle step per keypress (degrees)
        pitch_step: Pitch adjustment step (degrees)

        # IK parameters (SO101)
        l1: Upper arm length (meters)
        l2: Lower arm length (meters)

        # Initial arm positions
        initial_ee_x: Initial end-effector X position
        initial_ee_y: Initial end-effector Y position

        # Control loop settings
        control_freq: Control loop frequency (Hz)
    """
    # Network settings
    host_ip: str = "0.0.0.0"
    https_port: int = 8443
    websocket_port: int = 8442

    # SSL settings - use local certs by default (XLeRobot style)
    ssl_dir: str = field(default_factory=lambda: str(Path(__file__).parent))
    certfile: Optional[str] = None  # Auto-set based on ssl_dir
    keyfile: Optional[str] = None   # Auto-set based on ssl_dir

    # Web UI path for VR interface
    web_ui_path: str = field(default_factory=lambda: str(Path(__file__).parent / "web_ui"))

    # Feature flags
    enable_vr: bool = True
    enable_keyboard: bool = True

    # VR settings (legacy - replaced by delta control parameters)
    vr_to_robot_scale: float = 1.5  # VR 10cm movement → robot 15cm movement

    # VR delta control settings (Vector Wang style)
    vr_position_scale_xy: float = 0.7    # Scale for ee_x, ee_y (70 * 0.01)
    vr_position_scale_pan: float = 220.0  # Scale for shoulder pan (degrees per meter)
    vr_delta_limit: float = 0.01         # Max position delta per frame (meters)
    vr_angle_delta_limit: float = 8.0    # Max angle delta per frame (degrees)
    vr_angle_scale: float = 4.0          # Scale for wrist angles

    # Keyboard settings
    pos_step: float = 0.004  # 4mm per keypress
    angle_step: float = 1.0  # 1 degree per keypress
    pitch_step: float = 1.0  # 1 degree per keypress

    # IK parameters (AlohaMini URDF geometry)
    # Link lengths computed from URDF joint offsets
    l1: float = 0.1160  # Upper arm length (meters) - from joint3 Y-Z offset
    l2: float = 0.1342  # Lower arm length (meters) - from joint4 Y-Z offset

    # Initial arm positions (AlohaMini URDF coordinate system)
    # At home (j2=0, j3=0): ee_y=-0.0218, ee_z=0.0324
    # The arm is folded at home position (forearm points backward)
    # ee_y = forward direction (negative = behind shoulder)
    # ee_z = height (positive = up)
    initial_ee_y: float = -0.0218  # Forward position (URDF Y)
    initial_ee_z: float = 0.0324   # Height (URDF Z)

    # Workspace info
    # The arm workspace is primarily in negative Y (behind shoulder) due to
    # the backward-pointing forearm geometry in the URDF.
    # Maximum extension: y = -(l1+l2) ≈ -0.25m, z can vary
    # Minimum folded: y ≈ -(l2-l1) ≈ -0.02m at home

    # Control loop settings
    control_freq: float = 50.0  # Hz

    def __post_init__(self):
        """Set up derived values after initialization."""
        # Set SSL file paths if not specified
        if self.certfile is None:
            self.certfile = str(Path(self.ssl_dir) / "cert.pem")
        if self.keyfile is None:
            self.keyfile = str(Path(self.ssl_dir) / "key.pem")

    @property
    def ssl_files_exist(self) -> bool:
        """Check if SSL certificate files exist."""
        return (
            os.path.exists(self.certfile) and
            os.path.exists(self.keyfile)
        )

    def ensure_ssl_dir(self) -> bool:
        """Ensure SSL directory exists."""
        try:
            Path(self.ssl_dir).mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    @property
    def control_period(self) -> float:
        """Get control loop period in seconds."""
        return 1.0 / self.control_freq

    @classmethod
    def from_yaml(cls, path: str) -> "TeleopConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            TeleopConfig instance
        """
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration
        """
        import yaml
        from dataclasses import asdict
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# Default configuration instance
DEFAULT_CONFIG = TeleopConfig()
