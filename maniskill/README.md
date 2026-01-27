# AlohaMini ManiSkill3 Integration

Integration guide for using the AlohaMini dual-arm mobile robot in ManiSkill3 simulation environment.

## Overview

AlohaMini is a dual-arm mobile robot with the following configuration:
- **Mobile Base**: Virtual prismatic X/Y + rotation joints
- **Vertical Lift**: 1 DOF prismatic joint
- **Dual Arms**: Left/Right 6 DOF SO100 manipulators each

**Total DOF**: 16 (base 3 + lift 1 + left arm 6 + right arm 6)

## Directory Structure

```
maniskill/
├── agents/aloha_mini/           # Agent class files
│   ├── __init__.py
│   ├── base_agent.py            # AlohaMiniBaseAgent (abstract)
│   └── aloha_mini_so100_v2.py   # AlohaMiniSO100V2 (main agent)
├── assets/robots/aloha_mini/    # URDF and mesh files
│   ├── maniskill_so100_version.urdf
│   └── so100_meshes/            # STL/PLY mesh files
├── teleop/                      # Teleoperation module
│   ├── demo_teleop.py           # Keyboard IK teleop (recommended)
│   ├── demo_vr_teleop_stream.py # VR teleop + camera streaming
│   ├── controller.py            # TeleopController
│   ├── config.py                # TeleopConfig
│   ├── inputs/                  # Input handlers (keyboard, VR)
│   ├── kinematics/              # IK modules
│   └── web_ui_stream/           # VR web UI
├── examples/                    # Example scripts
│   ├── demo_ee_keyboard.py      # EE keyboard control
│   └── run_replicacad.py        # ReplicaCAD environment
├── scene_builder/replicacad/    # Modified scene builder
│   └── scene_builder.py
├── install.py                   # Installation script
├── pyproject.toml               # Package configuration
└── README.md
```

## Installation

### Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install UV (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to maniskill directory
cd maniskill

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
# IMPORTANT: Package name is 'mani-skill' (with HYPHEN), NOT 'mani_skill'
uv pip install mani-skill pygame websockets Pillow scipy

# Install AlohaMini agent
python install.py

# Verify installation
python install.py --check
```

### Using pip (Alternative)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
# IMPORTANT: Package name is 'mani-skill' (with HYPHEN), NOT 'mani_skill'
pip install mani-skill pygame websockets Pillow scipy

# Install AlohaMini agent
cd maniskill
python install.py

# Verify installation
python install.py --check
```

### install.py Options

| Option | Description |
|--------|-------------|
| `python install.py` | Install AlohaMini into ManiSkill |
| `python install.py --check` | Verify installation status |
| `python install.py --uninstall` | Remove AlohaMini from ManiSkill |
| `python install.py --help` | Show help message |

### What install.py does

1. Detects virtual environment (supports uv and standard venv)
2. Copies agent files to ManiSkill installation (`mani_skill/agents/robots/aloha_mini/`)
3. Copies URDF/mesh files to `~/.maniskill/data/robots/aloha_mini/`
4. Updates ReplicaCAD scene builder
5. Registers agent in ManiSkill

## Robot Agent

| Agent | UID | Description |
|-------|-----|-------------|
| **AlohaMiniSO100V2** | `aloha_mini_so100_v2` | Virtual base robot with SO100 arms |

> **Note**: Uses virtual base (prismatic X/Y + rotation) for stable locomotion.

## Quick Start

### Keyboard IK Teleoperation (Recommended)

```bash
cd maniskill/teleop
python demo_teleop.py --render
```

**Controls (XLeRobot Style)**:

| Left Arm | Right Arm | Function |
|----------|-----------|----------|
| Q/A | U/J | Shoulder Pan -/+ |
| W/S | I/K | End-Effector X (forward/back) |
| E/D | O/L | End-Effector Y (down/up) |
| R/F | P/; | Pitch -/+ |
| T/G | [/' | Wrist Roll -/+ |
| Y/H | ]/\ | Gripper close/open |

| General | Function |
|---------|----------|
| SPACE | Reset arms to initial position |
| X/ESC | Exit |

### VR Teleoperation (Camera Streaming)

```bash
cd maniskill/teleop
python demo_vr_teleop_stream.py
```

Access `https://<your-ip>:8443` from VR headset browser.

## Python API

```python
import gymnasium as gym
import mani_skill.envs

# Import agent to register
from mani_skill.agents.robots import aloha_mini

# Create environment
env = gym.make(
    "ReplicaCAD_SceneManipulation-v1",
    robot_uids="aloha_mini_so100_v2",
    render_mode="human",
    sim_backend="gpu",
    control_mode="pd_joint_pos",
    sensor_configs=dict(shader_pack="rt-fast"),
    human_render_camera_configs=dict(shader_pack="rt-fast"),
    enable_shadow=True,
)

obs, info = env.reset(options=dict(reconfigure=True))

while True:
    action = env.action_space.sample() * 0.1
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
```

## Controllers

### Action Space (pd_joint_pos)

| Index | Joint | Description |
|-------|-------|-------------|
| 0 | base_x | X velocity (forward/back) |
| 1 | base_y | Y velocity (left/right) |
| 2 | base_rot | Rotation velocity |
| 3 | lift | Lift position |
| 4-9 | left_arm | Left arm 6 joints |
| 10-15 | right_arm | Right arm 6 joints |

**Total 16 DOF**

## Shader Options

| Shader | Description | Performance |
|--------|-------------|-------------|
| `default` | Basic rasterizer | Fast |
| `rt-fast` | Fast ray tracing | Medium |
| `rt` | High quality ray tracing | Slow |

## Troubleshooting

### Package Name Error

**Wrong**: `uv pip install mani_skill` (underscore)
**Correct**: `uv pip install mani-skill` (hyphen)

### ManiSkill Not Found

Make sure:
1. Virtual environment is activated (`source .venv/bin/activate`)
2. ManiSkill is installed with correct name (`uv pip install mani-skill`)
3. Run `python install.py --check` to diagnose issues

### Black Screen

```python
env = gym.make(
    ...,
    sensor_configs=dict(shader_pack="default"),
    human_render_camera_configs=dict(shader_pack="default"),
    enable_shadow=True,
)
```

Make sure to call `env.render()` every step.

### Keyboard Input Not Working

Focus on the pygame window. Demo scripts automatically create a control window.

### ManiSkill Import Error

Make sure `install.py` ran successfully:
```bash
python install.py
python install.py --check
```

## Verification

After installation, verify everything works:

```bash
# 1. Check installation status
python install.py --check

# 2. Test import
python -c "from mani_skill.agents.robots.aloha_mini import AlohaMiniSO100V2; print('Success!')"

# 3. Check asset files
ls ~/.maniskill/data/robots/aloha_mini/

# 4. Run demo (optional)
cd teleop && python demo_teleop.py --render
```

## References

- [ManiSkill3 Documentation](https://maniskill.readthedocs.io/)
- [UV Package Installer](https://github.com/astral-sh/uv)
- [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) - Virtual base implementation reference
- [ReplicaCAD Dataset](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/scenes.html)
