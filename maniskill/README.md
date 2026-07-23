# AlohaMini ManiSkill3 Integration

Integration guide for using the AlohaMini dual-arm mobile robot in ManiSkill3 simulation environment.

## Overview

AlohaMini is a dual-arm mobile robot. Two official models are provided (see [Robots](#robots)):

- **AlohaMini 1** (`aloha_mini_1`) — SO100 arms + parallel gripper. 16 DOF actions (base 3 + lift 1 + 2 x arm 6).
- **AlohaMini 2** (`aloha_mini_2`) — official AlohaMini2 Pro arms + parallel gripper. 18 DOF actions (base 3 + lift 1 + 2 x (arm 6 + gripper 1)).

Both share the same base layout:
- **Mobile Base**: Virtual prismatic X/Y + rotation joints
- **Vertical Lift**: 1 DOF prismatic joint
- **Dual Arms**: Left/Right 6 DOF manipulators

## Directory Structure

```
maniskill/
├── agents/aloha_mini/           # Agent class files
│   ├── __init__.py
│   ├── base_agent.py            # AlohaMiniBaseAgent (abstract)
│   ├── aloha_mini_1.py          # AlohaMini1 (SO100 arms)
│   └── aloha_mini_2.py          # AlohaMini2 (AM2 Pro arms)
├── assets/robots/aloha_mini/    # AlohaMini 1 URDF + meshes
│   ├── aloha_mini_1.urdf
│   ├── meshes/ so100_meshes/ clamp_meshes/
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
├── setup.py                     # Package setup
└── README.md
```

## Installation

### Using UV (Recommended - Faster)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer.

```bash
# Install UV (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

uv pip install mani-skill pygame websockets Pillow

# Install AlohaMini
cd maniskill
python install.py
```

### Using pip (Alternative)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install mani-skill pygame websockets Pillow

# Install AlohaMini
cd maniskill
python install.py
```

### What install.py does

- Copies agent files to ManiSkill installation
- Copies the AlohaMini 1 URDF/meshes to `~/.maniskill/data/robots/aloha_mini/`
- Updates ReplicaCAD scene builder

### AlohaMini 2 assets (one extra step)

AlohaMini 2's URDF + meshes are too large for the repo — download them from
[Releases](https://github.com/MK040412/AlohaMini/releases/tag/urdf-assets-v1):

```bash
unzip aloha_mini_2_urdf.zip -d ~/.maniskill/data/robots/
# -> ~/.maniskill/data/robots/aloha_mini_2/aloha_mini_2.urdf
```

Verify both robots load:

```bash
cd maniskill
python view_urdf.py mini1
python view_urdf.py mini2
```

### Uninstall

```bash
python install.py --uninstall
```

## Robots

Two official robots. Everything else was a prototype and has been removed.

| Robot | Agent / UID | URDF | Get it from |
|-------|-------------|------|-------------|
| **AlohaMini 1** | `AlohaMini1` / `aloha_mini_1` | `aloha_mini_1.urdf` — SO100 arms + roboninecom parallel gripper | this repo (`assets/robots/aloha_mini/`); `python install.py` copies it to `~/.maniskill/data/robots/aloha_mini/` |
| **AlohaMini 2** | `AlohaMini2` / `aloha_mini_2` | `aloha_mini_2.urdf` — official AlohaMini2 Pro arms + parallel gripper, black motors | GitHub **Releases** of this repo (zip with meshes); extract to `~/.maniskill/data/robots/` |

> **Note**: Both use a virtual base (prismatic X/Y + rotation) for stable locomotion.

Upstream source models (reference only, not needed at runtime):

- `alohamini2pro.urdf` — [liyiteng/alohamini](https://github.com/liyiteng/alohamini) `AlohaMini2/urdf/alohamini2pro/urdf/` (the arm model AlohaMini 2 was converted from)
- `so_101.urdf.xacro` — [roboninecom/SO-ARM100-101-Parallel-Gripper](https://github.com/roboninecom/SO-ARM100-101-Parallel-Gripper) `simulation/so_arm_101_description/urdf/` (the parallel-gripper donor)

View either robot in the SAPIEN GUI:

```bash
python view_urdf.py            # list keys
python view_urdf.py mini2      # mini1 | mini2
```

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
    robot_uids="aloha_mini_1",
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
```

## References

- [ManiSkill3 Documentation](https://maniskill.readthedocs.io/)
- [UV Package Installer](https://github.com/astral-sh/uv)
- [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) - Virtual base implementation reference
- [ReplicaCAD Dataset](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/scenes.html)
