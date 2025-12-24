# AlohaMini
[![Join our Discord](https://img.shields.io/badge/Discord-Join%20chat-blue)](https://discord.gg/CacMUBaFgJ) [![Follow on X](https://img.shields.io/twitter/follow/liyitengx?style=social)](https://x.com/liyitengx)


AlohaMini is a dual-arm mobile robot with a motorized vertical lift — beautifully designed, fully 3D-printable, and affordable.  

Built for embodied AI research and real-world manipulation. Assemble at home in ~60 minutes, customize every part, and train or deploy with LeRobot.

Note:
Use ROS only if you need URDF visualization, RViz inspection, or Gazebo simulation.


## Updates

- [2025-12-24] — **ManiSkill3 Integration**: AlohaMini is now available in ManiSkill3 simulation! See [maniskill/README.md](maniskill/README.md) for details.
- [2025-12-18] — URDF and simulation assets are now available for visualization and simulation (`simulation/src/Aloha/`)
- [2025-11-27] STEP (CAD) files are now available (`/hardware/mobile_base/step/`)


### What Makes It Different

- **Motorized vertical lift** — 0–60 cm travel (floor-to-table reach)
- **5-camera perception system** — top, front, back, and dual arm cameras engineered for embodied-AI reproducibility
- **Completely open-source** — hardware and software freely available
- **LeRobot-compatible** — works out of the box
- **Clean, modern design** — built for both function and aesthetics
- **Low-cost & accessible** — engineered so anyone can build a capable robot at home


## 📸 Gallery

<p align="center">
  <img src="docs/media/alohamini_git.png" width="100%"/>
</p>


## ⚙️ Bill of Materials (BOM)
### Main Components

| Component | Model / Notes | Qty | Unit Price (USD) |
|-----------|---------------|-----|------------------|
| Servo motors | Feetech STS3215 (12V bus) | 16 | $13.89 |
| Motor control boards | Waveshare Bus Servo Adapter (A) | 2 | $10.55 |
| Compute platform | Raspberry Pi 5 (4/8GB) | 1 | $80 |
| Cameras | 720p USB cameras (2 for arms, 3 for mobile base) | 5 | $20 |
| Mobility system | Omni wheels | 3 | $36.00 |
| Battery | 12V Li-ion pack | 2 | $32.99 |
| Frame | 3D-printed body (ABS / PETG / PLA) | — | ~4kg filament (self-print) |
| **Total** | — | — | **~$600 (self-print)** |

Note: 
- Printable STL files under `/hardware/`
- Compute platform can be replaced with Jetson Nano or similar SBCs if desired.
- URDF files will be released soon

## Quick Start

Start building and running AlohaMini:

1. **Hardware acquisition** — Purchase components and 3D print parts  
   See **[BOM & 3D-Print](docs/BOM.md)**

1. **Assembly** — build the robot in ~60 minutes (SO-ARM pre-assembled)  
   See **[assembly guide](docs/hardware_assembly.md)**

1. **Software setup & teleoperation** — install, connect, and control the robot  
   See **[software guide](docs/software_setup.md)**





##  Product Line
| Model | Build | Rigidity | Target Users |
|---|---|---|---|
| **AlohaMini** | Fully 3D-printed | Standard | Education, makers, research labs, home builds |
| **AlohaMini Pro** | Hybrid **3D-print + metal** | **3×–5× stiffer** | Algorithm-focused researchers who need plug-and-play, reliable, and stable hardware — without assembly effort |

> Same URDF & control stack across both versions — only structural materials differ.

### AlohaMini Pro Access  
Pro hardware & kits available for advanced users.  
📩 **Contact to request early access.**

## Contact
Email: liyiteng+github@gmail.com  
WeChat: liyiteng  
Videos & tutorials soon on: Bilibili / YouTube / TikTok

## Team
AlohaMini is created by:  
**Li Yiteng** / **Wu Zhiyong**

##  Acknowledgements
Thanks to the open robotics community:  
**ALOHA · LeKiwi · SO-ARM100 · SO-ARM100-Track-Axis · Pi-0.5 · LeRobot · Hugging Face**

## ManiSkill3 Simulation

AlohaMini can be used in the [ManiSkill3](https://maniskill.readthedocs.io/) robotics simulation environment for training and testing manipulation and navigation tasks.

### Quick Install

```bash
# 1. Install ManiSkill3
pip install mani-skill

# 2. Download ReplicaCAD dataset
python -m mani_skill.utils.download_asset ReplicaCAD

# 3. Install AlohaMini into ManiSkill
cd maniskill
python install.py
```

### Run Demo

```bash
cd maniskill/examples
python demo_virtual_base.py --render
```

**Controls (FPS Style):**
- `W/S`: Forward/Backward
- `A/D`: Strafe Left/Right
- `Q/E`: Rotate Left/Right
- `R/F`: Lift Up/Down

### Robot Variants

| Variant | UID | Description |
|---------|-----|-------------|
| `aloha_mini` | Physical wheels | Wheel friction research |
| `aloha_mini_fixed` | Fixed base | Manipulation only |
| `aloha_mini_virtual` | Virtual mobile base **(Recommended)** | Navigation + Manipulation |

> **Recommended**: Use `aloha_mini_virtual` for stable navigation. It uses prismatic X/Y + rotation joints (same approach as XLeRobot).

See [maniskill/README.md](maniskill/README.md) for detailed documentation.

## Support AlohaMini
If you like this project:
- Star the repo
- Follow updates
- Join the community

