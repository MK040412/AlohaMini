# AlohaMini — ManiSkill3 Simulation

Simulate the AlohaMini dual-arm mobile robot in [ManiSkill3](https://maniskill.readthedocs.io/) / SAPIEN.
Two official robot models, ready-made tabletop pick/place/stack environments, a GUI robot viewer, and keyboard/VR teleoperation.

| Robot | UID | Arms | Actions | Assets |
|-------|-----|------|---------|--------|
| **AlohaMini 1** | `aloha_mini_1` | SO100 (5-DOF) + parallel gripper | 16 | in this repo (installed by `install.py`) |
| **AlohaMini 2** | `aloha_mini_2` | official AM2 Pro (6-DOF) + parallel gripper | 18 | [release zip](https://github.com/MK040412/AlohaMini/releases/tag/urdf-assets-v1) (step 2 below) |

Both are mobile: virtual planar base (x, y, rotation) + a vertical lift, with two arms on the lift carriage.

---

## 1. Setup

Requirements: Linux, Python 3.10–3.11, `wget` + `unzip`. A GPU is optional (physics runs on CPU); the GUI viewer needs a display + Vulkan.

```bash
git clone https://github.com/MK040412/AlohaMini.git
cd AlohaMini/maniskill

# Python environment (uv — https://github.com/astral-sh/uv)
uv venv --python 3.11
source .venv/bin/activate
uv pip install mani-skill pygame websockets Pillow

# Install both robot agents + AlohaMini 1 assets into ManiSkill
python install.py

# AlohaMini 2 assets are a separate ~100 MB download (too large for the repo)
wget https://github.com/MK040412/AlohaMini/releases/download/urdf-assets-v1/aloha_mini_2_urdf.zip
unzip aloha_mini_2_urdf.zip -d ~/.maniskill/data/robots/
rm aloha_mini_2_urdf.zip
```

<details>
<summary>No uv? Same thing with plain pip</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mani-skill pygame websockets Pillow
python install.py
wget https://github.com/MK040412/AlohaMini/releases/download/urdf-assets-v1/aloha_mini_2_urdf.zip
unzip aloha_mini_2_urdf.zip -d ~/.maniskill/data/robots/
rm aloha_mini_2_urdf.zip
```
</details>

### Verify

```bash
python install.py --check              # prints OK/FAIL per component, with the fix for each FAIL
python view_urdf.py mini1 --headless   # loads AlohaMini 1 in the simulator, no window
python view_urdf.py mini2 --headless   # loads AlohaMini 2 in the simulator, no window
```

Expected output of the last two:

```
[VIEW] aloha_mini_1: URDF loaded OK (...)
[VIEW] aloha_mini_2: URDF loaded OK (...)
```

> **If this crashes with `RuntimeError: The NVIDIA driver on your system is too old`**: pip resolved a torch built for a newer CUDA than your driver. Reinstall torch for your driver's CUDA and re-run — e.g. for a CUDA 12.8 driver:
> ```bash
> uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
> ```
> (check yours with `nvidia-smi`, top-right "CUDA Version")

`install.py` is idempotent — re-run it any time (e.g. after `pip install -U mani-skill`). `python install.py --uninstall` removes everything it added (the separately-downloaded AlohaMini 2 assets are kept).

---

## 2. Check the robots (GUI)

This is the normal way to verify the robots after setup — look at them, put them in a scene, drive one yourself. On a machine with a display:

```bash
# look at the model
python view_urdf.py mini1        # AlohaMini 1 in the SAPIEN viewer
python view_urdf.py mini2        # AlohaMini 2 (black motors)

# see it in a task scene
python demo.py table --render    # AlohaMini 1 at the cube-pick table
python demo.py empty --render    # AlohaMini 2 alone

# drive it yourself (keyboard IK — full key map in Section 6)
cd teleop && python demo_teleop.py --render
```

Close the window to exit. `python view_urdf.py` / `python demo.py --help` list all options.

---

## 3. Python API

Minimal example — spawn AlohaMini 2 in an empty scene and step it (works headless, no extra assets). Save it as e.g. `demo.py` and run `python demo.py` — any directory works:

```python
import gymnasium as gym
import mani_skill.envs                       # registers ManiSkill environments
import mani_skill.agents.robots.aloha_mini   # registers aloha_mini_1 / aloha_mini_2

env = gym.make(
    "Empty-v1",
    robot_uids="aloha_mini_2",     # or "aloha_mini_1"
    obs_mode="state",
    sim_backend="physx_cpu",
    reward_mode="none",
)
obs, info = env.reset(seed=0)
for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample() * 0)
env.close()
```

Add `render_mode="human"` and call `env.render()` each step to watch it live.

---

## 4. Task environments

The repo ships ready-made tabletop environments. They are registered by importing their module. Save the snippet **as a file inside `maniskill/`** (the imports resolve relative to the script's location) and run it from there:

```python
import gymnasium as gym
import data_gen.tasks        # registers the AlohaMini* environments (AlohaMini 1)
import vec_datagen.vec_env   # registers the AM2* environments (AlohaMini 2)

env = gym.make("AlohaMiniTablePick-v1", obs_mode="state",
               sim_backend="physx_cpu", reward_mode="none")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample() * 0)
```

| Environment | Robot | Task |
|-------------|-------|------|
| `AlohaMiniTablePick-v1` | 1 | pick a cube off a raised table (left arm) |
| `AlohaMiniPickPlace-v1` | 1 | pick the red cube, place it on a goal platform |
| `AlohaMiniStack-v1` | 1 | stack the red cube on the blue cube |
| `AlohaMiniPickCube-v1` | 1 | ManiSkill PickCube task adapted to AlohaMini |
| `AlohaMiniMultiYCB-v1` | 1 | multi-object YCB tabletop pick (needs YCB assets, below) |
| `AlohaMiniGripperView-v1` | 1 | close-up gripper inspection camera |
| `AM2VecPickPlace-v1` | 2 | table pick-place with per-env domain randomization, built for GPU batching |
| `AM2MultiObject-v1` | 2 | multiple colored cubes + place marker, per-env target |

Notes:

- `AlohaMiniMultiYCB-v1` needs the YCB object set once — see Section 7
- The `AM2*` environments also run batched on GPU: `gym.make(..., num_envs=16, sim_backend="physx_cuda")`
- These environments provide observations/success flags for scripted data generation, not shaped rewards — hence `reward_mode="none"`.

---

## 5. Control modes and action spaces

Both robots expose three control modes (pass `control_mode=...` to `gym.make`):

| Mode | Base | Arms | Use for |
|------|------|------|---------|
| `pd_joint_pos` (default) | velocity | absolute position | general control, teleop |
| `pd_joint_delta_pos` | velocity | delta position | RL policies |
| `pd_joint_pos_fixed_base` | position-held | absolute position | scripted manipulation / data generation |

Action layout (`pd_joint_pos`):

**AlohaMini 1 — 16 actions**

| Index | Group | Meaning |
|-------|-------|---------|
| 0–2 | base | x vel, y vel, yaw vel |
| 3 | lift | height (m) |
| 4–8 | left arm | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (rad) |
| 9 | left gripper | 0.0 closed → 0.037 open (m) |
| 10–14 | right arm | same 5 joints |
| 15 | right gripper | 0.0 closed → 0.037 open (m) |

**AlohaMini 2 — 18 actions**

| Index | Group | Meaning |
|-------|-------|---------|
| 0–2 | base | x vel, y vel, yaw vel |
| 3 | lift | height (m), travel −0.3…+0.3 |
| 4–9 | left arm | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll (rad) |
| 10 | left gripper | 0.0 closed → 0.037 open (m) |
| 11–16 | right arm | same 6 joints |
| 17 | right gripper | 0.0 closed → 0.037 open (m) |

Each gripper is one action: the two clamp joints are mirror-coupled in the controller. AlohaMini 2 additionally carries the 5 real cameras (front/back/chest + both wrists) as sensor cameras at their true mount poses.

---

## 6. Teleoperation

Keyboard IK teleop (AlohaMini 1):

```bash
cd teleop
python demo_teleop.py --render
```

| Left arm | Right arm | Function |
|----------|-----------|----------|
| Q/A | U/J | shoulder pan −/+ |
| W/S | I/K | end-effector forward/back |
| E/D | O/L | end-effector down/up |
| R/F | P/; | pitch −/+ |
| T/G | [/' | wrist roll −/+ |
| Y/H | ]/\ | gripper close/open |

SPACE resets the arms, X or ESC exits. Keep the pygame window focused.

VR teleop with camera streaming: `python demo_vr_teleop_stream.py`, then open `https://<your-ip>:8443` in the headset browser.

---

## 7. Demo runner + optional asset packs

`demo.py` runs every scene with one command — GUI with `--render`, headless smoke without it:

```bash
python demo.py empty --render                 # robot in an empty scene
python demo.py table --render                 # tabletop cube pick
python demo.py table                          # same scene, headless (steps 100x, prints OK)
```

Two scenes need a one-time optional asset download. Download the pack, then run the scene with the matching option — if the pack is missing, `demo.py` prints the exact download command instead of crashing:

```bash
# YCB object set (~1 GB) -> multi-object table
python -m mani_skill.utils.download_asset ycb
python demo.py ycb --render

# ReplicaCAD apartment (~2 GB) -> whole-apartment scene
python -m mani_skill.utils.download_asset ReplicaCAD
python demo.py replicacad --render --shader rt-fast
```

| Option | Values | Meaning |
|--------|--------|---------|
| `--robot` | `mini1` / `mini2` | robot choice, where the scene allows it (`empty`, `replicacad`) |
| `--render` | flag | open the SAPIEN viewer (default: headless) |
| `--steps` | int | headless step count (default 100) |
| `--shader` | `default` / `rt-fast` / `rt` | render quality, fast → best |

---

## 8. Data generation (GPU-parallel)

Episode generation always runs **batched on the GPU** (`physx_cuda`, N environments in lock-step + CuRobo batched motion planning) — never per-episode CPU loops. Requires a CUDA GPU and [CuRobo](https://curobo.org) in the venv; `--envs 16` is the ceiling on an 8 GB GPU.

```bash
# 3 batches x 16 parallel envs of "pick up the {color} cube" (per-env random target);
# only successful grasp+lift episodes are saved as .npz
python -m vec_datagen.gen --colors red,green,blue --batches 3

# with trajectory diversity + DART recovery + proprio randomization
python -m vec_datagen.gen --colors red,yellow,blue --batches 3 \
    --station-noise 0.02 --yaw-rand --grasp-dart 2 --arm-init-noise 0.05

# or drive it from a JSON task spec (keys = flag names)
python -m vec_datagen.gen --spec my_tasks.json
```

**LLM-driven generation**: the repo ships a Claude Code skill at `.claude/skills/alohamini-datagen/` — open the repo in Claude Code and just ask, e.g. *"generate 100 episodes of red/blue cube picking with yaw diversity"*; the skill maps the request onto the flags above.

Full documentation — episode anatomy, `.npz` schema, diversity knobs, GPU limits: **[vec_datagen/README.md](vec_datagen/README.md)**

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `install.py --check` prints FAIL | the FAIL line itself prints the exact fix command |
| `AlohaMini 2 assets: MISSING` | you skipped the release zip — run the `wget` + `unzip` from step 1 |
| `ImportError` on `mani_skill.agents.robots.aloha_mini` | re-run `python install.py` (also needed after reinstalling mani-skill) |
| GUI viewer fails / black screen | you need a display + Vulkan; use `--headless` to verify on a server, or `shader_pack="default"` and call `env.render()` every step |
| `RuntimeError: The NVIDIA driver on your system is too old` | torch was built for a newer CUDA than your driver — see the reinstall note in [Setup § Verify](#verify) |
| `NotImplementedError` in `get_reward` | pass `reward_mode="none"` (the task envs define success, not dense rewards) |
| keyboard teleop ignores keys | click the pygame window to focus it |

## Repository layout

```
maniskill/
├── install.py                # installer: agents + assets + registration (--check / --uninstall)
├── view_urdf.py              # GUI / headless robot viewer
├── demo.py                   # scene runner: empty | table | ycb | replicacad (--render / --robot)
├── agents/aloha_mini/        # AlohaMini1, AlohaMini2 agent classes (+ _validate_aloha_mini_2.py)
├── assets/robots/aloha_mini/ # AlohaMini 1 URDF + meshes (AlohaMini 2 comes from Releases)
├── data_gen/tasks.py         # AlohaMini* task environments (the rest of data_gen/ is research pipelines)
├── vec_datagen/vec_env.py    # AM2* GPU-batchable environments (+ CuRobo-based data-gen scripts)
├── teleop/                   # keyboard / VR teleoperation
└── scene_builder/            # ReplicaCAD scene-builder patch
```

Upstream source models (reference only, not needed at runtime):
[liyiteng/alohamini](https://github.com/liyiteng/alohamini) (`alohamini2pro.urdf` — the arm model AlohaMini 2 was converted from) ·
[roboninecom/SO-ARM100-101-Parallel-Gripper](https://github.com/roboninecom/SO-ARM100-101-Parallel-Gripper) (`so_101.urdf.xacro` — the parallel-gripper donor)

## References

- [ManiSkill3 documentation](https://maniskill.readthedocs.io/)
- [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) — virtual-base implementation reference
- [ReplicaCAD dataset](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/scenes.html)
