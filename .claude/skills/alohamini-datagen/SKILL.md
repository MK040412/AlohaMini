---
name: alohamini-datagen
description: Generate GPU-parallel robot manipulation data for AlohaMini 2 in ManiSkill (physx_cuda batched multi-object cube-pick episodes). Use when the user asks to generate task data/episodes/demonstrations, e.g. "generate 100 episodes of red/blue cube picking with yaw diversity".
---

# AlohaMini GPU data generation

Turn a natural-language task request into GPU-parallel episode generation.

## Hard rules

- Data generation ALWAYS runs GPU-batched (`physx_cuda`, N envs in lock-step). Never fall back to per-episode CPU loops.
- `--envs 16` is the ceiling on an 8 GB GPU (N>16 → OOM / PhysX error 700). Keep 16 unless the user's GPU is bigger.
- Requires CuRobo in the active venv, the AlohaMini 2 assets installed (`python maniskill/install.py --check`), and a CUDA GPU.
- Only successful grasp+lift episodes are saved; attempts = batches × envs, so expect fewer files than attempts.

## How to run

From `maniskill/`:

```bash
python -m vec_datagen.gen --colors red,green,blue --batches 3
```

Map the user's request onto the knobs:

| User asks for | Flag |
|---|---|
| which cubes are on the table (3–4 colors) | `--colors red,green,blue[,yellow]` (target color is per-env random → instruction "pick up the {color} cube") |
| roughly how many episodes | `--batches ceil(target / 16)` (attempts; successes are typically most of them) |
| approach/trajectory diversity | `--station-noise 0.02 --yaw-rand` |
| recovery/robustness data (DART) | `--grasp-dart 2` |
| proprio/initial-pose randomization | `--arm-init-noise 0.05` |
| separate output dataset | `--out-dir <dir>` and a fresh `--episode-base` (e.g. 700000) |

Multiple color sets = run the command once per set with distinct `--episode-base` values (see `run_foft_div_pipe.sh` for the pattern). A JSON spec is equivalent: `python -m vec_datagen.gen --spec tasks.json` (keys = flag names).

## Output

`.npz` per successful episode in the out dir (default `vec_datagen/instr_out/`): per-step qpos/actions, per-step object positions, 5-camera frames metadata, instruction color. Downstream rendering/training consumes these directly (`vla/build_dense_2cam.py`).

## Verify after generating

```bash
ls vec_datagen/instr_out/*.npz | wc -l    # episode count
```

Report to the user: attempts vs saved successes, diversity knobs used, output dir.
