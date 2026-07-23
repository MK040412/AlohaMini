# GPU-Parallel Data Generation

Scripted expert demonstrations for AlohaMini 2, generated **batched on the GPU**: N environments step in lock-step on `physx_cuda`, and every motion plan for all N environments is solved in a single [CuRobo](https://curobo.org) `plan_batch` call. There is no per-episode CPU loop anywhere — parallelism is the design, not an option.

Measured on one RTX 4060 (8 GB): **~2,500 episodes/hour at N=16**, ~8× the single-env CPU pipeline, with a near-100% success rate per batch.

## The task

"Pick up the **{color}** cube": 3–4 distinct-color cubes sit on a table (`AM2MultiObject-v1`), each environment gets a random target color, and the robot must navigate its base to the table, descend, grasp the right cube, and lift it. Episodes therefore contain **nav + manipulation jointly** — the base drive to the station is part of the recorded data, not a teleport.

## Quick start

```bash
# from maniskill/ — requires a CUDA GPU + CuRobo in the venv
python -m vec_datagen.gen --colors red,green,blue --batches 3
```

Each batch attempts `--envs` (default 16) episodes in parallel; **only successful grasp+lift episodes are saved**, so files ≤ attempts. Progress prints per batch: `grasp+lift 16/16 ... (2555 ep/h)`.

More episodes / more variety = more batches with different color sets and disjoint id ranges:

```bash
python -m vec_datagen.gen --colors red,green,blue      --batches 3 --episode-base 700000
python -m vec_datagen.gen --colors yellow,orange,purple --batches 3 --episode-base 720000
```

A JSON spec is equivalent (keys = flag names): `python -m vec_datagen.gen --spec tasks.json`

**LLM-driven**: open this repo in Claude Code and just ask — *"generate 100 episodes of red/blue cube picking with yaw diversity"*. The bundled skill (`.claude/skills/alohamini-datagen/`) maps the request onto these flags.

## Diversity knobs (all optional, all composable)

| Flag | What it randomizes | Why |
|------|--------------------|-----|
| `--station-noise 0.02` | base station XY offset (m) — the robot parks off-center | approach-trajectory diversity |
| `--yaw-rand` | grasp yaw over the 4 IK-verified yaws | wrist-pose diversity |
| `--grasp-dart 2` | N perturb→replan→re-grasp rounds at the grasp point | recovery data (DART) — teaches the policy to fix near-misses |
| `--arm-init-noise 0.05` | initial arm joint positions (rad) | proprioception robustness |

Scene-level randomization (cube layout per env, target color per env) is always on.

## How an episode is produced

```
settle at station (unrecorded)  ->  frame capture ready
drive base back to origin (unrecorded)
record: locate (5 steps, robot at origin, looking)
        approach (base ramp origin -> station, arm executes CuRobo hover plan)
        descend  (CuRobo plan down to grasp pose, TCP stops TIP_Z=0.1056 m above cube center)
        grasp    (close parallel gripper + dwell)
        lift     (raise; success = grasped AND cube_z > start + 5 cm)
```

The base ramp is recorded (COORD scheme) so downstream policies learn the nav signal; per-step **object positions are recorded live**, so replays show the cube actually lifting.

## Output format

One `ep_<id>.npz` per successful episode in `instr_out/` (or `--out-dir`):

| Key | Shape | Meaning |
|-----|-------|---------|
| `qpos` | (T, 20) | full robot joint positions per step |
| `action` | (T, 18) | commanded actions (base 3 + lift 1 + arm 6 + grip 1, ×2 arms) |
| `phase` | (T,) str | `locate / approach / descend / grasp / lift` |
| `obj_positions` | (T, K, 3) | live cube positions per step |
| `target_pos` | (T, 3) | target cube start position |
| `colors`, `target_color`, `target_idx` | — | scene colors + which one is the goal |
| `instruction` | str | `"pick up the {color} cube"` |
| `seed`, `cube_half` | — | episode id, cube half-size (m) |

Downstream: `vla/build_dense_2cam.py` renders these into 5-camera panel frames for VLA training (front + both wrists + chest + back, the robot's real camera set).

## Hard constraints (8 GB GPU)

- **N=16 is the ceiling** — N=24 → PhysX CUDA error 700, N≥32 → OOM (CuRobo warp + physx_cuda share the VRAM). Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Don't kill a run mid-batch: zombie processes hold VRAM and cascade the next run into OOM. Let a batch finish or clean up with `nvidia-smi` before relaunching.
- Colors are fixed per process (env reconfigure under `physx_cuda` crashes) — that's why multiple color sets = multiple `gen.py` invocations.

## Files

```
vec_datagen/
├── gen.py                  # spec-driven front-end (flags / JSON) — start here
├── vec_pick_gen_mo.py      # multi-object COORD generator (the engine gen.py drives)
├── vec_pick_gen.py         # single-cube predecessor (kept for reference)
├── vec_env.py              # AM2VecPickPlace-v1 / AM2MultiObject-v1 environments
├── curobo_pickplace.py     # CuRobo planning helpers (TCP frame, stations, plan_batch)
├── curobo_am2pro_left.yml  # CuRobo robot config for the AlohaMini 2 left arm
└── batched_ik.py           # batched DLS IK (pytorch_kinematics), CuRobo-free fallback
```
