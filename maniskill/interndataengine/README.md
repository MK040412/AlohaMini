# AlohaMini → InternDataEngine integration (WIP)

Files for running AlohaMini as a custom robot in the real
[InternDataEngine](https://github.com/InternRobotics/InternDataEngine) (Isaac Sim /
Nimbus / simbox), producing InternData-A1-style synthetic manipulation data.

These are copies of the files that live inside an InternDataEngine checkout; drop them in:

| File here | Goes to (in the InternDataEngine repo) |
|-----------|----------------------------------------|
| `ide_robot_aloha_mini.py` | `workflows/simbox/core/robots/aloha_mini.py` (+ register in `robots/__init__.py`) |
| `ide_robot_aloha_mini.yaml` | `workflows/simbox/core/configs/robots/aloha_mini.yaml` |
| `convert_aloha_urdf.py` | run once (Isaac Sim) to make `assets/aloha_mini/robot.usd` |

## Environment setup (verified 2026-07-01, Isaac Sim 4.5)
Run everything under `<isaacsim>/python.sh` (py3.10, torch cu128):
1. `pip install` InternDataEngine `requirements.txt`, then **`pip install "numpy<2"`** —
   numpy 2 breaks the Isaac Sim extensions (`_ARRAY_API not found`, URDF importer fails).
2. `bash scripts/download_assets.sh --min --with-curobo --with-drake --local-dir .`
   (needs `hf` on PATH). Symlink `workflows/simbox/{curobo,assets}` → `InternDataAssets/{curobo,assets}`.
3. CuRobo prebuilt is CUDA-11/torch-cu118 → **rebuild** against cu128:
   `rm -rf build src/curobo/curobolib/*.so;
   SETUPTOOLS_SCM_PRETEND_VERSION=0.7.6 CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9
   python.sh -m pip install -e . --no-deps --no-build-isolation`, then
   `pip install yourdfpy trimesh warp-lang networkx "numpy<2"`.
   RUN with `LD_LIBRARY_PATH=<isaacsim>/kit/python/lib/python3.10/site-packages/torch/lib`.

## Robot facts (from the converted USD, articulation DOF order)
- `0-2` root_x/y/z (virtual base, held), `3` vertical_move (lift),
  `4-8` left arm, `9-10` left fingers, `11-15` right arm, `16-17` right fingers.
- ee prims: `left_Fixed_Jaw` / `right_Fixed_Jaw`; arm bases `left_Base` / `right_Base`.

## Remaining
- CuRobo kinematics YAML per arm (`aloha_mini_left/right_arm.yml`) with collision spheres.
- A task config under `core/configs/tasks/basic/aloha_mini/`.
- Run `scripts/simbox/simbox_pipe.sh`, then the LMDB→LeRobot v2.1 converter.
