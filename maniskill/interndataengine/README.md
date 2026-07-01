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

## Pipeline status (2026-07-02) — runs end-to-end, physics-stability remains
The single-process pipeline (`configs/simbox/de_plan_and_render_template.yaml`, NOT the
multi-worker `de_pipe_template` which OOMs an 8 GB GPU) now takes the AlohaMini custom
robot through **load scene/robot/object → CuRobo grasp plan → Isaac Sim trajectory
execute → render**. It stops only at `PhysX error: Illegal BroadPhaseUpdateData`
(physics explosion during the scripted manipulation — same class as tuning a scripted
grasp; likely gripper convex-hull deep-penetration or too-aggressive joint velocity
limits). Fixes applied to reach this (also see the project memory):
1. `de_plan_and_render_template` (single process, fits 8 GB).
2. `pip install drake` (pydrake — the KPAM skill planner needs it).
3. Patch Isaac Sim deprecated shim `extsDeprecated/omni.isaac.core/.../materials/deformable_material.py` to re-export `DeformableMaterialView` from `isaacsim.core.api.materials.deformable_material_view`.
4. Task object path: use `../example_assets/...` (relative; the earlier `../../workflows/simbox/example_assets` double-counted). Keep RELATIVE (absolute breaks the USD load) and add a symlink `InternDataAssets/example_assets → workflows/simbox/example_assets` so `assets/../example_assets` resolves through the `assets` symlink for plain-open of `Aligned_grasp_sparse.npy`.
5. AlohaMini **controller** (`ide_controller_aloha_mini.py` → `core/controllers/aloha_mini_controller.py`, register in `__init__.py`).
6. CuRobo URDF copy: convert `continuous` joints → `revolute`+limit; **snap all joint axes to nearest principal X/Y/Z** (CuRobo only maps axis-aligned joints, else joint_type stays a str); set zero velocity/effort limits to 3.14/100.
7. Patch `nimbus/utils/utils.py init_env()`: `torch.backends.cudnn.enabled = False` (this Isaac torch's cuDNN fails to init; CuRobo trajopt works with native kernels).

Run: `LD_LIBRARY_PATH=<isaacsim>/kit/python/lib/python3.10/site-packages/torch/lib
PYTHONPATH=workflows/simbox <isaacsim>/python.sh launcher.py
--config configs/simbox/de_plan_and_render_template.yaml
--load_stage.scene_loader.args.cfg_path=<task> --load_stage.layout_random_generator.args.random_num=1 --debug`
