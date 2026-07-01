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

## ROOT CAUSE of the ~534 "Plan did not converge" — FIXED (2026-07-02)
It was **NOT** physics / collision / reachability / orientation (all red herrings). Every
grasp target fed to CuRobo had position = **FLT_MAX (3.4028e38)** because the robot was
placed at **Z = −inf**. Chain: `core/utils/region_sampler.py::A_on_B_region_sampler` sets
the robot Z from `compute_bbox(robot).min[2]`; **`compute_bbox`
(`core/utils/usd_geom_utils.py`) returned an EMPTY box (min=+FLT_MAX)** for the robot, so
`place_pos[2] = tgt_z_max + (obj_local_z − FLT_MAX) + shift_z = −inf`. The robot fell to
−inf → `left_Base` world Z = −inf → base transform Z = FLT_MAX → every grasp target
FLT_MAX → nothing could converge. Why the box was empty (two compounding reasons):
(1) the Isaac `Robot.prim` is the **ArticulationRoot *joint* prim** (`.../aloha_mini/root_joint`),
whose SIBLINGS — not children — hold the link meshes; (2) Isaac loads the URDF robot's
visual meshes as **instance proxies AND marks them invisible**, so
`UsdGeom.Imageable.ComputeWorldBound(default)` skips them.

**FIX = `ide_usd_geom_utils.py` → `core/utils/usd_geom_utils.py`** (drop-in): if the fast
`ComputeWorldBound(default, render)` path is empty, fall back to a manual union that reads
each mesh's local **extent/points directly** (ignores visibility), traverses **instance
proxies** (`Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))`),
and **walks UP to ancestor prims** until geometry is found (root_joint → `/World/task_0/aloha_mini`).
Note: the token is `UsdGeom.Tokens.render` (NOT `render_`; only `default_` has the underscore).
After this, the robot is placed on the floor (Z≈−0.006), the base transform is valid, and
**every grasp target is a real reachable position** — the single fix resolved all 534 failures.

Also move the robot within the 5-DOF arm's ~0.45 m reach (the arm can't reach a target
0.56 m away): `ide_task_pick_object.yaml` robot region `pos_range` `[0,-0.55,-0.765]` →
`[-0.20,-0.28,-0.765]` (robot euler is `[0,0,90]`).

**REMAINING (open-ended, same class as the 6-DOF grasp):** with valid, reachable targets,
CuRobo still can't converge a plan because it enforces the **full 6-DOF grasp pose** and the
under-actuated **5-DOF SO-100 arm can't achieve the bottle's side-grasp orientations**
(confirmed: not collision — shrinking CuRobo spheres to 0.006 didn't help; not position —
selected grasp `[-0.098, 0.08, 0.054]` is 0.14 m, well inside the reach envelope;
`constrain_grasp_approach` is a linear-approach constraint, not orientation relaxation, so it
doesn't help). Needs CuRobo pose-cost customization to relax one orientation DOF for the
under-actuated arm, or pre-filtering grasps to the top-down subset the 5-DOF arm can reach,
or a custom position+approach IK (as used on the ManiSkill side).
