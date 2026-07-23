# AlohaMini × InternDataEngine — Integration Plan

Goal: generate **InternData-A1-style** synthetic manipulation data for AlohaMini
(SO-100 dual arm + roboninecom parallel gripper) and export as **LeRobot v2.1**.
The user wants BOTH tracks:
- **(B)** the InternDataEngine *architecture* re-implemented on **ManiSkill** (runs now, `Basic_RL/.venv`).
- **(C)** AlohaMini added as a custom robot to the **real InternDataEngine on Isaac Sim** (`env_isaaclab`).

Reference repo (cloned to scratch): `github.com/InternRobotics/InternDataEngine` ("IDE" below).
Researched with GPT-5.5 (codex); this doc is the actionable synthesis + my review notes.

---

## PART 1 — InternDataEngine internals (how it actually works)

Pipeline **Load → Plan → Render → Store** (Nimbus engine). Component classes:
- Loader: `nimbus_extension/components/load/env_loader.py:EnvLoader` → `scene.wf.load_asset()`
- Randomizer: `.../load/env_randomizer.py:EnvRandomizer` → `scene.wf.randomization()`
- Planner: `.../planner/env_planner.py:EnvSeqPlanner` → `scene.wf.generate_seq()`
- Renderer: `.../render/env_renderer.py:EnvRenderer` → `scene.wf.seq_replay()`
- Writer: `.../store/env_writer.py:EnvWriter` → `task.save()/save_seq()`
- Workflow base `workflows/base.py:NimbusWorkFlow`; dual-arm impl `workflows/simbox_dual_workflow.py:SimBoxDualWorkFlow`.

Trajectory generation = **CuRobo** motion/IK:
`workflows/simbox/core/controllers/template_controller.py:TemplateController` uses
`curobo...MotionGen` + `IKSolver`, loading `robot_cfg` per `robot_file`. Skills in
`workflows/simbox/core/skills/` (`Pick`, `Place`, `Heuristic_Skill`). Dual-arm controller
`splitaloha_controller.py:SplitAlohaController` assumes **6 arm joints** `joint1..joint6`
(⚠ AlohaMini has **5** arm joints/side — key mismatch).

Task YAML keys (`workflows/simbox/utils/task_config_parser.py:TaskConfigParser`):
`name, task, task_id, asset_root, arena_file, env_map, offset, render, robots, objects,
regions, cameras, data, skills, distractors, fluid, neglect_collision_names`.
robot keys: `name, robot_config_file, euler, ignore_substring, use_batch, tcp_offset,
left/right_joint_home, left/right_gripper_home, constrain_grasp_approach`.

LMDB schema (`workflows/simbox/core/loggers/lmdb_logger.py:LmdbLogger.save`):
`<out>/<robot>/<task_dir>/<collect_info>/<ts>/{lmdb/{data,lock}.mdb,info.json, meta_info.pkl}`.
Keys: `json_data`; `states.{left,right}_joint.position`; `actions.*`;
`master_actions.{left,right}_joint.position`, `..._gripper.position`, `..._gripper.openness`;
images `images.{rgb,depth,seg}.<camera>/<step>`.

LeRobot v2.1 converter `policy/lmdb2lerobotv21/lmdb2lerobot_split_aloha_a1.py`
(dual-arm template to fork). Expects cameras `images.rgb.{head,hand_left,hand_right}`;
writes `states.*`/`actions.*` per arm (joint, gripper, ee/tcp poses) + `states.robot_to_env_pose`.
⚠ Converter expects `robot2env_pose` but `TemplateRobot.get_observations()` returns `T_world_base` —
needs an explicit remap.

---

## PART 2 — (B) ManiSkill port (runs in `Basic_RL/.venv` now)

New package `maniskill/data_gen/intern_engine/`:
```
__init__.py launcher.py config.py registry.py types.py pipeline.py
components/{load,plan,render,store}.py
workflows/aloha_mini.py  planners/aloha_pick_cube.py
writers/{lmdb_logger,lerobot_v21}.py
```
Classes mirror IDE: `ManiSkillDataEngine.run()` + `Load/Plan/Render/StoreStage`;
`types.py` `Scene/Sequence/Observations/ActionStep`; `ManiSkillEnvLoader`,
`AlohaMiniDomainRandomizer`, `AlohaScriptedPlanner`/`ReplayPlanner`, `ManiSkillRenderer`,
`AlohaLmdbWriter`, `LeRobotV21Writer`.

Reuse: env `data_gen/tasks.py` (`AlohaMiniTablePick-v1`, `AlohaMiniPickCube-v1`),
agent `agents/aloha_mini/aloha_mini_1.py` (modes `pd_joint_pos`,
`pd_joint_delta_pos`, `pd_joint_pos_fixed_base`; cams `cam_main/cam_left_wrist/cam_right_wrist`;
16-D action = base3+lift1+Larm5+Lgrip1+Rarm5+Rgrip1). Planner uses
`teleop/kinematics/so100_kinematics_v2.py:SO100KinematicsV2` (planar IK).

Camera map: `cam_main→images.rgb.head`, `cam_left_wrist→hand_left`, `cam_right_wrist→hand_right`.
State map: `states.{left,right}_joint.position` = 5 arm qpos; `states.{left,right}_gripper.position`
= gripper aperture (0..0.037). Keep arm features shape **(5,)** native (pad to (6,) ONLY if a
downstream SplitAloha policy demands it).

Example config `data_gen/configs/aloha_mini_table_pick.yaml` (load/plan/render/store stages,
`open_gripper: 0.037`, output `data_gen/output/aloha_mini_lerobot`, fps 30).
Validate: `tools/validate_parallel_gripper.py`, `tools/smoke_test_gripper.py`, then
`python -m data_gen.intern_engine.launcher --config ...`.

Risks: 5-vs-6 DOF; reuse the working DLS-IK + fixed-base control we already built;
make the LeRobot schema AlohaMini-specific.

---

## PART 3 — (C) Isaac Sim custom robot (in `env_isaaclab`)

Registration path (no `custom/robot.py` in this clone): `register_robot` decorator in
`workflows/simbox/core/robots/base_robot.py`, imported via `robots/__init__.py`, referenced
by robot-YAML `target_class`.

Add:
```
workflows/simbox/core/robots/aloha_mini_so100.py            # AlohaMiniSO100(TemplateRobot) @register_robot
workflows/simbox/core/controllers/alohamini_so100_controller.py  # AlohaMiniSO100Controller(TemplateController)
workflows/simbox/core/configs/robots/aloha_mini_so100.yaml  # fork split_aloha.yaml; target_class, path, robot_file, gripper_max_width=0.074
workflows/simbox/core/configs/curobo/aloha_mini_{left,right}_arm.yml   # MISSING upstream — must author
workflows/simbox/core/configs/tasks/basic/aloha_mini/.../task.yaml
workflows/simbox/example_assets/aloha_mini_so100/robot.usd  # URDF->USD
policy/lmdb2lerobotv21/lmdb2lerobot_aloha_mini_a1.py        # fork split_aloha converter
```
Edit `robots/__init__.py`, `controllers/__init__.py`, `loggers/{utils,lmdb_logger}.py`
(add `aloha_mini` to `log_dual_obs` + gripper-openness handling).

Joint names from current URDF: left `left_shoulder_pan/lift, left_elbow_flex, left_wrist_flex/roll`;
right mirror; grippers `*_finger_joint1/2` (limit 0..0.037). Authoritative description =
`maniskill/assets/robots/aloha_mini/aloha_mini_1.urdf` (NOT the old ROS `Aloha.urdf`,
which is a stale 6-joint layout). CuRobo per-arm cfg needs `base_link: left_Base`, `ee_link: left_tcp`,
collision spheres; if the full mobile/dual URDF is too much, make reduced fixed-base per-arm URDFs.

**Hardest/risky (do these carefully):**
1. URDF→USD articulation fidelity (joint order, mimic gripper, collision filtering, TCP prims).
2. **CuRobo robot configs must be authored** (the referenced dir is absent in the clone).
3. **5-vs-6 DOF** mismatch vs SplitAloha assumptions.
4. `robot2env_pose` vs `T_world_base` remap.
5. Bug in `SimBoxDualWorkFlow._record_rgb_depth` (depth/seg may be unassigned) → start **RGB-only**.

---

## Suggested order
1. (B) ManiSkill `intern_engine` skeleton + LeRobot writer + one working `AlohaMiniTablePick` episode → LeRobot v2.1. (validatable here)
2. Harden planner (reliable grasp) + domain randomization + the 4 tasks.
3. (C) Isaac Sim: USD export → robot/controller class → CuRobo cfg → fork converter → RGB-only run.
