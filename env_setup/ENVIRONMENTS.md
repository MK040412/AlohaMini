# AlohaMini — Simulation Environments (re-setup guide)

Two **separate** venvs are used. They cannot be merged into one because their
**torch builds conflict**: ManiSkill ships torch **cu128**, while Isaac Sim 4.5
pins torch **2.5.1 + cu118**. Keep them apart.

| Purpose | venv | python | torch | key pkgs |
|---|---|---|---|---|
| **ManiSkill** sim + data-gen (track B) | `/home/perelman/Basic_RL/.venv` | 3.11 | 2.11.0+cu128 | mani_skill 3.0.1, sapien 3.0.3, gymnasium 1.3.0, numpy 2.4.6 |
| **Isaac Sim** + InternDataEngine (track C) | `/home/perelman/env_isaaclab` | 3.10 | 2.5.1+cu118 | isaacsim 4.5.0.0, isaaclab 2.1.1 |

GPU: NVIDIA RTX 4060 (Ada, sm_89). Vulkan headless rendering works for both.

## Files here
- `requirements_maniskill.txt` — full `uv pip freeze` of the ManiSkill venv (130 pkgs).
- `requirements_isaac.txt` — full freeze of the Isaac venv (232 pkgs).
- `setup_maniskill_venv.sh` — recreate the ManiSkill venv + register the AlohaMini agent.
- `build_isaac_venv.sh` — the user's verified Isaac Sim 4.5 + IsaacLab 2.1.1 build script
  (creates `env_isaaclab`; torch cu118 pinned to avoid the IsaacLab #3524 hang).

## Re-setup

### ManiSkill venv
```bash
bash env_setup/setup_maniskill_venv.sh            # -> /home/perelman/Basic_RL/.venv
# verify
/home/perelman/Basic_RL/.venv/bin/python maniskill/tools/smoke_test_gripper.py
```

### Isaac Sim venv
```bash
bash env_setup/build_isaac_venv.sh                # -> /home/perelman/env_isaaclab
# verify
/home/perelman/env_isaaclab/bin/python /home/perelman/Basic_RL/_isaac_launch_test.py
```

### InternDataEngine (track C, into the Isaac venv)
```bash
source /home/perelman/env_isaaclab/bin/activate
git clone https://github.com/InternRobotics/InternDataEngine
uv pip install -r InternDataEngine/requirements.txt
# CuRobo must be installed separately (the simbox/curobo robot-config dir is NOT in the clone).
```

## Which venv for what
- `maniskill/` (robot URDF/agent, `data_gen/`, `tools/`, `intern_engine` track B) → **Basic_RL/.venv**.
- Isaac references on the machine: `/home/perelman/IsaacLab-SO_100`,
  `/home/perelman/SO-ARM101_MoveIt_IsaacSim`, `/home/perelman/BEHAVIOR-1K`.

Notes: the default `python3` is UV py3.11 and has **neither** sim stack. Always use the
explicit venv path. After editing the AlohaMini URDF/agent, re-run
`Basic_RL/.venv/bin/python maniskill/install.py` so `~/.maniskill/data` stays current.
