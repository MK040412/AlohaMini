#!/usr/bin/env bash
# Recreate the ManiSkill venv used for AlohaMini simulation + data generation.
# Mirrors /home/perelman/Basic_RL/.venv (mani_skill 3.0.1, sapien 3.0.3,
# torch 2.11.0+cu128, python 3.11). Uses uv (fast, reproducible).
#
# Usage:  bash setup_maniskill_venv.sh [VENV_DIR]
#   VENV_DIR defaults to /home/perelman/Basic_RL/.venv
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${1:-/home/perelman/Basic_RL/.venv}"
ALOHA_MANISKILL="$(cd "$HERE/../maniskill" && pwd)"

command -v uv >/dev/null || { echo "uv not found (expected /home/perelman/.local/bin/uv)"; exit 1; }

echo ">>> creating py3.11 venv at $VENV"
uv venv --python /usr/bin/python3.11 "$VENV"

echo ">>> installing pinned requirements (torch cu128 + mani_skill 3.0.1)"
uv pip install --python "$VENV/bin/python" -r "$HERE/requirements_maniskill.txt"

echo ">>> registering the AlohaMini agent + assets into mani_skill"
"$VENV/bin/python" "$ALOHA_MANISKILL/install.py"

echo ">>> verifying"
"$VENV/bin/python" - <<'PY'
import mani_skill, sapien, torch, gymnasium as gym
import data_gen  # noqa  (only works if run from maniskill/; otherwise import skip)
print("mani_skill", mani_skill.__version__, "sapien", sapien.__version__,
      "torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
echo ">>> ManiSkill venv ready: $VENV"
echo "    validate:  $VENV/bin/python $ALOHA_MANISKILL/tools/smoke_test_gripper.py"
