#!/usr/bin/env bash
# Version-matched uv venv for Isaac Sim 4.5.0 + Isaac Lab v2.1.1 (cu118 / py3.10)
# Derived from verified research (incl. IsaacLab issue #3524 + Sim 4.5.0 pip docs).
set -x
export OMNI_KIT_ACCEPT_EULA=YES
VENV=/home/perelman/env_isaaclab
LAB=/home/perelman/IsaacLab_2.1.1

# 1. fresh py3.10 uv venv (explicit python -> avoids #3524's silent 3.13)
rm -rf "$VENV"
uv venv --python /usr/bin/python3.10 "$VENV" || { echo "VENV_FAIL"; exit 1; }
source "$VENV/bin/activate"
python --version

# 2. real pip + build backend first
uv pip install --upgrade pip "setuptools==70.*" wheel || { echo "PIP_FAIL"; exit 1; }

# 3. torch FIRST, pinned cu118 (THE anti-spin-hang pin; RTX 4060 = Ada sm_89, cu118 native)
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118 || { echo "TORCH_FAIL"; exit 1; }

# 4. Isaac Sim 4.5.0 via pip (NVIDIA index; extscache pre-caches Kit extensions)
uv pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com || { echo "ISAACSIM_FAIL"; exit 1; }

# 5. Isaac Lab v2.1.1 (Sim-4.5.0-matched tag; NOT the failed 2.2.1)
rm -rf "$LAB"
git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git -b v2.1.1 "$LAB" || { echo "CLONE_FAIL"; exit 1; }
cd "$LAB"
./isaaclab.sh -i 2>&1 | tail -40

# 6. re-assert pins LAST (isaaclab -i may bump torch->cu128 / numpy>=2)
uv pip install "numpy==1.26.4" "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu118

# 7. verify
echo "=== VERIFY ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')); import numpy; print('numpy', numpy.__version__)"
echo "=== ISAAC UVENV BUILD DONE ==="
