#!/bin/bash
# Isaac Sim 5.1.0 + Isaac Lab 2.3.2 installation (pip workflow, no sudo).
#
# Why uv-managed Python 3.11: Isaac Sim 5.x pip wheels are Python-3.11-only
# (manylinux_2_35); this box has system Python 3.10 and no passwordless sudo,
# so we use uv standalone CPython instead of the deadsnakes PPA.
#
# Ref: docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html
#      isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
set -euo pipefail

ISAAC_VENV="$HOME/env_isaaclab"
ISAACLAB_DIR="$HOME/IsaacLab"
PHYSGATE_DIR="$HOME/Desktop/physgate"

export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "=== [1/7] Install Python 3.11 via uv ==="
uv python install 3.11

echo "=== [2/7] Create Python 3.11 venv at $ISAAC_VENV ==="
uv venv "$ISAAC_VENV" --python 3.11 --seed
# shellcheck disable=SC1091
source "$ISAAC_VENV/bin/activate"
python --version
pip install --upgrade pip

echo "=== [3/7] Install Isaac Sim 5.1.0 (multi-GB download, be patient) ==="
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

echo "=== [4/7] Install PyTorch 2.7.0+cu128 (after isaacsim, per official docs) ==="
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

echo "=== [5/7] Verify torch sees Blackwell sm_120 ==="
python -c "import torch; cap = torch.cuda.get_device_capability(); print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'capability', cap); assert cap == (12, 0), f'expected sm_120, got {cap}'"

echo "=== [6/7] Clone + install Isaac Lab 2.3.2 ==="
if [ ! -d "$ISAACLAB_DIR" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
cd "$ISAACLAB_DIR"
git fetch --tags
git checkout v2.3.2
./isaaclab.sh --install

echo "=== [7/7] Install physgate (editable + dev) into this venv ==="
cd "$PHYSGATE_DIR"
pip install -e ".[dev]"
pytest -q

echo "=== INSTALL COMPLETE ==="
