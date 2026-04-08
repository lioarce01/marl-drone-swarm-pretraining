#!/usr/bin/env bash
# ============================================================
#  MARL Drone Swarm — One-shot Setup (bash)
#  Run from the project root: bash scripts/install_deps.sh
#
#  Works on: Linux, macOS, WSL, Git Bash (Windows)
#  For native Windows CMD/PowerShell use install_deps.bat
# ============================================================

set -euo pipefail

# --- Helpers ---
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
die()   { echo -e "\033[1;31m[ERR ]\033[0m  $*" >&2; exit 1; }

# ============================================================
# 1. Python version check
# ============================================================
info "[1/6] Checking Python..."

# Try candidates in order: python3, python, py (Windows launcher)
PYTHON=""
for candidate in python3 python py; do
    if command -v "$candidate" &>/dev/null; then
        # On Windows, 'python' may resolve to the Store stub — it prints nothing useful
        # and exits 9009. Test it actually runs.
        if "$candidate" -c "import sys" &>/dev/null 2>&1; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    die "Python not found in PATH. Options:
  1. Add Python to PATH: open the Python installer → 'Modify' → check 'Add to PATH'
  2. Disable the Store alias: Settings → Apps → Advanced app settings → App execution aliases → turn off 'python'
  3. Use the full path: PYTHON=/c/Users/<you>/AppData/Local/Programs/Python/Python310/python.exe bash scripts/install_deps.sh"
fi

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ) ]]; then
    die "Python 3.10+ required, found $PY_VER. Install from https://python.org"
fi

if [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -ge 13 ]]; then
    warn "Python $PY_VER detected. gym-pybullet-drones on PyPI requires Python <3.13."
    warn "Will attempt GitHub source install instead. If it fails, use Python 3.11:"
    warn "  1. Install Python 3.11 from https://python.org"
    warn "  2. rm -rf .venv"
    warn "  3. py -3.11 -m venv .venv"
    warn "  4. bash scripts/install_deps.sh"
    echo
    PYBULLET_FROM_SOURCE=1
else
    PYBULLET_FROM_SOURCE=0
fi

ok "Python $PY_VER found at $(command -v $PYTHON)"

# ============================================================
# 2. Virtual environment
# ============================================================
info "[2/6] Creating virtual environment (.venv)..."
if [[ ! -d ".venv" ]]; then
    "$PYTHON" -m venv .venv
    ok "Virtual environment created."
else
    warn ".venv already exists — skipping creation."
fi

# Activate
if [[ -f ".venv/Scripts/activate" ]]; then
    # Git Bash / WSL on Windows
    source .venv/Scripts/activate
elif [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
else
    die "Could not find venv activation script."
fi
ok "Virtual environment activated."

# ============================================================
# 3. Upgrade pip
# ============================================================
info "[3/6] Upgrading pip..."
python -m pip install --upgrade pip --quiet
ok "pip upgraded."

# ============================================================
# 4. PyTorch with CUDA
# ============================================================
info "[4/6] Installing PyTorch..."
echo
echo "  Select your CUDA version:"
echo "    1) CUDA 12.8  (recommended — RTX 50 series / Blackwell, driver 572+)"
echo "    2) CUDA 12.1  (RTX 30/40 series)"
echo "    3) CUDA 11.8  (older / GTX 16xx / RTX 20xx)"
echo "    4) CPU only   (no GPU)"
echo
read -rp "  Enter choice [1]: " cuda_choice
cuda_choice="${cuda_choice:-1}"

case "$cuda_choice" in
    1) TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
    2) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    3) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
    4) TORCH_INDEX="" ;;
    *) warn "Invalid choice, defaulting to CUDA 12.8."
       TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
esac

if [[ -n "$TORCH_INDEX" ]]; then
    python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" --quiet
else
    python -m pip install torch torchvision torchaudio --quiet
fi
ok "PyTorch installed."

# ============================================================
# 5. Project dependencies
# ============================================================
info "[5/6] Installing project dependencies..."
python -m pip install \
    gymnasium \
    pettingzoo \
    numpy \
    pyyaml \
    tensorboard \
    tqdm \
    matplotlib \
    opencv-python \
    imageio \
    imageio-ffmpeg \
    pytest \
    --quiet

# gym-pybullet-drones: no Windows wheels on PyPI — always install from GitHub source
info "Installing gym-pybullet-drones from GitHub source..."
python -m pip install "git+https://github.com/utiasDSL/gym-pybullet-drones.git" && \
    ok "gym-pybullet-drones installed from source." || \
    die "gym-pybullet-drones install failed. Try manually:
  git clone https://github.com/utiasDSL/gym-pybullet-drones.git
  cd gym-pybullet-drones && python -m pip install -e . && cd .."
ok "Core dependencies installed."

info "Installing EPyMARL from source..."
python -m pip install "git+https://github.com/uoe-agents/epymarl.git" --quiet && \
    ok "EPyMARL installed." || \
    warn "EPyMARL install failed (needs git). Retry manually: python -m pip install git+https://github.com/uoe-agents/epymarl.git"

# ============================================================
# 6. Verification
# ============================================================
info "[6/6] Verifying installation..."
echo
python -c "import torch; print(f'  PyTorch : {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"
python -c "import gymnasium; print(f'  Gymnasium : {gymnasium.__version__}')"
python -c "
try:
    from gym_pybullet_drones.envs import HoverAviary
    e = HoverAviary()
    e.reset()
    e.close()
    print('  gym-pybullet-drones : OK')
except Exception as ex:
    print(f'  gym-pybullet-drones : FAILED ({ex})')
"
python -c "import tensorboard; print('  TensorBoard : OK')"

echo
echo "============================================================"
ok "Setup complete."
echo
echo "  Activate the environment in future sessions with:"
if [[ -f ".venv/Scripts/activate" ]]; then
    echo "    source .venv/Scripts/activate   # Git Bash / WSL"
else
    echo "    source .venv/bin/activate        # Linux / macOS"
fi
echo
echo "  Start training:"
echo "    bash scripts/run_train.sh mappo_mlp"
echo "============================================================"
