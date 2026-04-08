#!/usr/bin/env bash
# ============================================================
#  MARL Drone Swarm — Launch Training (bash)
#  Usage:   bash scripts/run_train.sh [config_name] [--resume path/to/ckpt.pt]
#  Example: bash scripts/run_train.sh mappo_mlp
#           bash scripts/run_train.sh mappo_mlp_dr --resume logs/mappo_mlp/checkpoints/ckpt_best.pt
#           bash scripts/run_train.sh mappo_gnn
# ============================================================

set -euo pipefail

# --- Activate venv ---
if [[ -f ".venv/Scripts/activate" ]]; then
    source .venv/Scripts/activate      # Git Bash / WSL on Windows
elif [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate           # Linux / macOS
else
    echo "[WARN] No .venv found — using system Python. Run scripts/install_deps.sh first."
fi

# --- Parse args ---
CONFIG="${1:-mappo_mlp}"
RESUME=""

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME="$2"; shift 2 ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

CONFIG_PATH="config/${CONFIG}.yaml"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERR] Config not found: $CONFIG_PATH"
    echo "      Available configs:"
    ls config/*.yaml | sed 's|config/||; s|\.yaml||' | xargs -I{} echo "        {}"
    exit 1
fi

echo "============================================================"
echo "  MARL Drone Swarm Training"
echo "  Config : $CONFIG_PATH"
[[ -n "$RESUME" ]] && echo "  Resume : $RESUME"
echo "============================================================"
echo

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CMD="python src/train.py --config $CONFIG_PATH"
[[ -n "$RESUME" ]] && CMD="$CMD --resume $RESUME"

exec $CMD
