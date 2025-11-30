#!/bin/bash
# Main wrapper script for ReinforcementEucher training with auto-device detection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Auto-detect device
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    DEFAULT_DEVICE="gpu"
else
    DEFAULT_DEVICE="cpu"
fi

# Allow override via --device flag
DEVICE="${DEFAULT_DEVICE}"
if [[ "$*" == *"--device"* ]]; then
    # Device specified in args, use it
    DEVICE_ARG=$(echo "$*" | grep -oE '--device\s+\w+' | awk '{print $2}')
    if [[ -n "${DEVICE_ARG}" ]]; then
        DEVICE="${DEVICE_ARG}"
    fi
fi

# Activate virtual environment if present
if [[ -d "${PROJECT_ROOT}/venv_vibe_eucher" ]]; then
    source "${PROJECT_ROOT}/venv_vibe_eucher/bin/activate"
fi

# Run training script
python "${SCRIPT_DIR}/train_reinforcement_eucher.py" --device "${DEVICE}" "$@"

