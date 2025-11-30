#!/bin/bash
# Main training script wrapper for EuchreZero with auto device detection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

# Activate virtual environment if it exists
if [ -d "venv_vibe_eucher" ]; then
    source venv_vibe_eucher/bin/activate
fi

# Auto-detect device
DEVICE="auto"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        DEVICE="gpu"
    fi
fi

# Override device if specified via --device flag
if [ "$1" = "--device" ]; then
    DEVICE="$2"
    shift 2
fi

# Default to 1-minute training if no duration specified
if [[ ! "$*" =~ --duration ]]; then
    python scripts/training/euchre_zero/train_euchre_zero.py \
        --device "${DEVICE}" \
        --duration 1m \
        "$@"
else
    python scripts/training/euchre_zero/train_euchre_zero.py \
        --device "${DEVICE}" \
        "$@"
fi

