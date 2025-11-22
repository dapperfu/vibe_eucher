#!/bin/bash
# Main training script wrapper with auto device detection

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
        DEVICE="cuda"
    fi
fi

# If device detection fails, use --device flag if provided
if [ "$1" = "--device" ]; then
    DEVICE="$2"
    shift 2
fi

# Run training script
python scripts/train_pytorch_ai.py --device "${DEVICE}" "$@"

