#!/bin/bash
# Main training script wrapper for PyTorch AI with auto device detection

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

# Detect GPU type and set appropriate batch size if using GPU
BATCH_SIZE_ARGS=""
if [ "$DEVICE" = "gpu" ] || [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 2>/dev/null || echo "0")
        if [ "${GPU_MEMORY}" -gt 0 ] && [ "${GPU_MEMORY}" -lt 13000 ]; then
            # RTX 3060 (12GB) configuration
            BATCH_SIZE_ARGS="--batch-size 256"
            echo "Detected RTX 3060 (12GB), using batch size 256"
        elif [ "${GPU_MEMORY}" -lt 25000 ]; then
            # P40 (24GB) configuration
            BATCH_SIZE_ARGS="--batch-size 128"
            echo "Detected P40 (24GB), using batch size 128"
        elif [ "${GPU_MEMORY}" -gt 0 ]; then
            # Other GPU
            BATCH_SIZE_ARGS="--batch-size 64"
            echo "Detected GPU with ${GPU_MEMORY}MB memory, using batch size 64"
        fi
    fi
fi

# Run training script
python scripts/training/pytorch_ai/train_pytorch_ai.py --device "${DEVICE}" ${BATCH_SIZE_ARGS} "$@"


