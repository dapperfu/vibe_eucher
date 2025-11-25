#!/bin/bash
# GPU-specific training script for PyTorch AI player

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"

# Activate virtual environment if it exists
if [ -d "venv_vibe_eucher" ]; then
    source venv_vibe_eucher/bin/activate
fi

# Force GPU device
DEVICE="gpu"

# Detect GPU type and set appropriate batch size
BATCH_SIZE=""
if command -v nvidia-smi &> /dev/null; then
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    if [ "${GPU_MEMORY}" -lt 13000 ]; then
        # RTX 3060 (12GB) configuration
        BATCH_SIZE="--batch-size 256"
        echo "Detected RTX 3060 (12GB), using batch size 256"
    elif [ "${GPU_MEMORY}" -lt 25000 ]; then
        # P40 (24GB) configuration
        BATCH_SIZE="--batch-size 128"
        echo "Detected P40 (24GB), using batch size 128"
    else
        # Other GPU
        BATCH_SIZE="--batch-size 64"
        echo "Detected GPU with ${GPU_MEMORY}MB memory, using batch size 64"
    fi
fi

# Run training script
python scripts/pytorch_ai/train_pytorch_ai.py \
    --device "${DEVICE}" \
    ${BATCH_SIZE} \
    "$@"


