#!/bin/bash
# CPU-specific training script for PyTorch AI player

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_DIR}"

# Activate virtual environment if it exists
if [ -d "venv_vibe_eucher" ]; then
    source venv_vibe_eucher/bin/activate
fi

# Force CPU device
# Optimized for 32GB RAM systems
python scripts/training/pytorch_ai/train_pytorch_ai.py \
    --device cpu \
    --batch-size 16 \
    "$@"


