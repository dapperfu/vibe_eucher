#!/bin/bash
# GPU-specific training script for Transformer RL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_DIR}"

# Activate virtual environment if it exists
if [ -d "venv_vibe_eucher" ]; then
    source venv_vibe_eucher/bin/activate
fi

# Force GPU device
DEVICE="gpu"

# Run training script
python scripts/transformer_rl/train_transformer_rl.py \
    --device "${DEVICE}" \
    "$@"

