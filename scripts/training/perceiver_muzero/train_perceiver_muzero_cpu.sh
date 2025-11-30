#!/bin/bash
# CPU-specific training script for EuchrePerceiverMuZero

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_DIR}"

# Activate virtual environment if it exists
if [ -d "venv_vibe_eucher" ]; then
    source venv_vibe_eucher/bin/activate
fi

# Force CPU device
# Default to 1-minute training if no duration specified
if [[ ! "$*" =~ --duration ]]; then
    python scripts/training/perceiver_muzero/train_perceiver_muzero.py \
        --device cpu \
        --duration 1m \
        "$@"
else
python scripts/training/perceiver_muzero/train_perceiver_muzero.py \
    --device cpu \
    "$@"
fi

