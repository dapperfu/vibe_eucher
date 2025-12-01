#!/bin/bash
# GPU-specific training script for EuchrePerceiverMuZero

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

# Run training script
# Default to 1-minute training if no duration specified
if [[ ! "$*" =~ --duration ]]; then
    python scripts/perceiver_muzero/train_perceiver_muzero.py \
        --device "${DEVICE}" \
        --duration 1m \
        "$@"
else
python scripts/perceiver_muzero/train_perceiver_muzero.py \
    --device "${DEVICE}" \
    "$@"
fi

