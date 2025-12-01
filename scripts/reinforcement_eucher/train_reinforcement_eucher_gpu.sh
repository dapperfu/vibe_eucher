#!/bin/bash
# GPU-specific wrapper for ReinforcementEucher training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate virtual environment if present
if [[ -d "${PROJECT_ROOT}/venv_vibe_eucher" ]]; then
    source "${PROJECT_ROOT}/venv_vibe_eucher/bin/activate"
fi

# Run training script with GPU device
python "${SCRIPT_DIR}/train_reinforcement_eucher.py" --device gpu "$@"

