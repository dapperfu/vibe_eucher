#!/bin/bash
# GPU-specific wrapper for EucherGo training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate virtual environment if it exists
if [ -d "${PROJECT_ROOT}/venv_vibe_eucher" ]; then
    source "${PROJECT_ROOT}/venv_vibe_eucher/bin/activate"
fi

# Force GPU device
python "${SCRIPT_DIR}/train_euchergo.py" --device cuda "$@"

