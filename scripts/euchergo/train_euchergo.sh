#!/bin/bash
# Main wrapper script for EucherGo training (auto-detects device)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate virtual environment if it exists
if [ -d "${PROJECT_ROOT}/venv_vibe_eucher" ]; then
    source "${PROJECT_ROOT}/venv_vibe_eucher/bin/activate"
fi

# Auto-detect device if not specified
DEVICE="${DEVICE:-auto}"

# Run training script with all arguments passed through
python "${PROJECT_ROOT}/scripts/training/train_euchergo.py" --device "${DEVICE}" "$@"

