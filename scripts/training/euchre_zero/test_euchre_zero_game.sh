#!/bin/bash
# Test script to run a complete game with EuchreZero bots

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_DIR}"

# Activate virtual environment if it exists
if [ -d "venv_vibe_eucher" ]; then
    source venv_vibe_eucher/bin/activate
fi

# Check if checkpoint provided
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path> [--device cpu|gpu]"
    echo "Example: $0 models/checkpoints/euchre_zero/checkpoint_final.pt"
    exit 1
fi

CHECKPOINT="$1"
shift

# Device (default: auto)
DEVICE="${1:-auto}"
if [ "$DEVICE" = "--device" ]; then
    DEVICE="${2:-auto}"
    shift 2
fi

# Run test script
python scripts/training/euchre_zero/test_euchre_zero_game.py \
    --checkpoint "${CHECKPOINT}" \
    --device "${DEVICE}" \
    "$@"

