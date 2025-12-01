#!/bin/bash
# Main training script wrapper for Transformer RL with auto device detection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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

# Default values
NUM_HANDS="${NUM_HANDS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
RISK_FACTOR="${RISK_FACTOR:-0.5}"
USE_CURRICULUM="${USE_CURRICULUM:-true}"

# Parse arguments
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num-hands)
            NUM_HANDS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --risk-factor)
            RISK_FACTOR="$2"
            shift 2
            ;;
        --no-curriculum)
            USE_CURRICULUM="false"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Build command
CMD="python3 scripts/training/train_transformer_rl.py"
CMD="$CMD --device $DEVICE"
CMD="$CMD --num-hands $NUM_HANDS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --risk-factor $RISK_FACTOR"

if [ "$USE_CURRICULUM" = "true" ]; then
    CMD="$CMD --use-curriculum"
fi

# Add any additional arguments
CMD="$CMD ${ARGS[@]}"

echo "Running: $CMD"
exec $CMD


