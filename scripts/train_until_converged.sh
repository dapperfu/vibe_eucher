#!/bin/bash
# Convergence Training Script - Train until model converges on target win rate
# 
# This script trains ML models until they achieve a target win rate (default 90%).
# 
# Setup:
#   1. Ensure virtual environment is activated: source venv_vibe_eucher/bin/activate
#   2. Ensure dependencies are installed: pip install -e .
#   3. Run this script: ./scripts/train_until_converged.sh
#
# What it does:
#   - Trains RL agents through self-play
#   - Monitors win rate over last 100 games (configurable)
#   - Stops when win rate >= target (default 90%)
#   - Uses default risk factors (0.5 for both trump and gameplay)
#   - Saves checkpoints every 100 games
#
# To customize:
#   - Change target win rate: --target-win-rate 0.95
#   - Change window size: --window-size 200
#   - Change risk factors: --trump-selection-risk 0.3 --gameplay-risk 0.7

set -e  # Exit on error

TARGET_WIN_RATE=${1:-0.90}  # Default 90%, can override: ./train_until_converged.sh 0.95
WINDOW_SIZE=${2:-100}        # Default 100 games, can override: ./train_until_converged.sh 0.90 200

echo "=========================================="
echo "Convergence Training Script"
echo "=========================================="
echo ""
echo "Setup:"
echo "  - Virtual environment: venv_vibe_eucher"
echo "  - Training mode: Until convergence"
echo "  - Target win rate: ${TARGET_WIN_RATE} (${TARGET_WIN_RATE%%.*}%)"
echo "  - Window size: ${WINDOW_SIZE} games"
echo "  - Risk factors: Default (0.5/0.5)"
echo ""
echo "Command to run:"
echo "  python scripts/train_models.py --mode train_rl --until-converged \\"
echo "    --target-win-rate ${TARGET_WIN_RATE} --window-size ${WINDOW_SIZE}"
echo ""
echo "Starting training (will stop when win rate >= ${TARGET_WIN_RATE%%.*}%)..."
echo ""

# Run the training command
python scripts/train_models.py --mode train_rl --until-converged \
    --target-win-rate "${TARGET_WIN_RATE}" --window-size "${WINDOW_SIZE}"

echo ""
echo "=========================================="
echo "Training complete! Model has converged."
echo "=========================================="

