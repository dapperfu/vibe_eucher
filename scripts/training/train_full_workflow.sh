#!/bin/bash
# Full Training Workflow Script - Complete training pipeline
# 
# This script runs the complete training workflow:
#   1. Collect training data through self-play
#   2. Train supervised models (sklearn)
#   3. Train RL agents until convergence
# 
# Setup:
#   1. Ensure virtual environment is activated: source venv_vibe_eucher/bin/activate
#   2. Ensure dependencies are installed: pip install -e .
#   3. Run this script: ./scripts/train_full_workflow.sh
#
# What it does:
#   Step 1: Collects training data (500 games)
#   Step 2: Trains supervised models on collected data
#   Step 3: Trains RL agents until 90% win rate
#
# Output:
#   - Training data: training_data/
#   - Trained models: models/
#   - Checkpoints: models/checkpoints/

set -e  # Exit on error

NUM_DATA_GAMES=${1:-500}        # Default 500 games for data collection
TARGET_WIN_RATE=${2:-0.90}      # Default 90% win rate target

echo "=========================================="
echo "Full Training Workflow Script"
echo "=========================================="
echo ""
echo "Setup:"
echo "  - Virtual environment: venv_vibe_eucher"
echo "  - Data collection games: ${NUM_DATA_GAMES}"
echo "  - Target win rate: ${TARGET_WIN_RATE} (${TARGET_WIN_RATE%%.*}%)"
echo ""
echo "Workflow:"
echo "  1. Collect training data (self-play)"
echo "  2. Train supervised models (sklearn)"
echo "  3. Train RL agents (until convergence)"
echo ""
echo "=========================================="
echo ""

# Step 1: Collect training data
echo "=========================================="
echo "Step 1: Collecting Training Data"
echo "=========================================="
echo ""
echo "Command:"
echo "  python scripts/train_models.py --mode self_play --num_games ${NUM_DATA_GAMES}"
echo ""
echo "Collecting ${NUM_DATA_GAMES} games of training data..."
echo ""

python scripts/train_models.py --mode self_play --num_games "${NUM_DATA_GAMES}"

echo ""
echo "Step 1 complete! Training data collected."
echo ""

# Step 2: Train supervised models
echo "=========================================="
echo "Step 2: Training Supervised Models"
echo "=========================================="
echo ""
echo "Command:"
echo "  python scripts/train_models.py --mode train"
echo ""
echo "Training sklearn models on collected data..."
echo ""

python scripts/train_models.py --mode train

echo ""
echo "Step 2 complete! Supervised models trained."
echo ""

# Step 3: Train RL agents until convergence
echo "=========================================="
echo "Step 3: Training RL Agents (Until Convergence)"
echo "=========================================="
echo ""
echo "Command:"
echo "  python scripts/train_models.py --mode train_rl --until-converged \\"
echo "    --target-win-rate ${TARGET_WIN_RATE}"
echo ""
echo "Training RL agents until ${TARGET_WIN_RATE%%.*}% win rate..."
echo ""

python scripts/train_models.py --mode train_rl --until-converged \
    --target-win-rate "${TARGET_WIN_RATE}"

echo ""
echo "=========================================="
echo "Full Training Workflow Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Training data: training_data/"
echo "  - Trained models: models/"
echo "  - Checkpoints: models/checkpoints/"
echo ""

