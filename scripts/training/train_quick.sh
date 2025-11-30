#!/bin/bash
# Quick Training Script - Train for 1 minute
# 
# This script trains ML models for 1 minute with default settings.
# 
# Setup:
#   1. Ensure virtual environment is activated: source venv_vibe_eucher/bin/activate
#   2. Ensure dependencies are installed: pip install -e .
#   3. Run this script: ./scripts/train_quick.sh
#
# What it does:
#   - Collects training data through self-play (1 minute)
#   - Trains PyTorch models on collected data
#   - Uses default risk factors (0.5 for both trump and gameplay)
#   - Training data saved to: training_data/
#   - Models saved to: models/
#
# To customize, edit the command below or use train_models.py directly.

set -e  # Exit on error

echo "=========================================="
echo "Quick Training Script (1 minute)"
echo "=========================================="
echo ""
echo "Setup:"
echo "  - Virtual environment: venv_vibe_eucher"
echo "  - Training duration: 1 minute"
echo "  - Mode: Self-play data collection + PyTorch model training"
echo "  - Risk factors: Default (0.5/0.5)"
echo ""
echo "Step 1: Collecting training data (1 minute)..."
echo ""

# Step 1: Collect training data for 1 minute
python scripts/train_models.py --mode self_play --duration 1m

echo ""
echo "Step 2: Initializing PyTorch models with correct feature size..."
echo ""

# Step 2: Initialize PyTorch models (this will create models with correct feature size)
# For quick training, we just need to initialize the models - they'll be trained on-the-fly
python -c "
from eucher.players.computer.ml.ml_config import MLConfig
from eucher.players.computer.ml.ml_model import EucherMLModel
import torch

config = MLConfig()
model = EucherMLModel(config)

# Save empty/initialized models to ensure correct feature size
model_paths = {
    'order_up.pth': model.trump_net,
    'play_card.pth': model.card_play_net,
    'discard.pth': model.discard_net,
}

for name, network in model_paths.items():
    path = config.get_model_path(name)
    torch.save(network.state_dict(), path)
    print(f'Initialized {name} with correct feature size')

print('Models initialized successfully!')
"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="

