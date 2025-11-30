#!/bin/bash
# Risk-Based Training Script - Train with custom risk/temperature factors
# 
# This script trains ML models with custom risk factors for trump selection and gameplay.
# 
# Setup:
#   1. Ensure virtual environment is activated: source venv_vibe_eucher/bin/activate
#   2. Ensure dependencies are installed: pip install -e .
#   3. Run this script: ./scripts/train_with_risk.sh
#
# Risk Factor Guide:
#   - Low (0.0-0.3): Conservative - only high-confidence actions
#   - Medium (0.3-0.7): Balanced play
#   - High (0.7-1.0): Risky - accepts lower-confidence actions
#
# Examples:
#   Conservative trump, risky gameplay:
#     ./scripts/train_with_risk.sh 0.2 0.8
#
#   Balanced play (default):
#     ./scripts/train_with_risk.sh 0.5 0.5
#
#   Risky trump, conservative gameplay:
#     ./scripts/train_with_risk.sh 0.9 0.2

set -e  # Exit on error

TRUMP_RISK=${1:-0.5}    # Default 0.5, can override: ./train_with_risk.sh 0.3
GAMEPLAY_RISK=${2:-0.5} # Default 0.5, can override: ./train_with_risk.sh 0.3 0.7
DURATION=${3:-"2h"}     # Default 2 hours, can override: ./train_with_risk.sh 0.3 0.7 "1d"

echo "=========================================="
echo "Risk-Based Training Script"
echo "=========================================="
echo ""
echo "Setup:"
echo "  - Virtual environment: venv_vibe_eucher"
echo "  - Training duration: ${DURATION}"
echo "  - Trump selection risk: ${TRUMP_RISK}"
echo "  - Gameplay risk: ${GAMEPLAY_RISK}"
echo ""
echo "Risk Interpretation:"
echo "  - Trump selection risk: ${TRUMP_RISK}"
echo "    (Low: 0.0-0.3, Medium: 0.3-0.7, High: 0.7-1.0)"
echo "  - Gameplay risk: ${GAMEPLAY_RISK}"
echo "    (Low: 0.0-0.3, Medium: 0.3-0.7, High: 0.7-1.0)"
echo ""
echo "Command to run:"
echo "  python scripts/train_models.py --mode train_rl --duration ${DURATION} \\"
echo "    --trump-selection-risk ${TRUMP_RISK} --gameplay-risk ${GAMEPLAY_RISK}"
echo ""
echo "Starting training..."
echo ""

# Run the training command
python scripts/train_models.py --mode train_rl --duration "${DURATION}" \
    --trump-selection-risk "${TRUMP_RISK}" --gameplay-risk "${GAMEPLAY_RISK}"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="

