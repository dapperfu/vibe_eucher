#!/usr/bin/env bash
# Score a trained EucherHandForest checkpoint.
#
# Usage:
#   scripts/training/evaluate_eucher_hand_forest.sh \
#     --checkpoint checkpoints/EucherHandForest/checkpoint_best.pth \
#     --data-path data/EucherHandForest/v1 \
#     --split test \
#     --output results/EucherHandForest/evaluation_test.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
exec "${ROOT}/venv_vibe_eucher/bin/python" \
    "${ROOT}/scripts/training/evaluate_eucher_hand_forest.py" \
    "$@"
