#!/usr/bin/env bash
# Score a trained EucherHeuristicTiebreak checkpoint.
#
# Usage:
#   scripts/training/evaluate_eucher_heuristic_tiebreak.sh \
#     --checkpoint checkpoints/EucherHeuristicTiebreak/checkpoint_best.pth \
#     --data-path data/EucherHeuristicTiebreak/v2 \
#     --split test \
#     --output results/EucherHeuristicTiebreak/evaluation_test.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
exec "${ROOT}/venv_vibe_eucher/bin/python" \
  "${ROOT}/scripts/training/evaluate_eucher_heuristic_tiebreak.py" \
  "$@"
