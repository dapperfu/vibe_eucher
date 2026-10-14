#!/usr/bin/env bash
# Train EucherHeuristicTiebreak on heuristic gray areas only.
#
# Usage:
#   scripts/training/train_eucher_heuristic_tiebreak.sh
#   scripts/training/train_eucher_heuristic_tiebreak.sh --hands 50 --epochs 5
#   scripts/training/train_eucher_heuristic_tiebreak.sh --collect-only
#
# Defaults live in configs/EucherHeuristicTiebreak.json.
# Extra arguments are passed through to the Python trainer.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
exec "${ROOT}/venv_vibe_eucher/bin/python" \
  "${ROOT}/scripts/training/train_eucher_heuristic_tiebreak.py" \
  "$@"
