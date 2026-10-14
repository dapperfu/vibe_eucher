#!/usr/bin/env bash
# Train EucherHandForest on hands from the hand generator.
#
# The generator deals hands of one game seed. Each hand is replayed while the
# dealer rotates. Seat 0's bids, plays, and discards train one random forest
# per decision.
#
# Usage:
#   scripts/training/train_eucher_hand_forest.sh
#   scripts/training/train_eucher_hand_forest.sh --collect-only
#   scripts/training/train_eucher_hand_forest.sh --no-resume --hand-seeds 2 --replays 4

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
exec "${ROOT}/venv_vibe_eucher/bin/python" \
    "${ROOT}/scripts/training/train_eucher_hand_forest.py" \
    "$@"
