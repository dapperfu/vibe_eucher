#!/usr/bin/env bash
# Replay one game seed against random AI and record the learner's decisions.
#
# The dealt hands match a game played with the same seed. Random opponents
# make different decisions on every run. That is a second training source
# beside ordinary self-play: same hands, different choices.
#
# Setup:
#   source venv_vibe_eucher/bin/activate
#
# Usage:
#   ./scripts/training/train_seeded_random.sh 42
#   ./scripts/training/train_seeded_random.sh 42 200
#   ./scripts/training/train_seeded_random.sh --seed 42 --num_games 200 --learner ml_sklearn
#
# The seed is the value printed when you start a game, or the value you type
# at the seed prompt. Seat 0 is the learner, which is the human seat.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${ROOT}/venv_vibe_eucher/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python"
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 SEED [NUM_GAMES]" >&2
    echo "   or: $0 --seed SEED [--num_games N] [--learner TYPE]" >&2
    exit 1
fi

cd "${ROOT}"
if [[ "$1" == --* ]]; then
    exec "${PYTHON}" scripts/training/train_models.py --mode seeded_random "$@"
fi

SEED="$1"
NUM_GAMES="${2:-100}"
exec "${PYTHON}" scripts/training/train_models.py \
    --mode seeded_random \
    --seed "${SEED}" \
    --num_games "${NUM_GAMES}"
