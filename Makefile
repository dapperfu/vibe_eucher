.PHONY: venv install test lint type-check format clean run help \
	train-pytorch-ai train-pytorch-ai-self-play train-pytorch-ai-self-play-tune \
	train-euchre-zero train-perceiver-muzero train-transformer-rl train-euchergo \
	train-ml-supervised train-ml-gan train-ml-rl train-ml-self-play train-ml-all \
	train-ml-seeded-random train-heuristic-tiebreak evaluate-heuristic-tiebreak \
	train-hand-forest evaluate-hand-forest

VENV := venv_vibe_eucher
PYTHON := ${VENV}/bin/python
PIP := ${VENV}/bin/pip
PYTEST := ${VENV}/bin/pytest
RUFF := ${VENV}/bin/ruff
MYPY := ${VENV}/bin/mypy
PYDOCSTYLE := ${VENV}/bin/pydocstyle

# Default target
.DEFAULT_GOAL := help

# Create virtual environment
${VENV}:
	python3 -m venv ${VENV}
	${PIP} install --upgrade pip

# Install dependencies
install: ${VENV}
	${PIP} install -e .

# Run tests
test: ${VENV}
	${PYTEST} tests/ -v

# Lint code
lint: ${VENV}
	${RUFF} check src/ tests/ main.py

# Type check
type-check: ${VENV}
	${MYPY} src/ main.py

# Format code
format: ${VENV}
	${RUFF} format src/ tests/ main.py

# Check documentation style
docstyle: ${VENV}
	${PYDOCSTYLE} src/ main.py

# Run the game
run: ${VENV}
	${PYTHON} main.py

# Clean generated files
clean:
	rm -rf ${VENV}
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf checkpoints/EucherHandForest
	rm -rf logs/EucherHandForest
	rm -rf results/EucherHandForest
	rm -rf data/EucherHandForest
	rm -rf checkpoints/EucherHeuristicTiebreak
	rm -rf logs/EucherHeuristicTiebreak
	rm -rf results/EucherHeuristicTiebreak
	rm -rf data/EucherHeuristicTiebreak
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# AI Bot Training Targets
train-pytorch-ai: ${VENV}
	${PYTHON} scripts/pytorch_ai/train_pytorch_ai.py

train-pytorch-ai-self-play: ${VENV}
	${PYTHON} scripts/pytorch_ai/train_pytorch_ai_self_play.py

train-pytorch-ai-self-play-tune: ${VENV}
	${PYTHON} scripts/pytorch_ai/train_pytorch_ai_self_play.py --tune-batch

train-euchre-zero: ${VENV}
	${PYTHON} scripts/eucher_zero/train_eucher_zero.py

train-perceiver-muzero: ${VENV}
	${PYTHON} scripts/perceiver_muzero/train_perceiver_muzero.py

train-transformer-rl: ${VENV}
	${PYTHON} scripts/transformer_rl/train_transformer_rl.py

train-euchergo: ${VENV}
	${PYTHON} scripts/euchergo/train_euchergo.py

train-ml-supervised: ${VENV}
	${PYTHON} scripts/training/train_models.py --mode train

train-ml-gan: ${VENV}
	${PYTHON} scripts/training/train_models.py --mode train_gan

train-ml-rl: ${VENV}
	${PYTHON} scripts/training/train_models.py --mode train_rl

train-ml-self-play: ${VENV}
	${PYTHON} scripts/training/train_models.py --mode self_play

# Replay one played-game seed against random AI.
# Usage: make train-ml-seeded-random SEED=42
#        make train-ml-seeded-random SEED=42 NUM_GAMES=50
train-ml-seeded-random: ${VENV}
	${PYTHON} scripts/training/train_models.py --mode seeded_random --seed "${SEED}" \
		--num_games "$(or ${NUM_GAMES},100)"

train-ml-all: ${VENV}
	${PYTHON} scripts/training/train_models.py --mode both

# Heuristic gray-area tie-break. Clear bids, plays, and discards never enter the dataset.
data/EucherHeuristicTiebreak/v2/manifest.json: configs/EucherHeuristicTiebreak.json \
		euchre/ai/gray_area.py euchre/ai/heuristic_player.py \
		euchre/training/gray_area_samples.py \
		scripts/training/train_eucher_heuristic_tiebreak.py ${VENV}
	${PYTHON} scripts/training/train_eucher_heuristic_tiebreak.py --collect-only \
		--config configs/EucherHeuristicTiebreak.json

checkpoints/EucherHeuristicTiebreak/checkpoint_best.pth: \
		data/EucherHeuristicTiebreak/v2/manifest.json \
		configs/EucherHeuristicTiebreak.json \
		euchre/ai/eucher_heuristic_tiebreak.py \
		scripts/training/train_eucher_heuristic_tiebreak.py ${VENV}
	${PYTHON} scripts/training/train_eucher_heuristic_tiebreak.py \
		--config configs/EucherHeuristicTiebreak.json \
		--data-path data/EucherHeuristicTiebreak/v2

results/EucherHeuristicTiebreak/evaluation_test.json: \
		checkpoints/EucherHeuristicTiebreak/checkpoint_best.pth \
		data/EucherHeuristicTiebreak/v2/manifest.json \
		scripts/training/evaluate_eucher_heuristic_tiebreak.py ${VENV}
	${PYTHON} scripts/training/evaluate_eucher_heuristic_tiebreak.py \
		--checkpoint checkpoints/EucherHeuristicTiebreak/checkpoint_best.pth \
		--data-path data/EucherHeuristicTiebreak/v2 \
		--split test \
		--output results/EucherHeuristicTiebreak/evaluation_test.json

train-heuristic-tiebreak: checkpoints/EucherHeuristicTiebreak/checkpoint_best.pth

evaluate-heuristic-tiebreak: results/EucherHeuristicTiebreak/evaluation_test.json

# Forests trained on hands from the hand generator.
data/EucherHandForest/v1/manifest.json: configs/EucherHandForest.json \
		eucher/training/hand_forest.py eucher/training/hand_study.py \
		eucher/training/self_play.py \
		scripts/training/train_eucher_hand_forest.py ${VENV}
	${PYTHON} scripts/training/train_eucher_hand_forest.py --collect-only \
		--config configs/EucherHandForest.json

checkpoints/EucherHandForest/checkpoint_best.pth: \
		data/EucherHandForest/v1/manifest.json \
		configs/EucherHandForest.json \
		eucher/training/hand_forest.py \
		scripts/training/train_eucher_hand_forest.py ${VENV}
	${PYTHON} scripts/training/train_eucher_hand_forest.py \
		--config configs/EucherHandForest.json \
		--data-path data/EucherHandForest/v1

results/EucherHandForest/evaluation_test.json: \
		checkpoints/EucherHandForest/checkpoint_best.pth \
		data/EucherHandForest/v1/manifest.json \
		scripts/training/evaluate_eucher_hand_forest.py ${VENV}
	${PYTHON} scripts/training/evaluate_eucher_hand_forest.py \
		--checkpoint checkpoints/EucherHandForest/checkpoint_best.pth \
		--data-path data/EucherHandForest/v1 \
		--split test \
		--output results/EucherHandForest/evaluation_test.json

train-hand-forest: checkpoints/EucherHandForest/checkpoint_best.pth

evaluate-hand-forest: results/EucherHandForest/evaluation_test.json

# Help
help:
	@echo "Available targets:"
	@echo ""
	@echo "Setup:"
	@echo "  make venv       - Create virtual environment"
	@echo "  make install   - Install dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Lint code with ruff"
	@echo "  make type-check - Type check with mypy"
	@echo "  make format     - Format code with ruff"
	@echo "  make docstyle   - Check documentation style"
	@echo "  make run        - Run the game"
	@echo ""
	@echo "AI Bot Training:"
	@echo "  make train-pytorch-ai              - Train PyTorch AI (supervised)"
	@echo "  make train-pytorch-ai-self-play    - Train PyTorch AI (self-play)"
	@echo "  make train-pytorch-ai-self-play-tune - Train PyTorch AI (self-play with batch tuning)"
	@echo "  make train-euchre-zero             - Train EuchreZero"
	@echo "  make train-perceiver-muzero        - Train PerceiverMuZero"
	@echo "  make train-transformer-rl          - Train Transformer RL"
	@echo "  make train-euchergo                - Train EucherGo"
	@echo "  make train-ml-supervised           - Train sklearn models (supervised)"
	@echo "  make train-ml-gan                  - Train GAN models"
	@echo "  make train-ml-rl                   - Train RL models"
	@echo "  make train-ml-self-play            - Collect self-play data (tqdm progress)"
	@echo "  make train-ml-seeded-random       - Replay SEED against random AI (SEED= required)"
	@echo "  make train-ml-all                  - Run full ML training workflow"
	@echo "  make train-heuristic-tiebreak    - Train EucherHeuristicTiebreak on gray areas"
	@echo "  make evaluate-heuristic-tiebreak - Score the tie-break model on held-out gray areas"
	@echo "  make train-hand-forest           - Train EucherHandForest on generated hands"
	@echo "  make evaluate-hand-forest        - Score EucherHandForest on held-out hands"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean      - Clean generated files"

