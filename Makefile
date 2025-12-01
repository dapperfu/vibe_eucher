.PHONY: venv install test lint type-check format clean run help \
	train-pytorch-ai train-pytorch-ai-self-play train-pytorch-ai-self-play-tune \
	train-euchre-zero train-perceiver-muzero train-transformer-rl train-euchergo \
	train-ml-supervised train-ml-gan train-ml-rl train-ml-self-play train-ml-all

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
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# AI Bot Training Targets
train-pytorch-ai: ${VENV}
	${PYTHON} scripts/training/train_pytorch_ai.py

train-pytorch-ai-self-play: ${VENV}
	${PYTHON} scripts/training/train_pytorch_ai_self_play.py

train-pytorch-ai-self-play-tune: ${VENV}
	${PYTHON} scripts/training/train_pytorch_ai_self_play.py --tune-batch

train-euchre-zero: ${VENV}
	${PYTHON} scripts/training/train_euchre_zero.py

train-perceiver-muzero: ${VENV}
	${PYTHON} scripts/training/train_perceiver_muzero.py

train-transformer-rl: ${VENV}
	${PYTHON} scripts/training/train_transformer_rl.py

train-euchergo: ${VENV}
	${PYTHON} scripts/training/train_euchergo.py

train-ml-supervised: ${VENV}
	${PYTHON} scripts/training/train_models.py --mode train

train-ml-gan: ${VENV}
	${PYTHON} scripts/training/train_models.py --mode train_gan

train-ml-rl: ${VENV}
	${PYTHON} scripts/training/train_models.py --mode train_rl

train-ml-self-play: ${VENV}
	${PYTHON} scripts/training/train_models.py --mode self_play

train-ml-all: ${VENV}
	${PYTHON} scripts/training/train_models.py --mode both

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
	@echo "  make train-ml-self-play            - Collect self-play data"
	@echo "  make train-ml-all                  - Run full ML training workflow"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean      - Clean generated files"

