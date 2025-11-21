.PHONY: venv install test lint type-check format clean run help

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

# Help
help:
	@echo "Available targets:"
	@echo "  make venv       - Create virtual environment"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Lint code with ruff"
	@echo "  make type-check - Type check with mypy"
	@echo "  make format     - Format code with ruff"
	@echo "  make docstyle   - Check documentation style"
	@echo "  make run        - Run the game"
	@echo "  make clean      - Clean generated files"

