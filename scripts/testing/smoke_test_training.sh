#!/bin/bash
# Smoke tests for training scripts
# Tests that all training scripts can at least start and show help/usage

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"

# Activate virtual environment if it exists
if [ -d "venv_vibe_eucher" ]; then
    source venv_vibe_eucher/bin/activate
fi

# Set PYTHONPATH to include project root
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

echo "Running smoke tests for training scripts..."
echo "=========================================="
echo ""

# Test 1: Transformer RL training script (main wrapper)
echo "Test 1: Transformer RL training script (main wrapper)"
if timeout 5 bash scripts/training/train_transformer_rl.sh --help > /dev/null 2>&1; then
    echo "  ✓ Transformer RL wrapper script works"
else
    echo "  ✗ Transformer RL wrapper script failed"
    exit 1
fi

# Test 2: PyTorch AI training script (main wrapper)
echo "Test 2: PyTorch AI training script (main wrapper)"
if timeout 5 bash scripts/training/train_pytorch_ai.sh --help > /dev/null 2>&1; then
    echo "  ✓ PyTorch AI wrapper script works"
else
    echo "  ✗ PyTorch AI wrapper script failed"
    exit 1
fi

# Test 3: EuchreZero training script (main wrapper)
echo "Test 3: EuchreZero training script (main wrapper)"
if timeout 5 bash scripts/training/train_eucher_zero.sh --help > /dev/null 2>&1; then
    echo "  ✓ EuchreZero wrapper script works"
else
    echo "  ✗ EuchreZero wrapper script failed"
    exit 1
fi

# Test 4: Main training script (train_models.py)
echo "Test 4: Main training script"
if timeout 5 python scripts/training/train_models.py --help > /dev/null 2>&1; then
    echo "  ✓ Main training script works"
else
    echo "  ✗ Main training script failed"
    exit 1
fi

# Test 5: GAN training mode (should fail gracefully with missing data)
echo "Test 5: GAN training mode (should fail gracefully)"
if timeout 5 python scripts/training/train_models.py --mode train_gan 2>&1 | grep -q "Training data file not found\|FileNotFoundError"; then
    echo "  ✓ GAN training fails gracefully with helpful error message"
else
    echo "  ✗ GAN training error handling needs improvement"
    exit 1
fi

echo ""
echo "=========================================="
echo "All smoke tests passed!"

