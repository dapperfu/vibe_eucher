# Training Scripts Organization

This document describes the consistent naming and organization of all training scripts.

## Main Training Scripts

All main training scripts are located in `scripts/` root directory and follow the pattern `train_<architecture>.sh`:

- **`train_euchre_zero.sh`** - Main wrapper for EuchreZero training (auto-detects device)
- **`train_transformer_rl.sh`** - Main wrapper for Transformer RL training (auto-detects device)
- **`train_pytorch_ai.sh`** - Main wrapper for PyTorch AI training (auto-detects device)

All main wrappers:
- Auto-detect GPU availability
- Support `--device` flag to override auto-detection
- Activate virtual environment if present
- Pass through all additional arguments to the underlying Python script

## Architecture-Specific Scripts

Each architecture has its Python training script and CPU/GPU variants in subdirectories:

### EuchreZero
- **Python script**: `scripts/training/euchre_zero/train_euchre_zero.py`
- **CPU variant**: `scripts/training/euchre_zero/train_euchre_zero_cpu.sh`
- **GPU variant**: `scripts/training/euchre_zero/train_euchre_zero_gpu.sh`

### Transformer RL
- **Python script**: `scripts/training/transformer_rl/train_transformer_rl.py`
- **CPU variant**: `scripts/training/transformer_rl/train_transformer_rl_cpu.sh`
- **GPU variant**: `scripts/training/transformer_rl/train_transformer_rl_gpu.sh`

### PyTorch AI
- **Python script**: `scripts/training/pytorch_ai/train_pytorch_ai.py`
- **CPU variant**: `scripts/training/pytorch_ai/train_pytorch_ai_cpu.sh`
- **GPU variant**: `scripts/training/pytorch_ai/train_pytorch_ai_gpu.sh`

## Workflow Scripts

Additional workflow scripts for specialized training scenarios:

- **`train_quick.sh`** - Quick training (30 minutes)
- **`train_full_workflow.sh`** - Full training workflow (self-play → supervised → RL)
- **`train_until_converged.sh`** - Train until convergence
- **`train_with_risk.sh`** - Train with custom risk factors

## General Training Script

- **`train_models.py`** - General training script supporting multiple modes:
  - `--mode self_play` - Collect training data
  - `--mode train` - Train supervised models (sklearn)
  - `--mode train_gan` - Train GAN models
  - `--mode train_rl` - Train RL agents
  - `--mode both` - Run multiple modes

## Usage Examples

### Using Main Wrappers (Recommended)

```bash
# Auto-detect device and train
./scripts/train_euchre_zero.sh --duration 1h
./scripts/train_transformer_rl.sh --num-hands 20000
./scripts/train_pytorch_ai.sh --epochs 10

# Force specific device
./scripts/train_euchre_zero.sh --device cpu --duration 30m
./scripts/train_transformer_rl.sh --device gpu --num-hands 50000
```

### Using CPU/GPU Variants Directly

```bash
# Force CPU training
./scripts/training/euchre_zero/train_euchre_zero_cpu.sh --duration 1h
./scripts/training/transformer_rl/train_transformer_rl_cpu.sh --num-hands 10000
./scripts/training/pytorch_ai/train_pytorch_ai_cpu.sh --epochs 5

# Force GPU training
./scripts/training/euchre_zero/train_euchre_zero_gpu.sh --duration 2h
./scripts/training/transformer_rl/train_transformer_rl_gpu.sh --num-hands 50000
./scripts/training/pytorch_ai/train_pytorch_ai_gpu.sh --epochs 20
```

### Using Python Scripts Directly

```bash
# Direct Python script invocation
python scripts/training/euchre_zero/train_euchre_zero.py --device auto --duration 1h
python scripts/training/transformer_rl/train_transformer_rl.py --device gpu --num-hands 20000
python scripts/training/pytorch_ai/train_pytorch_ai.py --device cpu --epochs 10
```

## Naming Convention Summary

- **Main wrappers**: `train_<architecture>.sh` (in `scripts/`)
- **Python scripts**: `train_<architecture>.py` (in `scripts/<architecture>/`)
- **CPU variants**: `train_<architecture>_cpu.sh` (in `scripts/<architecture>/`)
- **GPU variants**: `train_<architecture>_gpu.sh` (in `scripts/<architecture>/`)

All scripts follow this consistent pattern for easy discovery and usage.

