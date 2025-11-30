# Training Scripts Guide

This directory contains convenient shell scripts for training ML models with verbose output and clear setup instructions.

## Quick Start

All scripts are executable and can be run directly:

```bash
# Quick 30-minute training
./scripts/training/train_quick.sh

# Train until convergence (90% win rate)
./scripts/training/train_until_converged.sh

# Train with custom risk factors
./scripts/training/train_with_risk.sh 0.3 0.7

# Full training workflow
./scripts/training/train_full_workflow.sh

# Generate model input documentation
./scripts/docs/generate_docs.sh
```

## Scripts Overview

### train_quick.sh
Quick training for 30 minutes with default settings.

**Usage:**
```bash
./scripts/training/train_quick.sh
```

**What it does:**
- Trains RL agents for 30 minutes
- Uses default risk factors (0.5/0.5)
- Saves checkpoints every 100 games

### train_until_converged.sh
Train until model achieves target win rate.

**Usage:**
```bash
# Default: 90% win rate, 100 game window
./scripts/training/train_until_converged.sh

# Custom win rate: 95%
./scripts/training/train_until_converged.sh 0.95

# Custom win rate and window size
./scripts/training/train_until_converged.sh 0.95 200
```

**What it does:**
- Trains until win rate >= target (default 90%)
- Monitors win rate over last N games (default 100)
- Automatically stops when converged

### train_with_risk.sh
Train with custom risk/temperature factors.

**Usage:**
```bash
# Default: balanced play (0.5/0.5)
./scripts/training/train_with_risk.sh

# Conservative trump (0.2), risky gameplay (0.8)
./scripts/training/train_with_risk.sh 0.2 0.8

# Custom duration: 1 day
./scripts/training/train_with_risk.sh 0.3 0.7 "1d"
```

**Risk Factor Guide:**
- **Low (0.0-0.3)**: Conservative - only high-confidence actions
- **Medium (0.3-0.7)**: Balanced play
- **High (0.7-1.0)**: Risky - accepts lower-confidence actions

### train_full_workflow.sh
Complete training pipeline from data collection to convergence.

**Usage:**
```bash
# Default: 500 data games, 90% target win rate
./scripts/training/train_full_workflow.sh

# Custom data collection and target
./scripts/training/train_full_workflow.sh 1000 0.95
```

**What it does:**
1. Collects training data through self-play
2. Trains supervised models (sklearn)
3. Trains RL agents until convergence

### generate_docs.sh
Generate model input documentation.

**Usage:**
```bash
# Generate both markdown and HTML
./scripts/docs/generate_docs.sh

# Generate only markdown
./scripts/docs/generate_docs.sh markdown

# Generate only HTML
./scripts/docs/generate_docs.sh html
```

**Output:**
- `docs/model_inputs.md` - Markdown reference
- `docs/model_inputs.html` - Interactive HTML report

## Setup Requirements

All scripts require:

1. **Virtual environment activated:**
   ```bash
   source venv_vibe_eucher/bin/activate
   ```

2. **Dependencies installed:**
   ```bash
   pip install -e .
   ```

## Direct Python Commands

Each script displays the Python command it runs. You can also run these directly:

```bash
# Quick training
python scripts/training/train_models.py --mode train_rl --duration 30m

# Convergence training
python scripts/training/train_models.py --mode train_rl --until-converged --target-win-rate 0.90

# Risk-based training
python scripts/training/train_models.py --mode train_rl --duration 2h \
    --trump-selection-risk 0.3 --gameplay-risk 0.7

# Generate documentation
python scripts/docs/generate_input_docs.py --format both
```

## Output Locations

- **Training data**: `training_data/`
- **Trained models**: `models/`
- **Checkpoints**: `models/checkpoints/`
- **Training logs**: `training_logs/`
- **Documentation**: `docs/`

## Tips

- Scripts show verbose output with setup information
- Each script displays the exact Python command it runs
- All scripts can be run directly or customized by editing the command
- Check script headers for detailed usage examples

