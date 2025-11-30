# Self-Play Training Infrastructure

## Overview

The PyTorch AI training infrastructure has been completely redesigned to use **self-play training** instead of training on pre-collected data. The bots now play live games against each other, collect training data during gameplay, and train incrementally on that data.

## Key Features

### 1. **Live Self-Play Games**
   - Bots play real Euchre games against each other
   - Training data is collected automatically during gameplay
   - No need to pre-collect data files

### 2. **Epsilon-Greedy Exploration**
   - Configurable exploration probability (default: 10%)
   - Ensures diverse training data
   - Prevents overfitting to current model predictions

### 3. **Automatic Batch Size Tuning**
   - `--tune-batch` flag automatically finds optimal batch size
   - Tests batch sizes to maximize GPU memory usage
   - Prevents OOM errors

### 4. **Automatic Checkpoint Management**
   - Automatically resumes from latest checkpoint on startup
   - Saves checkpoint on Ctrl+C (SIGINT)
   - Periodic checkpoint saving (configurable interval)

### 5. **Configurable Player Combinations**
   - All PyTorch players (default)
   - PyTorch vs Heuristic
   - PyTorch vs Random
   - PyTorch vs AI
   - Custom combinations

## Usage

### Basic Training
```bash
# Train with default settings (all PyTorch players, 50 games per training)
python scripts/training/pytorch_ai/train_pytorch_ai_self_play.py

# Train with specific configuration
python scripts/training/pytorch_ai/train_pytorch_ai_self_play.py \
  --games-per-training 100 \
  --epsilon 0.15 \
  --player-config pytorch_vs_heuristic \
  --tune-batch
```

### Advanced Options
```bash
# Limit training cycles
python scripts/training/pytorch_ai/train_pytorch_ai_self_play.py --max-cycles 100

# Custom player configuration
python scripts/training/pytorch_ai/train_pytorch_ai_self_play.py \
  --player-config "pytorch,heuristic,pytorch,random"

# Start fresh (no resume)
python scripts/training/pytorch_ai/train_pytorch_ai_self_play.py --no-resume

# Custom learning rate
python scripts/training/pytorch_ai/train_pytorch_ai_self_play.py --learning-rate 1e-5
```

## Command Line Arguments

- `--device`: Device to train on (auto/cpu/gpu/cuda, default: auto)
- `--games-per-training`: Number of games before each training step (default: 50)
- `--epsilon`: Exploration probability 0.0-1.0 (default: 0.1)
- `--player-config`: Player combination (default: all_pytorch)
- `--max-cycles`: Maximum training cycles (default: unlimited)
- `--checkpoint-interval`: Save checkpoint every N cycles (default: 5)
- `--checkpoint-dir`: Checkpoint directory (default: models/checkpoints/pytorch_ai)
- `--data-dir`: Training data directory (default: training_data)
- `--tune-batch`: Automatically tune batch size for GPU memory
- `--batch-size`: Manual batch size (ignored if --tune-batch used)
- `--no-resume`: Don't resume from checkpoint (start fresh)
- `--learning-rate`: Learning rate (default: 5e-5)

## How It Works

1. **Game Play**: Runs N games (default: 50) with current model
2. **Data Collection**: Collects all decision points during gameplay
3. **Training**: Trains model for 1 epoch on collected data
4. **Model Update**: Updates all players with new model
5. **Repeat**: Continues until interrupted or max cycles reached

## Checkpoint Behavior

- **Auto-Resume**: Automatically loads latest checkpoint on startup
- **Ctrl+C**: Saves checkpoint immediately and exits gracefully
- **Periodic**: Saves checkpoint every N cycles (default: 5)

## Files Created/Modified

### New Files
- `scripts/training/pytorch_ai/train_pytorch_ai_self_play.py` - Main self-play training script
- `eucher/players/computer/ml/pytorch/training/self_play_trainer.py` - Self-play trainer class
- `eucher/players/computer/ml/pytorch/training/batch_tuner.py` - Batch size tuning utility

### Modified Files
- `eucher/players/computer/ml/pytorch/pytorch_player.py` - Added epsilon-greedy exploration
- `eucher/players/computer/ml/pytorch/training/data_manager.py` - Added collector-to-dataset conversion
- `eucher/players/computer/ml/pytorch/training/trainer.py` - Fixed action_type handling

## Differences from Old Training

| Old Training | New Self-Play Training |
|-------------|----------------------|
| Requires pre-collected `.npz` files | Collects data during gameplay |
| Static dataset | Dynamic, always fresh data |
| Manual data collection step | Automatic data collection |
| Trains on stale data | Trains on current model's decisions |

## Example Workflow

```bash
# Start training (will auto-resume if interrupted)
python scripts/training/pytorch_ai/train_pytorch_ai_self_play.py --tune-batch

# Training runs continuously:
# Cycle 1: Play 50 games → Train → Save checkpoint
# Cycle 2: Play 50 games → Train → Save checkpoint
# ...

# To stop: Press Ctrl+C (checkpoint saved automatically)
# To resume: Run same command (auto-resumes from checkpoint)
```

## Troubleshooting

**Loss not improving?**
- Try lower learning rate: `--learning-rate 1e-5`
- Increase exploration: `--epsilon 0.2`
- Check if enough games per training: `--games-per-training 100`

**GPU OOM errors?**
- Use `--tune-batch` to automatically find optimal batch size
- Or manually set smaller batch: `--batch-size 16`

**Want to start fresh?**
- Use `--no-resume` flag

