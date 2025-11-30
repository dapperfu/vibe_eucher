# Transformer RL Training Guide

This guide explains how to train the Transformer RL agent for Euchre.

## Quick Start

### Basic Training

```bash
# Train with default settings (10,000 hands, curriculum enabled)
python3 scripts/training/transformer_rl/train_transformer_rl.py

# Train with custom parameters
python3 scripts/training/transformer_rl/train_transformer_rl.py \
    --num-hands 20000 \
    --batch-size 128 \
    --learning-rate 0.0003 \
    --risk-factor 0.5 \
    --use-curriculum

# Train with duration (e.g., 10 minutes)
python3 scripts/training/transformer_rl/train_transformer_rl.py --duration 10m

# Train with epochs
python3 scripts/training/transformer_rl/train_transformer_rl.py --epochs 10
```

### Using the Shell Scripts

```bash
# GPU training
./scripts/training/transformer_rl/train_transformer_rl_gpu.sh

# CPU training
./scripts/training/transformer_rl/train_transformer_rl_cpu.sh

# Using the wrapper script (auto-detects device)
./scripts/train_transformer_rl.sh

# With environment variables
NUM_HANDS=50000 BATCH_SIZE=128 ./scripts/train_transformer_rl.sh

# With custom device
DEVICE=gpu ./scripts/train_transformer_rl.sh
```

## Command Line Options

### Training Parameters
- `--num-hands`: Number of hands to train on (default: 10000)
- `--batch-size`: Batch size for training (default: 64)
- `--update-frequency`: Update agent every N hands (default: 10)
- `--learning-rate`: Learning rate (default: 0.0003)
- `--gamma`: Discount factor (default: 0.99)

### Device and Model
- `--device`: Device to use (auto/cpu/gpu/cuda). Default: auto (auto-detects)
- `--model-path`: Path to load existing model from
- `--checkpoint-dir`: Directory to save checkpoints (default: models/checkpoints/transformer_rl)
- `--checkpoint-interval`: Save checkpoint every N hands (default: 100)
- `--no-resume`: Do not resume from latest checkpoint (start from scratch)

### Training Duration
- `--epochs`: Number of training epochs (alternative to --num-hands or --duration)
- `--duration`: Training duration in format like '1m', '10m', '1h', '2h', '1d' (alternative to --num-hands)
- `--num-hands`: Number of hands to train on (default: 10000, alternative to --epochs or --duration)

### Curriculum
- `--use-curriculum`: Enable curriculum training with progressive stages
- `--stage-threshold`: Performance threshold for stage progression (default: 0.7)

### Evaluation
- `--eval-interval`: Run evaluation every N hands (default: 500)
- `--eval-hands`: Number of hands for evaluation (default: 100)

### Risk Factors
- `--risk-factor`: Risk factor for decision-making 0.0-1.0 (default: 0.5)

### Training Orchestrator
- `--use-orchestrator`: Use training orchestrator for convergence tracking
- `--target-win-rate`: Target win rate for convergence (default: 0.6)
- `--max-games`: Maximum number of games (overrides num-hands if set)

## Training Stages

The curriculum training progresses through 7 stages:

1. **Stage 1: Basic Play** - Learn card legality and basic play rules
2. **Stage 2: Trick Strategy** - Learn trick strategy using self-play
3. **Stage 3: Deduction** - Add full deduction logic and perfect memory
4. **Stage 4: Bidding** - Introduce bidding and risk tuning
5. **Stage 5: Full RL** - Full hand reinforcement learning
6. **Stage 6: Multi-Agent** - Multi-agent self-play with varying risk
7. **Stage 7: Evaluation** - Evaluation against fixed bots

## Examples

### Train with GPU
```bash
python3 scripts/training/transformer_rl/train_transformer_rl.py --device gpu --num-hands 50000
# Or use the shell script
./scripts/training/transformer_rl/train_transformer_rl_gpu.sh --num-hands 50000
```

### Train with duration (cumulative training)
```bash
# Train for 10 minutes (will resume from latest checkpoint)
python3 scripts/training/transformer_rl/train_transformer_rl.py --duration 10m

# Train for 1 hour
python3 scripts/training/transformer_rl/train_transformer_rl.py --duration 1h
```

### Continue training from checkpoint (cumulative)
```bash
# Automatically resumes from latest checkpoint (default behavior)
python3 scripts/training/transformer_rl/train_transformer_rl.py --num-hands 20000

# Or specify a specific checkpoint UUID
python3 scripts/training/transformer_rl/train_transformer_rl.py \
    --model-path <checkpoint-uuid> \
    --num-hands 20000
```

### Train with convergence tracking
```bash
python3 scripts/training/transformer_rl/train_transformer_rl.py \
    --use-orchestrator \
    --target-win-rate 0.65 \
    --max-games 50000
```

### Train without curriculum (direct to full RL)
```bash
python3 scripts/training/transformer_rl/train_transformer_rl.py --num-hands 20000
```

## Output

Training will:
- Save checkpoints periodically to `models/checkpoints/transformer_rl/`
- Checkpoints are saved with UUID filenames in `.npz` format (compressed numpy)
- Display training progress and statistics
- Run periodic evaluations
- Save final model with UUID (also saved as `final_checkpoint.npz` for reference)

## Checkpoint System

The transformer RL system uses UUID-based cumulative checkpoints in `.npz` format:

### Features
- **UUID-based naming**: Each checkpoint has a unique UUID, enabling parallel training
- **Compressed format**: `.npz` format for efficient storage and syncing
- **Cumulative**: Checkpoints can be merged to combine training from multiple runs
- **Sync support**: Checkpoints can be synced between machines

### Checkpoint Management

```bash
# List all checkpoints
python3 -c "from src.ai_players.transformer_rl.checkpoint_manager import CumulativeCheckpointManager; from pathlib import Path; cm = CumulativeCheckpointManager(Path('models/checkpoints/transformer_rl')); print(cm.list_checkpoints())"

# Sync checkpoints from remote machine
python3 scripts/sync_checkpoints.py --remote-dir /path/to/remote/checkpoints

# Merge multiple checkpoints
python3 -c "from src.ai_players.transformer_rl.checkpoint_manager import CumulativeCheckpointManager; from pathlib import Path; cm = CumulativeCheckpointManager(Path('models/checkpoints/transformer_rl')); cm.merge_checkpoints(['uuid1', 'uuid2', 'uuid3'])"
```

### Parallel Training

Multiple training processes can run simultaneously:
- Each process generates unique UUID checkpoints
- No file conflicts between processes
- Checkpoints can be merged later

### Syncing Between Machines

```bash
# Sync checkpoints from remote directory
python3 scripts/sync_checkpoints.py \
    --local-dir models/checkpoints/transformer_rl \
    --remote-dir /mnt/shared/checkpoints/transformer_rl

# Sync and merge
python3 scripts/sync_checkpoints.py \
    --remote-dir /mnt/shared/checkpoints/transformer_rl \
    --merge
```

## Monitoring Training

The training script provides:
- Current stage (if using curriculum)
- Hands played
- Win rate
- Training loss (policy, value, entropy)
- Evaluation metrics

## Tips

1. **Start with curriculum**: Use `--use-curriculum` for better learning progression
2. **Adjust batch size**: Larger batch sizes (128-256) for more stable training
3. **Monitor convergence**: Use `--use-orchestrator` to track when agent reaches target performance
4. **Save frequently**: Lower `--checkpoint-interval` for more frequent saves
5. **GPU recommended**: Training is much faster on GPU

## Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Reduce `--update-frequency`

### Slow Training
- Use GPU: `--device cuda`
- Increase `--update-frequency` to train less frequently
- Reduce `--eval-interval`

### Not Learning
- Check learning rate (try 1e-4 to 1e-3)
- Ensure curriculum is progressing (check stage output)
- Verify rewards are being calculated correctly

