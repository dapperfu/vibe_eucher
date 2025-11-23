# How to Generate More Training Data

This guide explains the different methods for generating training data for your Euchre ML models.

## Overview

Training data is collected by running games and recording all decision points:
- **Order Up** decisions
- **Call Trump** decisions  
- **Play Card** decisions
- **Discard** decisions

Data is saved in NumPy `.npz` format (fast, binary) and optionally JSON format.

## Method 1: Simple Data Collection Script

The easiest way to collect data:

```bash
# Collect 1000 games of data
python scripts/collect_training_data.py --num_games 1000

# Specify output directory
python scripts/collect_training_data.py --num_games 500 --output_dir /path/to/data

# Custom data prefix
python scripts/collect_training_data.py --num_games 1000 --data_prefix my_data
```

**What it does:**
- Runs games with various player combinations (ML, AI, Random, Heuristic)
- Collects all decision points from computer players
- Saves to `training_data/` by default
- Creates `.npz` files (and optionally JSON/CSV)

## Method 2: Using train_models.py (Advanced)

More control with orchestrator, convergence tracking, and progress display:

```bash
# Collect data for 500 games
python scripts/train_models.py --mode self_play --num_games 500

# Collect data for a specific duration
python scripts/train_models.py --mode self_play --duration 30m

# Collect data until convergence (90% win rate)
python scripts/train_models.py --mode self_play --until-converged

# Custom convergence target
python scripts/train_models.py --mode self_play --until-converged --target-win-rate 0.95

# With custom risk factors
python scripts/train_models.py --mode self_play --num_games 1000 \
    --trump-selection-risk 0.3 --gameplay-risk 0.7
```

**Options:**
- `--mode self_play`: Only collect data (don't train)
- `--num_games N`: Run exactly N games
- `--duration TIME`: Run for time period (e.g., "30m", "2h", "1d")
- `--until-converged`: Run until win rate target reached
- `--target-win-rate FLOAT`: Target win rate (default 0.90)
- `--window-size N`: Games to consider for convergence (default 100)
- `--trump-selection-risk FLOAT`: Risk factor for trump decisions (0.0-1.0)
- `--gameplay-risk FLOAT`: Risk factor for gameplay decisions (0.0-1.0)

## Method 3: Python API (Programmatic)

For custom data collection logic:

```python
from pathlib import Path
from eucher.training.self_play import SelfPlayTrainer
from eucher.players.computer.ml.ml_config import MLConfig

# Create trainer
config = MLConfig()
trainer = SelfPlayTrainer(output_dir=config.training_data_dir)

# Collect data
trainer.run_training_round(
    num_games=1000,
    save_data=True
)

# Custom player combinations
custom_combos = [
    [("ML1", "ml"), ("ML2", "ml"), ("ML3", "ml"), ("ML4", "ml")],
    [("ML1", "ml"), ("AI1", "ai"), ("Random1", "random"), ("Random2", "random")],
]
trainer.run_training_round(
    num_games=500,
    player_combinations=custom_combos,
    save_data=True
)
```

## Method 4: Continuous Data Collection

For ongoing data collection while training:

```bash
# Collect data continuously for 1 hour
python scripts/train_models.py --mode self_play --duration 1h

# Collect data with periodic saves (every 100 games)
# The script automatically saves every 50-100 games
python scripts/train_models.py --mode self_play --num_games 10000
```

## Data Output Format

Data is saved to `training_data/` directory (or your specified directory):

```
training_data/
├── training_order_up.npz      # NumPy format (fast, binary, compressed)
├── training_call_trump.npz
├── training_play_card.npz
└── training_discard.npz
```

**File Format:**
- `.npz` files contain compressed NumPy arrays: `X` (features) and `y` (decisions)
- Training code automatically loads from `.npz` format
- Only binary `.npz` files are generated (no CSV or JSON files)
- JSON format is optional and slower

## Player Combinations

By default, data is collected from games with various player combinations:

1. **All ML players**: 4 ML players playing each other
2. **ML vs Random**: ML players vs random opponents
3. **ML + AI vs Random**: ML with AI partner vs random
4. **ML vs Heuristic**: ML players vs heuristic opponents
5. **Mixed**: ML, AI, Random, and Heuristic players

This diversity helps create robust training data.

## Tips for Better Data

1. **More Games = Better Data**: Collect at least 1000+ games for good coverage
2. **Diverse Opponents**: Use various player types (ML, AI, Random, Heuristic)
3. **Longer Sessions**: Run for hours/days for comprehensive data
4. **Incremental Collection**: Collect data in batches and combine
5. **Monitor Quality**: Review collected data occasionally to ensure quality

## Combining Multiple Data Collections

Data files are **overwritten** by default. To combine multiple collections:

```python
import numpy as np
from pathlib import Path

# Load existing data
data_dir = Path("training_data")
existing = np.load(data_dir / "training_play_card.npz")
X_old = existing["X"]
y_old = existing["y"]

# Load new data
new_data = np.load(data_dir / "training_play_card.npz")
X_new = new_data["X"]
y_new = new_data["y"]

# Combine
X_combined = np.concatenate([X_old, X_new], axis=0)
y_combined = np.concatenate([y_old, y_new], axis=0)

# Save combined
np.savez_compressed(
    data_dir / "training_play_card.npz",
    X=X_combined,
    y=y_combined
)
```

## Quick Examples

```bash
# Quick: 500 games
python scripts/collect_training_data.py --num_games 500

# Medium: 5000 games (takes ~30-60 minutes)
python scripts/collect_training_data.py --num_games 5000

# Large: 20000 games (takes ~2-4 hours)
python scripts/collect_training_data.py --num_games 20000

# Continuous: Run for 2 hours
python scripts/train_models.py --mode self_play --duration 2h

# Until convergence: Collect until 90% win rate
python scripts/train_models.py --mode self_play --until-converged
```

## Checking Data Size

After collection, check how much data you have:

```python
import numpy as np
from pathlib import Path

data_dir = Path("training_data")
for name in ["order_up", "call_trump", "play_card", "discard"]:
    file = data_dir / f"training_{name}.npz"
    if file.exists():
        data = np.load(file)
        print(f"{name}: {len(data['X'])} samples")
    else:
        print(f"{name}: No data file found")
```

## Next Steps

After collecting data:
1. Train supervised models: `python scripts/train_models.py --mode train`
2. Train GAN models: `python scripts/train_models.py --mode train_gan`
3. Train RL agents: `python scripts/train_models.py --mode train_rl`

