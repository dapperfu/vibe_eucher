# PerceiverMuZero Bot

## Overview

EuchrePerceiverMuZero uses a Perceiver architecture combined with MuZero-style MCTS. The Perceiver architecture allows efficient processing of variable-length sequences.

## Architecture

- **Perceiver Encoder**: Processes game state as tokens
- **Representation Network**: Encodes to latent representation
- **Dynamics Network**: Predicts next state
- **Prediction Network**: Outputs policy and value
- **MCTS Engine**: MuZero-style search with learned dynamics

## Decision Making

- Encodes game state as tokens
- Processes through Perceiver encoder
- Runs MCTS with learned dynamics
- Supports fast mode (feed-forward only, no MCTS)

## Training

```bash
# Basic training
python scripts/training/train_perceiver_muzero.py --duration 1h

# With custom parameters
python scripts/training/train_perceiver_muzero.py \
    --duration 2h \
    --num-games 20 \
    --num-simulations 200
```

## Configuration

- `num_simulations`: MCTS simulations per move (default: varies)
- `fast_mode`: Use feed-forward only (no MCTS) for faster play
- `model_path`: Path to model checkpoint
- `risk_factor`: Risk factor for decision making

## Usage

```python
from eucher.game import Game

# Standard mode (with MCTS)
player_config = [
    ("Perceiver1", "perceiver_muzero"),
    ("Perceiver2", "perceiver_muzero"),
    ("Perceiver3", "perceiver_muzero"),
    ("Perceiver4", "perceiver_muzero"),
]

# Fast mode (no MCTS, faster but weaker)
player_config = [
    ("Perceiver1", "perceiver_muzero_1"),  # 1 simulation = fast mode
    ("Perceiver2", "perceiver_muzero_1"),
    ("Perceiver3", "perceiver_muzero_1"),
    ("Perceiver4", "perceiver_muzero_1"),
]

game = Game(player_config)
```

## Strengths

- Efficient Perceiver architecture
- Handles variable-length game sequences well
- Fast mode available for tournaments
- Strong play with MCTS

## Weaknesses

- More complex architecture
- Requires extensive training
- Fast mode weaker than full MCTS

## Best For

- Competitive play after training
- Tournament play (fast mode)
- Research and experimentation

## See Also

- [EuchreZero Bot](euchre_zero.md) - Similar architecture
- [EucherGo Bot](euchergo.md) - Similar architecture with different design

