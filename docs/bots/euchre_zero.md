# EuchreZero Bot

## Overview

EuchreZero is an AlphaZero-inspired reinforcement learning system for Euchre. It uses Monte Carlo Tree Search (MCTS) guided by neural networks to make decisions.

## Architecture

- **Representation Network**: Encodes game state to latent representation
- **Dynamics Network**: Predicts next state and immediate reward
- **Prediction Network**: Outputs policy and value estimates
- **MCTS Engine**: Monte Carlo Tree Search with belief sampling
- **Deduction System**: Hidden card inference and probability tracking

## Decision Making

All decisions use MCTS search:
- Encodes current game state
- Runs MCTS simulations guided by neural networks
- Selects action with highest visit count from MCTS

## Training

```bash
# Basic training
python scripts/eucher_zero/train_euchre_zero.py --duration 1h

# With custom parameters
python scripts/eucher_zero/train_euchre_zero.py \
    --duration 2h \
    --num-games 20 \
    --num-simulations 200 \
    --batch-size 64
```

## Configuration

- `num_simulations`: MCTS simulations per move (default: 100)
- `model_path`: Path to model checkpoint
- `risk_factor`: Risk factor for decision making (0.0-1.0)

## Usage

```python
from eucher.game import Game

player_config = [
    ("EuchreZero1", "eucher_zero"),
    ("EuchreZero2", "eucher_zero"),
    ("EuchreZero3", "eucher_zero"),
    ("EuchreZero4", "eucher_zero"),
]

game = Game(player_config)
```

## Strengths

- Strong strategic play through MCTS
- Learns optimal strategies through self-play
- Handles hidden information well
- Can be trained to very high levels

## Weaknesses

- Computationally expensive (many simulations)
- Requires extensive training
- Slower decision-making than simpler bots

## Best For

- Competitive play after training
- Research and experimentation
- Discovering optimal strategies

## See Also

- [EucherGo Bot](euchergo.md) - Similar architecture with different design
- [PerceiverMuZero Bot](perceiver_muzero.md) - Perceiver-based variant

