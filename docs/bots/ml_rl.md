# ML Reinforcement Learning Bot

## Overview

The ML RL Bot learns by playing games and getting rewards or penalties. It doesn't need to watch other players - it learns by trial and error.

## Architecture

- **Reinforcement Learning**: Learns through self-play with reward signals
- **Reward Structure**: Gets positive rewards for wins, negative for losses
- **Exploration**: Uses epsilon-greedy exploration to try new strategies

## Decision Making

- Evaluates current game state
- Considers possible actions
- Chooses actions that have historically led to rewards
- Uses exploration to try new things occasionally

## Training

```bash
python scripts/training/train_models.py --mode train_rl
```

Or use the example script:
```bash
python examples/train_rl_example.py
```

## Configuration

- `backend`: "rl"
- `training_mode`: Whether in training mode (affects exploration)
- `trump_selection_risk`: Risk factor for trump decisions
- `gameplay_risk`: Risk factor for gameplay decisions

## Usage

```python
from eucher.game import Game

player_config = [
    ("RL1", "ml_rl"),
    ("RL2", "ml_rl"),
    ("RL3", "ml_rl"),
    ("RL4", "ml_rl"),
]

game = Game(player_config)
```

## Strengths

- Learns optimal strategies through experience
- Doesn't need pre-existing game data
- Can discover creative strategies
- Adapts to different opponents over time

## Weaknesses

- Needs many games to learn (slow to train)
- May make poor decisions early in training
- Can get stuck in suboptimal strategies
- Requires careful reward design

## Best For

- Long-term learning scenarios
- Adapting to specific opponents
- Discovering novel strategies

