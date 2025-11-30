# PyTorch Strategic Bot

## Overview

The PyTorch Strategic Bot is the most advanced bot, combining deep neural networks with strategic rule-based logic. It's like having a chess grandmaster who also uses a computer.

## Architecture

- **Deep Neural Network**: Sophisticated multi-layer neural network
- **Strategic Overrides**: Special rules for important situations (e.g., bower drawing)
- **Game State Tracking**: Remembers played cards and estimates opponent holdings
- **Risk Management**: Configurable risk factors for different play styles

## Decision Making

### Key Features

1. **Deep Neural Network**:
   - Processes complex game information simultaneously
   - Learns subtle patterns and relationships
   - Multiple layers for hierarchical feature learning

2. **Strategic Overrides**:
   - Special rules for critical situations
   - Knows when to "draw out" bowers (force opponents to play best cards)
   - Combines learned patterns with expert knowledge

3. **Game State Tracking**:
   - Remembers what cards have been played
   - Estimates what cards opponents might have
   - Tracks trick history and adapts strategy

4. **Risk Management**:
   - Uses risk factors to control aggressiveness
   - Can be tuned for different play styles

## Training

### Self-Play Training
```bash
python scripts/training/pytorch_ai/train_pytorch_ai_self_play.py \
    --games-per-training 100 \
    --epsilon 0.1
```

### With Custom Configuration
```bash
python scripts/training/pytorch_ai/train_pytorch_ai_self_play.py \
    --games-per-training 200 \
    --epsilon 0.15 \
    --player-config pytorch_vs_heuristic \
    --tune-batch
```

## Configuration

- `model_path`: Path to pre-trained model (optional)
- `device`: "cpu", "cuda", or None for auto
- `use_strategic_overrides`: Enable strategic overrides (default: True)
- `exploration_epsilon`: Exploration probability for training (0.0-1.0)

## Usage

```python
from eucher.game import Game

player_config = [
    ("PyTorch1", "pytorch_ai"),
    ("PyTorch2", "pytorch_ai"),
    ("PyTorch3", "pytorch_ai"),
    ("PyTorch4", "pytorch_ai"),
]

game = Game(player_config)
```

## Strengths

- Most sophisticated decision-making
- Combines learning with expert knowledge
- Adapts to game situations
- Can track opponent patterns
- Highly configurable

## Weaknesses

- Requires significant computational resources
- Needs training to be effective
- More complex to understand and debug
- May be slower than simpler bots

## Best For

- Competitive play
- Research and development
- Challenging opponents
- Discovering advanced strategies

