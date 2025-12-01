# Computer Bot Documentation

This directory contains detailed documentation for all computer bot types available in the Euchre game.

## Available Bots

### Basic Bots
- [Random Bot](random.md) - Makes all decisions randomly
- [Heuristic Bot](heuristic.md) - Rule-based bot with basic strategy
- [Weighted Heuristic Bot](weighted_heuristic.md) - Weighted heuristic bot with strategic decision-making

### Machine Learning Bots
- [ML Supervised Bot](ml_supervised.md) - Trained on collected game data
- [ML Reinforcement Learning Bot](ml_rl.md) - Learns through self-play
- [ML GAN Bot](ml_gan.md) - Uses Generative Adversarial Networks
- [PyTorch Strategic Bot](pytorch_strategic.md) - Deep neural network with strategic overrides

### Advanced Search-Based Bots
- [EuchreZero Bot](euchre_zero.md) - AlphaZero-style MCTS with neural networks
- [PerceiverMuZero Bot](perceiver_muzero.md) - Perceiver architecture with MuZero-style MCTS
- [EucherGo Bot](euchergo.md) - Hybrid MCTS and deep neural network engine

## Quick Reference

| Bot Type | Complexity | Strategy Level | Learning | Best Use Case |
|----------|------------|----------------|----------|---------------|
| Random | Very Low | None | No | Testing, baseline |
| Heuristic | Low | Basic | No | Learning, predictable play |
| Weighted Heuristic | Medium | Intermediate | No | Strategic play |
| ML Supervised | Medium-High | Advanced | Yes (from data) | Competitive play |
| ML RL | High | Advanced | Yes (from experience) | Long-term learning |
| ML GAN | Very High | Advanced | Yes (adversarial) | Research, creativity |
| PyTorch Strategic | Very High | Expert | Yes (hybrid) | Competitive, research |
| EuchreZero | Very High | Expert | Yes (MCTS+NN) | Competitive, research |
| PerceiverMuZero | Very High | Expert | Yes (MCTS+NN) | Competitive, research |
| EucherGo | Very High | Expert | Yes (MCTS+NN) | Competitive, research |

## Plugin System

All bots are registered as plugins and can be automatically discovered. See the [Plugin Architecture](../../eucher/plugins.py) for details on how to add new bot types.

## Usage

To use a bot in a game:

```python
from eucher.game import Game

# Create game with bot players
player_config = [
    ("Player 1", "euchergo"),
    ("Player 2", "heuristic"),
    ("Player 3", "weighted_heuristic"),
    ("Player 4", "random"),
]

game = Game(player_config)
```

## Training

Some bots require training before use:
- ML Supervised: Requires training data collection
- ML RL: Requires self-play training
- ML GAN: Requires adversarial training
- PyTorch Strategic: Can use pre-trained models or train from scratch
- EuchreZero: Requires self-play training with MCTS
- PerceiverMuZero: Requires self-play training with MCTS
- EucherGo: Requires self-play training with MCTS

See individual bot documentation for training instructions.


