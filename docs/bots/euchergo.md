# EucherGo Bot

## Overview

EucherGo is a competitive Euchre engine that learns optimal play using reinforcement learning, self-play, a policy network, a value network, and a Monte Carlo Tree Search loop, patterned after the AlphaZero/MuZero architecture used by DeepMind.

## Architecture

EucherGo uses a hybrid architecture combining:

1. **Deep Neural Networks**: Shared trunk with separate policy and value heads
2. **Monte Carlo Tree Search (MCTS)**: PUCT-based search with value network rollouts
3. **Belief Tracking**: Probability distributions for unknown cards
4. **Self-Play Training**: Learns by playing against itself

### Network Architecture

- **Shared Trunk**: Residual Conv1D or MLP stack with layer normalization
- **Policy Head**: Outputs action logits for all possible moves (order up, call trump, play cards, go alone)
- **Value Head**: Outputs scalar win probability
- **Action Masking**: Ensures only legal moves are considered

### State Representation

The state encoder creates a fixed-length tensor (712 dimensions) containing:
- Player hand: binary 24-vector (one per card)
- Trump suit: one-hot encoding (5-dim: None + 4 suits)
- Dealer/leader indices: one-hot encoding (4-dim each)
- Current trick cards: binary vectors (96-dim: 4 cards × 24 dims)
- Trick history: binary vectors (480-dim: 5 tricks × 4 cards × 24 dims)
- Belief map: probability distributions per player (96-dim: 4 players × 24 dims)
- Flags: can follow suit, must call trump, can go alone (3-dim)

### MCTS Search

- Uses PUCT (Polynomial Upper Confidence Trees) formula
- Value network provides rollouts (no random rollouts)
- Configurable number of simulations per move (default: 100)
- Temperature control for exploration
- Batch GPU evaluation support

## Decision Making

### Ordering Up
- Encodes current game state
- Runs MCTS search with action mask for order up/pass
- Selects action with highest probability

### Calling Trump
- Encodes current game state
- Runs MCTS search with action mask for suit selection
- Considers all available suits (excluding turned card suit)
- Handles "screw the dealer" rule (must choose)

### Playing Cards
- Encodes current game state including current trick
- Runs MCTS search with action mask for valid plays
- Considers suit following rules and trump dynamics
- Updates belief tracker after each play

### Going Alone
- Encodes current game state after trump is called
- Runs MCTS search to evaluate loner potential
- Considers hand strength and trick-taking potential

### Discarding
- Encodes current game state
- Runs MCTS search with action mask for discard options
- Selects card that minimizes hand weakness

## Training

### Self-Play Generation
- Plays games against itself or other bots
- Collects state-action pairs with MCTS-improved policies
- Stores game outcomes for value training

### Training Loop
- **Policy Loss**: Cross-entropy between network policy and MCTS-improved policy
- **Value Loss**: Mean squared error between network value and game outcome
- **Regularization**: L2 regularization on network weights
- **Batch Training**: Processes multiple games in parallel

### Configuration

Key training parameters:
- `num_simulations`: MCTS simulations per move (default: 100)
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Optimizer learning rate (default: 1e-3)
- `num_games`: Games per training iteration (default: 10)
- `checkpoint_interval`: Save checkpoint every N iterations (default: 5)

## Usage

### Basic Usage

```python
from eucher.game import Game

# Create game with EucherGo players
player_config = [
    ("EucherGo1", "euchergo"),
    ("EucherGo2", "euchergo"),
    ("EucherGo3", "euchergo"),
    ("EucherGo4", "euchergo"),
]

game = Game(player_config)
```

### With Custom Configuration

```python
from eucher.players.computer.euchergo.player import EucherGoPlayer
from eucher.players.computer.euchergo.config import EucherGoConfig

config = EucherGoConfig(
    num_simulations=200,  # More simulations for stronger play
    hidden_size=512,       # Larger network
    num_layers=6,          # Deeper network
)

player = EucherGoPlayer(
    config=config,
    model_path="models/checkpoints/euchergo/best_model.pt",
    risk_factor=0.0,  # Conservative play
)
```

### Training

```bash
# Basic training
python scripts/training/euchergo/train_euchergo.py --duration 1h

# With custom parameters
python scripts/training/euchergo/train_euchergo.py \
    --duration 2h \
    --num-games 20 \
    --num-simulations 200 \
    --batch-size 64 \
    --checkpoint-interval 10

# Resume from checkpoint
python scripts/training/euchergo/train_euchergo.py \
    --checkpoint models/checkpoints/euchergo/euchergo_iter_100.pt \
    --duration 1h
```

## Configuration

### Model Parameters
- `input_size`: Size of state tensor (default: 712)
- `hidden_size`: Hidden layer size (default: 256)
- `num_layers`: Number of residual layers (default: 4)
- `use_conv1d`: Use Conv1D instead of MLP (default: False)

### MCTS Parameters
- `num_simulations`: Simulations per move (default: 100)
- `exploration_constant`: PUCT exploration constant (default: 1.0)

### Training Parameters
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Learning rate (default: 1e-3)
- `num_games`: Games per iteration (default: 10)

## Strengths

- **Strong Strategic Play**: MCTS explores many possible game lines
- **Adaptive**: Learns optimal strategies through self-play
- **Belief Tracking**: Maintains probability distributions for unknown cards
- **Scalable**: Can be trained with more simulations for stronger play
- **GPU Accelerated**: Efficient batch processing for training

## Weaknesses

- **Computational Cost**: MCTS requires many simulations (slower decisions)
- **Training Time**: Requires extensive self-play training
- **Memory Usage**: Stores belief maps and game history
- **Complexity**: More complex than simpler bots

## Performance

Expected performance characteristics:
- **Training**: ~1000-10000 games per hour on modern GPU
- **Inference**: ~1-10 seconds per move (depending on simulations)
- **Win Rate**: Improves with training, can exceed 60% against heuristic bots

## Best For

- Competitive play after training
- Research and experimentation
- Discovering optimal strategies
- Challenging opponents

## See Also

- [EuchreZero Bot](euchre_zero.md) - Similar architecture with different network design
- [PerceiverMuZero Bot](perceiver_muzero.md) - Perceiver-based variant
- [Training Guide](../training_guide.md) - General ML training guide

