# EuchreZero Taxonomic Structure

This document describes how EuchreZero fits into the existing codebase taxonomic structure.

## Existing Taxonomic Patterns

The codebase follows these patterns:

### Player Implementations
- **Location**: `eucher/players/computer/{player_type}/`
- **Examples**:
  - `eucher/players/computer/ml/` - ML players
  - `eucher/players/computer/ml/pytorch/` - PyTorch-specific players
  - `eucher/players/computer/ml/pytorch/training/` - Training modules for PyTorch players

### Training Scripts
- **Location**: `scripts/training/{player_type}/`
- **Examples**:
  - `scripts/pytorch_ai/` - PyTorch AI training scripts
  - `scripts/transformer_rl/` - Transformer RL training scripts
  - `scripts/eucher_zero/` - EuchreZero training scripts (already exists)

### General Training Utilities
- **Location**: `eucher/training/` (flat structure)
- **Examples**:
  - `eucher/training/train_gan.py`
  - `eucher/training/train_rl.py`
  - `eucher/training/train_supervised.py`

## EuchreZero Structure

Following the existing patterns:

### Player Implementation
```
eucher/players/computer/euchre_zero/
├── __init__.py
├── networks/              # Neural networks (matches ml/models/ pattern)
├── mcts/                  # MCTS engine
├── deduction/             # Hidden information handling
├── training/              # Training modules (matches pytorch/training/ pattern)
├── rewards/               # Reward calculation
├── state_encoder.py
├── action_space.py
├── player.py              # PlayerProfile implementation
└── config.py
```

### Training Scripts
```
scripts/eucher_zero/       # Matches scripts/pytorch_ai/ pattern
├── train_euchre_zero.py
├── train_euchre_zero_cpu.sh
└── train_euchre_zero_gpu.sh
```

## Integration Points

### Game Integration
- Add to `eucher/game.py` in `_create_profile()` method:
  ```python
  elif profile_type == "eucher_zero":
      from eucher.players.computer.eucher_zero.player import EucherZeroPlayer
      return EuchreZeroPlayer(...)
  ```

### Import Paths
- Player: `from eucher.players.computer.eucher_zero.player import EucherZeroPlayer`
- Networks: `from eucher.players.computer.euchre_zero.networks.representation import RepresentationNetwork`
- Training modules: `from eucher.players.computer.euchre_zero.training.self_play import generate_self_play_game`
- Training scripts: `scripts/eucher_zero/train_euchre_zero.py` (executable script)

## Key Differences from Original Plan

1. **Training Scripts**: Located in `scripts/eucher_zero/` (not `eucher/training/euchre_zero/`)
   - Matches existing pattern: `scripts/pytorch_ai/`, `scripts/transformer_rl/`
   
2. **Training Modules**: Located in `eucher/players/computer/euchre_zero/training/`
   - Matches existing pattern: `eucher/players/computer/ml/pytorch/training/`

3. **No Separate Training Package**: Training utilities stay in `eucher/training/` (general utilities)
   - EuchreZero-specific training code is in the player directory

## Consistency with Existing Code

This structure ensures:
- ✅ Consistent with `ml/` and `pytorch/` player patterns
- ✅ Consistent with `scripts/pytorch_ai/` and `scripts/transformer_rl/` script patterns
- ✅ Clear separation between player implementation and training scripts
- ✅ Easy to find and maintain
- ✅ Follows existing import conventions

