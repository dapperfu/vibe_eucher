# EuchreZero Implementation Plan

## Overview
This document outlines the complete implementation plan for EuchreZero, an AlphaZero/MuZero-inspired reinforcement learning system for Euchre. The implementation will be integrated into the existing Euchre codebase structure.

## Architecture Overview

### Core Components
1. **Representation Network** - Encodes game state to latent vector
2. **Dynamics Network** - Predicts next state and immediate reward
3. **Prediction Network** - Outputs policy and value estimates
4. **MCTS Engine** - Monte Carlo Tree Search with belief sampling
5. **Deduction System** - Hidden card inference and probability tracking
6. **Training System** - Self-play, replay buffer, and curriculum learning
7. **Integration Layer** - Connects to existing game system

## Directory Structure

Matches the existing taxonomic structure of the codebase:

```
eucher/
├── players/
│   └── computer/
│       └── euchre_zero/                  # Player implementation (matches ml/, pytorch/ pattern)
│           ├── __init__.py
│           ├── networks/
│           │   ├── __init__.py
│           │   ├── representation.py      # Representation network
│           │   ├── dynamics.py            # Dynamics network
│           │   └── prediction.py          # Policy/value network
│           ├── mcts/
│           │   ├── __init__.py
│           │   ├── tree.py                # MCTS tree structure
│           │   ├── search.py              # MCTS search algorithm
│           │   └── belief_sampling.py     # Hidden state sampling
│           ├── deduction/
│           │   ├── __init__.py
│           │   ├── card_tracker.py        # Perfect memory tracking
│           │   ├── probability_map.py     # Deduction probability maps
│           │   └── inference.py          # Suit void and trump inference
│           ├── training/                  # Training modules (matches pytorch/training/ pattern)
│           │   ├── __init__.py
│           │   ├── self_play.py           # Self-play game generation
│           │   ├── replay_buffer.py      # Prioritized experience replay
│           │   ├── trainer.py             # Training loop
│           │   ├── curriculum.py          # Curriculum learning stages
│           │   └── evaluator.py           # Model evaluation metrics
│           ├── rewards/
│           │   ├── __init__.py
│           │   └── reward_calculator.py   # Reward computation
│           ├── state_encoder.py          # Game state to tensor encoding
│           ├── action_space.py            # Unified action space
│           ├── player.py                  # EuchreZero player profile
│           └── config.py                 # Configuration
├── training/                              # General training utilities (existing)
│   ├── train_gan.py
│   ├── train_rl.py
│   ├── train_supervised.py
│   └── ... (existing files)
└── scripts/                               # Training scripts
    └── training/                          # All training scripts
        └── euchre_zero/                  # Matches pytorch_ai/, transformer_rl/ pattern
        ├── train_euchre_zero.py          # Main training script
        ├── train_euchre_zero_cpu.sh       # CPU training wrapper
        └── train_euchre_zero_gpu.sh      # GPU training wrapper
```

## Component Specifications

### 1. Representation Network (`networks/representation.py`)

**Purpose**: Encode full game state into latent vector (256-512 dimensions)

**Input Features**:
- Player hand (5 cards) - one-hot encoded (24 cards × 5 positions)
- Upcard representation (24-dim one-hot)
- Discard status if dealer (boolean)
- Player seat (0-3) and partner seat (0-3)
- Trump context (5-dim: None, Hearts, Diamonds, Clubs, Spades)
- Bidding stage (3-dim: order_up, call_trump, playing)
- Score state (2-dim: team0_score, team1_score)
- Trick history (5 tricks × 4 cards × features)
- Perfect memory card state (24 cards × location: hand/played/discard)
- Deduction probability maps (24 cards × 3 players = 72 dims)
- Suit void inference (4 suits × 3 players = 12 dims)
- Trump count inference (3 players × counts = 3 dims)
- Risk factor (1-dim)

**Architecture**:
- Input processing layers for each feature group
- Concatenation layer
- 3-4 fully connected layers with ReLU
- Batch normalization
- Output: latent vector (256-512 dims)

**Key Functions**:
- `forward(state_tensor) -> latent_vector`
- `encode_game_state(game_state) -> tensor`

### 2. Dynamics Network (`networks/dynamics.py`)

**Purpose**: Predict next latent state and immediate reward after action

**Input**:
- Current latent state (256-512 dims)
- Action encoding (unified action space)

**Output**:
- Next latent state (256-512 dims)
- Immediate reward (1-dim)

**Architecture**:
- Concatenate latent + action
- 2-3 fully connected layers
- Split into state and reward heads
- State head: same size as input latent
- Reward head: single value

**Key Functions**:
- `forward(latent_state, action) -> (next_latent, reward)`

### 3. Prediction Network (`networks/prediction.py`)

**Purpose**: Output policy and value estimates for MCTS

**Input**:
- Latent state (256-512 dims)

**Output**:
- Policy logits over all actions (unified action space)
- Value estimate (1-dim)
- Risk-adjusted value estimate (1-dim)

**Architecture**:
- Input: latent state
- Shared layers (2-3 FC layers)
- Policy head: FC layer → action_space_size logits
- Value head: FC layer → 1 value
- Risk value head: FC layer → 1 value (with risk scaling)

**Key Functions**:
- `forward(latent_state) -> (policy_logits, value, risk_value)`

### 4. Unified Action Space (`action_space.py`)

**Purpose**: Define unified action encoding for all game phases

**Action Types**:
- Bidding: pass (0), order_up (1), call_hearts (2), call_diamonds (3), call_clubs (4), call_spades (5), stick_dealer (6)
- Discard: card_0 (7), card_1 (8), card_2 (9), card_3 (10), card_4 (11), card_5 (12)
- Play: card_0 (13), card_1 (14), card_2 (15), card_3 (16), card_4 (17)

**Total**: 18 actions (some may be invalid in certain states, but network learns via penalties)

**Key Functions**:
- `encode_action(action_type, card_index) -> int`
- `decode_action(action_id) -> (action_type, card_index)`
- `get_valid_actions(game_state) -> List[int]` (for evaluation only)

### 5. State Encoder (`state_encoder.py`)

**Purpose**: Convert game state to tensor representation

**Key Functions**:
- `encode_full_state(game, player_id) -> Dict[str, torch.Tensor]`
- `encode_hand(hand) -> torch.Tensor`
- `encode_trick_history(tricks) -> torch.Tensor`
- `encode_perfect_memory(game, player_id) -> torch.Tensor`

### 6. Deduction System

#### 6.1 Card Tracker (`deduction/card_tracker.py`)
- Track all 24 cards: location (hand/played/discard/unknown)
- Update as game progresses
- Perfect memory of all visible information

#### 6.2 Probability Map (`deduction/probability_map.py`)
- Maintain probability distributions: P(card | player)
- Update via Bayesian inference
- Support sampling for MCTS

#### 6.3 Inference Engine (`deduction/inference.py`)
- Suit void inference (player has no cards of suit)
- Trump count inference (estimated trump cards per player)
- Partner inference (likely cards in partner's hand)

**Key Functions**:
- `update_from_play(card, player_id)`
- `get_probability_map() -> np.ndarray`
- `sample_hidden_state() -> Dict[int, List[Card]]`
- `infer_suit_voids() -> Dict[int, Set[Suit]]`

### 7. MCTS Engine

#### 7.1 Tree Structure (`mcts/tree.py`)
- Node class with:
  - State (latent representation)
  - Visit count
  - Value sum
  - Risk-adjusted value sum
  - Children (action → node mapping)
  - Prior probability
  - Hidden state sample (for belief-based search)

#### 7.2 Search Algorithm (`mcts/search.py`)
- Selection: UCB1 with policy prior
- Expansion: Add new nodes
- Evaluation: Use prediction network
- Backpropagation: Update values with risk scaling
- Belief sampling: Sample hidden states for each rollout

**Key Functions**:
- `search(game_state, num_simulations, risk_factor) -> improved_policy`
- `select_action(node) -> action`
- `expand(node, action) -> new_node`
- `evaluate(node) -> (value, policy)`
- `backpropagate(node, value)`

#### 7.3 Belief Sampling (`mcts/belief_sampling.py`)
- Sample hidden card assignments from deduction probabilities
- Run multiple rollouts per sampled state
- Average results across samples

**Key Functions**:
- `sample_belief_states(deduction_map, num_samples) -> List[Dict]`
- `weight_rollouts(rollout_results, probabilities) -> weighted_policy`

### 8. Reward System (`rewards/reward_calculator.py`)

**Immediate Rewards**:
- Trick won: +1
- Trick lost: -1
- Drawing trump effectively: +1
- Burning trump uselessly: -1

**Bidding Rewards**:
- Successful call (3-4 tricks): +4
- Successful sweep: +5
- Failed call: -4
- Correct pass: 0
- Incorrect pass: -1

**Discard Rewards**:
- Strong discard: +1
- Harmful discard: -1

**Illegal Play Penalties**:
- Renege: -5 immediate
- Trick lost due to renege: -2
- Partner harmed: -3
- Hand penalty: -6

**Hand Outcome Rewards**:
- Calling team success: +3
- Calling team sweep: +4
- Calling team set: -4
- Defenders set caller: +3
- Defenders allow sweep: -1

**Risk Scaling**:
- `reward_scaled = reward * (1 + 0.5 * risk)`
- `penalty_scaled = penalty * (1 - 0.4 * risk)`

**Key Functions**:
- `calculate_immediate_reward(action, outcome) -> float`
- `calculate_hand_reward(hand_result, player_team) -> float`
- `apply_risk_scaling(reward, risk_factor, is_penalty) -> float`

### 9. Training System

#### 9.1 Self-Play (`training/self_play.py`)
- Run games with 4 EuchreZero players
- Each action guided by MCTS
- Store (state, improved_policy, final_outcome) tuples
- Support batched self-play for GPU efficiency

**Key Functions**:
- `generate_self_play_game(model, num_simulations, risk_factor) -> List[GameStep]`
- `run_batched_self_play(model, num_games, batch_size) -> List[GameStep]`

#### 9.2 Replay Buffer (`training/replay_buffer.py`)
- Prioritized experience replay
- Store (state, policy, value, reward) tuples
- Sample batches for training
- Support curriculum learning stages

**Key Functions**:
- `add_experience(state, policy, value, reward, priority)`
- `sample_batch(batch_size) -> batch`
- `update_priorities(indices, new_priorities)`

#### 9.3 Trainer (`training/trainer.py`)
- Training loop with multiple loss components
- Policy loss: KL divergence with MCTS improved policy
- Value loss: MSE with final outcome
- Risk value loss: MSE with risk-adjusted value
- Dynamics loss: MSE with predicted next state
- Reward loss: MSE with immediate reward

**Loss Function**:
```
total_loss = (
    λ_policy * policy_loss +
    λ_value * value_loss +
    λ_risk_value * risk_value_loss +
    λ_dynamics * dynamics_loss +
    λ_reward * reward_loss
)
```

**Key Functions**:
- `train_step(batch) -> losses`
- `evaluate_model(model, test_games) -> metrics`

#### 9.4 Curriculum (`training/curriculum.py`)
- Stage 1: Basic legality training (penalties only)
- Stage 2: Trick strategy (partial MCTS)
- Stage 3: Add deduction learning
- Stage 4: Introduce bidding
- Stage 5: Full hand MCTS
- Stage 6: Multi-agent self-play at scale

**Key Functions**:
- `get_current_stage(iteration) -> int`
- `get_stage_config(stage) -> Config`
- `should_advance_stage(metrics) -> bool`

#### 9.5 Evaluator (`training/evaluator.py`)
- Track evaluation metrics:
  - Tricks per hand
  - Win rate vs previous versions
  - Sweep and set rates
  - Bidding accuracy
  - Legality violation frequency
  - Value prediction error
  - MCTS branching efficiency
  - Tournament win rate

**Key Functions**:
- `evaluate_model(model, num_games) -> Dict[str, float]`
- `compare_models(model1, model2, num_games) -> Dict[str, float]`

### 10. Integration Layer

#### 10.1 EuchreZero Player (`player.py`)
- Implements `PlayerProfile` interface
- Uses MCTS for all decisions
- Integrates with existing game system

**Key Functions**:
- `decide_order_up(...) -> bool`
- `decide_call_trump(...) -> Optional[Suit]`
- `choose_card_to_discard(...) -> Card`
- `play_card(...) -> Card`

#### 10.2 Configuration (`config.py`)
- Model hyperparameters
- Training hyperparameters
- MCTS parameters
- Reward scaling factors
- Risk factor settings

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
1. Set up directory structure
2. Implement action space encoding
3. Implement state encoder
4. Create basic network architectures (skeleton)
5. Implement reward calculator
6. Create configuration system

### Phase 2: Neural Networks (Week 2-3)
1. Complete representation network
2. Complete dynamics network
3. Complete prediction network
4. Create unified model class
5. Add GPU support and batching

### Phase 3: Deduction System (Week 3-4)
1. Implement card tracker
2. Implement probability maps
3. Implement inference engine
4. Test deduction accuracy

### Phase 4: MCTS Engine (Week 4-5)
1. Implement tree structure
2. Implement search algorithm
3. Implement belief sampling
4. Test MCTS correctness
5. Optimize for performance

### Phase 5: Training Infrastructure (Week 5-6)
1. Implement self-play generator
2. Implement replay buffer
3. Implement trainer with all losses
4. Implement curriculum system
5. Implement evaluator

### Phase 6: Integration (Week 6-7)
1. Create EuchreZero player profile
2. Integrate with game system
3. Add CLI support for EuchreZero players
4. Test end-to-end gameplay

### Phase 7: Training and Optimization (Week 7-8)
1. Run initial training runs
2. Tune hyperparameters
3. Optimize GPU utilization
4. Implement distributed training (optional)
5. Create training scripts

### Phase 8: Evaluation and Refinement (Week 8+)
1. Run evaluation tournaments
2. Compare against existing players
3. Refine reward structure
4. Optimize MCTS parameters
5. Document results

## Training Scripts

### Main Training Script (`scripts/training/euchre_zero/train_euchre_zero.py`)

**Location**: Matches existing pattern (`scripts/training/pytorch_ai/`, `scripts/training/transformer_rl/`)

**Features**:
- Load or initialize model
- Run self-play games
- Train on replay buffer
- Save checkpoints
- Evaluate periodically
- Support curriculum learning
- GPU/CPU support
- Distributed training support (optional)

**Command Line Arguments**:
- `--config`: Config file path
- `--checkpoint`: Load checkpoint
- `--num_games`: Games per iteration
- `--num_iterations`: Training iterations
- `--batch_size`: Training batch size
- `--num_simulations`: MCTS simulations
- `--risk_factor`: Risk factor
- `--device`: cuda/cpu
- `--curriculum`: Enable curriculum learning
- `--evaluate_every`: Evaluation frequency

**Note**: Training modules (self_play, replay_buffer, trainer, etc.) are in `eucher/players/computer/euchre_zero/training/` (matches `pytorch/training/` pattern)

### Curriculum Training
- Integrated into main training script
- Stage-specific configurations
- Automatic stage advancement

### Evaluation Script
- Load model checkpoint
- Run evaluation games
- Generate metrics report
- Compare against baselines

## Integration with Existing System

### Game Integration
- Add `"euchre_zero"` profile type to `Game._create_profile()`
- EuchreZero player uses MCTS for all decisions
- Supports risk factor configuration
- Compatible with existing TUI

### CLI Integration
- Add `"euchre_zero"` profile type to `Game._create_profile()` in `eucher/game.py`
- Import: `from eucher.players.computer.euchre_zero.player import EuchreZeroPlayer`
- Support checkpoint loading via model path
- Configurable MCTS parameters
- Risk factor configuration
- Training scripts in `scripts/training/euchre_zero/` (matches existing pattern)

### Training Data Integration
- Can use existing training data for supervised pretraining (optional)
- Self-play generates new training data
- Compatible with existing data collection scripts

## Configuration Files

### Model Config (`euchre_zero_config.yaml`)
```yaml
model:
  latent_size: 512
  representation_layers: [1024, 512, 256]
  dynamics_layers: [256, 256]
  prediction_layers: [256, 128]
  action_space_size: 18

training:
  learning_rate: 0.001
  batch_size: 32
  num_games_per_iteration: 100
  num_iterations: 1000
  replay_buffer_size: 100000
  loss_weights:
    policy: 1.0
    value: 1.0
    risk_value: 0.5
    dynamics: 0.5
    reward: 0.1

mcts:
  num_simulations: 100
  exploration_constant: 1.0
  temperature: 1.0
  belief_samples: 10

rewards:
  trick_won: 1.0
  trick_lost: -1.0
  successful_call: 4.0
  failed_call: -4.0
  renege_penalty: -5.0

curriculum:
  enabled: true
  stages:
    - name: legality
      iterations: 100
      mcts_simulations: 10
    - name: trick_strategy
      iterations: 200
      mcts_simulations: 50
    # ... more stages
```

## Testing Strategy

### Unit Tests
- Network forward/backward passes
- Action encoding/decoding
- State encoding
- Reward calculation
- Deduction updates
- MCTS tree operations

### Integration Tests
- Self-play game generation
- Training loop
- Model checkpointing
- Player integration

### Performance Tests
- MCTS simulation speed
- GPU utilization
- Memory usage
- Training throughput

## Documentation

### Code Documentation
- NumPy-style docstrings for all functions
- Type hints throughout
- Architecture diagrams
- API documentation

### User Documentation
- Training guide
- Evaluation guide
- Configuration reference
- Integration guide

## Dependencies

### New Dependencies
- PyTorch (already present)
- NumPy (already present)
- Optional: Ray for distributed training

### No New Dependencies Required
- All other dependencies already in codebase

## Performance Considerations

### GPU Optimization
- Batch MCTS evaluations
- Parallel self-play games
- Efficient tensor operations
- Mixed precision training (optional)

### Memory Management
- Replay buffer size limits
- Gradient checkpointing (optional)
- Efficient state encoding
- MCTS tree pruning

### Scalability
- Support for distributed training
- Multi-GPU support
- Efficient data loading
- Checkpoint management

## Future Enhancements

1. **Partner Style Embeddings**: Learn partner playing styles
2. **Opponent Modeling**: Dedicated head for opponent behavior
3. **Endgame Solver**: Exact solver for final tricks
4. **Risk Personality Training**: Multiple risk profiles
5. **GAN Module**: Stylistic imitation (optional)
6. **Tournament System**: Automated model comparison
7. **Visualization Tools**: MCTS tree visualization, deduction maps

## Success Metrics

1. **Training Convergence**: Loss decreases over time
2. **Game Performance**: Win rate > 60% vs heuristic players
3. **Legality**: < 1% illegal move rate
4. **Bidding Accuracy**: > 70% correct bidding decisions
5. **Value Prediction**: Low MSE on final outcomes
6. **MCTS Efficiency**: Reasonable branching factor

## Risk Mitigation

1. **Complexity Management**: Implement in phases, test incrementally
2. **Performance Issues**: Profile early, optimize bottlenecks
3. **Training Instability**: Use gradient clipping, careful initialization
4. **Memory Issues**: Monitor usage, implement limits
5. **Integration Problems**: Test integration early and often

## Timeline Summary

- **Weeks 1-2**: Core infrastructure
- **Weeks 2-3**: Neural networks
- **Weeks 3-4**: Deduction system
- **Weeks 4-5**: MCTS engine
- **Weeks 5-6**: Training infrastructure
- **Weeks 6-7**: Integration
- **Weeks 7-8**: Training and optimization
- **Week 8+**: Evaluation and refinement

**Total Estimated Time**: 8-10 weeks for full implementation

