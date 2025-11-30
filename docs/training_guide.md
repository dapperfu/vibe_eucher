# ML Training Guide for Euchre AI

This guide explains how to train ML models for Euchre AI opponents using supervised learning, GANs, and reinforcement learning. The training system supports configurable durations, convergence detection, risk/temperature parameters, and comprehensive input documentation.

## Overview

The Euchre AI system supports three ML approaches:
1. **Supervised Learning (sklearn)**: Train on collected game data
2. **GANs**: Generate optimal play strategies through adversarial training
3. **Reinforcement Learning**: Learn through self-play with reward signals

## 1. Supervised Learning Training

### Collecting Training Data

First, collect training data by running games:

```bash
python scripts/data/collect_training_data.py --num_games 1000
```

This will:
- Run games with various player combinations
- Collect all decision points (order up, call trump, play card, discard)
- Save data to `training_data/` directory

### Training Models

Train sklearn models on collected data:

```bash
python scripts/training/train_models.py --mode train --model_type random_forest
```

Available model types:
- `random_forest`: Random Forest Classifier
- `gradient_boosting`: Gradient Boosting Classifier
- `neural_network`: Multi-layer Perceptron

### Using Trained Models

Trained models are automatically loaded when creating ML players:

```python
from eucher.game import Game

# Create game with ML players
player_config = [
    ("ML1", "ml"),
    ("ML2", "ml"),
    ("ML3", "ml"),
    ("ML4", "ml"),
]
game = Game(player_config)
```

## 2. GAN Training

**⚠️ IMPORTANT: The GAN backend REQUIRES training before use.**

The GAN model can be initialized without a trained model file, but it will use random weights and make essentially random decisions. **Training is mandatory** for the GAN backend to produce meaningful predictions.

### Training GAN Models

Train GAN models on expert/human decision data:

```bash
python scripts/training/train_models.py --mode train_gan
```

Or use the example script:

```bash
python examples/train_gan_example.py
```

The GAN training:
- Uses expert decisions as "real" examples
- Trains generator to produce similar decisions
- Trains discriminator to distinguish real from fake
- Saves model to `models/gan_model.pth`

**Note:** Training data must be collected first. The training script expects training data files in the `training_data/` directory (e.g., `training_play_card.json`).

### Using GAN Models

After training, create ML players with GAN backend:

```python
from eucher.players.computer.ml.player import MLPlayer

# Create GAN-based player (automatically loads trained model if available)
gan_player = MLPlayer(backend="gan", game_state_provider=get_game_state)
```

The `MLPlayer` will automatically load the trained model from `models/gan_model.pth` if it exists. **Without a trained model, the GAN backend will make random decisions.**

## 3. Reinforcement Learning Training

### Training RL Agents

Train RL agents through self-play:

```bash
python scripts/training/train_models.py --mode train_rl --num_games 1000
```

Or use the example script:

```bash
python examples/train_rl_example.py
```

The RL training:
- Plays games with current policy
- Collects (state, action, reward, next_state) experiences
- Updates Q-network using experience replay
- Gradually reduces exploration (epsilon decay)

### Using RL Agents

Create ML players with RL backend:

```python
from eucher.players.computer.ml.player import MLPlayer

# Create RL-based player
rl_player = MLPlayer(backend="rl", game_state_provider=get_game_state, training_mode=False)
```

## 4. Collecting Human Decisions

### Recording Human Decisions

To collect human decisions for training:

```python
from eucher.training.human_decision_recorder import HumanDecisionRecorder

recorder = HumanDecisionRecorder()
recorder.start_game("game_1")

# During gameplay, record decisions
recorder.record_order_up_decision(
    player_id=0,
    hand=player.hand,
    turned_card=turned_card,
    dealer_id=0,
    human_decision=True,
    game_state={"trick_number": 0, "tricks_won_team0": 0, "tricks_won_team1": 0},
)

# Save at end
recorder.save()
```

### Loading Expert Decisions

Load expert decisions from files:

```python
from eucher.training.expert_decision_loader import ExpertDecisionLoader

loader = ExpertDecisionLoader()
expert_decisions = loader.load_from_json(Path("expert_decisions.json"))

# Validate
is_valid, errors = loader.validate_expert_decisions(expert_decisions)

# Merge with training data
loader.merge_with_training_data(
    expert_decisions,
    training_data_file=Path("training_data/training_play_card.json"),
    output_file=Path("training_data/merged.json"),
)
```

## 5. Reviewing AI Decisions

### Interactive Review

Review AI decisions to validate playing style:

```bash
python scripts/review/review_decisions.py --decisions_file training_data/decisions_to_review.json
```

The review interface allows you to:
- View game state and AI decision
- Approve or reject decisions
- Rate decision quality (1-5)
- Provide feedback notes
- See alternative decisions

### Review Statistics

View review statistics:

```bash
python scripts/review/review_decisions.py --stats
```

### Export Reviews

Export reviews for analysis:

```bash
python scripts/review/review_decisions.py --export reviews.csv --export_format csv
```

## 6. Complete Training Workflow

### Recommended Workflow

1. **Initial Data Collection**:
   ```bash
   python scripts/collect_training_data.py --num_games 500
   ```

2. **Train Supervised Models**:
   ```bash
   python scripts/training/train_models.py --mode train
   ```

3. **Test Models in Games**:
   ```bash
   python main.py  # Use "ml" players
   ```

4. **Collect Human/Expert Decisions**:
   - Play games and record decisions
   - Or load expert decision files

5. **Train GAN Models** (required if using GAN backend):
   ```bash
   python scripts/training/train_models.py --mode train_gan
   ```
   **Note:** GAN models require training data and must be trained before use. Without training, the GAN backend will use random weights and make essentially random decisions.

6. **Train RL Agents** (optional):
   ```bash
   python scripts/training/train_models.py --mode train_rl --num_games 1000
   ```

7. **Review Decisions**:
   ```bash
   python scripts/review/review_decisions.py
   ```

8. **Iterate**: Collect more data, retrain, review, improve

## 7. Model Evaluation

### Testing Models

Run tests to verify models work correctly:

```bash
pytest tests/test_ml_models_gan.py
pytest tests/test_ml_models_rl.py
pytest tests/test_ml_training.py
pytest tests/test_decision_review.py
```

### Comparing Backends

Test different backends:

```python
# Supervised
player_supervised = MLPlayer(backend="supervised")

# GAN (requires trained model - see GAN Training section)
player_gan = MLPlayer(backend="gan")

# RL
player_rl = MLPlayer(backend="rl")
```

**Important Notes:**
- **GAN backend**: Must be trained before use. Without `models/gan_model.pth`, it will make random decisions.
- **RL backend**: Can be used without training, but will perform better after training.
- **Supervised backend**: Requires training data and model training.

## 8. Advanced Training Features

### Time-Based Training

Train for a specific duration:

```bash
# Train for 30 minutes
python scripts/training/train_models.py --mode train_rl --duration 30m

# Train for 2 hours
python scripts/training/train_models.py --mode train_rl --duration 2h

# Train for 1 day
python scripts/training/train_models.py --mode train_rl --duration 1d
```

### Convergence-Based Training

Train until the model achieves a target win rate:

```bash
# Train until 90% win rate (default)
python scripts/training/train_models.py --mode train_rl --until-converged

# Train until 95% win rate
python scripts/training/train_models.py --mode train_rl --until-converged --target-win-rate 0.95

# Use custom window size for convergence check
python scripts/training/train_models.py --mode train_rl --until-converged --window-size 200
```

### Risk/Temperature Parameters

Control player behavior with separate risk factors for trump selection and gameplay:

```bash
# Conservative trump selection, risky gameplay
python scripts/training/train_models.py --mode train_rl \
    --trump-selection-risk 0.3 \
    --gameplay-risk 0.8

# Balanced play (default: 0.5 for both)
python scripts/training/train_models.py --mode train_rl \
    --trump-selection-risk 0.5 \
    --gameplay-risk 0.5
```

**Understanding Temperature Thresholds:**
- **Low temperature (0.0-0.3)**: Conservative play - only accepts high-confidence actions (weight >= 0.3)
- **Medium temperature (0.3-0.7)**: Balanced play - accepts moderate-confidence actions
- **High temperature (0.7-1.0)**: Risky play - accepts lower-confidence actions (weight >= 0.7)

The ML models output decision weights (confidence scores) for each action. Actions with weight >= temperature threshold are considered valid.

### Decision Weight System

The ML models output decision weights (confidence scores) for each possible action:
- **Order Up**: Weight represents confidence in ordering up vs passing
- **Call Trump**: Weights for each suit (including pass)
- **Play Card**: Weights for each card in hand
- **Discard**: Weights for each card to discard

Temperature thresholds filter actions by these weights, allowing control over risk-taking behavior.

### Checkpointing

Save training progress at regular intervals:

```bash
# Save checkpoint every 50 games
python scripts/training/train_models.py --mode train_rl --checkpoint-interval 50
```

Checkpoints are saved to `models/checkpoints/` directory.

## 9. Model Input Documentation

### Generating Documentation

Generate human-readable documentation of all 236 model input features:

```bash
# Generate both markdown and HTML documentation
python scripts/docs/generate_input_docs.py

# Generate only markdown
python scripts/docs/generate_input_docs.py --format markdown

# Generate only HTML
python scripts/docs/generate_input_docs.py --format html

# Specify output directory
python scripts/docs/generate_input_docs.py --output-dir docs/
```

### Reviewing Model Inputs

**Markdown Documentation** (`docs/model_inputs.md`):
- Structured reference document
- Complete feature descriptions
- Feature categories and ranges
- Decision weight system explanation

**HTML Documentation** (`docs/model_inputs.html`):
- Interactive report with searchable features
- Filterable by category
- Visual feature breakdowns
- Example decision scenarios

The documentation includes:
- **Hand Encoding** (120 features): 5 cards × 24 one-hot encodings
- **Turned Card** (24 features): One-hot encoding of turned card
- **Trump Suit** (4 features): One-hot encoding of trump suit
- **Led Suit** (4 features): One-hot encoding of led suit
- **Trick Cards** (72 features): 3 cards × 24 one-hot encodings
- **Positional Features** (12 features): Player position, dealer, team, trick info

## 10. Tips and Best Practices

- **Data Quality**: More high-quality data is better than lots of poor data
- **Review Regularly**: Review AI decisions to catch issues early
- **Start Simple**: Begin with supervised learning, then try GAN/RL
- **Iterate**: Training is an iterative process - collect, train, test, repeat
- **Balance**: Use mix of player types in self-play for diverse training data
- **Validation**: Always validate models on held-out test data
- **Risk Factors**: Experiment with different risk factors to find optimal play style
- **Convergence**: Use convergence-based training for automatic stopping when model is ready
- **Documentation**: Review input documentation to understand what the model sees

## Troubleshooting

### Models Not Loading
- Check that model files exist in `models/` directory
- Verify file paths in MLConfig
- **GAN backend**: Ensure `models/gan_model.pth` exists after training

### Poor Performance
- **GAN backend**: If using untrained GAN, it will make random decisions. Train the model first.
- Collect more training data
- Try different model types
- Review decisions to identify issues
- Adjust hyperparameters

### Training Errors
- Check that training data exists
- Verify data format is correct
- Ensure sufficient memory/GPU resources
- **GAN training**: Ensure training data files (e.g., `training_play_card.json`) exist in `training_data/` directory

