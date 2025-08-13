# Euchre AI Model Package

This package contains PyTorch-based neural network models and training tools for creating AI euchre players that can learn from game data.

## Overview

The AI model package provides:

- **EuchreNeuralNetwork**: A PyTorch neural network for euchre decision making
- **GameStateEncoder**: Converts euchre game states to neural network inputs
- **TrainingPipeline**: Handles data loading, training, and model saving
- **ModelEvaluator**: Tests trained models by playing full games

## Architecture

### Neural Network Model

The `EuchreNeuralNetwork` is a multi-head neural network that makes three types of decisions:

1. **Trump Ordering**: Whether to order up the top card (binary classification)
2. **Card Selection**: Which card to play in a trick (5-class classification)
3. **Trump Selection**: Which suit to call as trump (4-class classification)

**Model Structure:**
- Input layer: 128 features
- Hidden layers: Configurable (default: 3 layers of 256 units)
- Output heads: Separate linear layers for each decision type
- Regularization: Dropout and layer normalization

### Game State Encoding

The `GameStateEncoder` converts euchre game states to 128-dimensional feature vectors:

- **Hand encoding**: 5 cards × 24 features per card = 120 features
- **Game context**: 8 features (position, dealer status, etc.)
- **Features per card**: Suit (one-hot), rank (one-hot), trump indicator, strength

## Usage

### Training a Model

```bash
# Basic training
python train_euchre_ai.py --data-dir data/games --epochs 100

# With custom parameters
python train_euchre_ai.py \
    --data-dir data/games \
    --output-dir models \
    --epochs 200 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --hidden-size 512 \
    --num-layers 4 \
    --dropout 0.3

# Generate sample data and train
python train_euchre_ai.py \
    --data-dir data/games \
    --generate-data \
    --evaluate
```

### Training Data Format

Training data should be JSON files with the following structure:

```json
{
    "game_id": "game_001",
    "actions": [
        {
            "type": "order_up",
            "player_hand": [
                {"suit": "hearts", "rank": 11, "is_trump": false},
                {"suit": "diamonds", "rank": 14, "is_trump": false}
            ],
            "top_card": {"suit": "hearts", "rank": 10, "is_trump": false},
            "ordered_up": true
        },
        {
            "type": "play_card",
            "player_hand": [...],
            "lead_suit": "hearts",
            "trump_suit": "hearts",
            "played_card": {...}
        },
        {
            "type": "call_trump",
            "dealer_hand": [...],
            "called_trump": "clubs"
        }
    ],
    "game_state": {...}
}
```

### Using a Trained Model

```python
from euchre.ai_model import EuchreNeuralNetwork, EuchreAIPlayer
from euchre.models import Player, PlayerType

# Load trained model
model = EuchreNeuralNetwork.load_model("models/euchre_model.pth")

# Create AI player
player = Player("AI_Player", PlayerType.AI)
ai_player = EuchreAIPlayer(model, player)

# Make decisions
should_order = ai_player.should_order_up(top_card)
card_to_play = ai_player.choose_card_to_play(lead_suit, trump_suit)
trump_suit = ai_player.choose_trump_suit(hand)
```

### Evaluating a Model

```python
from euchre.ai_model import ModelEvaluator

# Create evaluator
evaluator = ModelEvaluator("models/euchre_model.pth", num_games=100)

# Run evaluation
results = evaluator.evaluate_model()

# Generate report
report = evaluator.generate_evaluation_report(results)
print(report)

# Save results
evaluator.save_evaluation_results(results, "evaluation_results.json")
```

## Training Pipeline

The training pipeline handles:

1. **Data Loading**: Parses JSON game files and extracts training samples
2. **Data Preprocessing**: Converts game states to feature tensors
3. **Model Training**: Trains the neural network with validation
4. **Model Saving**: Saves the best model based on validation accuracy

### Training Process

1. **Data Preparation**: Game data is parsed into individual decision samples
2. **Feature Encoding**: Game states are converted to 128-dimensional tensors
3. **Batch Training**: Samples are batched and fed through the network
4. **Loss Calculation**: Cross-entropy loss for each decision type
5. **Optimization**: Adam optimizer with learning rate scheduling
6. **Validation**: Model performance is evaluated on held-out data

## Model Evaluation

The `ModelEvaluator` tests trained models by:

1. **Playing Full Games**: Uses the trained model to make all decisions
2. **Performance Metrics**: Tracks win rate, tricks won, decision accuracy
3. **Game Analysis**: Analyzes individual game outcomes
4. **Comparison**: Compares multiple models side-by-side

### Evaluation Metrics

- **Win Rate**: Percentage of games won
- **Average Tricks**: Average tricks won per game
- **Trump Calling Accuracy**: Accuracy of trump decisions
- **Card Selection Accuracy**: Accuracy of card playing decisions

## Advanced Features

### Custom Model Architectures

```python
# Create custom model
model = EuchreNeuralNetwork(
    input_size=256,      # Larger input
    hidden_size=512,     # Larger hidden layers
    num_layers=5,        # More layers
    dropout=0.4          # Higher dropout
)
```

### Data Augmentation

The training pipeline can generate synthetic training data:

```python
# Generate sample data
pipeline.generate_sample_data(num_games=10000)
```

### Model Comparison

```python
# Compare multiple models
comparison = evaluator.compare_models([
    "models/model_v1.pth",
    "models/model_v2.pth",
    "models/model_v3.pth"
], num_games=50)
```

## Performance Considerations

- **GPU Training**: Models automatically use CUDA if available
- **Batch Processing**: Configurable batch sizes for memory optimization
- **Data Loading**: Efficient data loading with PyTorch DataLoader
- **Model Checkpointing**: Automatic saving of best models

## File Structure

```
euchre/ai_model/
├── __init__.py              # Package initialization
├── euchre_nn.py            # Neural network model
├── game_state_encoder.py   # Game state encoding
├── training_pipeline.py    # Training pipeline
├── model_evaluator.py      # Model evaluation
└── README.md               # This file
```

## Dependencies

- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0 (for visualization)
- Seaborn >= 0.11.0 (for plotting)
- TQDM >= 4.64.0 (for progress bars)

## Examples

See the `train_euchre_ai.py` script for a complete training example, and the individual module docstrings for detailed API documentation. 