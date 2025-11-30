# ML Supervised Bot

## Overview

The ML Supervised Bot has been trained by watching thousands of games, learning patterns from real game data using supervised learning.

## Architecture

- **Supervised Learning**: Trained on collected game data
- **Model Types**: Supports Random Forest, Gradient Boosting, or Neural Networks
- **Feature Engineering**: Converts game state to numerical features

## Decision Making

- Looks at current game situation
- Converts everything into numerical features
- Feeds features into trained model
- Model predicts best decision based on learned patterns
- Uses risk factor to control confidence threshold

## Training

### Collecting Data
```bash
python scripts/data/collect_training_data.py --num_games 1000
```

### Training Models
```bash
python scripts/training/train_models.py --mode train --model_type random_forest
```

Available model types:
- `random_forest`: Random Forest Classifier
- `gradient_boosting`: Gradient Boosting Classifier
- `neural_network`: Multi-layer Perceptron

## Configuration

- `backend`: "supervised"
- `model_type`: "random_forest", "gradient_boosting", or "neural_network"
- `trump_selection_risk`: Risk factor for trump decisions (0.0-1.0)
- `gameplay_risk`: Risk factor for gameplay decisions (0.0-1.0)

## Usage

```python
from eucher.game import Game

player_config = [
    ("ML1", "ml_sklearn"),
    ("ML2", "ml_sklearn"),
    ("ML3", "ml_sklearn"),
    ("ML4", "ml_sklearn"),
]

game = Game(player_config)
```

## Strengths

- Learns from real game patterns
- Can discover strategies humans might not think of
- Adapts based on training data quality
- Multiple model types available

## Weaknesses

- Only as good as its training data
- Needs to be trained before it can play well
- May make mistakes if it encounters situations it hasn't seen before
- Can be slow if using complex models

## Best For

- Competitive play after proper training
- Learning from expert players
- Discovering new strategies

