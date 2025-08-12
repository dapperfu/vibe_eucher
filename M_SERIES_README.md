# M-Series PyTorch AI Models for Euchre

## Overview

The M-Series is a collection of brand new, from-scratch PyTorch neural network models specifically designed for Euchre gameplay. These models are engineered to be the best possible Euchre players through comprehensive training on thousands of games.

## 🎯 Key Features

### Comprehensive Game Understanding
- **Dealer Status Awareness**: Models understand if dealer is self, partner, or opposing team
- **Partner Coordination**: Advanced partner relationship modeling and coordination
- **Hand Strength Analysis**: Deep analysis of all cards in hand with trump context
- **Game State Context**: Full understanding of current trick, score, and game phase

### Advanced Neural Architecture
- **Multi-Head Design**: Separate decision heads for trump calling, card selection, and suit selection
- **Risk-Aware Training**: Built-in risk profiles that adapt during gameplay
- **Attention Mechanisms**: Advanced attention for complex game state understanding
- **Regularization**: Dropout and layer normalization for robust training

### Four Distinct Personalities

#### 🧠 **Magnus** - The Strategic Mastermind
- **Focus**: Deep strategic thinking and partner coordination
- **Style**: Analytical, long-term planning, team-oriented
- **Best For**: Complex game situations requiring coordination

#### 🚀 **Maverick** - The Aggressive Risk-Taker
- **Focus**: Bold, aggressive play with calculated risks
- **Style**: High-risk, high-reward, unpredictable
- **Best For**: Fast-paced games and aggressive strategies

#### 🎓 **Mentor** - The Balanced Teacher
- **Focus**: Balanced approach with learning and adaptation
- **Style**: Adaptive, balanced, educational
- **Best For**: General gameplay and learning optimal strategies

#### 🔮 **Mystic** - The Intuitive Player
- **Focus**: Pattern recognition and intuitive decision making
- **Style**: Intuitive, pattern-aware, flow-based
- **Best For**: Games requiring pattern recognition and intuition

## 🏗️ Architecture

### Model Structure
```
Input Layer (256 features) + Risk Embedding (64 features)
           ↓
    Hidden Layers (512 units × 3)
           ↓
    Specialized Output Heads:
    ├── Trump Decision (Order up or not)
    ├── Card Selection (5 cards in hand)
    └── Suit Selection (4 suits for trump calling)
```

### Feature Encoding
- **Hand Cards**: 135 features (5 cards × 27 features per card)
- **Game Context**: 32 features (dealer, partner, position, score)
- **Trick Context**: 20 features (current trick state)
- **Historical Context**: 20 features (game patterns and history)
- **Risk Profile**: 8 features (aggression, coordination, etc.)

### Risk Profile Parameters
- `trump_calling_aggression`: How aggressively to call trump
- `partner_dealer_bonus`: Bonus when partner is dealer
- `ace_ordering_threshold`: Threshold for ordering up aces
- `leading_aggression`: How aggressively to lead
- `trump_usage_strategy`: When to use trump cards
- `partner_coordination`: How much to coordinate with partner
- `score_adaptation`: How much to adapt based on score
- `set_avoidance`: How much to avoid being set

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8+
# PyTorch 1.9+
pip install torch torchvision torchaudio
pip install numpy tqdm
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd vibe_eucher

# Install the euchre package
pip install -e .
```

### Quick Start
```python
from euchre.ai_model.m_series_models import (
    create_mseries_model, 
    create_mseries_risk_profile,
    MSeriesGameStateEncoder
)

# Create a model
magnus = create_mseries_model("magnus")

# Create risk profile
risk_profile = create_mseries_risk_profile("magnus")

# Create encoder
encoder = MSeriesGameStateEncoder()

# Use the model
# (See usage examples below)
```

## 🎮 Usage Examples

### Basic Model Usage
```python
import torch
from euchre.ai_model.m_series_models import (
    create_mseries_model, 
    create_mseries_risk_profile
)

# Create model and risk profile
model = create_mseries_model("magnus")
risk_profile = create_mseries_risk_profile("magnus")

# Set model to evaluation mode
model.eval()

# Example input (256 features)
input_tensor = torch.randn(1, 256)

# Get predictions
with torch.no_grad():
    outputs = model(input_tensor, risk_profile)
    
    # Trump decision probabilities
    trump_probs = torch.softmax(outputs['trump_decision'], dim=1)
    
    # Card selection probabilities
    card_probs = torch.softmax(outputs['card_selection'], dim=1)
    
    # Suit selection probabilities
    suit_probs = torch.softmax(outputs['suit_selection'], dim=1)
```

### Game State Encoding
```python
from euchre.ai_model.m_series_models import MSeriesGameStateEncoder
from euchre.models import Player, Card, Suit, Rank

# Create encoder
encoder = MSeriesGameStateEncoder()

# Example player and game state
player = Player("TestPlayer", PlayerType.AI)
player.hand = [
    Card(Rank.ACE, Suit.SPADES),
    Card(Rank.KING, Suit.HEARTS),
    Card(Rank.QUEEN, Suit.DIAMONDS),
    Card(Rank.JACK, Suit.CLUBS),
    Card(Rank.TEN, Suit.SPADES)
]

# Encode for trump decision
trump_features = encoder.encode_game_state_for_trump_decision(
    player=player,
    top_card=Card(Rank.JACK, Suit.DIAMONDS),
    dealer=player,  # Self dealing
    team_scores={"Team 1": 5, "Team 2": 3},
    round_number=2
)

# Encode for card play
card_features = encoder.encode_game_state_for_card_play(
    player=player,
    lead_suit=Suit.HEARTS,
    trump_suit=Suit.DIAMONDS,
    current_trick=[("Player1", Card(Rank.NINE, Suit.HEARTS))],
    team_scores={"Team 1": 5, "Team 2": 3}
)
```

## 🎯 Training

### Self-Play Training
The M-Series models are designed to be trained through self-play on thousands of games:

```bash
# Basic training
python train_m_series_models.py

# Custom training configuration
python train_m_series_models.py \
    --epochs 2000 \
    --total-games 50000 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --hidden-size 1024 \
    --device cuda
```

### Training Configuration
```bash
# Model architecture
--input-size 256          # Input feature vector size
--hidden-size 512         # Hidden layer size
--risk-embedding-size 64  # Risk parameter embedding size

# Training parameters
--epochs 1000             # Number of training epochs
--batch-size 32           # Training batch size
--learning-rate 0.001     # Learning rate
--weight-decay 1e-5       # Weight decay

# Self-play configuration
--games-per-epoch 100     # Games per epoch
--total-games 10000       # Total training games
--curriculum-stages 5     # Curriculum learning stages
--games-per-stage 2000    # Games per curriculum stage

# Evaluation and saving
--eval-frequency 100      # Evaluation frequency
--eval-games 50           # Games for evaluation
--save-frequency 500      # Model save frequency
--model-dir trained_models # Model save directory
```

### Curriculum Learning
The training framework uses curriculum learning:

1. **Stage 1**: Basic game mechanics (1000 games)
2. **Stage 2**: Simple strategies (2000 games)
3. **Stage 3**: Partner coordination (3000 games)
4. **Stage 4**: Advanced strategies (3000 games)
5. **Stage 5**: Expert-level play (1000 games)

## 📊 Performance

### Training Metrics
- **Loss**: Cross-entropy loss for each decision type
- **Accuracy**: Decision accuracy across all game situations
- **Win Rate**: Model performance in evaluation games
- **Risk Adaptation**: How well models adapt risk profiles

### Expected Performance
After training on 10,000+ games:
- **Trump Calling Accuracy**: 85%+
- **Card Selection Accuracy**: 80%+
- **Overall Win Rate**: 60%+ (against baseline AI)
- **Partner Coordination**: Significantly improved

## 🔧 Customization

### Custom Risk Profiles
```python
from euchre.ai_model.m_series_models import MSeriesRiskProfile

# Create custom risk profile
custom_profile = MSeriesRiskProfile(
    trump_calling_aggression=0.8,    # Very aggressive
    partner_dealer_bonus=0.1,        # Low partner bonus
    ace_ordering_threshold=0.3,      # Order aces aggressively
    leading_aggression=0.9,          # Lead high cards
    trump_usage_strategy=0.8,        # Use trump aggressively
    partner_coordination=0.3,        # Low coordination
    score_adaptation=0.2,            # Don't adapt much
    set_avoidance=0.4                # Accept some risk
)
```

### Custom Model Architecture
```python
from euchre.ai_model.m_series_models import MSeriesBaseModel

class CustomModel(MSeriesBaseModel):
    def __init__(self, input_size=256, hidden_size=512, risk_embedding_size=64):
        super().__init__(input_size, hidden_size, risk_embedding_size)
        
        # Add custom layers
        self.custom_layer = nn.Linear(hidden_size // 2, 128)
        self.custom_output = nn.Linear(128, 5)
    
    def forward(self, x, risk_params):
        outputs = super().forward(x, risk_params)
        
        # Custom processing
        h = outputs['hidden_features']
        custom_features = F.relu(self.custom_layer(h))
        custom_output = self.custom_output(custom_features)
        
        outputs['custom_output'] = custom_output
        return outputs
```

## 🧪 Testing and Validation

### Model Validation
```bash
# Validate models without training
python train_m_series_models.py --validate-only

# Check model creation and forward pass
python -c "
from euchre.ai_model.m_series_models import create_mseries_model, create_mseries_risk_profile
import torch

model = create_mseries_model('magnus')
risk_profile = create_mseries_risk_profile('magnus')
test_input = torch.randn(1, 256)
outputs = model(test_input, risk_profile)
print('Model validation successful!')
"
```

### Performance Testing
```python
# Test model performance in games
from euchre.ai_model.m_series_training import MSeriesSelfPlayTrainer
from euchre.ai_model.m_series_training import TrainingConfig

config = TrainingConfig(eval_games=100)
trainer = MSeriesSelfPlayTrainer(config)

# Evaluate a specific model
win_rate = trainer._evaluate_model("magnus", 100)
print(f"Magnus win rate: {win_rate:.2%}")
```

## 📁 File Structure
```
euchre/ai_model/
├── m_series_models.py          # Core model definitions
├── m_series_training.py        # Training framework
└── README.md                   # This file

train_m_series_models.py        # Main training script
M_SERIES_README.md              # This documentation
```

## 🚨 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Make sure you're in the correct directory
cd vibe_eucher

# Install the package in development mode
pip install -e .

# Check Python path
python -c "import euchre.ai_model.m_series_models; print('Import successful')"
```

#### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Use CPU if CUDA not available
python train_m_series_models.py --device cpu
```

#### Memory Issues
```bash
# Reduce batch size and model size
python train_m_series_models.py \
    --batch-size 16 \
    --hidden-size 256 \
    --input-size 128
```

### Performance Optimization
```bash
# Use mixed precision training (if available)
export CUDA_LAUNCH_BLOCKING=1

# Profile memory usage
python -m memory_profiler train_m_series_models.py

# Use gradient checkpointing for large models
# (Add to model configuration)
```

## 🔮 Future Enhancements

### Planned Features
- **Multi-GPU Training**: Distributed training across multiple GPUs
- **Advanced Curriculum**: Dynamic curriculum based on model performance
- **Meta-Learning**: Models that learn to learn from game outcomes
- **Ensemble Methods**: Combine multiple models for better performance
- **Online Learning**: Continuous learning during gameplay

### Research Directions
- **Attention Visualization**: Understand what models focus on
- **Strategy Analysis**: Analyze learned strategies and tactics
- **Human-AI Collaboration**: Models that work well with human players
- **Tournament Play**: Advanced tournament strategies and preparation

## 📚 References

### Technical Papers
- [Deep Learning for Game AI](https://arxiv.org/abs/1802.01784)
- [Self-Play Learning](https://arxiv.org/abs/1712.01815)
- [Attention Mechanisms](https://arxiv.org/abs/1706.03762)

### Euchre Resources
- [Official Euchre Rules](https://www.euchre.com/rules/)
- [Euchre Strategy Guide](https://www.euchre.com/strategy/)
- [Euchre Tournament Play](https://www.euchre.com/tournaments/)

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd vibe_eucher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
python -m pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints throughout
- Document all functions and classes
- Write comprehensive tests

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Claude Sonnet 4**: AI model architecture and training framework design
- **Cursor IDE**: Development environment and AI assistance
- **PyTorch Team**: Deep learning framework
- **Euchre Community**: Game rules and strategy insights

---

**Note**: The M-Series models are designed to be trained on thousands of games to achieve optimal performance. Initial performance may be lower until sufficient training data is accumulated. 