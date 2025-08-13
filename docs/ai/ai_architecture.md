# Euchre AI Architecture Documentation

This document provides a comprehensive overview of the two different types of AI implementations in the Euchre system: **Rule-Based AI Profiles** and **Neural Network AI Models**.

## Overview

The Euchre system implements two distinct approaches to artificial intelligence:

1. **Rule-Based AI Profiles** - Traditional rule-based systems with configurable risk parameters
2. **Neural Network AI Models** - Machine learning models trained on game data, including the new **M-Series AI**

## 1. Rule-Based AI Profiles

### Architecture Overview

The rule-based AI system is built on a hierarchical class structure that provides consistent behavior patterns while allowing for personality customization through risk parameters.

```
BaseAI (Abstract Base Class)
├── AggressiveAI
├── ConservativeAI  
├── BalancedAI
└── OpportunisticAI
```

### Core Components

#### BaseAI Class
- **Location**: `euchre/ai/base_ai.py`
- **Purpose**: Provides common functionality and abstract methods for all AI players
- **Key Features**:
  - Risk ratio management (0.0 = conservative, 1.0 = aggressive)
  - Common card evaluation logic
  - Trump suit analysis
  - Left bower detection

#### AI Profiles

| Profile | Risk Ratio | Description | Strategy |
|---------|------------|-------------|----------|
| **AggressiveAI** | 0.6-1.0 | High risk tolerance, leads with high cards | Orders up aggressively, plays high cards, takes calculated risks |
| **ConservativeAI** | 0.0-0.4 | Low risk tolerance, plays safe | Only orders up with strong hands, plays conservatively, avoids being set |
| **BalancedAI** | 0.3-0.7 | Moderate risk tolerance, adaptive play | Balanced approach, adapts to game situation |
| **OpportunisticAI** | 0.4-0.8 | Seizes opportunities, calculated risks | Orders up when advantageous, plays situationally |

### Decision Making Process

#### Trump Calling (Order Up)
```python
def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
    hand_strength = self._evaluate_hand_for_trump(top_card.suit, top_card, is_partner_dealing)
    base_threshold = 15.0 - (self.risk_ratio * 7.0)
    return hand_strength >= base_threshold
```

#### Card Selection
```python
def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
    if not lead_suit:  # Leading
        return max(self.hand, key=lambda c: self._card_value(c, trump_suit))
    elif self.has_suit(lead_suit):  # Following suit
        cards_of_suit = self.get_cards_of_suit(lead_suit)
        return max(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
    else:  # Cannot follow suit
        return self._choose_off_suit_card(trump_suit)
```

### Risk Parameter System

The risk ratio system controls multiple aspects of play:

- **Trump Calling Threshold**: Lower risk = higher threshold required
- **Card Play Aggression**: Higher risk = more aggressive card selection
- **Set Avoidance**: Lower risk = more conservative play to avoid being set

## 2. Neural Network AI Models

### Architecture Overview

The neural network AI system uses PyTorch-based deep learning models trained on thousands of games to make optimal decisions. This system includes both the original **EuchreNN** and the new **M-Series AI** models.

```
Neural Network Models
├── EuchreNN (Original)
└── M-Series AI
    ├── MagnusModel (Strategic Mastermind)
    ├── MaverickModel (Aggressive Risk-Taker)
    ├── MentorModel (Balanced Teacher)
    └── MysticModel (Intuitive Player)
```

### Core Components

#### EuchreNN (Original Model)
- **Location**: `euchre/ai_model/euchre_nn.py`
- **Architecture**: Multi-head neural network with risk-aware decision making
- **Input Size**: 128 features
- **Hidden Layers**: Configurable (default: 3 layers of 256 units)
- **Output Heads**: 
  - Trump ordering (binary classification)
  - Card selection (5-class classification)
  - Trump selection (4-class classification)

#### M-Series AI Models

The M-Series represents a significant advancement in AI architecture, designed specifically for Euchre with deep understanding of game mechanics.

##### MSeriesBaseModel
- **Location**: `euchre/ai_model/m_series_models.py`
- **Input Size**: 256 features (expandable)
- **Hidden Size**: 512 units (configurable)
- **Architecture**: Advanced neural network with specialized layers

**Key Features:**
- **Risk Profile Integration**: 8-dimensional risk parameter system
- **Specialized Layers**: Partner intuition, card memory, strategic planning, trick analysis
- **Attention Mechanisms**: Multi-head attention for risk parameters
- **Advanced Encoding**: Comprehensive game state understanding

##### M-Series Model Variants

| Model | Personality | Specialization | Key Features |
|-------|-------------|----------------|--------------|
| **Magnus** | Strategic Mastermind | Deep strategic planning, partner coordination | Long-term strategy, partner intuition, game flow analysis |
| **Maverick** | Aggressive Risk-Taker | Bold play, calculated risks | Risk assessment, aggressive strategies, opportunistic play |
| **Mentor** | Balanced Teacher | Learning and adaptation | Balanced approach, learning from games, optimal strategy teaching |
| **Mystic** | Intuitive Player | Pattern recognition, subtle cues | Game pattern intuition, subtle signal detection, flow-based decisions |

### Risk Profile System

The M-Series uses an advanced 8-dimensional risk profile:

```python
@dataclass
class MSeriesRiskProfile:
    trump_calling_aggression: float = 0.5    # Trump calling aggressiveness
    partner_dealer_bonus: float = 0.2        # Bonus when partner is dealer
    ace_ordering_threshold: float = 0.6      # Threshold for ordering up aces
    leading_aggression: float = 0.5          # How aggressively to lead
    trump_usage_strategy: float = 0.5        # When to use trump cards
    partner_coordination: float = 0.7        # Partner coordination level
    score_adaptation: float = 0.6            # Score-based adaptation
    set_avoidance: float = 0.8               # Set avoidance priority
```

### Game State Encoding

The M-Series uses a sophisticated 256-dimensional encoding system:

#### Feature Breakdown
- **Hand Encoding**: 135 features (5 cards × 27 features per card)
- **Game Context**: 32 features (dealer, partner, position, score, etc.)
- **Trick Context**: 20 features (current trick state)
- **Historical Features**: 20 features (game history and patterns)
- **Strategic Features**: 49 features (advanced strategy and coordination)

#### Card Encoding (27 features per card)
- **Suit**: One-hot encoding (4 features)
- **Rank**: One-hot encoding (6 features)
- **Trump Status**: Trump indicator (1 feature)
- **Card Strength**: Normalized rank value (1 feature)
- **Bower Status**: Right/Left bower indicators (2 features)
- **Context Features**: 13 additional contextual features

### Decision Making Process

#### Forward Pass Architecture
```python
def forward(self, x: torch.Tensor, risk_params: MSeriesRiskProfile) -> Dict[str, torch.Tensor]:
    # Risk parameter embedding
    risk_embedding = self.risk_embedding(risk_params.get_risk_vector())
    
    # Input processing with risk integration
    x_with_risk = torch.cat([x, risk_embedding], dim=1)
    
    # Specialized feature processing
    partner_features = self.partner_intuition_layer(h)
    card_memory_features = self.card_memory_layer(h)
    strategic_features = self.strategic_planning_layer(h)
    trick_features = self.trick_analysis_layer(h)
    
    # Feature fusion and output generation
    combined_features = torch.cat([partner_features, card_memory_features, 
                                 strategic_features, trick_features], dim=1)
    
    return {
        'trump_decision': self.trump_decision_head(h),
        'card_selection': self.card_selection_head(h),
        'suit_selection': self.suit_selection_head(h)
    }
```

## 3. Comparison Table

| Aspect | Rule-Based AI | Neural Network AI | M-Series AI |
|--------|---------------|-------------------|-------------|
| **Complexity** | Low | Medium | High |
| **Training Required** | No | Yes | Yes |
| **Decision Speed** | Fast | Medium | Medium |
| **Adaptability** | Limited | High | Very High |
| **Memory Usage** | Minimal | Medium | High |
| **Explainability** | High | Low | Medium |
| **Customization** | Risk parameters | Model weights | Risk profiles + weights |
| **Performance** | Consistent | Variable | High (when trained) |

## 4. Usage Examples

### Creating Rule-Based AI
```python
from euchre.ai.ai_factory import AIFactory

# Create aggressive AI
aggressive_player = AIFactory.create_ai_player("Alice", "aggressive", risk_ratio=0.8)

# Create conservative AI
conservative_player = AIFactory.create_ai_player("Bob", "conservative", risk_ratio=0.2)
```

### Creating M-Series AI
```python
from euchre.ai_model.m_series_models import MagnusModel, MSeriesRiskProfile

# Create Magnus model
magnus = MagnusModel(input_size=256, hidden_size=512)

# Configure risk profile
risk_profile = MSeriesRiskProfile(
    trump_calling_aggression=0.7,
    partner_coordination=0.9,
    set_avoidance=0.8
)

# Make decisions
trump_probs = magnus.get_trump_decision_probs(game_state, risk_profile)
card_probs = magnus.get_card_selection_probs(game_state, risk_profile)
```

## 5. Training and Deployment

### Training Pipeline
- **Data Collection**: Games played by rule-based AI or human players
- **Feature Extraction**: Game states converted to feature vectors
- **Model Training**: PyTorch training with validation
- **Model Evaluation**: Performance testing on held-out data

### Deployment
- **Model Loading**: Trained models loaded at runtime
- **Inference**: Real-time decision making during games
- **Performance Monitoring**: Continuous evaluation and improvement

## 6. Future Development

The M-Series AI represents the cutting edge of Euchre AI development, with plans for:

- **Continuous Learning**: Models that improve through self-play
- **Advanced Coordination**: Better partner coordination algorithms
- **Meta-Learning**: Adaptation to different playing styles
- **Ensemble Methods**: Combining multiple models for optimal performance

This architecture provides a robust foundation for both immediate gameplay needs (rule-based AI) and advanced AI research (neural network models), ensuring the system can serve both casual players and serious AI researchers. 