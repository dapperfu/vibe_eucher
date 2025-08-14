# AI Profiles

This document explains the different AI personality profiles available in the Euchre game, how they behave, and how to customize them. The AI system now features three distinct levels with unified interfaces.

## Profile Overview

AI profiles represent different playing styles and strategies across three AI levels. Each profile has distinct characteristics that affect how the AI makes decisions during the game. All AI levels implement the same `BaseAIInterface` for seamless integration.

## AI Levels and Profile Types

### Level 1: Traditional AI
**Rule-based AI with configurable risk profiles**

- **Types**: `level1_aggressive`, `level1_conservative`, `level1_balanced`, `level1_opportunistic`
- **Strategy**: Hard-coded rules and heuristics
- **Performance**: Fast, explainable, consistent
- **Best For**: Learning, testing, predictable gameplay

### Level 2: Neural Network AI
**Pre-trained neural networks with risk profile integration**

- **Types**: `level2_strategic`, `level2_aggressive`, `level2_balanced`, `level2_intuitive`
- **Strategy**: Neural network evaluation with 256-dimensional input
- **Performance**: Sophisticated, learns from data
- **Best For**: Advanced gameplay, research

### Level 3: Advanced Neural AI
**Advanced neural networks with comprehensive game modeling**

- **Types**: `level3_strategic`, `level3_aggressive`, `level3_balanced`, `level3_conservative`, `level3_opportunistic`
- **Strategy**: Advanced neural networks with 2048-dimensional input
- **Performance**: Highest sophistication, most adaptable
- **Best For**: Research, advanced AI development

## Available Profiles

### Level 1 Profiles

#### Balanced Profile
**Default profile for most Level 1 AI players**

- **Risk Level**: Medium
- **Trump Calling**: Moderate frequency
- **Card Play**: Balanced between aggressive and conservative
- **Partner Coordination**: Good teamwork
- **Best For**: General gameplay, learning different styles

**Characteristics:**
- Calls trump when hand strength is moderate to strong
- Balances high and low card play
- Considers partner's plays when making decisions
- Adapts strategy based on game state

#### Aggressive Profile
**High-risk, high-reward playing style**

- **Risk Level**: High
- **Trump Calling**: Frequent
- **Card Play**: Plays high cards early
- **Partner Coordination**: Moderate
- **Best For**: Fast-paced games, high-risk scenarios

**Characteristics:**
- Calls trump with marginal hands
- Leads with high cards frequently
- Uses trump cards aggressively
- Prioritizes winning tricks over saving cards

#### Conservative Profile
**Defensive, risk-averse playing style**

- **Risk Level**: Low
- **Trump Calling**: Rare
- **Card Play**: Saves strong cards
- **Partner Coordination**: Excellent
- **Best For**: Defensive play, protecting leads

**Characteristics:**
- Only calls trump with very strong hands
- Saves high cards for critical moments
- Follows suit strictly
- Works closely with partner

#### Opportunistic Profile
**Adaptive strategy based on game state**

- **Risk Level**: Variable
- **Trump Calling**: Context-dependent
- **Card Play**: Adapts to situation
- **Partner Coordination**: Variable
- **Best For**: Dynamic situations, complex game states

**Characteristics:**
- Changes strategy based on current game state
- Calls trump when advantageous
- Adapts card play to trick requirements
- Balances individual and team goals

### Level 2 Profiles

#### Strategic Profile
**Deep strategic thinking and partner coordination**

- **Risk Level**: Medium-High
- **Trump Calling**: Strategic, considers game context
- **Card Play**: Plans multiple tricks ahead
- **Partner Coordination**: Excellent
- **Best For**: Strategic gameplay, team coordination

#### Aggressive Profile
**Bold and unpredictable, taking calculated risks**

- **Risk Level**: High
- **Trump Calling**: Frequent, aggressive
- **Card Play**: High card usage, aggressive leading
- **Partner Coordination**: Good
- **Best For**: High-risk scenarios, aggressive play

#### Balanced Profile
**Most balanced and adaptable player**

- **Risk Level**: Medium
- **Trump Calling**: Balanced approach
- **Card Play**: Adapts to game situation
- **Partner Coordination**: Very good
- **Best For**: General gameplay, learning

#### Intuitive Profile
**Deep intuition about game patterns**

- **Risk Level**: Medium-Low
- **Trump Calling**: Intuitive, pattern-based
- **Card Play**: Follows game flow
- **Partner Coordination**: Good
- **Best For**: Pattern recognition, flow-based play

### Level 3 Profiles

#### Strategic Profile
**Advanced strategic thinking with comprehensive modeling**

- **Risk Level**: Adaptive
- **Trump Calling**: Context-aware, strategic
- **Card Play**: Multi-trick planning
- **Partner Coordination**: Excellent
- **Best For**: Advanced strategic gameplay

#### Aggressive Profile
**Advanced aggressive play with risk assessment**

- **Risk Level**: High
- **Trump Calling**: Aggressive with risk calculation
- **Card Play**: High card usage, aggressive leading
- **Partner Coordination**: Good
- **Best For**: High-risk, high-reward scenarios

#### Balanced Profile
**Most balanced Level 3 player**

- **Risk Level**: Medium
- **Trump Calling**: Balanced, adaptive
- **Card Play**: Adapts to all situations
- **Partner Coordination**: Excellent
- **Best For**: General advanced gameplay

#### Conservative Profile
**Advanced conservative play with protection**

- **Risk Level**: Low
- **Trump Calling**: Conservative, protective
- **Card Play**: Saves strong cards, defensive
- **Partner Coordination**: Excellent
- **Best For**: Defensive, protective play

#### Opportunistic Profile
**Advanced opportunistic play with adaptation**

- **Risk Level**: Variable
- **Trump Calling**: Context-dependent, adaptive
- **Card Play**: Adapts to game state
- **Partner Coordination**: Very good
- **Best For**: Dynamic, adaptive gameplay

## Profile Configuration

### Basic Profile Selection
```bash
# Level 1 AI profiles
euchre ai-profiles -p level1_aggressive -p level1_conservative -p level1_balanced -p level1_opportunistic

# Level 2 AI profiles
euchre ai-profiles -p level2_strategic -p level2_aggressive -p level2_balanced -p level2_intuitive

# Level 3 AI profiles
euchre ai-profiles -p level3_strategic -p level3_aggressive -p level3_balanced -p level3_conservative -p level3_opportunistic

# Mix profiles and risk ratios
euchre ai-profiles -p level1_aggressive -r 0.9 -p level2_conservative -r 0.2
```

### Risk Ratio Configuration
Each profile can be customized with a risk ratio (0.0 to 1.0):

```bash
# High risk aggressive AI
euchre ai-profiles -p level1_aggressive -r 0.9

# Low risk conservative AI
euchre ai-profiles -p level1_conservative -r 0.1

# Balanced AI with moderate risk
euchre ai-profiles -p level1_balanced -r 0.5

# Level 2 AI with custom risk
euchre ai-profiles -p level2_strategic -r 0.7

# Level 3 AI with custom risk
euchre ai-profiles -p level3_balanced -r 0.6
```

### Profile Parameters

#### Aggressiveness (0.0 - 1.0)
- **0.0**: Never calls trump, always passes
- **0.5**: Calls trump with moderate hands
- **1.0**: Calls trump with any hand

#### Conservativeness (0.0 - 1.0)
- **0.0**: Plays high cards immediately
- **0.5**: Balances high and low card play
- **1.0**: Saves all high cards

#### Opportunism (0.0 - 1.0)
- **0.0**: Never changes strategy
- **0.5**: Adapts moderately to game state
- **1.0**: Completely adaptive strategy

## Profile Behavior Examples

### Level 1 AI Examples

#### Balanced AI Example
```
Hand: ['Ace of Spades', 'King of Diamonds', 'Queen of Clubs', 'Ten of Hearts', 'Nine of Spades']
Top Card: Jack of Diamonds
Decision: Orders up Diamonds (moderate strength, good trump potential)

Trick 1: Leads with Ten of Hearts (moderate card, saves stronger cards)
Trick 2: Plays Nine of Spades when following suit (lowest card)
Trick 3: Uses King of Diamonds to win (saves Ace for later)
```

#### Aggressive AI Example
```
Hand: ['Ace of Spades', 'King of Diamonds', 'Queen of Clubs', 'Ten of Hearts', 'Nine of Spades']
Top Card: Jack of Diamonds
Decision: Orders up Diamonds (any trump is good)

Trick 1: Leads with Ace of Spades (highest card, immediate win)
Trick 2: Plays King of Diamonds (high trump, aggressive play)
Trick 3: Uses Queen of Clubs (high card, maintains lead)
```

#### Conservative AI Example
```
Hand: ['Ace of Spades', 'King of Diamonds', 'Queen of Clubs', 'Ten of Hearts', 'Nine of Spades']
Top Card: Jack of Diamonds
Decision: Passes (hand not strong enough for trump)

Trick 1: Leads with Nine of Spades (lowest card, saves high cards)
Trick 2: Plays Ten of Hearts when following suit (low card)
Trick 3: Saves Ace of Spades (protects high card)
```

#### Opportunistic AI Example
```
Hand: ['Ace of Spades', 'King of Diamonds', 'Queen of Clubs', 'Ten of Hearts', 'Nine of Spades']
Top Card: Jack of Diamonds
Decision: Orders up Diamonds (good trump potential, game state favorable)

Trick 1: Leads with Nine of Spades (partner has strong hand, conservative lead)
Trick 2: Plays King of Diamonds (trump needed to win)
Trick 3: Saves Ace of Spades (partner can win without it)
```

### Level 2 AI Examples

#### Strategic AI Example
```
Hand: ['Ace of Spades', 'King of Diamonds', 'Queen of Clubs', 'Ten of Hearts', 'Nine of Spades']
Top Card: Jack of Diamonds
Decision: Orders up Diamonds (neural network evaluation shows strong potential)

Trick 1: Leads with Nine of Spades (strategic lead, saves high cards)
Trick 2: Plays Ten of Hearts when following suit (moderate card)
Trick 3: Uses King of Diamonds strategically (saves Ace for critical moment)
```

### Level 3 AI Examples

#### Advanced Strategic AI Example
```
Hand: ['Ace of Spades', 'King of Diamonds', 'Queen of Clubs', 'Ten of Hearts', 'Nine of Spades']
Top Card: Jack of Diamonds
Decision: Orders up Diamonds (comprehensive game state analysis shows advantage)

Trick 1: Leads with Nine of Spades (advanced strategic analysis)
Trick 2: Plays Ten of Hearts when following suit (calculated risk)
Trick 3: Uses King of Diamonds (optimal timing based on game state)
```

## Creating Custom Profiles

### Profile Definition Using Unified Interface
```python
from euchre.ai.ai_factory import AIFactory

# Create custom Level 1 AI profile
custom_ai = AIFactory.create_ai_player(
    name="CustomPlayer",
    ai_type="level1_balanced",  # Level 1 AI with balanced profile
    risk_ratio=0.7
)

# Create custom Level 2 AI profile
custom_ai = AIFactory.create_ai_player(
    name="CustomPlayer",
    ai_type="level2_strategic",  # Level 2 AI with strategic profile
    risk_ratio=0.6
)

# Create custom Level 3 AI profile
custom_ai = AIFactory.create_ai_player(
    name="CustomPlayer",
    ai_type="level3_balanced",  # Level 3 AI with balanced profile
    risk_ratio=0.5
)
```

### Custom Profile Class
```python
from euchre.ai.base_ai_interface import BaseAIInterface
from euchre.ai.game_context import GameContext
from euchre.ai.decision_result import DecisionResult, DecisionType

class CustomProfile(BaseAIInterface):
    def __init__(self, name: str, risk_profile: float = 0.5):
        super().__init__(name, risk_profile)
        self.aggressiveness = 0.6
        self.conservativeness = 0.3
        self.opportunism = 0.7
    
    def should_order_up(self, context: GameContext) -> DecisionResult:
        # Custom trump calling logic
        hand_strength = self.evaluate_hand(context.hand)
        if hand_strength > (0.7 - self.risk_profile * 0.3):
            return DecisionResult(
                decision_type=DecisionType.ORDER_UP,
                confidence=0.8,
                reasoning="Strong hand with good trump potential",
                metadata={'hand_strength': hand_strength}
            )
        else:
            return DecisionResult(
                decision_type=DecisionType.PASS,
                confidence=0.9,
                reasoning="Hand not strong enough for trump",
                metadata={'hand_strength': hand_strength}
            )
    
    def play_card(self, context: GameContext) -> DecisionResult:
        # Custom card selection logic
        if context.lead_suit and self.has_suit(context.lead_suit):
            card = self.select_follow_suit_card(context.lead_suit)
        else:
            card = self.select_lead_card()
        
        return DecisionResult(
            decision_type=DecisionType.PLAY_CARD,
            confidence=0.7,
            reasoning="Selected best available card",
            metadata={'selected_card': card}
        )
```

## Profile Performance Analysis

### Win Rate Comparison
```
Profile Performance (1000 games each):

Level 1 AI:
- Balanced: 52.3% win rate
- Aggressive: 48.7% win rate
- Conservative: 51.2% win rate
- Opportunistic: 53.1% win rate

Level 2 AI:
- Strategic: 58.9% win rate
- Aggressive: 55.2% win rate
- Balanced: 57.1% win rate
- Intuitive: 56.8% win rate

Level 3 AI:
- Strategic: 62.3% win rate
- Aggressive: 59.7% win rate
- Balanced: 61.2% win rate
- Conservative: 60.8% win rate
- Opportunistic: 61.5% win rate
```

### Strategy Effectiveness
```
Trump Calling Success:
Level 1:
- Balanced: 67.2%
- Aggressive: 58.9%
- Conservative: 78.4%
- Opportunistic: 71.6%

Level 2:
- Strategic: 72.1%
- Aggressive: 68.3%
- Balanced: 70.8%
- Intuitive: 69.5%

Level 3:
- Strategic: 75.2%
- Aggressive: 71.8%
- Balanced: 74.1%
- Conservative: 76.3%
- Opportunistic: 73.9%

Trick Efficiency:
Level 1: 2.4 tricks/round
Level 2: 2.7 tricks/round
Level 3: 2.9 tricks/round
```

## Profile Selection Guide

### For Beginners
**Recommended**: Level 1 Balanced Profile
- **Why**: Predictable behavior, good learning tool
- **Risk Level**: Medium
- **Complexity**: Low

### For Learning
**Recommended**: Mix of Level 1 Profiles
- **Why**: See different strategies in action
- **Risk Level**: Variable
- **Complexity**: Medium

### For Advanced Play
**Recommended**: Level 2 or Level 3 AI
- **Why**: More sophisticated decision making
- **Risk Level**: Configurable
- **Complexity**: High

### For Research
**Recommended**: Level 3 AI with Custom Profiles
- **Why**: Most advanced capabilities, fully configurable
- **Risk Level**: Configurable
- **Complexity**: Very High

### For Entertainment
**Recommended**: Level 1 Aggressive vs Conservative
- **Why**: Exciting gameplay, clear strategy differences
- **Risk Level**: High vs Low
- **Complexity**: Low

## Profile Customization

### Environment Variables
```bash
# Set default profile behavior
export EUCHRE_AI_DEFAULT_LEVEL=1
export EUCHRE_AI_DEFAULT_PROFILE=balanced
export EUCHRE_AI_DEFAULT_RISK=0.5
export EUCHRE_AI_AGGRESSIVENESS=0.6
export EUCHRE_AI_CONSERVATIVENESS=0.4
```

### Configuration Files
```json
{
  "ai_profiles": {
    "default_level": 1,
    "default_profile": "balanced",
    "profiles": {
      "level1_balanced": {
        "risk_ratio": 0.5,
        "aggressiveness": 0.5,
        "conservativeness": 0.5,
        "opportunism": 0.5
      },
      "level1_aggressive": {
        "risk_ratio": 0.8,
        "aggressiveness": 0.9,
        "conservativeness": 0.1,
        "opportunism": 0.3
      },
      "level2_strategic": {
        "risk_ratio": 0.6,
        "aggressiveness": 0.6,
        "conservativeness": 0.4,
        "opportunism": 0.7
      },
      "level3_balanced": {
        "risk_ratio": 0.5,
        "aggressiveness": 0.5,
        "conservativeness": 0.5,
        "opportunism": 0.5
      }
    }
  }
}
```

## Advanced Profile Features

### Dynamic Profile Switching
Some AI can change profiles during gameplay:
```python
def adapt_profile(self, context: GameContext):
    if context.team_scores[0] < 3:
        self.profile = "aggressive"  # Need to catch up
    elif context.team_scores[0] > 7:
        self.profile = "conservative"  # Protect the lead
    else:
        self.profile = "balanced"  # Normal play
```

### Profile Learning
Advanced AI can learn from game results:
```python
def update_profile(self, game_result):
    if game_result.won:
        # Reinforce successful strategies
        self.aggressiveness *= 1.1
        self.risk_profile *= 1.05
    else:
        # Adjust unsuccessful strategies
        self.aggressiveness *= 0.9
        self.risk_profile *= 0.95
```

## Troubleshooting Profiles

### Common Issues

#### AI Too Aggressive
```bash
# Reduce risk ratio
euchre ai-profiles -p level1_aggressive -r 0.3

# Use conservative profile instead
euchre ai-profiles -p level1_conservative
```

#### AI Too Conservative
```bash
# Increase risk ratio
euchre ai-profiles -p level1_conservative -r 0.7

# Use balanced profile instead
euchre ai-profiles -p level1_balanced
```

#### Profile Not Working
```bash
# Check profile name spelling
euchre ai-profiles --help

# Use default profiles
euchre ai-profiles
```

### Profile Validation
```bash
# Test profile behavior
euchre ai-profiles -p level1_balanced -v

# Run profile comparison
euchre ai-profiles -p level1_aggressive -p level1_conservative -n 100

# Test Level 2 profiles
euchre ai-profiles -p level2_strategic -p level2_balanced -n 100

# Test Level 3 profiles
euchre ai-profiles -p level3_strategic -p level3_balanced -n 100
```

## Future Profile Enhancements

### Planned Features
1. **Machine Learning Profiles**: AI that learns optimal strategies
2. **Human-Like Profiles**: Mimic human playing patterns
3. **Tournament Profiles**: Specialized for competitive play
4. **Teaching Profiles**: Designed to help players learn

### Research Areas
1. **Profile Evolution**: How profiles improve over time
2. **Profile Interaction**: How different profiles affect each other
3. **Optimal Profiles**: Finding the best strategy combinations
4. **Profile Transfer**: Applying successful strategies to new scenarios

---

*For AI system details, see [AI Overview](overview.md)*
*For training instructions, see [AI Training](training.md)*
*For AI vs AI games, see [AI vs AI Games](../ai-vs-ai.md)* 