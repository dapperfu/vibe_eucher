# AI vs AI Games

This guide explains how to run AI vs AI euchre games, tournaments, and analysis. Perfect for researchers, developers, and anyone interested in watching AI strategies compete.

## Quick Start

### Basic AI vs AI Game
```bash
# Run a single AI vs AI game
euchre ai-vs-ai

# With verbose logging
euchre --verbose ai-vs-ai

# With debug logging
euchre --very-verbose ai-vs-ai
```

### What You'll See
```
Starting AI-only Euchre game...
Dealer: Alice
Top card: Queen of Hearts
Initial hands:
Alice: ['Ace of Spades', 'Ten of Spades', 'Queen of Clubs', 'Ace of Diamonds', 'Ten of Diamonds']
Bob: ['King of Spades', 'Nine of Spades', 'Jack of Clubs', 'King of Diamonds', 'Nine of Diamonds']
Charlie: ['Queen of Spades', 'Ace of Clubs', 'Ten of Clubs', 'Queen of Diamonds', 'Ace of Hearts']
David: ['Jack of Spades', 'King of Clubs', 'Nine of Clubs', 'Jack of Diamonds', 'King of Hearts']

=== Starting Round 1 ===
Alice called DIAMONDS as trump!

--- Trick 1 ---
First trick - starting with Bob
Bob plays Nine of Spades
Trick so far: Bob: Nine of Spades
Charlie plays Ten of Clubs
Trick so far: Bob: Nine of Spades, Charlie: Ten of Clubs
David plays Nine of Clubs
Trick so far: Bob: Nine of Spades, Charlie: Ten of Clubs, David: Nine of Clubs
Alice plays Ten of Spades
Trick so far: Bob: Nine of Spades, Charlie: Ten of Clubs, David: Nine of Clubs, Alice: Ten of Spades
Trick won by Alice!
```

## AI Profile Tournaments

### Different AI Personalities
```bash
# Run game with different AI profiles
euchre ai-profiles -p aggressive -p conservative -p balanced -p opportunistic

# Custom risk ratios
euchre ai-profiles -r 0.8 -r 0.3 -r 0.5 -r 0.7

# Mix profiles and risk ratios
euchre ai-profiles -p aggressive -r 0.9 -p conservative -r 0.2
```

### AI Profile Types

#### Aggressive AI
- **Characteristics**: Calls trump frequently, plays high cards early
- **Risk Level**: High
- **Best For**: Fast-paced games, high-risk scenarios

#### Conservative AI
- **Characteristics**: Passes on marginal hands, saves strong cards
- **Risk Level**: Low
- **Best For**: Defensive play, protecting leads

#### Balanced AI
- **Characteristics**: Mixes aggressive and conservative strategies
- **Risk Level**: Medium
- **Best For**: General gameplay, learning different styles

#### Opportunistic AI
- **Characteristics**: Adapts strategy based on game state
- **Risk Level**: Variable
- **Best For**: Dynamic situations, complex game states

## Tournament Mode

### Basic Tournaments
```bash
# 100-game tournament
euchre tournament --num-games 100

# Short form
euchre tournament -n 250

# Default tournament (100 games)
euchre tournament
```

### Tournament Results
Tournaments provide comprehensive statistics:
```
=== TOURNAMENT RESULTS ===
Total Games: 100
Team 1 Wins: 52 (52.0%)
Team 2 Wins: 48 (48.0%)

Average Game Length: 3.2 rounds
Longest Game: 7 rounds
Shortest Game: 1 round

Top Performers:
- Alice: 28 wins, 72% win rate
- Bob: 24 wins, 48% win rate
- Charlie: 24 wins, 48% win rate
- David: 24 wins, 48% win rate
```

## Neural Network Tournaments

### Trained AI Models
```bash
# Basic neural tournament
euchre run-neural-games

# Custom models
euchre run-neural-games -m1 Alice -m2 Bob

# Custom number of games
euchre run-neural-games -n 500

# Combined options
euchre run-neural-games -m1 Sherlock -m2 Watson -n 2000
```

### Model Types
- **Integer Models**: Fast, lightweight AI
- **Float Models**: More precise, computationally intensive
- **Hybrid Models**: Balance of speed and accuracy

## Mass Game Analysis

### Running Thousands of Games
```bash
# Default mass games (10,000)
euchre run-mass-games

# Custom number of games
euchre run-mass-games --num-games 25000

# Short form
euchre run-mass-games -n 50000
```

### Analysis Output
Mass games generate comprehensive statistics:
```
=== MASS GAME ANALYSIS ===
Games Completed: 25,000
Processing Time: 45.2 seconds
Games per Second: 553.1

Win Distribution:
- Team 1: 12,487 wins (49.9%)
- Team 2: 12,513 wins (50.1%)

Performance Metrics:
- Average Game Length: 3.1 rounds
- Trump Calling Success: 67.3%
- Euchre Rate: 12.8%
- Perfect Game Rate: 3.2%
```

## Advanced AI vs AI Features

### Logged Games (No Interface)
```bash
# Run without ncurses interface
euchre logged-game

# With verbose logging
euchre --verbose logged-game
```

### Ncurses Interface
```bash
# Interactive ncurses interface
euchre ncurses
```

## AI Training and Evolution

### Self-Play Training
```bash
# Train AI using self-play
euchre train-self-play --num-games 100000 --players Alice Bob Charlie David

# Custom training
euchre train-self-play -n 50000 -p Sherlock -p Watson -p Moriarty -p Irene
```

### Training Process
1. **Game Generation**: AI plays thousands of games against itself
2. **Performance Analysis**: Track win rates, trick counts, strategies
3. **Strategy Refinement**: Adjust decision-making parameters
4. **Validation**: Test against different opponents
5. **Iteration**: Repeat process to improve performance

### Training Results
```
=== TRAINING RESULTS ===
Training Games: 100,000
Training Time: 2 hours 15 minutes

Performance Improvement:
- Initial Win Rate: 48.2%
- Final Win Rate: 67.8%
- Improvement: +19.6%

Strategy Evolution:
- Trump Calling: More conservative
- Card Play: Better suit management
- Partner Coordination: Improved teamwork
```

## Analysis and Insights

### What to Look For

#### Strategy Patterns
- **Trump Selection**: When and why AI calls trump
- **Card Play**: How AI manages their hand
- **Partner Coordination**: Teamwork effectiveness
- **Risk Management**: Aggressive vs. conservative play

#### Performance Metrics
- **Win Rates**: Overall success of different strategies
- **Trick Efficiency**: Average tricks won per round
- **Trump Success**: Percentage of successful trump calls
- **Game Length**: How quickly games are won

#### Learning Progress
- **Improvement Over Time**: How AI strategies evolve
- **Strategy Adaptation**: How AI responds to different opponents
- **Error Reduction**: Fewer mistakes over time

### Common Analysis Questions

1. **Which AI profile performs best?**
   - Run tournaments with different profiles
   - Compare win rates and strategies

2. **How do AI strategies differ?**
   - Watch games with verbose logging
   - Analyze decision-making patterns

3. **What makes a successful AI?**
   - Study high-performing models
   - Identify common strategies

4. **How do AI players learn?**
   - Run training sessions
   - Track performance improvements

## Customizing AI vs AI Games

### Environment Variables
```bash
# Set AI behavior parameters
export EUCHRE_AI_AGGRESSIVENESS=0.7
export EUCHRE_AI_RISK_TOLERANCE=0.5
export EUCHRE_AI_LEARNING_RATE=0.1

# Run game with custom settings
euchre ai-vs-ai
```

### Configuration Files
Create custom AI profiles:
```json
{
  "name": "CustomAI",
  "profile": "balanced",
  "risk_ratio": 0.6,
  "aggressiveness": 0.7,
  "conservativeness": 0.3,
  "opportunism": 0.8
}
```

## Performance Considerations

### Hardware Requirements
- **CPU**: Multi-core processor recommended for mass games
- **Memory**: 4GB+ RAM for large tournaments
- **Storage**: SSD recommended for fast game logging

### Optimization Tips
```bash
# Run games in parallel
euchre run-mass-games -n 100000 --parallel 8

# Use GPU acceleration (if available)
euchre run-neural-games --device cuda

# Optimize memory usage
euchre run-mass-games --memory-efficient
```

## Troubleshooting

### Common Issues

#### Games Run Slowly
- Reduce number of games
- Use fewer AI profiles
- Check system resources

#### Memory Issues
- Use memory-efficient mode
- Reduce batch sizes
- Close other applications

#### AI Behavior Issues
- Check AI profile settings
- Verify training data
- Reset AI to default state

### Getting Help
```bash
# Check command help
euchre ai-vs-ai --help

# Run with verbose logging
euchre --very-verbose ai-vs-ai

# Check system resources
top
free -h
```

## Research Applications

### Academic Research
- **Game Theory**: Study strategic decision making
- **Machine Learning**: AI training and evolution
- **Psychology**: Human vs. AI behavior comparison

### Industry Applications
- **Gaming**: AI opponent development
- **Education**: Strategy learning tools
- **Testing**: Game rule validation

### Data Collection
```bash
# Collect game data for analysis
euchre run-mass-games --save-data --output-format json

# Analyze specific aspects
euchre analyze-games --focus trump-selection
euchre analyze-games --focus card-play
euchre analyze-games --focus teamwork
```

## Next Steps

### For Beginners
1. **Start Simple**: Run basic AI vs AI games
2. **Watch Games**: Use verbose logging to understand strategies
3. **Try Profiles**: Experiment with different AI personalities

### For Researchers
1. **Run Experiments**: Use mass game runner for large-scale analysis
2. **Train Models**: Use self-play training to improve AI
3. **Analyze Results**: Use analysis tools to study patterns

### For Developers
1. **Study Code**: Review AI implementation
2. **Create Profiles**: Build custom AI personalities
3. **Contribute**: Improve AI algorithms

---

*For AI system details, see [AI Overview](ai/overview.md)*
*For training instructions, see [AI Training](ai/training.md)*
*For analysis tools, see [Game Analysis](tools/analysis.md)* 