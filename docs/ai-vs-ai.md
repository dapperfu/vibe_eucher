# AI vs AI Games

This guide explains how to run AI vs AI euchre games, tournaments, and analysis using the new unified AI interface. Perfect for researchers, developers, and anyone interested in watching AI strategies compete across three distinct AI levels.

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

### Different AI Levels and Profiles
```bash
# Level 1 AI profiles (Traditional rule-based)
euchre ai-profiles -p level1_aggressive -p level1_conservative -p level1_balanced -p level1_opportunistic

# Level 2 AI profiles (Neural network)
euchre ai-profiles -p level2_strategic -p level2_aggressive -p level2_balanced -p level2_intuitive

# Level 3 AI profiles (Advanced neural network)
euchre ai-profiles -p level3_strategic -p level3_aggressive -p level3_balanced -p level3_conservative -p level3_opportunistic

# Mix AI levels
euchre ai-profiles -p level1_aggressive -p level2_strategic -p level3_balanced -p level1_conservative

# Custom risk ratios
euchre ai-profiles -r 0.8 -r 0.3 -r 0.5 -r 0.7

# Mix profiles and risk ratios
euchre ai-profiles -p level1_aggressive -r 0.9 -p level2_conservative -r 0.2
```

### AI Level Types

#### Level 1: Traditional AI
- **Types**: `level1_aggressive`, `level1_conservative`, `level1_balanced`, `level1_opportunistic`
- **Strategy**: Rule-based decision making with configurable risk profiles
- **Performance**: Fast, explainable, consistent behavior
- **Best For**: Learning, testing, and predictable gameplay

#### Level 2: Neural Network AI
- **Types**: `level2_strategic`, `level2_aggressive`, `level2_balanced`, `level2_intuitive`
- **Strategy**: Pre-trained neural networks with 256-dimensional input
- **Performance**: Sophisticated decision making, learns from data
- **Best For**: Advanced gameplay and research

#### Level 3: Advanced Neural AI
- **Types**: `level3_strategic`, `level3_aggressive`, `level3_balanced`, `level3_conservative`, `level3_opportunistic`
- **Strategy**: Advanced neural networks with 2048-dimensional input
- **Performance**: Highest level of sophistication and adaptability
- **Best For**: Research, advanced AI development, and cutting-edge gameplay

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

### Level-Specific Tournaments
```bash
# Level 1 AI tournament
euchre tournament --ai-types level1_aggressive level1_conservative level1_balanced level1_opportunistic

# Level 2 AI tournament
euchre tournament --ai-types level2_strategic level2_aggressive level2_balanced level2_intuitive

# Level 3 AI tournament
euchre tournament --ai-types level3_strategic level3_aggressive level3_balanced level3_conservative level3_opportunistic
```

### Cross-Level Tournaments
```bash
# Tournament between different AI levels
euchre tournament --ai-types level1_balanced level2_strategic level3_balanced level1_aggressive

# Comprehensive tournament with all levels
euchre comprehensive-tournament
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
- Alice (Level 1 Balanced): 28 wins, 72% win rate
- Bob (Level 2 Strategic): 24 wins, 48% win rate
- Charlie (Level 3 Aggressive): 24 wins, 48% win rate
- David (Level 1 Conservative): 24 wins, 48% win rate

AI Level Performance:
- Level 1 AI: 52 wins (52.0%)
- Level 2 AI: 24 wins (24.0%)
- Level 3 AI: 24 wins (24.0%)
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
- **Level 2 Models**: 256-dimensional input, trained on game data
- **Level 3 Models**: 2048-dimensional input, advanced architectures
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

AI Level Performance:
- Level 1 AI: 12,234 wins (48.9%)
- Level 2 AI: 6,383 wins (25.5%)
- Level 3 AI: 6,383 wins (25.5%)

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

AI Level Improvements:
- Level 1: +15.2% improvement
- Level 2: +22.1% improvement
- Level 3: +28.7% improvement
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

#### AI Level Comparison
- **Level 1 vs Level 2**: Rule-based vs. neural network performance
- **Level 2 vs Level 3**: Basic vs. advanced neural network capabilities
- **Cross-Level Learning**: How different levels learn from each other

### Common Analysis Questions

1. **Which AI level performs best?**
   - Run tournaments with different AI levels
   - Compare win rates and strategies

2. **How do AI strategies differ between levels?**
   - Watch games with verbose logging
   - Analyze decision-making patterns

3. **What makes a successful AI?**
   - Study high-performing models
   - Identify common strategies

4. **How do AI players learn?**
   - Run training sessions
   - Track performance improvements

5. **Which risk profiles work best?**
   - Test different risk ratios
   - Analyze performance patterns

## Customizing AI vs AI Games

### Environment Variables
```bash
# Set AI behavior parameters
export EUCHRE_AI_AGGRESSIVENESS=0.7
export EUCHRE_AI_RISK_TOLERANCE=0.5
export EUCHRE_AI_LEARNING_RATE=0.1

# Set AI level preferences
export EUCHRE_AI_DEFAULT_LEVEL=2
export EUCHRE_AI_DEFAULT_PROFILE=strategic

# Run game with custom settings
euchre ai-vs-ai
```

### Configuration Files
Create custom AI profiles:
```json
{
  "name": "CustomAI",
  "ai_level": 2,
  "ai_type": "strategic",
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
- **GPU**: Optional for Level 2/3 AI acceleration

### Optimization Tips
```bash
# Run games in parallel
euchre run-mass-games -n 100000 --parallel 8

# Use GPU acceleration (if available)
euchre run-neural-games --device cuda

# Optimize memory usage
euchre run-mass-games --memory-efficient

# Use specific AI levels for performance
euchre run-mass-games --ai-levels 1  # Only Level 1 AI for speed
```

## Troubleshooting

### Common Issues

#### Games Run Slowly
- Reduce number of games
- Use fewer AI profiles
- Use Level 1 AI for faster execution
- Check system resources

#### Memory Issues
- Use memory-efficient mode
- Reduce batch sizes
- Close other applications
- Use Level 1 AI for lower memory usage

#### AI Behavior Issues
- Check AI profile settings
- Verify training data
- Reset AI to default state
- Check AI level compatibility

#### PyTorch Issues (Level 2/3 AI)
```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify model files
ls -la euchre/ai_model/
```

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
- **AI Architecture**: Compare different AI approaches

### Industry Applications
- **Gaming**: AI opponent development
- **Education**: Strategy learning tools
- **Testing**: Game rule validation
- **AI Development**: Benchmark different AI systems

### Data Collection
```bash
# Collect game data for analysis
euchre run-mass-games --save-data --output-format json

# Analyze specific aspects
euchre analyze-games --focus trump-selection
euchre analyze-games --focus card-play
euchre analyze-games --focus teamwork

# Compare AI levels
euchre analyze-games --ai-levels 1 2 3 --focus performance
```

## Next Steps

### For Beginners
1. **Start Simple**: Run basic AI vs AI games with Level 1 AI
2. **Watch Games**: Use verbose logging to understand strategies
3. **Try Profiles**: Experiment with different AI personalities
4. **Compare Levels**: See how different AI levels perform

### For Researchers
1. **Run Experiments**: Use mass game runner for large-scale analysis
2. **Train Models**: Use self-play training to improve AI
3. **Analyze Results**: Use analysis tools to study patterns
4. **Compare Architectures**: Test different AI level approaches

### For Developers
1. **Study Code**: Review AI implementation
2. **Create Profiles**: Build custom AI personalities
3. **Contribute**: Improve existing AI algorithms
4. **Extend System**: Add new AI levels or capabilities

---

*For AI system details, see [AI Overview](ai/overview.md)*
*For training instructions, see [AI Training](ai/training.md)*
*For analysis tools, see [Game Analysis](tools/analysis.md)* 