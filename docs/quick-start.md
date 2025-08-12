# Quick Start Guide

Get up and running with the Euchre CLI Game in under 5 minutes!

## 🚀 Installation (2 minutes)

### 1. Clone and Setup
```bash
git clone https://github.com/dapperfu/vibe_eucher.git
cd vibe_eucher
```

### 2. Install Dependencies
```bash
make install
```

### 3. Verify Installation
```bash
venv/bin/python -m euchre.cli_main --help
```

## 🎮 First Game (1 minute)

### Start a Quick AI vs AI Game
```bash
make run
```

This will start a full AI vs AI game where you can watch the computer play against itself.

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
```

## 👥 Play Against AI (2 minutes)

### Start a Human vs AI Game
```bash
venv/bin/python -m euchre.cli_main play --player-name "YourName"
```

### Game Flow
1. **Setup**: You'll be assigned to a team (Team 1 or Team 2)
2. **Dealing**: 5 cards are dealt to each player
3. **Trump Selection**: Players can call trump or pass
4. **Tricks**: 5 rounds of card playing
5. **Scoring**: Points awarded based on tricks won

### Your Turn
When it's your turn, you'll see:
```
Your hand: ['Ace of Spades', 'Ten of Spades', 'Queen of Clubs', 'Ace of Diamonds', 'Ten of Diamonds']
Choose a card (1-5): 
```

Enter the number (1-5) corresponding to the card you want to play.

## 🎯 Basic Strategy

### Trump Selection
- **Strong Hand**: Call trump if you have multiple high cards
- **Weak Hand**: Pass and let others call trump
- **Risk Assessment**: Consider if you can win 3+ tricks

### Card Play
- **Follow Suit**: Always play the same suit if possible
- **Lead High**: Play high cards when leading
- **Save Trump**: Don't waste trump cards unnecessarily

### Team Play
- **Coordinate**: Work with your partner
- **Count Tricks**: Keep track of how many each team has won
- **Risk Management**: Don't risk losing if you're ahead

## 🔧 Quick Commands

### Essential Commands
```bash
# Play a game
make run

# Human vs AI
venv/bin/python -m euchre.cli_main play -n "YourName"

# Watch AI vs AI
venv/bin/python -m euchre.cli_main ai-vs-ai

# Get help
venv/bin/python -m euchre.cli_main --help
```

### Verbose Mode
```bash
# Get detailed game information
venv/bin/python -m euchre.cli_main --verbose play -n "YourName"

# Get debug information
venv/bin/python -m euchre.cli_main --very-verbose ai-vs-ai
```

## 📚 What's Next?

### For Players
1. **Learn the Rules**: Read [Game Rules](game-rules.md) for complete understanding
2. **Try Different AI**: Experiment with [AI Profiles](ai/profiles.md)
3. **Practice**: Play multiple games to improve your strategy

### For Developers
1. **Study the Code**: Explore the [Architecture](architecture/core.md)
2. **Run Tests**: Use `make test` to validate functionality
3. **Contribute**: Check [Development Setup](development/setup.md)

### For Researchers
1. **AI Training**: Use [AI Training](ai/training.md) commands
2. **Analysis**: Explore [Game Analysis](tools/analysis.md) tools
3. **Benchmarking**: Run performance tests with [Benchmarking](tools/benchmarking.md)

## 🆘 Need Help?

### Common Issues
- **"Python command not found"**: Install Python 3.8+
- **"Permission denied"**: Check directory permissions
- **"Import errors"**: Ensure virtual environment is activated

### Getting Help
```bash
# Command help
venv/bin/python -m euchre.cli_main --help

# Specific command help
venv/bin/python -m euchre.cli_main play --help
```

### Documentation
- **Installation Issues**: [Installation Guide](installation.md)
- **Game Rules**: [Game Rules](game-rules.md)
- **AI Features**: [AI Overview](ai/overview.md)
- **CLI Reference**: [CLI Commands](cli-commands.md)

## 🎉 You're Ready!

You now have:
- ✅ A working Euchre game installation
- ✅ Experience with AI vs AI games
- ✅ Knowledge of basic gameplay
- ✅ Understanding of key commands

**Next step**: Play your first game and explore the advanced features!

---

*For complete game rules, see [Game Rules](game-rules.md)*
*For human vs AI gameplay, see [Human vs AI Guide](human-vs-ai.md)*
*For advanced features, see [AI Overview](ai/overview.md)* 