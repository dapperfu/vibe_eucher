# Game Controls

This document explains how to control the Euchre game during gameplay, including card selection, navigation, and interface options.

## Basic Controls

### Card Selection
During your turn, you'll see your hand displayed with numbers:

```
Your hand: ['Ace of Spades', 'Ten of Spades', 'Queen of Clubs', 'Ace of Diamonds', 'Ten of Diamonds']
Choose a card (1-5): 
```

**How to Play a Card:**
- **Type the number** (1-5) corresponding to the card you want to play
- **Press Enter** to confirm your choice
- **Card 1** = Leftmost card (Ace of Spades in this example)
- **Card 5** = Rightmost card (Ten of Diamonds in this example)

### Navigation Commands

#### Help Commands
```bash
help          # Show available commands
?             # Same as help
```

#### Game Status
```bash
status        # Show current game state
hand          # Display your current hand
trick         # Show current trick state
score         # Display current scores
```

#### Game Control
```bash
quit          # Exit the current game
exit          # Same as quit
restart       # Start a new game (if supported)
```

## Game Interface

### Main Game Display
```
=== Round 1 ===
Trump Suit: Diamonds
Trump Caller: Alice

Current Trick:
Lead: Bob (Nine of Spades)
Charlie: Ten of Clubs
David: Nine of Clubs
Your turn!

Your hand: ['Ace of Spades', 'Ten of Spades', 'Queen of Clubs', 'Ace of Diamonds', 'Ten of Diamonds']
Choose a card (1-5): 
```

### Game State Information
The interface shows:
- **Current round** and trump suit
- **Who called trump** and when
- **Current trick** with cards played so far
- **Your hand** with numbered cards
- **Game scores** and trick counts

### Trick Display
```
--- Trick 2 ---
Starting with Alice (winner of previous trick)
Alice plays Ten of Diamonds
Trick so far: Alice: Ten of Diamonds
Bob plays Nine of Diamonds
Trick so far: Alice: Ten of Diamonds, Bob: Nine of Diamonds
Charlie plays Queen of Spades
Trick so far: Alice: Ten of Diamonds, Bob: Nine of Diamonds, Charlie: Queen of Spades
Your turn!
```

## Trump Selection Controls

### First Round - Ordering Up
When the top card is revealed:
```
Top card: Queen of Hearts
Bob, do you want to order up Hearts? (y/n): 
```

**Your Options:**
- **y** or **yes**: Order up the top card as trump
- **n** or **no**: Pass on the top card
- **help**: Get more information about trump selection

### Second Round - Calling Trump
If no one orders up:
```
No one ordered up the top card.
Bob, what suit would you like to call as trump?
Options: Hearts, Diamonds, Clubs, Spades, or pass: 
```

**Your Options:**
- **Hearts**: Call Hearts as trump
- **Diamonds**: Call Diamonds as trump
- **Clubs**: Call Clubs as trump
- **Spades**: Call Spades as trump
- **pass**: Decline to call trump

## Advanced Controls

### Verbose Mode
Run the game with detailed information:
```bash
euchre --verbose play -n "YourName"
```

**Additional Information Displayed:**
- Detailed card rankings
- AI decision explanations
- Trick analysis
- Strategy hints

### Very Verbose Mode
Get debug-level information:
```bash
euchre --very-verbose play -n "YourName"
```

**Debug Information:**
- Internal game state
- AI evaluation details
- Performance metrics
- Error tracking

### Quiet Mode
Minimal output for automated play:
```bash
euchre --quiet play -n "YourName"
```

## Keyboard Shortcuts

### Quick Commands
- **Ctrl+C**: Interrupt current game
- **Ctrl+D**: End input (EOF)
- **Tab**: Auto-complete (if supported)
- **Arrow Keys**: Navigate command history (if supported)

### Input Shortcuts
- **1-5**: Quick card selection
- **y/n**: Quick yes/no responses
- **Enter**: Confirm selection
- **Backspace**: Correct input errors

## Error Handling

### Invalid Input
If you enter invalid input:
```
Invalid input. Please enter a number between 1 and 5.
Choose a card (1-5): 
```

**Common Errors:**
- **Wrong number**: Enter a number not in your hand
- **Invalid suit**: Call a suit that's not available
- **Out of turn**: Try to play when it's not your turn

### Recovery
- **Re-enter**: Simply type the correct input
- **Help**: Use `help` command for guidance
- **Status**: Check current game state with `status`

## Game Flow Controls

### Round Progression
```
=== Round 1 Complete ===
Final trick counts: Alice: 3, Bob: 1, Charlie: 1, David: 0
Team 1 (Alice & Charlie): 1 points
Team 2 (Bob & David): 0 points
Game Score - Team 1: 1, Team 2: 0

Press Enter to continue to next round...
```

**Controls:**
- **Enter**: Continue to next round
- **quit**: Exit the game
- **status**: Review current scores

### Game Completion
```
🎉 GAME OVER! 🎉
Team 1 (Alice & Charlie) wins!

Final Score - Team 1: 10, Team 2: 6

Press Enter to exit...
```

**Options:**
- **Enter**: Exit the game
- **restart**: Start a new game (if supported)

## Accessibility Features

### Text-Based Interface
- **No graphics required**: Works on all terminals
- **Screen reader friendly**: Clear text output
- **Keyboard only**: No mouse required

### Adjustable Output
- **Verbose levels**: Choose amount of information
- **Clear formatting**: Easy to read text
- **Consistent structure**: Predictable layout

## Troubleshooting Controls

### Common Issues

#### Game Won't Respond
```bash
# Check if game is waiting for input
# Look for prompt like "Choose a card (1-5):"

# Try pressing Enter
# Use Ctrl+C to interrupt if stuck
```

#### Controls Not Working
```bash
# Verify you're in the right mode
# Check for error messages
# Use help command for guidance
```

#### Game Crashes
```bash
# Restart the game
# Check for error logs
# Use verbose mode for debugging
```

### Getting Help
```bash
# In-game help
help

# Command-line help
euchre play --help

# General help
euchre --help
```

## Customization

### Environment Variables
```bash
# Set default verbosity
export EUCHRE_VERBOSE=1

# Set input timeout
export EUCHRE_INPUT_TIMEOUT=30

# Set output format
export EUCHRE_OUTPUT_FORMAT=detailed
```

### Configuration Files
Create custom control settings:
```json
{
  "controls": {
    "auto_confirm": false,
    "show_hints": true,
    "input_timeout": 30,
    "verbosity": "normal"
  }
}
```

## Best Practices

### For New Players
1. **Start Simple**: Use basic controls first
2. **Read Prompts**: Pay attention to what the game is asking
3. **Use Help**: Don't hesitate to type `help`
4. **Practice**: Play a few games to get comfortable

### For Experienced Players
1. **Use Shortcuts**: Master quick card selection
2. **Enable Verbose**: Get detailed game information
3. **Customize**: Adjust settings for your preferences
4. **Experiment**: Try different control options

### For Developers
1. **Test Controls**: Verify all commands work
2. **Error Handling**: Ensure graceful error recovery
3. **User Experience**: Make controls intuitive
4. **Documentation**: Keep this guide updated

## Next Steps

### Practice Controls
1. **Play a Game**: Use basic controls in a real game
2. **Try Commands**: Experiment with different commands
3. **Use Help**: Learn about advanced features

### Advanced Features
1. **Verbose Mode**: Enable detailed information
2. **Customization**: Adjust game settings
3. **Shortcuts**: Master keyboard shortcuts

### Troubleshooting
1. **Common Issues**: Learn to solve problems
2. **Getting Help**: Know where to find assistance
3. **Reporting Bugs**: Help improve the game

---

*For gameplay instructions, see [Human vs AI Guide](human-vs-ai.md)*
*For game rules, see [Game Rules](game-rules.md)*
*For CLI commands, see [CLI Commands Reference](cli-commands.md)* 