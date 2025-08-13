# Human vs AI Guide

This guide teaches you how to play Euchre as a human player against AI opponents. You'll learn the controls, strategies, and how to interact with the game.

## Starting a Human vs AI Game

### Basic Command
```bash
# Start a game with you as a human player
euchre play --player-name "YourName"

# Or use the short form
euchre play -n "YourName"
```

### Customizing AI Opponents
```bash
# Choose your AI opponents' names
euchre play -n "YourName" -a "Sherlock" -a "Watson" -a "Moriarty"

# Use default names (Alice, Bob, Charlie)
euchre play -n "YourName"
```

### Example Session
```bash
$ euchre play -n "Player1"
Welcome to Euchre, Player1!
Starting AI-only Euchre game...
Game started! Dealing cards...
```

## Game Setup

When you start a game, you'll see:

1. **Player Assignment**: You'll be assigned to a team (Team 1 or Team 2)
2. **Initial Deal**: 5 cards are dealt to each player
3. **Top Card**: A card is revealed that can become trump
4. **Your Hand**: Your 5 cards are displayed

### Example Game Setup
```
Dealer: Alice
Top card: Queen of Hearts
Initial hands:
Player1: ['Ace of Spades', 'Ten of Spades', 'Queen of Clubs', 'Ace of Diamonds', 'Ten of Diamonds']
Alice: ['King of Spades', 'Nine of Spades', 'Jack of Clubs', 'King of Diamonds', 'Nine of Diamonds']
Bob: ['Queen of Spades', 'Ace of Clubs', 'Ten of Clubs', 'Queen of Diamonds', 'Ace of Hearts']
Charlie: ['Jack of Spades', 'King of Clubs', 'Nine of Clubs', 'Jack of Diamonds', 'King of Hearts']
```

## Trump Selection Phase

### Understanding Trump
- **Trump suit**: The highest-ranking suit for the round
- **Left Bower**: The Jack of the same color as trump (e.g., if Diamonds is trump, Jack of Hearts becomes the Left Bower)
- **Right Bower**: The Jack of the trump suit

### Trump Selection Process
1. **First Round**: Players can "order up" the top card to make its suit trump
2. **Second Round**: If no one orders up, players can call any suit as trump
3. **Dealer's Choice**: If no one calls, the dealer must pick a trump suit

### Your Decisions
- **Order Up**: Accept the top card as trump (you get the card)
- **Pass**: Decline to make the top card trump
- **Call Trump**: Choose a different suit as trump (second round only)

## Playing the Game

### Trick Structure
Each round consists of 5 tricks:
1. **Lead**: First player plays any card
2. **Follow Suit**: Other players must play cards of the same suit if possible
3. **Trump**: If you can't follow suit, you can play any card (including trump)
4. **Winning**: Highest trump card wins, or highest card of the lead suit if no trump

### During Your Turn
1. **View the Trick**: See what cards have been played
2. **Choose Your Card**: Select from your remaining cards
3. **Follow Rules**: Ensure you're following suit if possible
4. **Strategic Play**: Consider when to play high cards vs. low cards

### Example Trick
```
--- Trick 1 ---
Bob plays Nine of Spades
Charlie plays Ten of Clubs
David plays Nine of Clubs
Your turn! Choose a card:

Your hand: ['Ace of Spades', 'Ten of Spades', 'Queen of Clubs', 'Ace of Diamonds', 'Ten of Diamonds']
```

## Game Controls

### Card Selection
- **Number Input**: Enter the number corresponding to your card (1-5)
- **Card Display**: Cards are numbered from left to right in your hand
- **Confirmation**: Confirm your choice before playing

### Navigation
- **Help**: Type `help` or `?` for assistance
- **Quit**: Type `quit` or `exit` to end the game
- **Status**: Type `status` to see current game state

## Strategy Tips

### Trump Selection
- **Strong Trump Hand**: If you have multiple trump cards, consider calling trump
- **Weak Hand**: If your hand is weak, you might want to pass
- **Partner's Strength**: Consider your partner's potential strength

### Card Play
- **Lead High**: When leading, consider playing high cards to win the trick
- **Save Trump**: Don't waste trump cards on tricks you can't win
- **Follow Suit**: Always follow suit when possible to avoid penalties
- **Trump Low**: Use low trump cards to win tricks when you can't follow suit

### Team Play
- **Communication**: Pay attention to your partner's plays
- **Trick Counting**: Keep track of how many tricks each team has won
- **Risk Assessment**: Don't risk losing if your team is already ahead

## Scoring

### Point System
- **3-4 Tricks**: 1 point
- **5 Tricks**: 2 points
- **Euchre**: If the trump-calling team doesn't win 3+ tricks, the opposing team gets 2 points
- **Game End**: First team to reach 10 points wins

### Example Scoring
```
=== Round 1 Complete ===
Final trick counts: Player1: 2, Alice: 1, Bob: 1, Charlie: 1
Team 1 (Player1 & Charlie): 1 points
Team 2 (Alice & Bob): 0 points
Game Score - Team 1: 1, Team 2: 0
```

## Advanced Features

### Verbose Mode
```bash
# Get detailed game information
euchre --verbose play -n "YourName"
```

### Very Verbose Mode
```bash
# Get debug-level information
euchre --very-verbose play -n "YourName"
```

### Custom AI Profiles
```bash
# Play against different AI personalities
euchre ai-profiles -p aggressive -p conservative -p balanced -p opportunistic
```

## Common Scenarios

### Scenario 1: Strong Trump Hand
**Your Hand**: ['Ace of Diamonds', 'King of Diamonds', 'Jack of Hearts', 'Ten of Spades', 'Nine of Clubs']
**Top Card**: Queen of Diamonds
**Decision**: Order up Diamonds as trump (you have 3 trump cards)

### Scenario 2: Weak Hand
**Your Hand**: ['Nine of Spades', 'Ten of Clubs', 'Queen of Hearts', 'King of Clubs', 'Ace of Hearts']
**Top Card**: Jack of Diamonds
**Decision**: Pass (your hand is weak and you don't want to risk calling trump)

### Scenario 3: Following Suit
**Lead Suit**: Spades
**Your Hand**: ['Ace of Spades', 'Ten of Diamonds', 'Queen of Clubs', 'King of Hearts', 'Nine of Clubs']
**Decision**: Play Ace of Spades (you must follow suit and it's your highest Spade)

## Troubleshooting

### Game Won't Start
- Ensure you're using the correct command: `euchre play -n "YourName"`
- Check that the virtual environment is activated
- Verify installation with `euchre --help`

### Controls Not Working
- Use number keys (1-5) to select cards
- Type `help` for assistance
- Ensure you're following the game prompts

### Game Crashes
- Check for error messages
- Try running with verbose mode: `euchre --verbose play -n "YourName"`
- Report issues on the GitHub repository

## Practice Tips

1. **Start Simple**: Play a few games to get familiar with the interface
2. **Watch AI**: Observe how AI players make decisions
3. **Experiment**: Try different strategies to see what works
4. **Learn from Losses**: Analyze why you lost tricks or rounds
5. **Practice Trump Selection**: This is often the most critical decision

## Next Steps

Once you're comfortable with human vs AI play:

1. **Try Different AI Profiles**: Experiment with aggressive, conservative, and balanced AI
2. **Watch AI vs AI Games**: Use `euchre ai-vs-ai` to see how AI strategies differ
3. **Explore Advanced Features**: Check out [CLI Commands Reference](cli-commands.md)
4. **Learn Game Analysis**: Use [Game Analysis Tools](tools/analysis.md) to review your games

---

*For complete game rules, see [Game Rules](game-rules.md)*
*For advanced AI features, see [AI Overview](ai/overview.md)* 