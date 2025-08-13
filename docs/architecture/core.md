# Core Game Engine

The core game engine provides the fundamental building blocks for the Euchre game, including deck management, game state tracking, trick management, and scoring systems.

## Architecture Overview

```
Core Game Engine
├── Deck (deck.py)
├── Game State (game_state.py)
├── Trick Manager (trick_manager.py)
├── Scoring (scoring.py)
└── Game Session (game_session.py)
```

## Deck Management

### Deck Class (`euchre/core/deck.py`)

The `Deck` class manages the 24-card euchre deck and provides card dealing functionality.

#### Key Features
- **24-Card Deck**: Contains 9, 10, J, Q, K, A of each suit
- **Card Dealing**: Distributes cards to players
- **Top Card Management**: Handles the card that can become trump
- **Deck Reset**: Restores deck to initial state

#### Core Methods

```python
class Deck:
    def __init__(self) -> None:
        """Initialize a new euchre deck."""
        
    def reset(self) -> None:
        """Reset the deck to its initial state."""
        
    def deal_cards(self, num_players: int) -> List[List[Card]]:
        """Deal 5 cards to each player.
        
        Returns:
            List of hands, one for each player
        """
        
    def draw_top_card(self) -> Card:
        """Draw the top card that can become trump.
        
        Returns:
            The top card from the deck
        """
```

#### Usage Example

```python
from euchre.core.deck import Deck

# Create and use a deck
deck = Deck()
hands = deck.deal_cards(4)  # Deal to 4 players
top_card = deck.draw_top_card()  # Get trump candidate
```

#### Card Representation

Cards are represented with:
- **Suit**: Hearts, Diamonds, Clubs, Spades
- **Rank**: 9, 10, J, Q, K, A
- **Value**: Numeric ranking for comparison
- **Trump Status**: Whether the card is currently trump

## Game State Management

### Game State Manager (`euchre/core/game_state.py`)

The `GameStateManager` tracks the current state of the game, including scores, dealer, and trump information.

#### Key Features
- **Score Tracking**: Maintains team and individual scores
- **Dealer Management**: Tracks current dealer and rotation
- **Trump Information**: Stores current trump suit and caller
- **Game Status**: Tracks whether game is active or complete

#### Core Methods

```python
class GameStateManager:
    def __init__(self) -> None:
        """Initialize the game state manager."""
        
    def initialize_game(self, players: List[Player]) -> None:
        """Initialize a new game with the given players."""
        
    def set_dealer(self, player: Player) -> None:
        """Set the current dealer."""
        
    def set_trump_suit(self, trump_suit: Suit, caller: Player) -> None:
        """Set the trump suit and record who called it."""
        
    def update_scores(self, team1_score: int, team2_score: int) -> None:
        """Update the current game scores."""
        
    def is_game_over(self) -> bool:
        """Check if the game is over (team has 10+ points)."""
        
    def get_winner(self) -> str:
        """Get the winning team name."""
```

#### State Information

```python
# Example game state
game_state = {
    "dealer": "Alice",
    "trump_suit": "Diamonds",
    "trump_caller": "Bob",
    "trump_caller_team": 1,
    "game_scores": {"Team 1": 4, "Team 2": 2},
    "round_number": 3,
    "game_active": True
}
```

## Trick Management

### Trick Manager (`euchre/core/trick_manager.py`)

The `TrickManager` handles individual tricks, including card playing, winner determination, and trick completion.

#### Key Features
- **Trick Creation**: Start new tricks
- **Card Playing**: Add cards to current trick
- **Winner Determination**: Calculate trick winners based on trump and lead suit
- **Trick History**: Track completed tricks for the round

#### Core Methods

```python
class TrickManager:
    def __init__(self) -> None:
        """Initialize the trick manager."""
        
    def start_new_trick(self) -> None:
        """Start a new trick."""
        
    def play_card(self, player: Player, card: Card) -> None:
        """Play a card in the current trick."""
        
    def complete_trick(self) -> Player:
        """Complete the current trick and determine the winner.
        
        Returns:
            The player who won the trick
        """
        
    def get_current_trick(self) -> Optional[Trick]:
        """Get the current trick in progress."""
        
    def get_round_results(self) -> List[int]:
        """Get the trick counts for each player in the current round."""
```

#### Trick Structure

```python
class Trick:
    def __init__(self):
        self.cards_played: List[Tuple[Player, Card]] = []
        self.lead_suit: Optional[Suit] = None
        self.winner: Optional[Player] = None
        self.is_complete: bool = False
```

#### Card Comparison Logic

```python
def _card_beats(self, card1: Card, card2: Card) -> bool:
    """Determine if card1 beats card2.
    
    Trump cards beat non-trump cards.
    Higher trump cards beat lower trump cards.
    Within the same suit, higher rank wins.
    """
```

## Scoring System

### Scoring Manager (`euchre/core/scoring.py`)

The `ScoringManager` handles all scoring calculations and determines when games and rounds are complete.

#### Key Features
- **Round Scoring**: Calculate points for each round
- **Game Completion**: Determine when a team reaches 10 points
- **Euchre Detection**: Identify when trump-calling team gets set
- **Team Score Tracking**: Maintain cumulative game scores

#### Core Methods

```python
class ScoringManager:
    def __init__(self) -> None:
        """Initialize the scoring manager."""
        
    def score_round(self, players: List[Player], trump_caller_team: int) -> Tuple[int, int]:
        """Score a completed round.
        
        Args:
            players: List of players in the game
            trump_caller_team: Team that called trump (0 or 1)
            
        Returns:
            Tuple of (team1_score, team2_score)
        """
        
    def is_game_over(self, players: List[Player]) -> bool:
        """Check if the game is over.
        
        Returns:
            True if any team has 10+ points
        """
        
    def is_team_set(self, players: List[Player], trump_caller_team: int) -> bool:
        """Check if the trump-calling team got set.
        
        Returns:
            True if trump-calling team won fewer than 3 tricks
        """
        
    def get_team_scores(self, players: List[Player]) -> Dict[str, int]:
        """Get the current team scores.
        
        Returns:
            Dictionary mapping team names to scores
        """
```

#### Scoring Rules

1. **Basic Scoring**
   - Win 3-4 tricks: 1 point
   - Win all 5 tricks: 2 points

2. **Euchre Bonus**
   - Trump-calling team gets set (wins <3 tricks): Opposing team gets 2 points

3. **Game End**
   - First team to reach 10 points wins

#### Score Calculation Example

```python
# Example round scoring
trump_caller_team = 0  # Team 1 called trump
team1_tricks = 3       # Team 1 won 3 tricks
team2_tricks = 2       # Team 2 won 2 tricks

# Team 1 gets 1 point (won 3 tricks)
# Team 2 gets 0 points
team1_score, team2_score = 1, 0
```

## Game Session Management

### Game Session (`euchre/core/game_session.py`)

The `GameSession` class manages the overall game flow and coordinates between different components.

#### Key Features
- **Game Coordination**: Orchestrates deck, state, tricks, and scoring
- **Round Management**: Handles round transitions and game completion
- **Player Coordination**: Manages player turns and game flow
- **Session Persistence**: Saves and loads game sessions

#### Core Methods

```python
class GameSession:
    def __init__(self, players: List[Player]) -> None:
        """Initialize a new game session."""
        
    def start_new_game(self) -> None:
        """Start a new game."""
        
    def start_new_round(self) -> None:
        """Start a new round within the current game."""
        
    def play_trick(self) -> Player:
        """Play a single trick.
        
        Returns:
            The player who won the trick
        """
        
    def complete_round(self) -> Tuple[int, int]:
        """Complete the current round and return scores."""
        
    def save_session(self, filename: str) -> None:
        """Save the current game session to a file."""
        
    def load_session(self, filename: str) -> None:
        """Load a game session from a file."""
```

## Integration Example

### Complete Game Flow

```python
from euchre.core.deck import Deck
from euchre.core.game_state import GameStateManager
from euchre.core.trick_manager import TrickManager
from euchre.core.scoring import ScoringManager

# Initialize components
deck = Deck()
game_state = GameStateManager()
trick_manager = TrickManager()
scoring = ScoringManager()

# Start a game
deck.reset()
hands = deck.deal_cards(4)
top_card = deck.draw_top_card()

# Play a round
for trick_num in range(5):
    trick_manager.start_new_trick()
    
    # Players play cards...
    # ... (card playing logic)
    
    winner = trick_manager.complete_trick()
    # Update trick counts...

# Score the round
team1_score, team2_score = scoring.score_round(players, trump_caller_team)
```

## Performance Considerations

### Memory Management
- **Card Objects**: Reused across games to reduce object creation
- **State Tracking**: Minimal memory footprint for game state
- **Trick History**: Limited to current round to prevent memory growth

### Computational Efficiency
- **Card Comparison**: Optimized for fast winner determination
- **Score Calculation**: Efficient algorithms for round scoring
- **State Updates**: Minimal computation for state changes

## Error Handling

### Validation
- **Player Count**: Ensures exactly 4 players
- **Card Validity**: Validates card plays and suit following
- **State Consistency**: Maintains game state integrity

### Exception Handling
- **Invalid Moves**: Graceful handling of rule violations
- **State Errors**: Recovery from corrupted game state
- **Resource Issues**: Handling file I/O and memory problems

## Testing

### Unit Tests
- **Component Isolation**: Test each component independently
- **Edge Cases**: Cover unusual game scenarios
- **Performance**: Validate computational efficiency

### Integration Tests
- **Full Game Flow**: Test complete game scenarios
- **Component Interaction**: Verify proper coordination
- **Error Scenarios**: Test error handling and recovery

---

*For game logic details, see [Game Logic](game-logic.md)*
*For player system information, see [Player System](players.md)*
*For development setup, see [Development Setup](../development/setup.md)* 