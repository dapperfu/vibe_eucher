"""Main game logic for the euchre card game."""

from typing import List, Optional, Tuple
import random
from .models import (
    Player, PlayerType, Card, Suit, Rank, GameState
)


class EuchreGame:
    """Main euchre game controller."""
    
    def __init__(self) -> None:
        """Initialize a new euchre game."""
        self.players: List[Player] = []
        self.game_state: Optional[GameState] = None
        self.deck: List[Card] = []
        self._initialize_deck()
        
    def _initialize_deck(self) -> None:
        """Initialize the deck with euchre cards (9-A of each suit)."""
        self.deck.clear()
        for suit in Suit:
            for rank in Rank:
                self.deck.append(Card(rank=rank, suit=suit))
                
    def add_player(self, name: str, player_type: PlayerType) -> None:
        """Add a player to the game.
        
        Parameters
        ----------
        name : str
            The player's name
        player_type : PlayerType
            Whether the player is human or AI
        """
        player = Player(name=name, player_type=player_type)
        self.players.append(player)
        
    def start_new_game(self) -> None:
        """Start a new game with the current players."""
        if len(self.players) != 4:
            raise ValueError("Euchre requires exactly 4 players")
            
        # Reset all players
        for player in self.players:
            player.clear_hand()
            player.tricks_won = 0
            player.is_dealer = False
            
        # Set dealer (rotate each game)
        dealer_index = random.randint(0, 3)
        self.players[dealer_index].is_dealer = True
        
        # Initialize game state
        self.game_state = GameState(
            players=self.players,
            current_player_index=(dealer_index + 1) % 4,  # Start with player after dealer
            trump_suit=None,
            dealer_index=dealer_index,
            round_number=1,
            team1_score=0,
            team2_score=0
        )
        
        self._deal_cards()
        
    def _deal_cards(self) -> None:
        """Deal 5 cards to each player."""
        # Shuffle deck
        random.shuffle(self.deck)
        
        # Deal 5 cards to each player
        for i, player in enumerate(self.players):
            start_idx = i * 5
            end_idx = start_idx + 5
            player.hand = self.deck[start_idx:end_idx]
            
        # Remove dealt cards from deck
        self.deck = self.deck[20:]
        
    def get_player_hand(self, player_name: str) -> List[Card]:
        """Get the hand of a specific player.
        
        Parameters
        ----------
        player_name : str
            The name of the player
            
        Returns
        -------
        List[Card]
            The player's hand
        """
        for player in self.players:
            if player.name == player_name:
                return player.hand.copy()
        return []
        
    def play_ai_turn(self, player: Player) -> Card:
        """Play a turn for an AI player.
        
        Parameters
        ----------
        player : Player
            The AI player
            
        Returns
        -------
        Card
            The card the AI chose to play
        """
        # Simple AI: just play the first card in hand
        # In a real implementation, this would be much more sophisticated
        if not player.hand:
            raise ValueError("AI player has no cards to play")
            
        card = player.hand[0]
        player.remove_card(card)
        return card
        
    def is_game_over(self) -> bool:
        """Check if the game is over.
        
        Returns
        -------
        bool
            True if the game is over
        """
        if not self.game_state:
            return True
            
        return (self.game_state.team1_score >= 10 or 
                self.game_state.team2_score >= 10)
        
    def get_winner(self) -> Optional[str]:
        """Get the winning team.
        
        Returns
        -------
        Optional[str]
            The name of the winning team, or None if game not over
        """
        if not self.is_game_over():
            return None
            
        if self.game_state and self.game_state.team1_score >= 10:
            return "Team 1"
        elif self.game_state and self.game_state.team2_score >= 10:
            return "Team 2"
        return None


class AIPlayer:
    """AI player logic for euchre."""
    
    def __init__(self, player: Player) -> None:
        """Initialize AI player.
        
        Parameters
        ----------
        player : Player
            The player this AI controls
        """
        self.player = player
        
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play.
        
        Parameters
        ----------
        lead_suit : Optional[Suit]
            The suit that was led (if any)
        trump_suit : Optional[Suit]
            The current trump suit
            
        Returns
        -------
        Card
            The card to play
        """
        # Simple strategy: play the highest card of the lead suit if possible
        # Otherwise, play the lowest card
        
        if lead_suit and self.player.has_suit(lead_suit):
            # Must follow suit
            cards_of_suit = self.player.get_cards_of_suit(lead_suit)
            return max(cards_of_suit, key=lambda c: c.rank.value)
        else:
            # Can play any card - play the lowest
            return min(self.player.hand, key=lambda c: c.rank.value)
            
    def should_order_up(self, top_card: Card) -> bool:
        """Decide whether to order up the top card.
        
        Parameters
        ----------
        top_card : Card
            The top card that could be ordered up
            
        Returns
        -------
        bool
            True if the AI should order up the card
        """
        # Simple strategy: order up if we have 2+ cards of that suit
        cards_of_suit = self.player.get_cards_of_suit(top_card.suit)
        return len(cards_of_suit) >= 2 