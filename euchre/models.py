"""Game models for the euchre card game."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    """Card suits in euchre."""
    
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"


class Rank(Enum):
    """Card ranks in euchre (9, 10, J, Q, K, A)."""
    
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass
class Card:
    """A playing card in the euchre game."""
    
    rank: Rank
    suit: Suit
    is_trump: bool = False
    
    def __str__(self) -> str:
        """String representation of the card."""
        rank_str = self.rank.name.title()
        suit_str = self.suit.name.title()
        trump_indicator = " (Trump)" if self.is_trump else ""
        return f"{rank_str} of {suit_str}{trump_indicator}"
    
    def __lt__(self, other: "Card") -> bool:
        """Compare cards for ordering."""
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank.value < other.rank.value
        
    def __hash__(self) -> int:
        """Hash function for Card objects."""
        return hash((self.rank, self.suit, self.is_trump))
        
    def __eq__(self, other: object) -> bool:
        """Equality comparison for Card objects."""
        if not isinstance(other, Card):
            return False
        return (self.rank == other.rank and 
                self.suit == other.suit and 
                self.is_trump == other.is_trump)


class PlayerType(Enum):
    """Type of player in the game."""
    
    HUMAN = "human"
    AI = "ai"


class Player:
    """A player in the euchre game."""
    
    def __init__(self, name: str, player_type: PlayerType) -> None:
        """Initialize a new player.
        
        Parameters
        ----------
        name : str
            The player's name
        player_type : PlayerType
            Whether the player is human or AI
        """
        self.name: str = name
        self.player_type: PlayerType = player_type
        self.hand: List[Card] = []
        self.tricks_won: int = 0
        self.is_dealer: bool = False
        
    def add_card(self, card: Card) -> None:
        """Add a card to the player's hand.
        
        Parameters
        ----------
        card : Card
            The card to add to the hand
        """
        self.hand.append(card)
        
    def remove_card(self, card: Card) -> None:
        """Remove a card from the player's hand.
        
        Parameters
        ----------
        card : Card
            The card to remove from the hand
        """
        if card in self.hand:
            self.hand.remove(card)
            
    def clear_hand(self) -> None:
        """Clear all cards from the player's hand."""
        self.hand.clear()
        
    def get_hand_size(self) -> int:
        """Get the number of cards in the player's hand.
        
        Returns
        -------
        int
            The number of cards in the hand
        """
        return len(self.hand)
        
    def has_suit(self, suit: Suit) -> bool:
        """Check if the player has any cards of a specific suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to check for
            
        Returns
        -------
        bool
            True if the player has cards of the specified suit
        """
        return any(card.suit == suit for card in self.hand)
        
    def get_cards_of_suit(self, suit: Suit) -> List[Card]:
        """Get all cards of a specific suit from the player's hand.
        
        Parameters
        ----------
        suit : Suit
            The suit to get cards for
            
        Returns
        -------
        List[Card]
            List of cards of the specified suit
        """
        return [card for card in self.hand if card.suit == suit]
        
    def __str__(self) -> str:
        """String representation of the player."""
        return f"{self.name} ({self.player_type.value})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the player."""
        return f"Player(name='{self.name}', player_type={self.player_type.value}, hand_size={len(self.hand)})"


@dataclass
class GameState:
    """Current state of the euchre game."""
    
    players: List[Player]
    current_player_index: int
    trump_suit: Optional[Suit]
    dealer_index: int
    round_number: int
    team1_score: int
    team2_score: int
    
    def get_current_player(self) -> Player:
        """Get the current player.
        
        Returns
        -------
        Player
            The current player
        """
        return self.players[self.current_player_index]
    
    def next_player(self) -> None:
        """Move to the next player."""
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        
    def get_dealer(self) -> Player:
        """Get the dealer player.
        
        Returns
        -------
        Player
            The dealer player
        """
        return self.players[self.dealer_index] 