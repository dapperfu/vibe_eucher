"""Game models for the euchre card game."""

from typing import List, Optional, Dict, Any, Tuple
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
        trump_indicator = " *" if self.is_trump else ""
        return f"{rank_str} of {suit_str}{trump_indicator}"
    
    def short_str(self) -> str:
        """Short string representation for display."""
        rank_map = {
            Rank.NINE: "9",
            Rank.TEN: "10", 
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A"
        }
        suit_map = {
            Suit.HEARTS: "♥",
            Suit.DIAMONDS: "♦",
            Suit.CLUBS: "♣",
            Suit.SPADES: "♠"
        }
        
        rank_str = rank_map.get(self.rank, str(self.rank.value))
        suit_str = suit_map.get(self.suit, self.suit.name[0].upper())
        
        if self.is_trump:
            return f"{rank_str}{suit_str}*"
        else:
            return f"{rank_str}{suit_str}"
    
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


@dataclass
class Trick:
    """A trick in the euchre game."""
    
    lead_suit: Optional[Suit] = None
    cards_played: List[Tuple["Player", Card]] = None
    
    def __post_init__(self) -> None:
        """Initialize cards_played if None."""
        if self.cards_played is None:
            self.cards_played = []
    
    def add_card(self, player: "Player", card: Card) -> None:
        """Add a card to the trick."""
        if not self.lead_suit:
            self.lead_suit = card.suit
        self.cards_played.append((player, card))
    
    def get_winner(self, trump_suit: Optional[Suit]) -> Tuple["Player", Card]:
        """Get the winner of this trick."""
        if not self.cards_played:
            raise ValueError("No cards played in trick")
            
        winner = self.cards_played[0][0]
        winning_card = self.cards_played[0][1]
        
        for player, card in self.cards_played[1:]:
            if self._card_beats(card, winning_card, trump_suit):
                winner = player
                winning_card = card
                
        return winner, winning_card
    
    def format_as_table(self, dealer_index: int, players: List["Player"], trump_suit: Optional[Suit] = None) -> str:
        """Format the trick as a table showing what each player played.
        
        Parameters
        ----------
        dealer_index : int
            Index of the dealer in the players list
        players : List[Player]
            List of all players in the game
        trump_suit : Optional[Suit]
            The current trump suit for highlighting trump cards
            
        Returns
        -------
        str
            Formatted table string showing the trick
        """
        if not self.cards_played:
            return "No cards played yet"
            
        # Create a list of player names starting from left of dealer
        # In euchre, play goes clockwise from left of dealer
        player_order = []
        for i in range(4):
            player_idx = (dealer_index + 1 + i) % 4
            player_order.append(players[player_idx])
        
        # Create the table header
        table_lines = []
        table_lines.append("┌" + "─" * 50 + "┐")
        
        # Header row with player names
        header = "│"
        for player in player_order:
            header += f" {player.name:>10} │"
        table_lines.append(header)
        
        # Separator line
        table_lines.append("├" + "─" * 50 + "┤")
        
        # Row showing what each player played (compact format)
        cards_row = "│"
        for player in player_order:
            # Find what this player played in this trick
            card_played = None
            for trick_player, card in self.cards_played:
                if trick_player == player:
                    card_played = card
                    break
            
            if card_played:
                compact_card = self._get_compact_card_format(card_played, trump_suit)
                cards_row += f" {compact_card:>10} │"
            else:
                cards_row += " " + " " * 10 + "│"
        table_lines.append(cards_row)
        
        # Bottom border
        table_lines.append("└" + "─" * 50 + "┘")
        
        return "\n".join(table_lines)
    
    def _get_compact_card_format(self, card: Card, trump_suit: Optional[Suit] = None) -> str:
        """Get a compact representation of the card (e.g., 'A♥', 'K♣').
        
        Parameters
        ----------
        card : Card
            The card to format
        trump_suit : Optional[Suit]
            The current trump suit for highlighting trump cards
            
        Returns
        -------
        str
            Compact card representation
        """
        # Get rank symbol
        rank_symbol = self._get_rank_symbol(card.rank)
        
        # Get suit symbol
        suit_symbol = self._get_suit_symbol(card.suit)
        
        # Add trump indicator if it's a trump card
        trump_indicator = ""
        if trump_suit and self._is_trump_card(card, trump_suit):
            trump_indicator = "*"
        
        return f"{rank_symbol}{suit_symbol}{trump_indicator}"
    
    def _get_rank_symbol(self, rank: Rank) -> str:
        """Get a symbol representation of the rank.
        
        Parameters
        ----------
        rank : Rank
            The rank to get symbol for
            
        Returns
        -------
        str
            Symbol representation of the rank
        """
        rank_symbols = {
            Rank.NINE: "9",
            Rank.TEN: "10",
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A"
        }
        return rank_symbols.get(rank, rank.value)
    
    def _get_suit_symbol(self, suit: Suit) -> str:
        """Get a symbol representation of the suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to get symbol for
            
        Returns
        -------
        str
            Symbol representation of the suit
        """
        suit_symbols = {
            Suit.HEARTS: "♥",
            Suit.DIAMONDS: "♦", 
            Suit.CLUBS: "♣",
            Suit.SPADES: "♠"
        }
        return suit_symbols.get(suit, suit.value.title())
    
    def _card_beats(self, card1: Card, card2: Card, trump_suit: Optional[Suit]) -> bool:
        """Determine if card1 beats card2."""
        # Check if cards are trump (including left bower)
        is_trump1 = self._is_trump_card(card1, trump_suit)
        is_trump2 = self._is_trump_card(card2, trump_suit)
        
        # Trump cards beat non-trump cards
        if is_trump1 and not is_trump2:
            return True
        if not is_trump1 and is_trump2:
            return False
            
        # If both are trump or both are non-trump, compare ranks
        if is_trump1 == is_trump2:
            return card1.rank.value > card2.rank.value
            
        # If one follows lead suit and other doesn't, lead suit wins
        if self.lead_suit:
            if card1.suit == self.lead_suit and card2.suit != self.lead_suit:
                return True
            if card2.suit == self.lead_suit and card1.suit != self.lead_suit:
                return False
                
        # Same suit, compare ranks
        if card1.suit == card2.suit:
            return card1.rank.value > card2.rank.value
            
        # Different suits, neither trump, neither follows lead - first card wins
        return False
    
    def _is_trump_card(self, card: Card, trump_suit: Optional[Suit]) -> bool:
        """Check if a card is a trump card (including left bower)."""
        if not trump_suit:
            return False
            
        # Right bower (jack of trump suit)
        if card.rank == Rank.JACK and card.suit == trump_suit:
            return True
            
        # Left bower (jack of same color as trump)
        if card.rank == Rank.JACK:
            if (trump_suit == Suit.HEARTS and card.suit == Suit.DIAMONDS) or \
               (trump_suit == Suit.DIAMONDS and card.suit == Suit.HEARTS) or \
               (trump_suit == Suit.CLUBS and card.suit == Suit.SPADES) or \
               (trump_suit == Suit.SPADES and card.suit == Suit.CLUBS):
                return True
                
        # Regular trump suit cards
        return card.suit == trump_suit


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