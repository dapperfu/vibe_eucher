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
    
    def unicode_str(self) -> str:
        """Enhanced Unicode string representation with emoji modifiers."""
        rank_map = {
            Rank.NINE: "9",
            Rank.TEN: "10", 
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A"
        }
        suit_map = {
            Suit.HEARTS: "♥️",
            Suit.DIAMONDS: "♦️",
            Suit.CLUBS: "♣️",
            Suit.SPADES: "♠️"
        }
        
        rank_str = rank_map.get(self.rank, str(self.rank.value))
        suit_str = suit_map.get(self.suit, self.suit.name[0].upper())
        
        if self.is_trump:
            return f"{rank_str}{suit_str}*"
        else:
            return f"{rank_str}{suit_str}"
    
    def compact_str(self) -> str:
        """Compact string representation for tight displays."""
        rank_map = {
            Rank.NINE: "9",
            Rank.TEN: "10", 
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A"
        }
        suit_map = {
            Suit.HEARTS: "♥️",
            Suit.DIAMONDS: "♦️",
            Suit.CLUBS: "♣️",
            Suit.SPADES: "♠️"
        }
        
        rank_str = rank_map.get(self.rank, str(self.rank.value))
        suit_str = suit_map.get(self.suit, self.suit.name[0].upper())
        
        return f"{rank_str}{suit_str}"
    
    @classmethod
    def get_suit_symbol(cls, suit: Suit, use_emoji: bool = True) -> str:
        """Get the Unicode symbol for a suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to get the symbol for
        use_emoji : bool
            Whether to use emoji modifiers (♥️ vs ♥)
            
        Returns
        -------
        str
            The Unicode symbol for the suit
        """
        if use_emoji:
            suit_map = {
                Suit.HEARTS: "♥️",
                Suit.DIAMONDS: "♦️",
                Suit.CLUBS: "♣️",
                Suit.SPADES: "♠️"
            }
        else:
            suit_map = {
                Suit.HEARTS: "♥",
                Suit.DIAMONDS: "♦",
                Suit.CLUBS: "♣",
                Suit.SPADES: "♠"
            }
        
        return suit_map.get(suit, suit.name[0].upper())
    
    @classmethod
    def get_rank_symbol(cls, rank: Rank) -> str:
        """Get the display symbol for a rank.
        
        Parameters
        ----------
        rank : Rank
            The rank to get the symbol for
            
        Returns
        -------
        str
            The display symbol for the rank
        """
        rank_map = {
            Rank.NINE: "9",
            Rank.TEN: "10", 
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A"
        }
        
        return rank_map.get(rank, str(rank.value))
    
    @classmethod
    def format_hand(cls, cards: List["Card"], use_emoji: bool = True, show_trump: bool = False) -> str:
        """Format a hand of cards for display.
        
        Parameters
        ----------
        cards : List[Card]
            List of cards to format
        use_emoji : bool
            Whether to use emoji modifiers
        show_trump : bool
            Whether to show trump indicators
            
        Returns
        -------
        str
            Formatted hand string
        """
        if not cards:
            return "Empty hand"
        
        formatted_cards = []
        for card in cards:
            if show_trump and card.is_trump:
                formatted_cards.append(f"{cls.get_rank_symbol(card.rank)}{cls.get_suit_symbol(card.suit, use_emoji)}*")
            else:
                formatted_cards.append(f"{cls.get_rank_symbol(card.rank)}{cls.get_suit_symbol(card.suit, use_emoji)}")
        
        return " ".join(formatted_cards)
    
    @property
    def value(self) -> int:
        """Get the numeric value of the card."""
        return self.rank.value
    
    @property
    def is_face_card(self) -> bool:
        """Check if the card is a face card (J, Q, K)."""
        return self.rank in [Rank.JACK, Rank.QUEEN, Rank.KING]
    
    @property
    def is_high_card(self) -> bool:
        """Check if the card is a high card (10, J, Q, K, A)."""
        return self.rank.value >= 10
    
    @property
    def is_low_card(self) -> bool:
        """Check if the card is a low card (9)."""
        return self.rank.value == 9
    
    def is_trump_card(self, trump_suit: Optional[Suit]) -> bool:
        """Check if this card is a trump card for the given trump suit."""
        if not trump_suit:
            return False
            
        # Right bower (jack of trump suit)
        if self.rank == Rank.JACK and self.suit == trump_suit:
            return True
            
        # Left bower (jack of same color as trump)
        if self.rank == Rank.JACK:
            if (trump_suit == Suit.HEARTS and self.suit == Suit.DIAMONDS) or \
               (trump_suit == Suit.DIAMONDS and self.suit == Suit.HEARTS) or \
               (trump_suit == Suit.CLUBS and self.suit == Suit.SPADES) or \
               (trump_suit == Suit.SPADES and self.suit == Suit.CLUBS):
                return True
                
        # Regular trump suit cards
        return self.suit == trump_suit
    
    def beats(self, other: "Card", trump_suit: Optional[Suit], lead_suit: Optional[Suit]) -> bool:
        """Determine if this card beats another card in a trick.
        
        Parameters
        ----------
        other : Card
            The card to compare against
        trump_suit : Optional[Suit]
            The current trump suit
        lead_suit : Optional[Suit]
            The suit that was led in the trick
            
        Returns
        -------
        bool
            True if this card beats the other card
        """
        if not isinstance(other, Card):
            return False
            
        # Get trump values for both cards
        this_trump_value = self.get_trump_value(trump_suit)
        other_trump_value = other.get_trump_value(trump_suit)
        
        # If both cards are trump, compare their trump values
        if this_trump_value > 0 and other_trump_value > 0:
            return this_trump_value > other_trump_value
        
        # If only this card is trump, it wins
        if this_trump_value > 0 and other_trump_value == 0:
            return True
        
        # If only other card is trump, it wins
        if this_trump_value == 0 and other_trump_value > 0:
            return False
        
        # If neither card is trump, check if they follow lead suit
        if lead_suit:
            this_follows_lead = self.suit == lead_suit
            other_follows_lead = other.suit == lead_suit
            
            # If one follows lead and other doesn't, lead suit wins
            if this_follows_lead and not other_follows_lead:
                return True
            if other_follows_lead and not this_follows_lead:
                return False
            
            # If both follow lead or neither follows, compare ranks
            if this_follows_lead == other_follows_lead:
                return self.rank.value > other.rank.value
        
        # If no lead suit or same suit, compare ranks
        if self.suit == other.suit:
            return self.rank.value > other.rank.value
        
        # Different suits, neither trump, neither follows lead - first card wins
        return False
    
    def get_trump_value(self, trump_suit: Optional[Suit]) -> int:
        """Get the trump value of this card for comparison.
        
        Trump hierarchy:
        1. Right Bower (Jack of trump suit) = 7
        2. Left Bower (Jack of same color) = 6  
        3. Ace of trump = 5
        4. King of trump = 4
        5. Queen of trump = 3
        6. 10 of trump = 2
        7. 9 of trump = 1
        8. Non-trump cards = 0
        
        Parameters
        ----------
        trump_suit : Optional[Suit]
            The current trump suit
            
        Returns
        -------
        int
            Trump value (higher is better)
        """
        if not trump_suit:
            return 0
            
        # Right Bower (Jack of trump suit)
        if self.rank == Rank.JACK and self.suit == trump_suit:
            return 7
            
        # Left Bower (Jack of same color as trump)
        if self.rank == Rank.JACK:
            if (trump_suit == Suit.HEARTS and self.suit == Suit.DIAMONDS) or \
               (trump_suit == Suit.DIAMONDS and self.suit == Suit.HEARTS) or \
               (trump_suit == Suit.CLUBS and self.suit == Suit.SPADES) or \
               (trump_suit == Suit.SPADES and self.suit == Suit.CLUBS):
                return 6
        
        # Regular trump suit cards
        if self.suit == trump_suit:
            if self.rank == Rank.ACE:
                return 5
            elif self.rank == Rank.KING:
                return 4
            elif self.rank == Rank.QUEEN:
                return 3
            elif self.rank == Rank.TEN:
                return 2
            elif self.rank == Rank.NINE:
                return 1
        
        # Not a trump card
        return 0
    
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
    winner: Optional["Player"] = None
    
    def __post_init__(self) -> None:
        """Initialize cards_played if None."""
        if self.cards_played is None:
            self.cards_played = []
    
    def add_card(self, player: "Player", card: Card) -> None:
        """Add a card to the trick."""
        if not self.lead_suit:
            self.lead_suit = card.suit
        self.cards_played.append((player, card))
    
    def is_complete(self) -> bool:
        """Check if the trick is complete (all 4 players have played).
        
        Returns
        -------
        bool
            True if the trick is complete
        """
        return len(self.cards_played) == 4
    
    @property
    def num_cards_played(self) -> int:
        """Get the number of cards played so far."""
        return len(self.cards_played)
    
    @property
    def is_started(self) -> bool:
        """Check if the trick has started (at least one card played)."""
        return len(self.cards_played) > 0
    
    @property
    def current_leader(self) -> Optional["Player"]:
        """Get the current leader of the trick."""
        if not self.cards_played:
            return None
        return self.cards_played[0][0]
    
    @property
    def current_winning_card(self) -> Optional[Card]:
        """Get the current winning card in the trick."""
        if not self.cards_played:
            return None
        
        winning_card = self.cards_played[0][1]
        for _, card in self.cards_played[1:]:
            if self._card_beats(card, winning_card, None):  # No trump context yet
                winning_card = card
        return winning_card
    
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
    
    def get_players_who_played(self) -> List["Player"]:
        """Get list of players who have played in this trick.
        
        Returns
        -------
        List[Player]
            List of players who have played
        """
        return [player for player, _ in self.cards_played]
    
    def get_cards_played(self) -> List[Card]:
        """Get list of cards played in this trick.
        
        Returns
        -------
        List[Card]
            List of cards played
        """
        return [card for _, card in self.cards_played]
    
    def get_player_card(self, player: "Player") -> Optional[Card]:
        """Get the card played by a specific player.
        
        Parameters
        ----------
        player : Player
            The player to get the card for
            
        Returns
        -------
        Optional[Card]
            The card played by the player, or None if not played yet
        """
        for p, card in self.cards_played:
            if p == player:
                return card
        return None
    
    def has_player_played(self, player: "Player") -> bool:
        """Check if a specific player has played in this trick.
        
        Parameters
        ----------
        player : Player
            The player to check
            
        Returns
        -------
        bool
            True if the player has played
        """
        return any(p == player for p, _ in self.cards_played)
    
    def get_legal_plays_for_player(self, player: "Player", trump_suit: Optional[Suit] = None) -> List[Card]:
        """Get legal cards a player can play in this trick.
        
        Parameters
        ----------
        player : Player
            The player to get legal plays for
        trump_suit : Optional[Suit]
            The current trump suit
            
        Returns
        -------
        List[Card]
            List of legal cards to play
        """
        if self.has_player_played(player):
            return []  # Already played
        
        if not self.lead_suit:
            return player.hand.copy()  # Leading, can play anything
        
        # Must follow suit if possible
        cards_of_lead_suit = player.get_cards_of_suit(self.lead_suit)
        if cards_of_lead_suit:
            return cards_of_lead_suit
        
        # Can't follow suit, can play anything
        return player.hand.copy()
    
    def reset(self) -> None:
        """Reset the trick to initial state."""
        self.lead_suit = None
        self.cards_played.clear()
        self.winner = None
    
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
        if trump_suit and card.is_trump_card(trump_suit):
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
        return rank_symbols.get(rank, str(rank.value))
    
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
        """Determine if card1 beats card2 using proper trump hierarchy."""
        # Use the Card class's beats method for consistent logic
        return card1.beats(card2, trump_suit, self.lead_suit)
    
    def _is_trump_card(self, card: Card, trump_suit: Optional[Suit]) -> bool:
        """Check if a card is a trump card (including left bower)."""
        return card.is_trump_card(trump_suit)


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
        self.score: int = 0
        self.is_dealer: bool = False
        self.team: Optional[int] = None
        
    def add_card(self, card: Card) -> None:
        """Add a card to the player's hand.
        
        Parameters
        ----------
        card : Card
            The card to add to the hand
        """
        self.hand.append(card)
        
    def remove_card(self, card: Card) -> bool:
        """Remove a card from the player's hand.
        
        Parameters
        ----------
        card : Card
            The card to remove from the hand
            
        Returns
        -------
        bool
            True if card was removed, False if not found
        """
        try:
            self.hand.remove(card)
            return True
        except ValueError:
            return False
            
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
    
    @property
    def has_cards(self) -> bool:
        """Check if the player has any cards."""
        return len(self.hand) > 0
    
    @property
    def hand_is_full(self) -> bool:
        """Check if the player's hand is full (5 cards)."""
        return len(self.hand) == 5
    
    @property
    def hand_is_empty(self) -> bool:
        """Check if the player's hand is empty."""
        return len(self.hand) == 0
        
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
    
    def get_trump_cards(self, trump_suit: Suit) -> List[Card]:
        """Get all trump cards from the player's hand.
        
        Parameters
        ----------
        trump_suit : Suit
            The current trump suit
            
        Returns
        -------
        List[Card]
            List of trump cards (including left bower)
        """
        trump_cards = []
        for card in self.hand:
            if card.is_trump_card(trump_suit):
                trump_cards.append(card)
        return trump_cards
    
    def get_high_cards(self, threshold: int = 10) -> List[Card]:
        """Get high-value cards from the player's hand.
        
        Parameters
        ----------
        threshold : int
            Minimum rank value to consider "high"
            
        Returns
        -------
        List[Card]
            List of high-value cards
        """
        return [card for card in self.hand if card.rank.value >= threshold]
    
    def get_low_cards(self, threshold: int = 9) -> List[Card]:
        """Get low-value cards from the player's hand.
        
        Parameters
        ----------
        threshold : int
            Maximum rank value to consider "low"
            
        Returns
        -------
        List[Card]
            List of low-value cards
        """
        return [card for card in self.hand if card.rank.value <= threshold]
    
    def can_follow_suit(self, lead_suit: Suit) -> bool:
        """Check if the player can follow the lead suit.
        
        Parameters
        ----------
        lead_suit : Suit
            The suit that was led
            
        Returns
        -------
        bool
            True if the player has cards of the lead suit
        """
        return self.has_suit(lead_suit)
    
    def get_legal_plays(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> List[Card]:
        """Get all legal cards the player can play.
        
        Parameters
        ----------
        lead_suit : Optional[Suit]
            The suit that was led (None if leading)
        trump_suit : Optional[Suit]
            The current trump suit
            
        Returns
        -------
        List[Card]
            List of legal cards to play
        """
        if not lead_suit:
            return self.hand.copy()
        
        # Must follow suit if possible
        cards_of_lead_suit = self.get_cards_of_suit(lead_suit)
        if cards_of_lead_suit:
            return cards_of_lead_suit
        
        # Can play any card if can't follow suit
        return self.hand.copy()
    
    def evaluate_hand_strength(self, trump_suit: Optional[Suit] = None) -> float:
        """Evaluate the overall strength of the player's hand.
        
        Parameters
        ----------
        trump_suit : Optional[Suit]
            The current trump suit
            
        Returns
        -------
        float
            Hand strength score (higher is better)
        """
        if not self.hand:
            return 0.0
        
        score = 0.0
        
        # Base score from card values
        for card in self.hand:
            score += card.rank.value
        
        # Bonus for trump cards
        if trump_suit:
            trump_cards = self.get_trump_cards(trump_suit)
            score += len(trump_cards) * 5.0
        
        # Bonus for high cards
        high_cards = self.get_high_cards(10)
        score += len(high_cards) * 2.0
        
        # Bonus for face cards
        face_cards = [card for card in self.hand if card.is_face_card]
        score += len(face_cards) * 1.5
        
        return score
    
    def set_team(self, team: int) -> None:
        """Set the player's team.
        
        Parameters
        ----------
        team : int
            Team number (0 or 1)
        """
        if team not in [0, 1]:
            raise ValueError("Team must be 0 or 1")
        self.team = team
    
    def get_team_name(self) -> str:
        """Get the player's team name.
        
        Returns
        -------
        str
            Team name ("Team 1" or "Team 2")
        """
        if self.team is None:
            return "No Team"
        return f"Team {self.team + 1}"
    
    def reset_round_stats(self) -> None:
        """Reset round-specific statistics."""
        self.tricks_won = 0
        self.clear_hand()
    
    def reset_game_stats(self) -> None:
        """Reset game-specific statistics."""
        self.score = 0
        self.is_dealer = False
        self.reset_round_stats()
        
    def __str__(self) -> str:
        """String representation of the player."""
        return f"{self.name} ({self.player_type.value})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the player."""
        team_info = f", team={self.get_team_name()}" if self.team is not None else ""
        dealer_info = ", dealer" if self.is_dealer else ""
        return f"Player(name='{self.name}', player_type={self.player_type.value}, hand_size={len(self.hand)}{team_info}{dealer_info})"


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