"""Card and Deck classes for Euchre game."""

from enum import Enum
from typing import List, Optional


class Suit(Enum):
    """Card suits in Euchre."""

    HEARTS = "Hearts"
    DIAMONDS = "Diamonds"
    CLUBS = "Clubs"
    SPADES = "Spades"

    def unicode_symbol(self) -> str:
        """
        Return Unicode symbol for the suit.

        Returns
        -------
        str
            Unicode symbol for the suit.
        """
        symbol_map = {
            Suit.HEARTS: "♥",
            Suit.DIAMONDS: "♦",
            Suit.CLUBS: "♣",
            Suit.SPADES: "♠",
        }
        return symbol_map[self]


class Rank(Enum):
    """Card ranks in Euchre (9, 10, J, Q, K, A)."""

    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class Card:
    """Represents a single playing card in Euchre."""

    def __init__(self, suit: Suit, rank: Rank) -> None:
        """
        Initialize a card.

        Parameters
        ----------
        suit : Suit
            The suit of the card.
        rank : Rank
            The rank of the card.
        """
        self.suit: Suit = suit
        self.rank: Rank = rank

    def __eq__(self, other: object) -> bool:
        """
        Check if two cards are equal.

        Parameters
        ----------
        other : object
            The other card to compare.

        Returns
        -------
        bool
            True if cards are equal, False otherwise.
        """
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self) -> int:
        """
        Return hash of the card.

        Returns
        -------
        int
            Hash value of the card.
        """
        return hash((self.suit, self.rank))

    def __repr__(self) -> str:
        """
        Return string representation of the card with Unicode suit symbols.

        Returns
        -------
        str
            String representation of the card (e.g., "K❤️", "A♠️").
        """
        rank_str = {
            Rank.NINE: "9",
            Rank.TEN: "10",
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A",
        }[self.rank]
        suit_symbol = self.suit.unicode_symbol()
        return f"{rank_str}{suit_symbol}"

    def __str__(self) -> str:
        """
        Return human-readable string representation.

        Returns
        -------
        str
            Human-readable string representation.
        """
        rank_str = {
            Rank.NINE: "9",
            Rank.TEN: "10",
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A",
        }[self.rank]
        return f"{rank_str} of {self.suit.value}"

    def is_same_color(self, other: "Card") -> bool:
        """
        Check if card is the same color as another card.

        Parameters
        ----------
        other : Card
            The other card to compare.

        Returns
        -------
        bool
            True if cards are the same color, False otherwise.
        """
        red_suits = {Suit.HEARTS, Suit.DIAMONDS}
        black_suits = {Suit.CLUBS, Suit.SPADES}
        return (self.suit in red_suits and other.suit in red_suits) or (
            self.suit in black_suits and other.suit in black_suits
        )

    def compare_to(
        self, other: "Card", trump_suit: Optional[Suit] = None, led_suit: Optional[Suit] = None
    ) -> int:
        """
        Compare this card to another card.

        Parameters
        ----------
        other : Card
            The other card to compare to.
        trump_suit : Optional[Suit]
            The current trump suit, if any.
        led_suit : Optional[Suit]
            The suit that was led in the trick, if any.

        Returns
        -------
        int
            -1 if this card is less than other, 0 if equal, 1 if greater.
        """
        if trump_suit is None:
            # No trump: standard comparison
            if self.suit != other.suit:
                return 0  # Different suits, can't compare without trump
            if self.rank.value < other.rank.value:
                return -1
            if self.rank.value > other.rank.value:
                return 1
            return 0

        # With trump: check if either is trump
        self_is_trump = self._is_trump(trump_suit)
        other_is_trump = other._is_trump(trump_suit)

        # Trump always beats non-trump
        if self_is_trump and not other_is_trump:
            return 1
        if not self_is_trump and other_is_trump:
            return -1

        # Both trump or both non-trump
        if self_is_trump:
            return self._compare_trump(other, trump_suit)
        # Both non-trump
        if led_suit is not None:
            # Must follow led suit if possible
            if self.suit == led_suit and other.suit != led_suit:
                return 1
            if self.suit != led_suit and other.suit == led_suit:
                return -1
            if self.suit == led_suit and other.suit == led_suit:
                # Same suit, compare ranks
                if self.rank.value < other.rank.value:
                    return -1
                if self.rank.value > other.rank.value:
                    return 1
                return 0
        # Can't compare non-trump of different suits
        return 0

    def _is_trump(self, trump_suit: Suit) -> bool:
        """
        Check if this card is a trump card.

        Parameters
        ----------
        trump_suit : Suit
            The current trump suit.

        Returns
        -------
        bool
            True if card is trump, False otherwise.
        """
        # Right Bower (Jack of trump suit)
        if self.rank == Rank.JACK and self.suit == trump_suit:
            return True
        # Left Bower (Jack of same color as trump)
        if self.rank == Rank.JACK:
            trump_card = Card(trump_suit, Rank.ACE)  # Dummy card for color check
            if self.is_same_color(trump_card):
                return True
        # Regular trump suit card
        return self.suit == trump_suit

    def _compare_trump(self, other: "Card", trump_suit: Suit) -> int:
        """
        Compare two trump cards.

        Parameters
        ----------
        other : Card
            The other trump card to compare.
        trump_suit : Suit
            The current trump suit.

        Returns
        -------
        int
            -1 if this card is less than other, 0 if equal, 1 if greater.
        """
        # Get trump ranks
        self_rank = self._get_trump_rank(trump_suit)
        other_rank = other._get_trump_rank(trump_suit)

        if self_rank < other_rank:
            return -1
        if self_rank > other_rank:
            return 1
        return 0

    def _get_trump_rank(self, trump_suit: Suit) -> int:
        """
        Get the rank of this card in trump ordering.

        Parameters
        ----------
        trump_suit : Suit
            The current trump suit.

        Returns
        -------
        int
            Trump rank (higher is better).
        """
        # Right Bower (Jack of trump suit) is highest
        if self.rank == Rank.JACK and self.suit == trump_suit:
            return 7
        # Left Bower (Jack of same color as trump) is second
        if self.rank == Rank.JACK:
            trump_card = Card(trump_suit, Rank.ACE)
            if self.is_same_color(trump_card):
                return 6
        # Regular trump suit cards
        if self.suit == trump_suit:
            rank_map = {
                Rank.ACE: 5,
                Rank.KING: 4,
                Rank.QUEEN: 3,
                Rank.TEN: 2,
                Rank.NINE: 1,
            }
            return rank_map.get(self.rank, 0)
        return 0


class Deck:
    """Represents a 24-card Euchre deck."""

    def __init__(self) -> None:
        """Initialize a new Euchre deck with 24 cards."""
        self.cards: List[Card] = []
        self._create_deck()

    def _create_deck(self) -> None:
        """Create the 24-card Euchre deck."""
        ranks = [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]
        suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]

        self.cards = [Card(suit, rank) for suit in suits for rank in ranks]

    def shuffle(self) -> None:
        """Shuffle the deck."""
        import random

        random.shuffle(self.cards)

    def deal(self, num_cards: int) -> List[Card]:
        """
        Deal a specified number of cards from the deck.

        Parameters
        ----------
        num_cards : int
            Number of cards to deal.

        Returns
        -------
        List[Card]
            List of dealt cards.

        Raises
        ------
        ValueError
            If there are not enough cards in the deck.
        """
        if len(self.cards) < num_cards:
            raise ValueError(f"Not enough cards in deck. Requested {num_cards}, have {len(self.cards)}")
        dealt = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt

    def draw_one(self) -> Card:
        """
        Draw a single card from the deck.

        Returns
        -------
        Card
            The drawn card.

        Raises
        ------
        ValueError
            If the deck is empty.
        """
        if not self.cards:
            raise ValueError("Cannot draw from empty deck")
        return self.cards.pop(0)

    def __len__(self) -> int:
        """
        Return the number of cards remaining in the deck.

        Returns
        -------
        int
            Number of cards in the deck.
        """
        return len(self.cards)

