"""Base class for all computer players in Euchre."""

from abc import ABC
from typing import List, Optional

from eucher.cards import Card, Rank, Suit
from eucher.players.base import PlayerProfile
from eucher.rules import RulesEngine


class ComputerPlayer(PlayerProfile, ABC):
    """Base class for all computer-controlled players."""

    def __init__(self) -> None:
        """Initialize the computer player."""
        self.rules = RulesEngine()

    def _count_trump_cards(self, hand: List[Card], trump_suit: Suit) -> int:
        """
        Count how many trump cards are in a hand.

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        int
            Number of trump cards.
        """
        count = 0
        for card in hand:
            # Right Bower (Jack of trump suit)
            if card.rank == Rank.JACK and card.suit == trump_suit:
                count += 1
            # Left Bower (Jack of same color as trump)
            elif card.rank == Rank.JACK:
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    count += 1
            # Regular trump suit card
            elif card.suit == trump_suit:
                count += 1
        return count

    def _has_bower(self, hand: List[Card], trump_suit: Suit) -> bool:
        """
        Check if hand has a bower (Right or Left).

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        bool
            True if hand has a bower, False otherwise.
        """
        for card in hand:
            if card.rank == Rank.JACK:
                # Right Bower
                if card.suit == trump_suit:
                    return True
                # Left Bower
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    return True
        return False

    def _has_strong_trump(self, hand: List[Card], trump_suit: Suit) -> bool:
        """
        Check if hand has strong trump cards (Ace or King).

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        bool
            True if hand has strong trump, False otherwise.
        """
        for card in hand:
            if card.suit == trump_suit and card.rank in [Rank.ACE, Rank.KING]:
                return True
        return False

    def _card_value(self, card: Card, trump_suit: Optional[Suit], led_suit: Optional[Suit] = None) -> int:
        """
        Get the value of a card for comparison.

        Parameters
        ----------
        card : Card
            The card to value.
        trump_suit : Optional[Suit]
            The trump suit.
        led_suit : Optional[Suit]
            The led suit (unused but kept for compatibility).

        Returns
        -------
        int
            Card value (higher is better).
        """
        if trump_suit is not None:
            # Right Bower (Jack of trump suit) is highest
            if card.rank == Rank.JACK and card.suit == trump_suit:
                return 7
            # Left Bower (Jack of same color as trump) is second
            if card.rank == Rank.JACK:
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    return 6
            # Regular trump suit cards
            if card.suit == trump_suit:
                rank_map = {
                    Rank.ACE: 5,
                    Rank.KING: 4,
                    Rank.QUEEN: 3,
                    Rank.TEN: 2,
                    Rank.NINE: 1,
                }
                return rank_map.get(card.rank, 0)
        return card.rank.value

    def _can_win_trick(
        self, card: Card, trick_cards: List[Card], led_suit: Suit, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Check if a card can win the current trick.

        Parameters
        ----------
        card : Card
            The card to check.
        trick_cards : List[Card]
            Cards already played.
        led_suit : Suit
            The led suit.
        trump_suit : Optional[Suit]
            The trump suit.

        Returns
        -------
        bool
            True if card can win, False otherwise.
        """
        if not trick_cards:
            return True

        current_winner = trick_cards[0]
        for c in trick_cards[1:]:
            if c.compare_to(current_winner, trump_suit, led_suit) > 0:
                current_winner = c

        return card.compare_to(current_winner, trump_suit, led_suit) > 0

    def _is_trump_card(self, card: Card, trump_suit: Suit) -> bool:
        """
        Check if a card is a trump card.

        Parameters
        ----------
        card : Card
            The card to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        bool
            True if card is trump, False otherwise.
        """
        # Right Bower
        if card.rank == Rank.JACK and card.suit == trump_suit:
            return True
        # Left Bower
        if card.rank == Rank.JACK:
            trump_card = Card(trump_suit, Rank.ACE)
            if card.is_same_color(trump_card):
                return True
        # Regular trump
        return card.suit == trump_suit

