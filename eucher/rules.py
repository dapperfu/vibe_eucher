"""Rules engine for Euchre game validation."""

from typing import List, Optional

from src.cards import Card, Suit


class RulesEngine:
    """Engine for validating game rules and determining trick winners."""

    def __init__(self) -> None:
        """Initialize the rules engine."""
        pass

    def can_play_card(
        self,
        card: Card,
        hand: List[Card],
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
    ) -> bool:
        """
        Check if a card can be legally played.

        Parameters
        ----------
        card : Card
            The card to check.
        hand : List[Card]
            The player's current hand.
        led_suit : Optional[Suit]
            The suit that was led in the trick, if any.
        trump_suit : Optional[Suit]
            The current trump suit, if any.

        Returns
        -------
        bool
            True if the card can be played, False otherwise.
        """
        if card not in hand:
            return False

        # If no suit was led, any card can be played
        if led_suit is None:
            return True

        # Check if player can follow suit
        can_follow = self._can_follow_suit(hand, led_suit, trump_suit)

        if can_follow:
            # Must follow suit if possible
            return self._card_matches_led_suit(card, led_suit, trump_suit)
        # Can play any card if cannot follow suit
        return True

    def _can_follow_suit(self, hand: List[Card], led_suit: Suit, trump_suit: Optional[Suit]) -> bool:
        """
        Check if player can follow the led suit.

        Parameters
        ----------
        hand : List[Card]
            The player's hand.
        led_suit : Suit
            The suit that was led.
        trump_suit : Optional[Suit]
            The current trump suit.

        Returns
        -------
        bool
            True if player can follow suit, False otherwise.
        """
        for card in hand:
            if self._card_matches_led_suit(card, led_suit, trump_suit):
                return True
        return False

    def _card_matches_led_suit(
        self, card: Card, led_suit: Suit, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Check if a card matches the led suit.

        Parameters
        ----------
        card : Card
            The card to check.
        led_suit : Suit
            The suit that was led.
        trump_suit : Optional[Suit]
            The current trump suit.

        Returns
        -------
        bool
            True if card matches led suit, False otherwise.
        """
        if trump_suit is None:
            return card.suit == led_suit

        # With trump, check for bowers
        # Right Bower (Jack of trump) doesn't match non-trump led suit
        # Left Bower (Jack of same color) doesn't match non-trump led suit
        if card.rank.value == 11:  # Jack
            if card.suit == trump_suit:
                # Right Bower - only matches if trump was led
                return led_suit == trump_suit
            # Check if Left Bower
            trump_card = Card(trump_suit, card.rank)
            if card.is_same_color(trump_card):
                # Left Bower - only matches if trump was led
                return led_suit == trump_suit

        # Regular cards match their suit
        if card.suit == led_suit:
            return True

        # Trump cards match if trump was led
        if card.suit == trump_suit and led_suit == trump_suit:
            return True

        return False

    def determine_trick_winner(
        self, played_cards: List[Card], players: List[int], led_suit: Suit, trump_suit: Optional[Suit]
    ) -> int:
        """
        Determine the winner of a trick.

        Parameters
        ----------
        played_cards : List[Card]
            Cards played in the trick (in order of play).
        players : List[int]
            Player indices who played each card.
        led_suit : Suit
            The suit that was led.
        trump_suit : Optional[Suit]
            The current trump suit.

        Returns
        -------
        int
            Index of the winning player.

        Raises
        ------
        ValueError
            If the number of cards and players don't match.
        """
        if len(played_cards) != len(players):
            raise ValueError("Number of cards must match number of players")

        if not played_cards:
            raise ValueError("No cards played in trick")

        winner_idx = 0
        winner_card = played_cards[0]

        for i in range(1, len(played_cards)):
            card = played_cards[i]
            comparison = card.compare_to(winner_card, trump_suit, led_suit)
            if comparison > 0:
                winner_card = card
                winner_idx = i

        return players[winner_idx]

    def get_valid_plays(
        self, hand: List[Card], led_suit: Optional[Suit], trump_suit: Optional[Suit]
    ) -> List[Card]:
        """
        Get all valid cards that can be played from a hand.

        Parameters
        ----------
        hand : List[Card]
            The player's hand.
        led_suit : Optional[Suit]
            The suit that was led, if any.
        trump_suit : Optional[Suit]
            The current trump suit, if any.

        Returns
        -------
        List[Card]
            List of valid cards that can be played.
        """
        valid = []
        for card in hand:
            if self.can_play_card(card, hand, led_suit, trump_suit):
                valid.append(card)
        return valid

