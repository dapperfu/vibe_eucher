"""Heuristic-based player using rule-based logic."""

from typing import TYPE_CHECKING, List, Optional

from eucher.cards import Card, Suit
from eucher.players.computer.base import ComputerPlayer

if TYPE_CHECKING:
    from eucher.players.base import Player


class HeuristicPlayer(ComputerPlayer):
    """Heuristic-based player using rule-based logic (formerly SimpleRuleBasedProfile)."""

    def __init__(self) -> None:
        """Initialize a heuristic-based player."""
        super().__init__()

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Decide whether to order up using simple rules.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up.
        dealer_id : int
            The ID of the dealer.
        trump_suit : Optional[Suit]
            Current trump suit if already determined.

        Returns
        -------
        bool
            True to order up, False to pass.
        """
        if trump_suit is not None:
            return False

        potential_trump = turned_card.suit
        trump_count = self._count_trump_cards(player.hand, potential_trump)

        # Order up if we have 2+ trump cards
        if trump_count >= 2:
            return True

        # Order up if we have Right or Left Bower
        if self._has_bower(player.hand, potential_trump):
            return True

        return False

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Decide which suit to call as trump using simple rules.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up.
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit.

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        if trump_suit is not None:
            if must_choose:
                # Must return something, but trump already set
                return trump_suit
            return None

        forbidden_suit = turned_card.suit
        suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]

        best_suit = None
        best_count = 0

        for suit in suits:
            if suit == forbidden_suit:
                continue
            count = self._count_trump_cards(player.hand, suit)
            if count > best_count:
                best_count = count
                best_suit = suit

        # Call trump if we have at least 2 cards of that suit
        if best_count >= 2:
            return best_suit

        # Call trump if we have a bower
        if best_suit is not None and self._has_bower(player.hand, best_suit):
            return best_suit

        # If must choose, return best suit even if weak
        if must_choose and best_suit is not None:
            return best_suit

        return None

    def choose_card_to_discard(self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None) -> Card:
        """
        Choose a card to discard using simple rules.

        Parameters
        ----------
        player : Player
            The dealer player.
        turned_card : Optional[Card]
            The card that was ordered up, if available.
        ordered_up_by : Optional[str]
            Name of the player who ordered up, if available.

        Returns
        -------
        Card
            The card to discard.
        """
        if not player.hand:
            raise ValueError("Player has no cards to discard")
        # Discard the lowest card
        return min(player.hand, key=lambda c: c.rank.value)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Choose a card to play using simple rules.

        Parameters
        ----------
        player : Player
            The player making the decision.
        led_suit : Optional[Suit]
            The suit that was led, if any.
        trump_suit : Optional[Suit]
            The current trump suit, if any.
        trick_cards : List[Card]
            Cards already played in the trick.
        trick_player_ids : List[int]
            Player IDs who played each card in trick_cards (same order).

        Returns
        -------
        Card
            The card to play.
        """
        valid_cards = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)
        if not valid_cards:
            return player.hand[0]

        # If leading, play highest card
        if led_suit is None:
            return max(valid_cards, key=lambda c: self._card_value(c, trump_suit))

        # Try to win with lowest winning card, or play lowest
        winning_cards = []
        for card in valid_cards:
            if self._can_win_trick(card, trick_cards, led_suit, trump_suit):
                winning_cards.append(card)

        if winning_cards:
            return min(winning_cards, key=lambda c: self._card_value(c, trump_suit))

        # Can't win, play lowest
        return min(valid_cards, key=lambda c: self._card_value(c, trump_suit))
