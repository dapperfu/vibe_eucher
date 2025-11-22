"""Random player that makes all decisions randomly."""

import random
from typing import TYPE_CHECKING, List, Optional

from eucher.cards import Card, Suit
from eucher.players.computer.base import ComputerPlayer

if TYPE_CHECKING:
    from eucher.players.base import Player


class RandomPlayer(ComputerPlayer):
    """Random player that makes all decisions randomly (formerly RandomProfile)."""

    def __init__(self) -> None:
        """Initialize a random player."""
        super().__init__()

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Randomly decide whether to order up the turned card.

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
            Randomly True to order up, False to pass.
        """
        return random.choice([True, False])

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Randomly decide which suit to call as trump (or pass).

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up (cannot be chosen).
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit (cannot pass).

        Returns
        -------
        Optional[Suit]
            Randomly chosen suit, or None to pass (only if must_choose=False).
        """
        if trump_suit is not None:
            return None

        forbidden_suit = turned_card.suit
        available_suits = [suit for suit in Suit if suit != forbidden_suit]

        if must_choose:
            # Must choose a suit
            return random.choice(available_suits)
        else:
            # Can pass or choose a suit
            choices: List[Optional[Suit]] = [None] + available_suits
            return random.choice(choices)

    def choose_card_to_discard(self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None) -> Card:
        """
        Randomly choose a card to discard.

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
            A randomly chosen card from the hand.
        """
        if not player.hand:
            raise ValueError("Player has no cards to discard")
        return random.choice(player.hand)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Randomly choose a card to play from valid plays.

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
            A randomly chosen valid card to play.
        """
        valid_cards = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)

        if not valid_cards:
            # Fallback if no valid cards (should not happen)
            return player.hand[0]

        return random.choice(valid_cards)
