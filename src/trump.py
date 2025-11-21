"""Trump selection logic for Euchre game."""

from typing import List, Optional

from src.cards import Card, Suit
from src.players import Player


class TrumpSelector:
    """Handles the two-round trump selection process."""

    def __init__(self, players: List[Player]) -> None:
        """
        Initialize the trump selector.

        Parameters
        ----------
        players : List[Player]
            List of all players in the game.
        """
        self.players = players
        self.trump_suit: Optional[Suit] = None
        self.turned_card: Optional[Card] = None
        self.dealer_id: int = 0

    def select_trump(self, turned_card: Card, dealer_id: int) -> Optional[Suit]:
        """
        Run the two-round trump selection process.

        Parameters
        ----------
        turned_card : Card
            The card that was turned up.
        dealer_id : int
            The ID of the dealer.

        Returns
        -------
        Optional[Suit]
            The selected trump suit, or None if all passed (redeal).
        """
        self.turned_card = turned_card
        self.dealer_id = dealer_id
        self.trump_suit = None

        # Round 1: Order Up
        trump_suit = self._round_one_order_up()
        if trump_suit is not None:
            self.trump_suit = trump_suit
            return trump_suit

        # Round 2: Call Trump
        trump_suit = self._round_two_call_trump()
        if trump_suit is not None:
            self.trump_suit = trump_suit
            return trump_suit

        # All passed - redeal
        return None

    def _round_one_order_up(self) -> Optional[Suit]:
        """
        Execute Round 1: Order Up phase.

        Returns
        -------
        Optional[Suit]
            The trump suit if ordered up, None otherwise.
        """
        # Start with player left of dealer
        start_idx = (self.dealer_id + 1) % len(self.players)

        for i in range(len(self.players)):
            player_idx = (start_idx + i) % len(self.players)
            player = self.players[player_idx]

            # Dealer cannot order up in round 1
            if player_idx == self.dealer_id:
                continue

            decision = player.decide_order_up(self.turned_card, self.dealer_id, None)
            if decision:
                # Ordered up - dealer picks up and discards
                dealer = self.players[self.dealer_id]
                dealer.receive_card(self.turned_card)
                discard = dealer.choose_card_to_discard()
                dealer.remove_card(discard)
                return self.turned_card.suit

        return None

    def _round_two_call_trump(self) -> Optional[Suit]:
        """
        Execute Round 2: Call Trump phase.

        Implements "screw the dealer" rule: if all pass before dealer,
        dealer must choose a suit.

        Returns
        -------
        Optional[Suit]
            The trump suit if called, None otherwise (should not happen with screw the dealer).
        """
        # Start with player left of dealer
        start_idx = (self.dealer_id + 1) % len(self.players)
        forbidden_suit = self.turned_card.suit

        # Check all players before dealer
        for i in range(len(self.players) - 1):  # Exclude dealer
            player_idx = (start_idx + i) % len(self.players)
            player = self.players[player_idx]

            decision = player.decide_call_trump(self.turned_card, None, must_choose=False)
            if decision is not None and decision != forbidden_suit:
                return decision

        # All passed before dealer - "screw the dealer" rule
        dealer = self.players[self.dealer_id]
        decision = dealer.decide_call_trump(self.turned_card, None, must_choose=True)
        if decision is not None and decision != forbidden_suit:
            return decision

        # Fallback: dealer must choose something, even if it's the only option
        # This should not happen, but handle it gracefully
        suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
        for suit in suits:
            if suit != forbidden_suit:
                return suit

        return None

    def get_trump_suit(self) -> Optional[Suit]:
        """
        Get the currently selected trump suit.

        Returns
        -------
        Optional[Suit]
            The trump suit, or None if not selected.
        """
        return self.trump_suit

