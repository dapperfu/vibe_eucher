"""Trade-in handler for Midwest 9-10 trade-in variant."""

from typing import Dict, List, Optional

from eucher.cards import Card, Rank, Suit
from eucher.players import Player


class TradeInHandler:
    """Handles the Midwest 9-10 trade-in variant."""

    def __init__(self, players: List[Player], kitty: List[Card], dealer_id: int, tui=None) -> None:
        """
        Initialize the trade-in handler.

        Parameters
        ----------
        players : List[Player]
            List of all players in the game.
        kitty : List[Card]
            The three cards remaining in the deck (kitty).
        dealer_id : int
            The ID of the dealer.
        tui
            Optional TUI object for logging decisions.
        """
        self.players = players
        self.kitty = kitty.copy()
        self.dealer_id = dealer_id
        self.tui = tui

    def check_trade_in_eligible(self, player: Player) -> Optional[List[Card]]:
        """
        Check if a player has eligible cards for trade-in.

        A player is eligible if they have exactly 3 cards that are all 9s or 10s.
        (Note: Since there are only 2 cards per suit that are 9s or 10s, we check
        for 3 cards total across any suits, not necessarily the same suit.)

        Parameters
        ----------
        player : Player
            The player to check.

        Returns
        -------
        Optional[List[Card]]
            The three eligible cards if found, None otherwise.
        """
        # Find all 9s and 10s in the player's hand
        nines_and_tens = [card for card in player.hand if card.rank in (Rank.NINE, Rank.TEN)]

        # If player has exactly 3 cards that are 9s or 10s, they're eligible
        if len(nines_and_tens) == 3:
            return nines_and_tens

        return None

    def process_trade_in(self) -> Optional[Player]:
        """
        Process the trade-in phase.

        Checks players clockwise from dealer. The first eligible player
        who wants to trade-in will exchange their 3 trash cards for the 3 kitty cards.

        Returns
        -------
        Optional[Player]
            The player who traded-in, or None if no one traded-in.
        """
        # Safety check: ensure kitty has exactly 3 cards
        if len(self.kitty) != 3:
            return None

        # Start with player left of dealer (same order as trump selection)
        start_idx = (self.dealer_id + 1) % len(self.players)

        # Check each player clockwise from dealer
        for i in range(len(self.players)):
            player_idx = (start_idx + i) % len(self.players)
            player = self.players[player_idx]

            # Check if player is eligible
            eligible_cards = self.check_trade_in_eligible(player)
            if eligible_cards is None:
                continue

            # Ask player if they want to trade-in
            decision = player.decide_trade_in(eligible_cards)

            # Log decision
            if self.tui is not None and hasattr(self.tui, "log_trade_in_decision"):
                self.tui.log_trade_in_decision(player.name, decision, eligible_cards)

            if decision:
                # Perform the trade-in
                self._execute_trade_in(player, eligible_cards)

                # Log the trade-in
                if self.tui is not None and hasattr(self.tui, "log_trade_in"):
                    self.tui.log_trade_in(player.name, eligible_cards, self.kitty)

                return player

        return None

    def _execute_trade_in(self, player: Player, trash_cards: List[Card]) -> None:
        """
        Execute the trade-in exchange.

        Parameters
        ----------
        player : Player
            The player trading in.
        trash_cards : List[Card]
            The three cards to discard.
        """
        # Remove trash cards from player's hand
        for card in trash_cards:
            player.remove_card(card)

        # Add kitty cards to player's hand
        for card in self.kitty:
            player.receive_card(card)

        # Clear kitty (cards have been transferred)
        self.kitty.clear()

