"""Trump selection logic for Euchre game."""

from typing import List, Optional

from eucher.cards import Card, Suit
from eucher.players import Player


class TrumpSelector:
    """Handles the two-round trump selection process."""

    def __init__(self, players: List[Player], tui=None) -> None:
        """
        Initialize the trump selector.

        Parameters
        ----------
        players : List[Player]
            List of all players in the game.
        tui
            Optional TUI object for logging decisions.
        """
        self.players = players
        self.trump_suit: Optional[Suit] = None
        self.turned_card: Optional[Card] = None
        self.dealer_id: int = 0
        self.tui = tui
        self.trump_maker_name: Optional[str] = None
        # Decision history for post-hand inspection / logging
        self.order_up_decisions: List[tuple[str, bool]] = []
        self.call_trump_decisions: List[tuple[str, Optional[Suit]]] = []
        # Track "screw the dealer" rule
        self.screw_the_dealer_occurred: bool = False
        # Track "going alone" decision
        self.going_alone: bool = False
        self.going_alone_player_id: Optional[int] = None

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

        # Log round 1 start
        if self.tui is not None and hasattr(self.tui, "current_hand_log"):
            self.tui.current_hand_log.append("\nRound 1: Order Up")

        # Round 1: Order Up
        trump_suit = self._round_one_order_up()
        if trump_suit is not None:
            self.trump_suit = trump_suit
            # Log trump selected
            if self.tui is not None and hasattr(self.tui, "log_trump_selected"):
                if self.trump_maker_name:
                    self.tui.log_trump_selected(trump_suit, self.trump_maker_name)
            # Ask trump maker if they want to go alone
            self._ask_going_alone()
            return trump_suit

        # Log round 2 start
        if self.tui is not None:
            if hasattr(self.tui, "current_hand_log"):
                self.tui.current_hand_log.append("\nRound 2: Call Trump")

        # Round 2: Call Trump
        trump_suit = self._round_two_call_trump()
        if trump_suit is not None:
            self.trump_suit = trump_suit
            # Log trump selected
            if self.tui is not None and hasattr(self.tui, "log_trump_selected"):
                if self.trump_maker_name:
                    self.tui.log_trump_selected(trump_suit, self.trump_maker_name)
            # Ask trump maker if they want to go alone
            self._ask_going_alone()
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

            decision = player.decide_order_up(self.turned_card, self.dealer_id, None)
            
            # Log decision
            # Store and log decision
            self.order_up_decisions.append((player.name, decision))
            if self.tui is not None and hasattr(self.tui, "log_order_up_decision"):
                self.tui.log_order_up_decision(player.name, decision)
            
            if decision:
                # Ordered up - dealer picks up and discards
                self.trump_maker_name = player.name
                dealer = self.players[self.dealer_id]
                dealer.receive_card(self.turned_card)
                # Pass who ordered up so the message can be correct
                # If dealer ordered up themselves, pass None for ordered_up_by
                ordered_up_by = None if player_idx == self.dealer_id else self.trump_maker_name
                discard = dealer.choose_card_to_discard(self.turned_card, ordered_up_by)
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
            
            # Safety check: reject forbidden suit even if player returns it
            if decision == forbidden_suit:
                decision = None  # Treat as pass if player incorrectly returned forbidden suit
            
            # Log decision
            # Store and log decision
            self.call_trump_decisions.append((player.name, decision))
            if self.tui is not None and hasattr(self.tui, "log_call_trump_decision"):
                self.tui.log_call_trump_decision(player.name, decision)
            
            if decision is not None and decision != forbidden_suit:
                self.trump_maker_name = player.name
                return decision

        # All passed before dealer - "screw the dealer" rule
        dealer = self.players[self.dealer_id]
        self.screw_the_dealer_occurred = True  # Track that screw the dealer occurred
        decision = dealer.decide_call_trump(self.turned_card, None, must_choose=True)
        
        # Safety check: reject forbidden suit even if dealer returns it
        if decision == forbidden_suit:
            # Dealer must choose, so pick first available suit
            suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
            for suit in suits:
                if suit != forbidden_suit:
                    decision = suit
                    break
        
        # Log dealer's decision
        # Store and log dealer decision
        self.call_trump_decisions.append((dealer.name, decision))
        if self.tui is not None and hasattr(self.tui, "log_call_trump_decision"):
            self.tui.log_call_trump_decision(dealer.name, decision)
        
        if decision is not None and decision != forbidden_suit:
            self.trump_maker_name = dealer.name
            return decision

        # Fallback: dealer must choose something, even if it's the only option
        # This should not happen, but handle it gracefully
        suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
        for suit in suits:
            if suit != forbidden_suit:
                return suit

        return None

    def get_decision_history(self) -> dict:
        """
        Return a dict containing the recorded order-up and call-trump decisions for the hand.

        Returns
        -------
        dict
            Keys: `order_up` -> List[tuple[player_name, bool]], `call_trump` -> List[tuple[player_name, Optional[Suit]]]
        """
        return {"order_up": self.order_up_decisions.copy(), "call_trump": self.call_trump_decisions.copy()}

    def _ask_going_alone(self) -> None:
        """
        Ask the trump maker if they want to go alone.

        This is called after trump is selected, before the first card is led.
        """
        if self.trump_suit is None or self.trump_maker_name is None:
            return

        # Find the trump maker player
        trump_maker = None
        for player in self.players:
            if player.name == self.trump_maker_name:
                trump_maker = player
                break

        if trump_maker is None:
            return

        # Ask the player if they want to go alone
        decision = trump_maker.decide_going_alone(self.trump_suit)

        if decision:
            self.going_alone = True
            self.going_alone_player_id = trump_maker.player_id

            # Log going alone decision
            if self.tui is not None and hasattr(self.tui, "log_message"):
                self.tui.log_message(f"{trump_maker.name} is going alone!")
            if self.tui is not None and hasattr(self.tui, "current_hand_log"):
                self.tui.current_hand_log.append(f"{trump_maker.name} is going alone!")

    def get_trump_suit(self) -> Optional[Suit]:
        """
        Get the currently selected trump suit.

        Returns
        -------
        Optional[Suit]
            The trump suit, or None if not selected.
        """
        return self.trump_suit

