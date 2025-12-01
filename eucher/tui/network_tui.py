"""Network-aware TUI for multiplayer Euchre."""

from typing import Dict, List, Optional, TYPE_CHECKING

from eucher.cards import Card, Suit
from eucher.players import Player
from eucher.tui import TextTUI
from eucher.network.protocol import DecisionType, serialize_card, serialize_cards, serialize_suit

if TYPE_CHECKING:
    from eucher.network.client import EuchreClient


class NetworkTUI(TextTUI):
    """Network-aware TUI that sends decisions over network and receives game state."""

    def __init__(
        self,
        client: "EuchreClient",
        player_id: int,
        **kwargs,
    ) -> None:
        """
        Initialize network TUI.

        Parameters
        ----------
        client : EuchreClient
            Network client connection.
        player_id : int
            This player's ID.
        **kwargs
            Additional arguments passed to TextTUI.
        """
        super().__init__(**kwargs)
        self.client = client
        self.player_id = player_id
        self._waiting_for_decision = False

    def _wait_for_decision_request(self) -> Optional[Dict]:
        """
        Wait for a decision request from the server.

        Returns
        -------
        Optional[Dict]
            Decision request message, or None if error.
        """
        while True:
            message = self.client.receive_message()
            if message is None:
                return None

            msg_type = message.get("type")
            if msg_type == "DECISION_REQUEST":
                return message
            elif msg_type == "GAME_STATE":
                # Update local game state from server
                self._update_game_state(message["data"])
            elif msg_type == "ERROR":
                print(f"Error from server: {message['data'].get('message', 'Unknown error')}")
                return None

    def _update_game_state(self, state_data: Dict) -> None:
        """
        Update local game state from server broadcast.

        Parameters
        ----------
        state_data : Dict
            Game state data from server.
        """
        # Update scores
        if "scores" in state_data:
            self.team_scores = tuple(state_data["scores"])

        # Update current trick
        if "current_trick_cards" in state_data:
            from eucher.network.protocol import deserialize_cards
            self.current_trick_cards = deserialize_cards(state_data["current_trick_cards"])
        if "current_trick_player_ids" in state_data:
            self.current_trick_player_ids = state_data["current_trick_player_ids"]

        # Update dealer
        if "dealer_id" in state_data:
            self.dealer_id = state_data["dealer_id"]

        # Update trump suit
        if "trump_suit" in state_data:
            from eucher.network.protocol import deserialize_suit
            self.trump_suit = deserialize_suit(state_data["trump_suit"])

        # Update players if provided
        if "players" in state_data and self.players:
            for player_data in state_data["players"]:
                pid = player_data["player_id"]
                if pid < len(self.players):
                    # Update player hand if it's this player
                    if pid == self.player_id and "hand" in player_data:
                        from eucher.network.protocol import deserialize_cards
                        self.players[pid].hand = deserialize_cards(player_data["hand"])

    def get_order_up_decision(
        self, player: Player, turned_card: Card, dealer_id: int
    ) -> bool:
        """
        Get order up decision (network version).

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The turned card.
        dealer_id : int
            Dealer ID.

        Returns
        -------
        bool
            True to order up, False to pass.
        """
        # Wait for decision request from server
        request = self._wait_for_decision_request()
        if request is None:
            raise ConnectionError("Lost connection to server")

        # Get decision using parent class method
        decision = super().get_order_up_decision(player, turned_card, dealer_id)

        # Send decision to server
        self.client.send_decision(DecisionType.ORDER_UP, decision)
        return decision

    def get_call_trump_decision(
        self,
        player: Player,
        turned_card: Card,
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Get call trump decision (network version).

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The turned card.
        must_choose : bool
            Whether player must choose a suit.

        Returns
        -------
        Optional[Suit]
            Selected trump suit, or None to pass.
        """
        # Wait for decision request from server
        request = self._wait_for_decision_request()
        if request is None:
            raise ConnectionError("Lost connection to server")

        # Get decision using parent class method
        decision = super().get_call_trump_decision(player, turned_card, must_choose)

        # Send decision to server
        self.client.send_decision(DecisionType.CALL_TRUMP, decision)
        return decision

    def get_discard_decision(
        self,
        player: Player,
        turned_card: Optional[Card] = None,
        ordered_up_by: Optional[str] = None,
    ) -> Card:
        """
        Get discard decision (network version).

        Parameters
        ----------
        player : Player
            The dealer player.
        turned_card : Optional[Card]
            The turned card.
        ordered_up_by : Optional[str]
            Name of player who ordered up.

        Returns
        -------
        Card
            Card to discard.
        """
        # Wait for decision request from server
        request = self._wait_for_decision_request()
        if request is None:
            raise ConnectionError("Lost connection to server")

        # Get decision using parent class method
        decision = super().get_discard_decision(player, turned_card, ordered_up_by)

        # Send decision to server
        self.client.send_decision(DecisionType.DISCARD, decision)
        return decision

    def get_play_card_decision(
        self,
        player: Player,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Get play card decision (network version).

        Parameters
        ----------
        player : Player
            The player making the decision.
        led_suit : Optional[Suit]
            The suit that was led.
        trump_suit : Optional[Suit]
            The trump suit.
        trick_cards : List[Card]
            Cards played so far in the trick.
        trick_player_ids : List[int]
            Player IDs who played each card.

        Returns
        -------
        Card
            Card to play.
        """
        # Wait for decision request from server
        request = self._wait_for_decision_request()
        if request is None:
            raise ConnectionError("Lost connection to server")

        # Get decision using parent class method
        decision = super().get_play_card_decision(
            player, led_suit, trump_suit, trick_cards, trick_player_ids
        )

        # Send decision to server
        self.client.send_decision(DecisionType.PLAY_CARD, decision)
        return decision

    def get_going_alone_decision(self, player: Player, trump_suit: Suit) -> bool:
        """
        Get going alone decision (network version).

        Parameters
        ----------
        player : Player
            The player making the decision.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        bool
            True to go alone, False otherwise.
        """
        # Wait for decision request from server
        request = self._wait_for_decision_request()
        if request is None:
            raise ConnectionError("Lost connection to server")

        # Get decision using parent class method
        decision = super().get_going_alone_decision(player, trump_suit)

        # Send decision to server
        self.client.send_decision(DecisionType.GOING_ALONE, decision)
        return decision

