"""Server-side network TUI wrapper for multiplayer Euchre."""

from typing import Dict, List, Optional, TYPE_CHECKING

from eucher.cards import Card, Suit
from eucher.players import Player
from eucher.tui import TextTUI

if TYPE_CHECKING:
    from eucher.network.server import EuchreServer
    from eucher.network.protocol import DecisionType, serialize_card, serialize_cards, serialize_suit


class ServerNetworkTUI(TextTUI):
    """Server-side TUI that handles both local and network players."""

    def __init__(
        self,
        server: "EuchreServer",
        local_player_id: int = 0,
        **kwargs,
    ) -> None:
        """
        Initialize server network TUI.

        Parameters
        ----------
        server : EuchreServer
            Network server instance.
        local_player_id : int
            ID of the local player (default: 0).
        **kwargs
            Additional arguments passed to TextTUI.
        """
        super().__init__(**kwargs)
        self.server = server
        self.local_player_id = local_player_id

    def get_order_up_decision(
        self, player: Player, turned_card: Card, dealer_id: int
    ) -> bool:
        """
        Get order up decision (local or network).

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
        if player.player_id == self.local_player_id:
            # Local player - use normal TUI
            return super().get_order_up_decision(player, turned_card, dealer_id)
        else:
            # Network player - request via server
            context = {
                "turned_card": serialize_card(turned_card),
                "dealer_id": dealer_id,
                "player_hand": [serialize_card(c) for c in player.hand],
            }
            decision = self.server.request_decision(
                player.player_id,
                DecisionType.ORDER_UP,
                context,
            )
            if decision is None:
                raise ConnectionError(f"Failed to get decision from player {player.player_id}")
            return bool(decision)

    def get_call_trump_decision(
        self,
        player: Player,
        turned_card: Card,
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Get call trump decision (local or network).

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
        if player.player_id == self.local_player_id:
            # Local player - use normal TUI
            return super().get_call_trump_decision(player, turned_card, must_choose)
        else:
            # Network player - request via server
            context = {
                "turned_card": serialize_card(turned_card),
                "must_choose": must_choose,
                "player_hand": [serialize_card(c) for c in player.hand],
            }
            decision = self.server.request_decision(
                player.player_id,
                DecisionType.CALL_TRUMP,
                context,
            )
            if decision is None:
                raise ConnectionError(f"Failed to get decision from player {player.player_id}")
            return deserialize_suit(decision) if decision is not None else None

    def get_discard_decision(
        self,
        player: Player,
        turned_card: Optional[Card] = None,
        ordered_up_by: Optional[str] = None,
    ) -> Card:
        """
        Get discard decision (local or network).

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
        if player.player_id == self.local_player_id:
            # Local player - use normal TUI
            return super().get_discard_decision(player, turned_card, ordered_up_by)
        else:
            # Network player - request via server
            context = {
                "player_hand": [serialize_card(c) for c in player.hand],
                "turned_card": serialize_card(turned_card) if turned_card else None,
                "ordered_up_by": ordered_up_by,
            }
            decision = self.server.request_decision(
                player.player_id,
                DecisionType.DISCARD,
                context,
            )
            if decision is None:
                raise ConnectionError(f"Failed to get decision from player {player.player_id}")
            return deserialize_card(decision)

    def get_play_card_decision(
        self,
        player: Player,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Get play card decision (local or network).

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
        if player.player_id == self.local_player_id:
            # Local player - use normal TUI
            return super().get_play_card_decision(
                player, led_suit, trump_suit, trick_cards, trick_player_ids
            )
        else:
            # Network player - request via server
            context = {
                "player_hand": serialize_cards(player.hand),
                "led_suit": serialize_suit(led_suit),
                "trump_suit": serialize_suit(trump_suit),
                "trick_cards": serialize_cards(trick_cards),
                "trick_player_ids": trick_player_ids,
            }
            decision = self.server.request_decision(
                player.player_id,
                DecisionType.PLAY_CARD,
                context,
            )
            if decision is None:
                raise ConnectionError(f"Failed to get decision from player {player.player_id}")
            return deserialize_card(decision)

    def get_going_alone_decision(self, player: Player, trump_suit: Suit) -> bool:
        """
        Get going alone decision (local or network).

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
        if player.player_id == self.local_player_id:
            # Local player - use normal TUI
            return super().get_going_alone_decision(player, trump_suit)
        else:
            # Network player - request via server
            context = {
                "trump_suit": serialize_suit(trump_suit),
                "player_hand": [serialize_card(c) for c in player.hand],
            }
            decision = self.server.request_decision(
                player.player_id,
                DecisionType.GOING_ALONE,
                context,
            )
            if decision is None:
                raise ConnectionError(f"Failed to get decision from player {player.player_id}")
            return bool(decision)

