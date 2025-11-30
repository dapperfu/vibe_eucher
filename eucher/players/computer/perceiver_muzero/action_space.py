"""Unified action space for EucherPerceiverMuZero."""

from enum import IntEnum
from typing import Optional, Tuple

import torch

from eucher.cards import Suit


class ActionType(IntEnum):
    """Action types in unified action space."""

    PASS = 0
    ORDER_UP = 1
    CALL_HEARTS = 2
    CALL_DIAMONDS = 3
    CALL_CLUBS = 4
    CALL_SPADES = 5
    GO_ALONE = 6
    TRADE_IN = 7
    PICKUP = 8
    DISCARD_0 = 9
    DISCARD_1 = 10
    DISCARD_2 = 11
    DISCARD_3 = 12
    DISCARD_4 = 13
    DISCARD_5 = 14
    PLAY_0 = 15
    PLAY_1 = 16
    PLAY_2 = 17
    PLAY_3 = 18
    PLAY_4 = 19


ACTION_SPACE_SIZE = 20


class ActionEncoder:
    """Encode/decode actions in unified space."""

    @staticmethod
    def encode_bid_action(
        action: str, suit: Optional[Suit] = None, go_alone: bool = False
    ) -> int:
        """
        Encode bidding action.

        Parameters
        ----------
        action : str
            Action type: "pass", "order_up", "call", "go_alone", "trade_in", "pickup".
        suit : Optional[Suit]
            Suit for "call" action.
        go_alone : bool
            Whether to go alone (for order_up or call).

        Returns
        -------
        int
            Action ID.

        Raises
        ------
        ValueError
            If action is invalid.
        """
        if action == "pass":
            return ActionType.PASS
        elif action == "order_up":
            if go_alone:
                return ActionType.GO_ALONE
            return ActionType.ORDER_UP
        elif action == "call":
            if suit is None:
                raise ValueError("Suit required for call action")
            suit_map = {
                Suit.HEARTS: ActionType.CALL_HEARTS,
                Suit.DIAMONDS: ActionType.CALL_DIAMONDS,
                Suit.CLUBS: ActionType.CALL_CLUBS,
                Suit.SPADES: ActionType.CALL_SPADES,
            }
            if go_alone:
                # For go_alone with call, we use GO_ALONE action
                # The suit is encoded separately in the game state
                return ActionType.GO_ALONE
            return suit_map[suit]
        elif action == "go_alone":
            return ActionType.GO_ALONE
        elif action == "trade_in":
            return ActionType.TRADE_IN
        elif action == "pickup":
            return ActionType.PICKUP
        else:
            raise ValueError(f"Unknown bid action: {action}")

    @staticmethod
    def encode_discard_action(card_index: int) -> int:
        """
        Encode discard action.

        Parameters
        ----------
        card_index : int
            Index of card to discard (0-5).

        Returns
        -------
        int
            Action ID.

        Raises
        ------
        ValueError
            If card_index is invalid.
        """
        if not 0 <= card_index <= 5:
            raise ValueError(f"Invalid discard index: {card_index}")
        return ActionType.DISCARD_0 + card_index

    @staticmethod
    def encode_play_action(card_index: int) -> int:
        """
        Encode play action.

        Parameters
        ----------
        card_index : int
            Index of card to play (0-4).

        Returns
        -------
        int
            Action ID.

        Raises
        ------
        ValueError
            If card_index is invalid.
        """
        if not 0 <= card_index <= 4:
            raise ValueError(f"Invalid play index: {card_index}")
        return ActionType.PLAY_0 + card_index

    @staticmethod
    def encode_action_to_tensor(action_id: int, device: str = "cpu") -> torch.Tensor:
        """
        Encode action ID as one-hot tensor.

        Parameters
        ----------
        action_id : int
            Action ID (0-19).
        device : str
            Device for tensor.

        Returns
        -------
        torch.Tensor
            One-hot action tensor [action_space_size].
        """
        tensor = torch.zeros(ACTION_SPACE_SIZE, device=device)
        if 0 <= action_id < ACTION_SPACE_SIZE:
            tensor[action_id] = 1.0
        return tensor

    @staticmethod
    def decode_action(action_id: int) -> Tuple[str, Optional[int], Optional[Suit]]:
        """
        Decode action to (type, card_index, suit).

        Parameters
        ----------
        action_id : int
            Action ID to decode.

        Returns
        -------
        Tuple[str, Optional[int], Optional[Suit]]
            (action_type, card_index, suit).

        Raises
        ------
        ValueError
            If action_id is invalid.
        """
        if action_id == ActionType.PASS:
            return ("pass", None, None)
        elif action_id == ActionType.ORDER_UP:
            return ("order_up", None, None)
        elif action_id == ActionType.CALL_HEARTS:
            return ("call", None, Suit.HEARTS)
        elif action_id == ActionType.CALL_DIAMONDS:
            return ("call", None, Suit.DIAMONDS)
        elif action_id == ActionType.CALL_CLUBS:
            return ("call", None, Suit.CLUBS)
        elif action_id == ActionType.CALL_SPADES:
            return ("call", None, Suit.SPADES)
        elif action_id == ActionType.GO_ALONE:
            return ("go_alone", None, None)
        elif action_id == ActionType.TRADE_IN:
            return ("trade_in", None, None)
        elif action_id == ActionType.PICKUP:
            return ("pickup", None, None)
        elif ActionType.DISCARD_0 <= action_id <= ActionType.DISCARD_5:
            return ("discard", action_id - ActionType.DISCARD_0, None)
        elif ActionType.PLAY_0 <= action_id <= ActionType.PLAY_4:
            return ("play", action_id - ActionType.PLAY_0, None)
        else:
            raise ValueError(f"Unknown action ID: {action_id}")

