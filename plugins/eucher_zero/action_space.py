"""Unified action space for EucherZero."""

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
    STICK_DEALER = 6
    DISCARD_0 = 7
    DISCARD_1 = 8
    DISCARD_2 = 9
    DISCARD_3 = 10
    DISCARD_4 = 11
    DISCARD_5 = 12
    PLAY_0 = 13
    PLAY_1 = 14
    PLAY_2 = 15
    PLAY_3 = 16
    PLAY_4 = 17


ACTION_SPACE_SIZE = 18


class ActionEncoder:
    """Encode/decode actions in unified space."""

    @staticmethod
    def encode_bid_action(action: str, suit: Optional[Suit] = None) -> int:
        """
        Encode bidding action.

        Parameters
        ----------
        action : str
            Action type: "pass", "order_up", "call", "stick_dealer".
        suit : Optional[Suit]
            Suit for "call" action.

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
            return suit_map[suit]
        elif action == "stick_dealer":
            return ActionType.STICK_DEALER
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
            Action ID (0-17).
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
        elif ActionType.CALL_HEARTS <= action_id <= ActionType.CALL_SPADES:
            suit_map = {
                ActionType.CALL_HEARTS: Suit.HEARTS,
                ActionType.CALL_DIAMONDS: Suit.DIAMONDS,
                ActionType.CALL_CLUBS: Suit.CLUBS,
                ActionType.CALL_SPADES: Suit.SPADES,
            }
            return ("call", None, suit_map[action_id])
        elif action_id == ActionType.STICK_DEALER:
            return ("stick_dealer", None, None)
        elif ActionType.DISCARD_0 <= action_id <= ActionType.DISCARD_5:
            return ("discard", action_id - ActionType.DISCARD_0, None)
        elif ActionType.PLAY_0 <= action_id <= ActionType.PLAY_4:
            return ("play", action_id - ActionType.PLAY_0, None)
        else:
            raise ValueError(f"Unknown action ID: {action_id}")

