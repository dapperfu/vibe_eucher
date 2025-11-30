"""Unified action space for ReinforcementEucher."""

from enum import IntEnum
from typing import List, Optional, Tuple

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
    def encode_bid_action(action: str, suit: Optional[Suit] = None, go_alone: bool = False) -> int:
        """Encode bidding action.

        Parameters
        ----------
        action : str
            Action type: "pass", "order_up", "call", "go_alone".
        suit : Optional[Suit]
            Suit for "call" action.
        go_alone : bool
            Whether to go alone (only valid with order_up or call).

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
                return ActionType.GO_ALONE  # Simplified: go alone is separate action
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
        elif action == "go_alone":
            return ActionType.GO_ALONE
        else:
            raise ValueError(f"Unknown bid action: {action}")

    @staticmethod
    def encode_discard_action(card_index: int) -> int:
        """Encode discard action.

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
        """Encode play action.

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
    def decode_action(action_id: int) -> Tuple[str, Optional[int], Optional[Suit], bool]:
        """Decode action to (type, card_index, suit, go_alone).

        Parameters
        ----------
        action_id : int
            Action ID to decode.

        Returns
        -------
        Tuple[str, Optional[int], Optional[Suit], bool]
            (action_type, card_index, suit, go_alone).

        Raises
        ------
        ValueError
            If action_id is invalid.
        """
        if action_id == ActionType.PASS:
            return ("pass", None, None, False)
        elif action_id == ActionType.ORDER_UP:
            return ("order_up", None, None, False)
        elif action_id == ActionType.GO_ALONE:
            return ("go_alone", None, None, True)
        elif ActionType.CALL_HEARTS <= action_id <= ActionType.CALL_SPADES:
            suit_map = {
                ActionType.CALL_HEARTS: Suit.HEARTS,
                ActionType.CALL_DIAMONDS: Suit.DIAMONDS,
                ActionType.CALL_CLUBS: Suit.CLUBS,
                ActionType.CALL_SPADES: Suit.SPADES,
            }
            return ("call", None, suit_map[action_id], False)
        elif ActionType.DISCARD_0 <= action_id <= ActionType.DISCARD_5:
            return ("discard", action_id - ActionType.DISCARD_0, None, False)
        elif ActionType.PLAY_0 <= action_id <= ActionType.PLAY_4:
            return ("play", action_id - ActionType.PLAY_0, None, False)
        else:
            raise ValueError(f"Unknown action ID: {action_id}")

    @staticmethod
    def get_legal_actions_mask(legal_actions: List[int], device: str = "cpu") -> torch.Tensor:
        """Create legal actions mask tensor.

        Parameters
        ----------
        legal_actions : List[int]
            List of legal action IDs.
        device : str
            Device for tensor.

        Returns
        -------
        torch.Tensor
            Binary mask tensor [ACTION_SPACE_SIZE].
        """
        mask = torch.zeros(ACTION_SPACE_SIZE, device=device)
        for action_id in legal_actions:
            if 0 <= action_id < ACTION_SPACE_SIZE:
                mask[action_id] = 1.0
        return mask

    @staticmethod
    def apply_legal_mask(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """Apply legal action mask to logits.

        Parameters
        ----------
        logits : torch.Tensor
            Action logits [batch_size, ACTION_SPACE_SIZE].
        legal_mask : torch.Tensor
            Legal action mask [batch_size, ACTION_SPACE_SIZE] or [ACTION_SPACE_SIZE].

        Returns
        -------
        torch.Tensor
            Masked logits (illegal actions set to large negative value).
        """
        if legal_mask.dim() == 1:
            legal_mask = legal_mask.unsqueeze(0).expand_as(logits)

        # Set illegal actions to very negative value
        masked_logits = logits.clone()
        masked_logits[legal_mask == 0] = float("-inf")
        return masked_logits


