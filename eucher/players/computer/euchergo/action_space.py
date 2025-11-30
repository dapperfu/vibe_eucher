"""Action space encoding/decoding for EucherGo."""

from typing import Optional, Tuple

from eucher.cards import Card, Suit


class ActionEncoder:
    """Encode and decode actions for EucherGo.

    Action space:
    - PASS (0): Pass during trump selection
    - ORDER_UP (1): Order up the turned card
    - CALL_HEARTS (2): Call Hearts as trump
    - CALL_DIAMONDS (3): Call Diamonds as trump
    - CALL_CLUBS (4): Call Clubs as trump
    - CALL_SPADES (5): Call Spades as trump
    - GO_ALONE (6): Go alone after calling trump
    - DISCARD_0 to DISCARD_5 (7-12): Discard card at index (supports 6 cards for dealer)
    - PLAY_0 to PLAY_4 (13-17): Play card at index
    """

    ACTION_SPACE_SIZE = 18

    PASS = 0
    ORDER_UP = 1
    CALL_HEARTS = 2
    CALL_DIAMONDS = 3
    CALL_CLUBS = 4
    CALL_SPADES = 5
    GO_ALONE = 6
    DISCARD_START = 7
    DISCARD_END = 12  # Now supports 0-5 (6 cards)
    PLAY_START = 13  # Shifted by 1 to accommodate DISCARD_5
    PLAY_END = 17

    @staticmethod
    def encode_order_up() -> int:
        """
        Encode order up action.

        Returns
        -------
        int
            Action ID.
        """
        return ActionEncoder.ORDER_UP

    @staticmethod
    def encode_pass() -> int:
        """
        Encode pass action.

        Returns
        -------
        int
            Action ID.
        """
        return ActionEncoder.PASS

    @staticmethod
    def encode_call_trump(suit: Suit) -> int:
        """
        Encode call trump action.

        Parameters
        ----------
        suit : Suit
            Suit to call as trump.

        Returns
        -------
        int
            Action ID.
        """
        suit_map = {
            Suit.HEARTS: ActionEncoder.CALL_HEARTS,
            Suit.DIAMONDS: ActionEncoder.CALL_DIAMONDS,
            Suit.CLUBS: ActionEncoder.CALL_CLUBS,
            Suit.SPADES: ActionEncoder.CALL_SPADES,
        }
        return suit_map[suit]

    @staticmethod
    def encode_go_alone() -> int:
        """
        Encode go alone action.

        Returns
        -------
        int
            Action ID.
        """
        return ActionEncoder.GO_ALONE

    @staticmethod
    def encode_discard(card_index: int) -> int:
        """
        Encode discard action.

        Parameters
        ----------
        card_index : int
            Index of card to discard (0-5). Supports 6 cards for dealer after ordering up.

        Returns
        -------
        int
            Action ID.
        """
        if not 0 <= card_index <= 5:
            raise ValueError(f"Invalid card index: {card_index}")
        return ActionEncoder.DISCARD_START + card_index

    @staticmethod
    def encode_play_card(card_index: int) -> int:
        """
        Encode play card action.

        Parameters
        ----------
        card_index : int
            Index of card to play (0-4).

        Returns
        -------
        int
            Action ID.
        """
        if not 0 <= card_index <= 4:
            raise ValueError(f"Invalid card index: {card_index}")
        return ActionEncoder.PLAY_START + card_index

    @staticmethod
    def decode_action(action_id: int) -> Tuple[str, Optional[int], Optional[Suit]]:
        """
        Decode action ID to action type and parameters.

        Parameters
        ----------
        action_id : int
            Action ID.

        Returns
        -------
        Tuple[str, Optional[int], Optional[Suit]]
            (action_type, card_index, suit)
        """
        if action_id == ActionEncoder.PASS:
            return ("pass", None, None)
        elif action_id == ActionEncoder.ORDER_UP:
            return ("order_up", None, None)
        elif action_id == ActionEncoder.CALL_HEARTS:
            return ("call", None, Suit.HEARTS)
        elif action_id == ActionEncoder.CALL_DIAMONDS:
            return ("call", None, Suit.DIAMONDS)
        elif action_id == ActionEncoder.CALL_CLUBS:
            return ("call", None, Suit.CLUBS)
        elif action_id == ActionEncoder.CALL_SPADES:
            return ("call", None, Suit.SPADES)
        elif action_id == ActionEncoder.GO_ALONE:
            return ("go_alone", None, None)
        elif ActionEncoder.DISCARD_START <= action_id <= ActionEncoder.DISCARD_END:
            card_index = action_id - ActionEncoder.DISCARD_START
            return ("discard", card_index, None)
        elif ActionEncoder.PLAY_START <= action_id <= ActionEncoder.PLAY_END:
            card_index = action_id - ActionEncoder.PLAY_START
            return ("play", card_index, None)
        else:
            raise ValueError(f"Invalid action ID: {action_id}")

