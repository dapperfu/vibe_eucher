"""Network protocol for Euchre multiplayer.

Handles message serialization and deserialization for client-server communication.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from eucher.cards import Card, Rank, Suit


class MessageType(Enum):
    """Message types for network protocol."""

    GAME_STATE = "GAME_STATE"
    DECISION_REQUEST = "DECISION_REQUEST"
    DECISION_RESPONSE = "DECISION_RESPONSE"
    PLAYER_JOINED = "PLAYER_JOINED"
    PLAYER_DISCONNECTED = "PLAYER_DISCONNECTED"
    ERROR = "ERROR"
    POSITION_SELECT = "POSITION_SELECT"
    POSITION_ASSIGNED = "POSITION_ASSIGNED"
    GAME_START = "GAME_START"
    CONNECTION_ACK = "CONNECTION_ACK"


class DecisionType(Enum):
    """Types of decisions that can be requested."""

    ORDER_UP = "ORDER_UP"
    CALL_TRUMP = "CALL_TRUMP"
    DISCARD = "DISCARD"
    PLAY_CARD = "PLAY_CARD"
    GOING_ALONE = "GOING_ALONE"
    TRADE_IN = "TRADE_IN"


def serialize_card(card: Card) -> Dict[str, Any]:
    """
    Serialize a Card to a dictionary.

    Parameters
    ----------
    card : Card
        The card to serialize.

    Returns
    -------
    Dict[str, Any]
        Serialized card representation.
    """
    return {
        "suit": card.suit.value,
        "rank": card.rank.value,
    }


def deserialize_card(data: Dict[str, Any]) -> Card:
    """
    Deserialize a Card from a dictionary.

    Parameters
    ----------
    data : Dict[str, Any]
        Serialized card representation.

    Returns
    -------
    Card
        The deserialized card.
    """
    suit = Suit(data["suit"])
    rank = Rank(data["rank"])
    return Card(suit, rank)


def serialize_cards(cards: List[Card]) -> List[Dict[str, Any]]:
    """
    Serialize a list of cards.

    Parameters
    ----------
    cards : List[Card]
        List of cards to serialize.

    Returns
    -------
    List[Dict[str, Any]]
        List of serialized card representations.
    """
    return [serialize_card(card) for card in cards]


def deserialize_cards(data: List[Dict[str, Any]]) -> List[Card]:
    """
    Deserialize a list of cards.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        List of serialized card representations.

    Returns
    -------
    List[Card]
        List of deserialized cards.
    """
    return [deserialize_card(card_data) for card_data in data]


def serialize_suit(suit: Optional[Suit]) -> Optional[str]:
    """
    Serialize a Suit enum.

    Parameters
    ----------
    suit : Optional[Suit]
        The suit to serialize.

    Returns
    -------
    Optional[str]
        Serialized suit value, or None.
    """
    return suit.value if suit is not None else None


def deserialize_suit(data: Optional[str]) -> Optional[Suit]:
    """
    Deserialize a Suit enum.

    Parameters
    ----------
    data : Optional[str]
        Serialized suit value.

    Returns
    -------
    Optional[Suit]
        The deserialized suit, or None.
    """
    return Suit(data) if data is not None else None


def create_message(
    msg_type: MessageType,
    data: Dict[str, Any],
    player_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create a network message.

    Parameters
    ----------
    msg_type : MessageType
        The type of message.
    data : Dict[str, Any]
        Message data payload.
    player_id : Optional[int]
        Optional player ID associated with the message.

    Returns
    -------
    Dict[str, Any]
        Complete message dictionary.
    """
    message = {
        "type": msg_type.value,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }
    if player_id is not None:
        message["player_id"] = player_id
    return message


def serialize_message(message: Dict[str, Any]) -> bytes:
    """
    Serialize a message to JSON bytes.

    Parameters
    ----------
    message : Dict[str, Any]
        Message dictionary.

    Returns
    -------
    bytes
        JSON-encoded message bytes.
    """
    return json.dumps(message).encode("utf-8")


def deserialize_message(data: bytes) -> Dict[str, Any]:
    """
    Deserialize a message from JSON bytes.

    Parameters
    ----------
    data : bytes
        JSON-encoded message bytes.

    Returns
    -------
    Dict[str, Any]
        Message dictionary.

    Raises
    ------
    json.JSONDecodeError
        If the data is not valid JSON.
    """
    return json.loads(data.decode("utf-8"))


def create_decision_request(
    decision_type: DecisionType,
    player_id: int,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a decision request message.

    Parameters
    ----------
    decision_type : DecisionType
        Type of decision being requested.
    player_id : int
        ID of player who needs to make the decision.
    context : Dict[str, Any]
        Context data for the decision (e.g., valid cards, current trick state).

    Returns
    -------
    Dict[str, Any]
        Decision request message.
    """
    return create_message(
        MessageType.DECISION_REQUEST,
        {
            "decision_type": decision_type.value,
            "context": context,
        },
        player_id=player_id,
    )


def create_decision_response(
    decision_type: DecisionType,
    decision: Any,
) -> Dict[str, Any]:
    """
    Create a decision response message.

    Parameters
    ----------
    decision_type : DecisionType
        Type of decision being responded to.
    decision : Any
        The decision value (bool, Card, Suit, etc.).

    Returns
    -------
    Dict[str, Any]
        Decision response message.
    """
    # Serialize decision based on type
    serialized_decision: Any
    if isinstance(decision, Card):
        serialized_decision = serialize_card(decision)
    elif isinstance(decision, Suit):
        serialized_decision = decision.value
    elif isinstance(decision, bool):
        serialized_decision = decision
    elif decision is None:
        serialized_decision = None
    else:
        serialized_decision = decision

    return create_message(
        MessageType.DECISION_RESPONSE,
        {
            "decision_type": decision_type.value,
            "decision": serialized_decision,
        },
    )


def parse_decision_response(message: Dict[str, Any]) -> Any:
    """
    Parse a decision response message.

    Parameters
    ----------
    message : Dict[str, Any]
        Decision response message.

    Returns
    -------
    Any
        The decision value (deserialized if needed).
    """
    data = message["data"]
    decision_type = DecisionType(data["decision_type"])
    decision = data["decision"]

    # Deserialize based on decision type
    if decision_type == DecisionType.PLAY_CARD or decision_type == DecisionType.DISCARD:
        if decision is not None:
            return deserialize_card(decision)
        return None
    elif decision_type == DecisionType.CALL_TRUMP:
        if decision is not None:
            return deserialize_suit(decision)
        return None
    elif decision_type == DecisionType.ORDER_UP or decision_type == DecisionType.GOING_ALONE or decision_type == DecisionType.TRADE_IN:
        return bool(decision) if decision is not None else False
    else:
        return decision

