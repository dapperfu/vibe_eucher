"""Standard decision format definitions and utilities."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from eucher.cards import Card, Suit


class DecisionType(Enum):
    """Types of decisions in Euchre."""

    ORDER_UP = "order_up"
    CALL_TRUMP = "call_trump"
    PLAY_CARD = "play_card"
    DISCARD = "discard"


@dataclass
class DecisionRecord:
    """Standard format for a decision record."""

    decision_id: str
    game_id: str
    player_id: int
    decision_type: DecisionType
    game_state: Dict[str, Any]  # Encoded game state features
    decision_value: Any  # The decision made (bool, Suit, Card, etc.)
    metadata: Dict[str, Any]  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation.
        """
        data = asdict(self)
        data["decision_type"] = self.decision_type.value
        # Convert numpy arrays to lists
        if isinstance(data["game_state"], dict):
            for key, value in data["game_state"].items():
                if isinstance(value, np.ndarray):
                    data["game_state"][key] = value.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionRecord":
        """
        Create from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary representation.

        Returns
        -------
        DecisionRecord
            Decision record instance.
        """
        data = data.copy()
        data["decision_type"] = DecisionType(data["decision_type"])
        return cls(**data)


def validate_decision(decision_record: DecisionRecord) -> tuple[bool, Optional[str]]:
    """
    Validate a decision record.

    Parameters
    ----------
    decision_record : DecisionRecord
        The decision record to validate.

    Returns
    -------
    tuple[bool, Optional[str]]
        (is_valid, error_message)
    """
    # Check required fields
    if not decision_record.decision_id:
        return False, "decision_id is required"
    if not decision_record.game_id:
        return False, "game_id is required"
    if decision_record.player_id < 0 or decision_record.player_id > 3:
        return False, "player_id must be 0-3"
    if not decision_record.game_state:
        return False, "game_state is required"

    # Validate decision value based on type
    if decision_record.decision_type == DecisionType.ORDER_UP:
        if not isinstance(decision_record.decision_value, bool):
            return False, "order_up decision must be bool"
    elif decision_record.decision_type == DecisionType.CALL_TRUMP:
        if decision_record.decision_value is not None:
            if not isinstance(decision_record.decision_value, (str, Suit)):
                return False, "call_trump decision must be Suit or None"
    elif decision_record.decision_type == DecisionType.PLAY_CARD:
        if not isinstance(decision_record.decision_value, (dict, Card, int)):
            return False, "play_card decision must be Card, dict, or int (card index)"
    elif decision_record.decision_type == DecisionType.DISCARD:
        if not isinstance(decision_record.decision_value, (dict, Card, int)):
            return False, "discard decision must be Card, dict, or int (card index)"

    return True, None


def convert_decision_to_standard_format(
    decision_type: str,
    game_state: Dict[str, Any],
    decision: Any,
    game_id: str,
    player_id: int,
    decision_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DecisionRecord:
    """
    Convert a decision to standard format.

    Parameters
    ----------
    decision_type : str
        Type of decision.
    game_state : Dict[str, Any]
        Game state features.
    decision : Any
        The decision value.
    game_id : str
        Game identifier.
    player_id : int
        Player identifier.
    decision_id : Optional[str]
        Decision identifier. If None, generates one.
    metadata : Optional[Dict[str, Any]]
        Additional metadata.

    Returns
    -------
    DecisionRecord
        Standardized decision record.
    """
    if decision_id is None:
        decision_id = f"{game_id}_{player_id}_{decision_type}_{datetime.now().isoformat()}"

    if metadata is None:
        metadata = {}

    metadata["timestamp"] = datetime.now().isoformat()

    return DecisionRecord(
        decision_id=decision_id,
        game_id=game_id,
        player_id=player_id,
        decision_type=DecisionType(decision_type),
        game_state=game_state,
        decision_value=decision,
        metadata=metadata,
    )


def save_decisions(decisions: List[DecisionRecord], filepath: str) -> None:
    """
    Save decisions to JSON file.

    Parameters
    ----------
    decisions : List[DecisionRecord]
        List of decision records.
    filepath : str
        Path to save file.
    """
    data = [decision.to_dict() for decision in decisions]
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_decisions(filepath: str) -> List[DecisionRecord]:
    """
    Load decisions from JSON file.

    Parameters
    ----------
    filepath : str
        Path to load file from.

    Returns
    -------
    List[DecisionRecord]
        List of decision records.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    return [DecisionRecord.from_dict(item) for item in data]

