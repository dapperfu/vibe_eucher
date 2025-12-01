"""Decision playback system for recording and replaying human player decisions.

This module provides functionality to record human player decisions during gameplay
and replay them when using the same seed, allowing players to test different
strategies against the same game scenarios.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from eucher.cards import Card, Suit


class DecisionType(Enum):
    """Types of decisions that can be recorded and replayed."""

    ORDER_UP = "order_up"
    CALL_TRUMP = "call_trump"
    DISCARD = "discard"
    PLAY_CARD = "play_card"
    TRADE_IN = "trade_in"
    GOING_ALONE = "going_alone"


class DecisionPlayback:
    """Manages recording and playback of human player decisions."""

    def __init__(
        self,
        record_file: Optional[Union[str, Path]] = None,
        playback_file: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize decision playback system.

        Parameters
        ----------
        record_file : Optional[Union[str, Path]]
            Path to file where decisions will be recorded. If None, recording is disabled.
        playback_file : Optional[Union[str, Path]]
            Path to file containing decisions to replay. If None, playback is disabled.
        """
        self.record_file = Path(record_file) if record_file else None
        self.playback_file = Path(playback_file) if playback_file else None
        self.recorded_decisions: Dict[str, List[Dict[str, Any]]] = {}
        self.playback_decisions: Dict[str, List[Dict[str, Any]]] = {}
        self.current_seed: Optional[str] = None
        self.decision_counter: int = 0

        # Load playback decisions if file exists
        if self.playback_file and self.playback_file.exists():
            self._load_playback_decisions()

    def _load_playback_decisions(self) -> None:
        """Load decisions from playback file."""
        try:
            with open(self.playback_file, "r") as f:
                data = json.load(f)
                self.playback_decisions = data.get("decisions", {})
        except Exception as e:
            raise ValueError(f"Failed to load playback decisions: {e}")

    def start_game(self, seed: Union[int, str]) -> None:
        """
        Start recording/playback for a new game.

        Parameters
        ----------
        seed : Union[int, str]
            Game seed (integer or UUID string).
        """
        self.current_seed = str(seed)
        self.decision_counter = 0

        # Initialize recording for this seed if not already present
        if self.record_file and self.current_seed not in self.recorded_decisions:
            self.recorded_decisions[self.current_seed] = []

    def get_playback_decision(
        self, decision_type: DecisionType, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Get a decision from playback if available.

        Parameters
        ----------
        decision_type : DecisionType
            Type of decision to get.
        context : Optional[Dict[str, Any]]
            Optional context for matching decisions (not currently used, but available for future use).

        Returns
        -------
        Optional[Any]
            The decision value if found, None otherwise.
        """
        if not self.playback_file or not self.current_seed:
            return None

        seed_decisions = self.playback_decisions.get(self.current_seed, [])
        if self.decision_counter >= len(seed_decisions):
            return None

        decision_record = seed_decisions[self.decision_counter]
        if decision_record.get("type") != decision_type.value:
            # Decision type mismatch - playback file may be out of sync
            return None

        return decision_record.get("value")

    def record_decision(
        self, decision_type: DecisionType, value: Any, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a decision for later playback.

        Parameters
        ----------
        decision_type : DecisionType
            Type of decision being recorded.
        value : Any
            The decision value (bool, Card, Suit, etc.).
        context : Optional[Dict[str, Any]]
            Optional context information (not currently used, but available for future use).
        """
        if not self.record_file or not self.current_seed:
            return

        # Convert value to serializable format
        serialized_value = self._serialize_value(value)

        decision_record = {
            "type": decision_type.value,
            "value": serialized_value,
            "counter": self.decision_counter,
        }
        if context:
            decision_record["context"] = context

        if self.current_seed not in self.recorded_decisions:
            self.recorded_decisions[self.current_seed] = []
        self.recorded_decisions[self.current_seed].append(decision_record)
        self.decision_counter += 1

        # Save after each decision
        self._save_decisions()

    def _serialize_value(self, value: Any) -> Any:
        """
        Serialize a decision value to JSON-compatible format.

        Parameters
        ----------
        value : Any
            The value to serialize.

        Returns
        -------
        Any
            Serialized value.
        """
        if isinstance(value, bool):
            return value
        elif isinstance(value, Card):
            return {"rank": value.rank.value, "suit": value.suit.value}
        elif isinstance(value, Suit):
            return value.value
        elif value is None:
            return None
        else:
            return str(value)

    def _deserialize_value(self, decision_type: DecisionType, serialized_value: Any) -> Any:
        """
        Deserialize a decision value from JSON format.

        Parameters
        ----------
        decision_type : DecisionType
            Type of decision (used to determine deserialization method).
        serialized_value : Any
            Serialized value.

        Returns
        -------
        Any
            Deserialized value.
        """
        if decision_type == DecisionType.ORDER_UP:
            return bool(serialized_value)
        elif decision_type == DecisionType.CALL_TRUMP:
            if serialized_value is None:
                return None
            return Suit(serialized_value)
        elif decision_type == DecisionType.DISCARD or decision_type == DecisionType.PLAY_CARD:
            if isinstance(serialized_value, dict):
                from eucher.cards import Rank

                return Card(Suit(serialized_value["suit"]), Rank(serialized_value["rank"]))
            return serialized_value
        elif decision_type == DecisionType.TRADE_IN or decision_type == DecisionType.GOING_ALONE:
            return bool(serialized_value)
        else:
            return serialized_value

    def get_playback_decision_deserialized(
        self, decision_type: DecisionType, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Get a decision from playback and deserialize it.

        Parameters
        ----------
        decision_type : DecisionType
            Type of decision to get.
        context : Optional[Dict[str, Any]]
            Optional context for matching decisions.

        Returns
        -------
        Optional[Any]
            The deserialized decision value if found, None otherwise.
        """
        serialized_value = self.get_playback_decision(decision_type, context)
        if serialized_value is None:
            return None
        return self._deserialize_value(decision_type, serialized_value)

    def _save_decisions(self) -> None:
        """Save recorded decisions to file."""
        if not self.record_file:
            return

        try:
            # Create parent directory if it doesn't exist
            self.record_file.parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            data = {
                "decisions": self.recorded_decisions,
            }
            with open(self.record_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # Don't raise - recording failures shouldn't break gameplay
            print(f"Warning: Failed to save decision recording: {e}")

    def finish_game(self) -> None:
        """Finish recording/playback for the current game."""
        self.current_seed = None
        self.decision_counter = 0

