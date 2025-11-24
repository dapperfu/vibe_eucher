"""Record human decisions during gameplay for training."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.cards import Card, Suit
from src.ml_features import GameStateEncoder
from src.training.decision_format import (
    DecisionRecord,
    DecisionType,
    convert_decision_to_standard_format,
    save_decisions,
    validate_decision,
)


class HumanDecisionRecorder:
    """Records human decisions during gameplay."""

    def __init__(self, output_file: Optional[Path] = None) -> None:
        """
        Initialize the human decision recorder.

        Parameters
        ----------
        output_file : Optional[Path]
            File to save recorded decisions. If None, uses default.
        """
        from src.ml_config import MLConfig

        self.config = MLConfig()
        if output_file is None:
            output_file = self.config.get_training_data_path("human_decisions.json")
        self.output_file = Path(output_file)
        self.encoder = GameStateEncoder()
        self.recorded_decisions: List[DecisionRecord] = []
        self.current_game_id: Optional[str] = None

    def start_game(self, game_id: str) -> None:
        """
        Start recording a new game.

        Parameters
        ----------
        game_id : str
            Unique identifier for the game.
        """
        self.current_game_id = game_id

    def record_order_up_decision(
        self,
        player_id: int,
        hand: List[Card],
        turned_card: Card,
        dealer_id: int,
        human_decision: bool,
        game_state: Dict[str, Any],
    ) -> DecisionRecord:
        """
        Record a human order up decision.

        Parameters
        ----------
        player_id : int
            ID of the player.
        hand : List[Card]
            Player's hand.
        turned_card : Card
            The turned card.
        dealer_id : int
            ID of the dealer.
        human_decision : bool
            Human's decision (True to order up, False to pass).
        game_state : Dict[str, Any]
            Additional game state.

        Returns
        -------
        DecisionRecord
            Recorded decision.
        """
        # Encode game state
        features = self.encoder.encode_game_state(
            hand=hand,
            turned_card=turned_card,
            trump_suit=None,
            led_suit=None,
            trick_cards=[],
            player_id=player_id,
            dealer_id=dealer_id,
            team=player_id % 2,
            trick_number=game_state.get("trick_number", 0),
            tricks_won_team0=game_state.get("tricks_won_team0", 0),
            tricks_won_team1=game_state.get("tricks_won_team1", 0),
        )

        decision_record = convert_decision_to_standard_format(
            decision_type=DecisionType.ORDER_UP.value,
            game_state={"features": features.numpy().tolist()},
            decision=human_decision,
            game_id=self.current_game_id or "unknown",
            player_id=player_id,
            metadata={
                "turned_card": f"{turned_card.rank.value}_{turned_card.suit.value}",
                "dealer_id": dealer_id,
            },
        )

        self.recorded_decisions.append(decision_record)
        return decision_record

    def record_call_trump_decision(
        self,
        player_id: int,
        hand: List[Card],
        turned_card: Card,
        human_decision: Optional[Suit],
        game_state: Dict[str, Any],
    ) -> DecisionRecord:
        """
        Record a human call trump decision.

        Parameters
        ----------
        player_id : int
            ID of the player.
        hand : List[Card]
            Player's hand.
        turned_card : Card
            The turned card.
        human_decision : Optional[Suit]
            Human's decision (suit to call or None to pass).
        game_state : Dict[str, Any]
            Additional game state.

        Returns
        -------
        DecisionRecord
            Recorded decision.
        """
        # Encode game state
        features = self.encoder.encode_game_state(
            hand=hand,
            turned_card=turned_card,
            trump_suit=None,
            led_suit=None,
            trick_cards=[],
            player_id=player_id,
            dealer_id=0,
            team=player_id % 2,
            trick_number=game_state.get("trick_number", 0),
            tricks_won_team0=game_state.get("tricks_won_team0", 0),
            tricks_won_team1=game_state.get("tricks_won_team1", 0),
        )

        decision_value = human_decision.value if human_decision else None

        decision_record = convert_decision_to_standard_format(
            decision_type=DecisionType.CALL_TRUMP.value,
            game_state={"features": features.numpy().tolist()},
            decision=decision_value,
            game_id=self.current_game_id or "unknown",
            player_id=player_id,
            metadata={
                "turned_card": f"{turned_card.rank.value}_{turned_card.suit.value}",
            },
        )

        self.recorded_decisions.append(decision_record)
        return decision_record

    def record_play_card_decision(
        self,
        player_id: int,
        hand: List[Card],
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        human_decision: Card,
        game_state: Dict[str, Any],
    ) -> DecisionRecord:
        """
        Record a human play card decision.

        Parameters
        ----------
        player_id : int
            ID of the player.
        hand : List[Card]
            Player's hand.
        led_suit : Optional[Suit]
            Suit that was led.
        trump_suit : Optional[Suit]
            Current trump suit.
        trick_cards : List[Card]
            Cards already played.
        human_decision : Card
            Human's card choice.
        game_state : Dict[str, Any]
            Additional game state.

        Returns
        -------
        DecisionRecord
            Recorded decision.
        """
        # Encode game state
        features = self.encoder.encode_game_state(
            hand=hand,
            turned_card=None,
            trump_suit=trump_suit,
            led_suit=led_suit,
            trick_cards=trick_cards,
            player_id=player_id,
            dealer_id=0,
            team=player_id % 2,
            trick_number=game_state.get("trick_number", 0),
            tricks_won_team0=game_state.get("tricks_won_team0", 0),
            tricks_won_team1=game_state.get("tricks_won_team1", 0),
        )

        # Encode decision as card index
        decision_value = self.encoder.card_to_index.get(human_decision, -1)

        decision_record = convert_decision_to_standard_format(
            decision_type=DecisionType.PLAY_CARD.value,
            game_state={"features": features.numpy().tolist()},
            decision=decision_value,
            game_id=self.current_game_id or "unknown",
            player_id=player_id,
            metadata={
                "card": f"{human_decision.rank.value}_{human_decision.suit.value}",
                "led_suit": led_suit.value if led_suit else None,
                "trump_suit": trump_suit.value if trump_suit else None,
            },
        )

        self.recorded_decisions.append(decision_record)
        return decision_record

    def record_discard_decision(
        self,
        player_id: int,
        hand: List[Card],
        human_decision: Card,
        game_state: Dict[str, Any],
    ) -> DecisionRecord:
        """
        Record a human discard decision.

        Parameters
        ----------
        player_id : int
            ID of the player.
        hand : List[Card]
            Player's hand (6 cards).
        human_decision : Card
            Human's discard choice.
        game_state : Dict[str, Any]
            Additional game state.

        Returns
        -------
        DecisionRecord
            Recorded decision.
        """
        # Encode game state
        features = self.encoder.encode_game_state(
            hand=hand,
            turned_card=None,
            trump_suit=None,
            led_suit=None,
            trick_cards=[],
            player_id=player_id,
            dealer_id=player_id,
            team=player_id % 2,
            trick_number=game_state.get("trick_number", 0),
            tricks_won_team0=game_state.get("tricks_won_team0", 0),
            tricks_won_team1=game_state.get("tricks_won_team1", 0),
        )

        # Encode decision as card index
        decision_value = self.encoder.card_to_index.get(human_decision, -1)

        decision_record = convert_decision_to_standard_format(
            decision_type=DecisionType.DISCARD.value,
            game_state={"features": features.numpy().tolist()},
            decision=decision_value,
            game_id=self.current_game_id or "unknown",
            player_id=player_id,
            metadata={
                "card": f"{human_decision.rank.value}_{human_decision.suit.value}",
            },
        )

        self.recorded_decisions.append(decision_record)
        return decision_record

    def save(self) -> None:
        """Save recorded decisions to file."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        save_decisions(self.recorded_decisions, str(self.output_file))
        print(f"Saved {len(self.recorded_decisions)} decisions to {self.output_file}")

    def load_existing(self) -> List[DecisionRecord]:
        """
        Load existing decisions from file.

        Returns
        -------
        List[DecisionRecord]
            List of loaded decisions.
        """
        if self.output_file.exists():
            from src.training.decision_format import load_decisions

            self.recorded_decisions = load_decisions(str(self.output_file))
            return self.recorded_decisions
        return []

    def merge_with_existing(self) -> None:
        """Merge new decisions with existing ones from file."""
        existing = self.load_existing()
        existing_ids = {d.decision_id for d in existing}
        new_decisions = [d for d in self.recorded_decisions if d.decision_id not in existing_ids]
        self.recorded_decisions = existing + new_decisions

    def validate_all(self) -> tuple[bool, List[str]]:
        """
        Validate all recorded decisions.

        Returns
        -------
        tuple[bool, List[str]]
            (all_valid, error_messages)
        """
        errors = []
        for decision in self.recorded_decisions:
            is_valid, error = validate_decision(decision)
            if not is_valid:
                errors.append(f"{decision.decision_id}: {error}")
        return len(errors) == 0, errors

