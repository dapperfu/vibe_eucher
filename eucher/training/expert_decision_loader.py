"""Load expert decisions from various sources."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from eucher.cards import Card, Rank, Suit
from eucher.players.computer.ml.ml_features import GameStateEncoder
from eucher.training.decision_format import DecisionRecord, DecisionType, load_decisions, validate_decision


class ExpertDecisionLoader:
    """Load expert decisions from various sources."""

    def __init__(self) -> None:
        """Initialize the expert decision loader."""
        self.encoder = GameStateEncoder()

    def load_from_json(self, filepath: Path) -> List[DecisionRecord]:
        """
        Load expert decisions from JSON file.

        Parameters
        ----------
        filepath : Path
            Path to JSON file.

        Returns
        -------
        List[DecisionRecord]
            List of decision records.
        """
        return load_decisions(str(filepath))

    def load_from_game_log(self, log_file: Path) -> List[DecisionRecord]:
        """
        Load decisions from a game log file.

        Parameters
        ----------
        log_file : Path
            Path to game log file.

        Returns
        -------
        List[DecisionRecord]
            List of decision records.
        """
        # This is a placeholder - would parse actual game log format
        with open(log_file, "r") as f:
            log_data = json.load(f)

        decisions = []
        # Parse log data and convert to DecisionRecord format
        # Implementation depends on log format
        return decisions

    def convert_from_annotated_file(
        self, annotated_file: Path, format_type: str = "csv"
    ) -> List[DecisionRecord]:
        """
        Convert decisions from annotated file format.

        Parameters
        ----------
        annotated_file : Path
            Path to annotated file.
        format_type : str
            Format type: "csv", "json", etc.

        Returns
        -------
        List[DecisionRecord]
            List of decision records.
        """
        decisions = []
        if format_type == "csv":
            import csv

            with open(annotated_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse CSV row and convert to DecisionRecord
                    # This is a placeholder - actual implementation depends on CSV format
                    pass
        elif format_type == "json":
            with open(annotated_file, "r") as f:
                data = json.load(f)
                # Parse JSON and convert to DecisionRecord
                # This is a placeholder
                pass

        return decisions

    def validate_expert_decisions(self, decisions: List[DecisionRecord]) -> tuple[bool, List[str]]:
        """
        Validate expert decisions against game rules.

        Parameters
        ----------
        decisions : List[DecisionRecord]
            List of decisions to validate.

        Returns
        -------
        tuple[bool, List[str]]
            (all_valid, error_messages)
        """
        errors = []
        for decision in decisions:
            is_valid, error = validate_decision(decision)
            if not is_valid:
                errors.append(f"{decision.decision_id}: {error}")

            # Additional validation: check if decision makes sense given game state
            # This would require more context about the game rules
            # For now, just basic format validation

        return len(errors) == 0, errors

    def merge_with_training_data(
        self,
        expert_decisions: List[DecisionRecord],
        training_data_file: Path,
        output_file: Path,
    ) -> None:
        """
        Merge expert decisions with existing training data.

        Parameters
        ----------
        expert_decisions : List[DecisionRecord]
            Expert decisions to merge.
        training_data_file : Path
            Path to existing training data file.
        output_file : Path
            Path to save merged data.
        """
        # Load existing training data
        existing_data = []
        if training_data_file.exists():
            with open(training_data_file, "r") as f:
                existing_data = json.load(f)

        # Convert expert decisions to training data format
        expert_data = []
        for decision in expert_decisions:
            expert_data.append({
                "game_id": decision.game_id,
                "player_id": decision.player_id,
                "features": decision.game_state.get("features", []),
                "decision": decision.decision_value,
            })

        # Merge
        merged_data = existing_data + expert_data

        # Save
        with open(output_file, "w") as f:
            json.dump(merged_data, f, indent=2)

        print(f"Merged {len(expert_decisions)} expert decisions with {len(existing_data)} existing records")
        print(f"Saved {len(merged_data)} total records to {output_file}")

