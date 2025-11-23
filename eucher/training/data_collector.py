"""Data collection for training ML models."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from eucher.cards import Card, Suit
from eucher.players.computer.ml.ml_config import MLConfig


@dataclass
class TrainingExample:
    """A single training example."""

    # Game state features (will be encoded)
    hand: List[Card]
    turned_card: Optional[Card]
    trump_suit: Optional[Suit]
    led_suit: Optional[Suit]
    trick_cards: List[Card]
    player_id: int
    dealer_id: int
    team: int
    trick_number: int
    tricks_won_team0: int
    tricks_won_team1: int

    # Action taken
    action_type: str  # "order_up", "call_trump", "play_card", "discard"
    action_value: str  # JSON-serializable action (bool, suit name, card repr, etc.)

    # Outcome
    outcome: Optional[float] = None  # Expected value or reward
    tricks_won_by_team: Optional[List[int]] = None  # Final tricks won [team0, team1]
    points_scored: Optional[List[int]] = None  # Points scored [team0, team1]

    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.

        Returns
        -------
        dict
            Dictionary representation.
        """
        result = asdict(self)
        # Convert cards to serializable format
        result["hand"] = [repr(card) for card in self.hand]
        result["turned_card"] = repr(self.turned_card) if self.turned_card else None
        result["trump_suit"] = self.trump_suit.value if self.trump_suit else None
        result["led_suit"] = self.led_suit.value if self.led_suit else None
        result["trick_cards"] = [repr(card) for card in self.trick_cards]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingExample":
        """
        Create from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation.

        Returns
        -------
        TrainingExample
            The training example.
        """
        # Parse cards from strings (simplified - would need proper card parsing)
        # For now, store as strings and parse when loading
        return cls(**data)


class GameDataCollector:
    """Collects training data during gameplay."""

    def __init__(self, output_dir: Optional[Path] = None, config: Optional[MLConfig] = None) -> None:
        """
        Initialize the data collector.

        Parameters
        ----------
        output_dir : Optional[Path]
            Directory to save collected data. If None, uses default from MLConfig.
        config : Optional[MLConfig]
            ML configuration. If None, creates a new config.
        """
        self.config = config or MLConfig()
        if output_dir is None:
            output_dir = self.config.training_data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.examples: List[TrainingExample] = []
        self.current_hand_examples: List[TrainingExample] = []
        self.current_game_id: Optional[str] = None
        self.game_outcomes: List[Dict[str, Any]] = []

    def start_game(self, game_id: str) -> None:
        """
        Start tracking a new game.

        Parameters
        ----------
        game_id : str
            Unique identifier for the game.
        """
        self.current_game_id = game_id

    def start_new_hand(self) -> None:
        """Start collecting data for a new hand."""
        self.current_hand_examples = []

    def _record_order_up_decision_internal(
        self,
        hand: List[Card],
        turned_card: Card,
        player_id: int,
        dealer_id: int,
        team: int,
        decision: bool,
        tricks_won: Optional[List[int]] = None,
        points_scored: Optional[List[int]] = None,
    ) -> None:
        """
        Record an order_up decision (internal method).

        Parameters
        ----------
        hand : List[Card]
            Player's hand.
        turned_card : Card
            The turned card.
        player_id : int
            ID of the player.
        dealer_id : int
            ID of the dealer.
        team : int
            Team of the player.
        decision : bool
            True to order up, False to pass.
        tricks_won : Optional[List[int]]
            Final tricks won by each team.
        points_scored : Optional[List[int]]
            Points scored by each team.
        """
        example = TrainingExample(
            hand=hand.copy(),
            turned_card=turned_card,
            trump_suit=None,
            led_suit=None,
            trick_cards=[],
            player_id=player_id,
            dealer_id=dealer_id,
            team=team,
            trick_number=0,
            tricks_won_team0=0,
            tricks_won_team1=0,
            action_type="order_up",
            action_value=str(decision),
            tricks_won_by_team=tricks_won,
            points_scored=points_scored,
        )
        self.current_hand_examples.append(example)

    def record_order_up_decision(
        self,
        player_id: int,
        hand: List[Card],
        turned_card: Card,
        dealer_id: int,
        decision: bool,
        game_state: Dict[str, Any],
    ) -> None:
        """
        Record an order_up decision (wrapper for self_play compatibility).

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
        decision : bool
            True to order up, False to pass.
        game_state : Dict[str, Any]
            Game state dictionary with trick_number, tricks_won_team0, tricks_won_team1.
        """
        team = player_id % 2
        self._record_order_up_decision_internal(
            hand=hand,
            turned_card=turned_card,
            player_id=player_id,
            dealer_id=dealer_id,
            team=team,
            decision=decision,
            tricks_won=None,
            points_scored=None,
        )

    def _record_call_trump_decision_internal(
        self,
        hand: List[Card],
        turned_card: Card,
        player_id: int,
        dealer_id: int,
        team: int,
        decision: Optional[Suit],
        tricks_won: Optional[List[int]] = None,
        points_scored: Optional[List[int]] = None,
    ) -> None:
        """
        Record a call_trump decision (internal method).

        Parameters
        ----------
        hand : List[Card]
            Player's hand.
        turned_card : Card
            The turned card.
        player_id : int
            ID of the player.
        dealer_id : int
            ID of the dealer.
        team : int
            Team of the player.
        decision : Optional[Suit]
            Suit chosen, or None to pass.
        tricks_won : Optional[List[int]]
            Final tricks won by each team.
        points_scored : Optional[List[int]]
            Points scored by each team.
        """
        example = TrainingExample(
            hand=hand.copy(),
            turned_card=turned_card,
            trump_suit=None,
            led_suit=None,
            trick_cards=[],
            player_id=player_id,
            dealer_id=dealer_id,
            team=team,
            trick_number=0,
            tricks_won_team0=0,
            tricks_won_team1=0,
            action_type="call_trump",
            action_value=decision.value if decision else "pass",
            tricks_won_by_team=tricks_won,
            points_scored=points_scored,
        )
        self.current_hand_examples.append(example)

    def record_call_trump_decision(
        self,
        player_id: int,
        hand: List[Card],
        turned_card: Card,
        decision: Optional[Suit],
        game_state: Dict[str, Any],
    ) -> None:
        """
        Record a call_trump decision (wrapper for self_play compatibility).

        Parameters
        ----------
        player_id : int
            ID of the player.
        hand : List[Card]
            Player's hand.
        turned_card : Card
            The turned card.
        decision : Optional[Suit]
            Suit chosen, or None to pass.
        game_state : Dict[str, Any]
            Game state dictionary with trick_number, tricks_won_team0, tricks_won_team1.
        """
        team = player_id % 2
        dealer_id = game_state.get("dealer_id", 0)
        self._record_call_trump_decision_internal(
            hand=hand,
            turned_card=turned_card,
            player_id=player_id,
            dealer_id=dealer_id,
            team=team,
            decision=decision,
            tricks_won=None,
            points_scored=None,
        )

    def record_play_card_decision(
        self,
        player_id: int,
        hand: List[Card],
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        decision: Card,
        game_state: Dict[str, Any],
    ) -> None:
        """
        Record a play_card decision (wrapper for self_play compatibility).

        Parameters
        ----------
        player_id : int
            ID of the player.
        hand : List[Card]
            Player's hand before playing.
        led_suit : Optional[Suit]
            Suit that was led.
        trump_suit : Optional[Suit]
            Current trump suit.
        trick_cards : List[Card]
            Cards already played in trick.
        decision : Card
            Card that was played.
        game_state : Dict[str, Any]
            Game state dictionary with trick_number, tricks_won_team0, tricks_won_team1.
        """
        team = player_id % 2
        trick_number = game_state.get("trick_number", 0)
        tricks_won_team0 = game_state.get("tricks_won_team0", 0)
        tricks_won_team1 = game_state.get("tricks_won_team1", 0)
        dealer_id = game_state.get("dealer_id", 0)
        self._record_card_play_internal(
            hand=hand,
            trump_suit=trump_suit,
            led_suit=led_suit,
            trick_cards=trick_cards,
            player_id=player_id,
            dealer_id=dealer_id,
            team=team,
            trick_number=trick_number,
            tricks_won_team0=tricks_won_team0,
            tricks_won_team1=tricks_won_team1,
            card_played=decision,
            tricks_won=None,
            points_scored=None,
        )

    def _record_card_play_internal(
        self,
        hand: List[Card],
        trump_suit: Optional[Suit],
        led_suit: Optional[Suit],
        trick_cards: List[Card],
        player_id: int,
        dealer_id: int,
        team: int,
        trick_number: int,
        tricks_won_team0: int,
        tricks_won_team1: int,
        card_played: Card,
        tricks_won: Optional[List[int]] = None,
        points_scored: Optional[List[int]] = None,
    ) -> None:
        """
        Record a card play decision (internal method).

        Parameters
        ----------
        hand : List[Card]
            Player's hand before playing.
        trump_suit : Optional[Suit]
            Current trump suit.
        led_suit : Optional[Suit]
            Suit that was led.
        trick_cards : List[Card]
            Cards already played in trick.
        player_id : int
            ID of the player.
        dealer_id : int
            ID of the dealer.
        team : int
            Team of the player.
        trick_number : int
            Current trick number.
        tricks_won_team0 : int
            Tricks won by team 0 so far.
        tricks_won_team1 : int
            Tricks won by team 1 so far.
        card_played : Card
            Card that was played.
        tricks_won : Optional[List[int]]
            Final tricks won by each team.
        points_scored : Optional[List[int]]
            Points scored by each team.
        """
        example = TrainingExample(
            hand=hand.copy(),
            turned_card=None,
            trump_suit=trump_suit,
            led_suit=led_suit,
            trick_cards=trick_cards.copy(),
            player_id=player_id,
            dealer_id=dealer_id,
            team=team,
            trick_number=trick_number,
            tricks_won_team0=tricks_won_team0,
            tricks_won_team1=tricks_won_team1,
            action_type="play_card",
            action_value=repr(card_played),
            tricks_won_by_team=tricks_won,
            points_scored=points_scored,
        )
        self.current_hand_examples.append(example)

    def record_discard_decision(
        self,
        player_id: int,
        hand: List[Card],
        decision: Card,
        game_state: Dict[str, Any],
    ) -> None:
        """
        Record a discard decision (wrapper for self_play compatibility).

        Parameters
        ----------
        player_id : int
            ID of the player (dealer).
        hand : List[Card]
            Player's hand (6 cards) before discarding.
        decision : Card
            Card that was discarded.
        game_state : Dict[str, Any]
            Game state dictionary with trick_number, tricks_won_team0, tricks_won_team1.
        """
        team = player_id % 2
        dealer_id = game_state.get("dealer_id", player_id)
        trump_suit = game_state.get("trump_suit", None)
        self._record_discard_internal(
            hand=hand,
            trump_suit=trump_suit,
            player_id=player_id,
            dealer_id=dealer_id,
            team=team,
            card_discarded=decision,
            tricks_won=None,
            points_scored=None,
        )

    def _record_discard_internal(
        self,
        hand: List[Card],
        trump_suit: Optional[Suit],
        player_id: int,
        dealer_id: int,
        team: int,
        card_discarded: Card,
        tricks_won: Optional[List[int]] = None,
        points_scored: Optional[List[int]] = None,
    ) -> None:
        """
        Record a discard decision (internal method).

        Parameters
        ----------
        hand : List[Card]
            Player's hand (6 cards) before discarding.
        trump_suit : Optional[Suit]
            Current trump suit.
        player_id : int
            ID of the player (dealer).
        dealer_id : int
            ID of the dealer.
        team : int
            Team of the player.
        card_discarded : Card
            Card that was discarded.
        tricks_won : Optional[List[int]]
            Final tricks won by each team.
        points_scored : Optional[List[int]]
            Points scored by each team.
        """
        example = TrainingExample(
            hand=hand.copy(),
            turned_card=None,
            trump_suit=trump_suit,
            led_suit=None,
            trick_cards=[],
            player_id=player_id,
            dealer_id=dealer_id,
            team=team,
            trick_number=0,
            tricks_won_team0=0,
            tricks_won_team1=0,
            action_type="discard",
            action_value=repr(card_discarded),
            tricks_won_by_team=tricks_won,
            points_scored=points_scored,
        )
        self.current_hand_examples.append(example)

    def finish_hand(self, tricks_won: List[int], points_scored: List[int]) -> None:
        """
        Finish collecting data for a hand and update outcomes.

        Parameters
        ----------
        tricks_won : List[int]
            Tricks won by each team [team0, team1].
        points_scored : List[int]
            Points scored by each team [team0, team1].
        """
        # Update all examples from this hand with final outcomes
        for example in self.current_hand_examples:
            example.tricks_won_by_team = tricks_won
            example.points_scored = points_scored
            # Calculate outcome (simplified: points difference for player's team)
            if example.team == 0:
                example.outcome = float(points_scored[0] - points_scored[1])
            else:
                example.outcome = float(points_scored[1] - points_scored[0])

        # Add to main collection
        self.examples.extend(self.current_hand_examples)
        self.current_hand_examples = []

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Save collected examples to a file.

        Parameters
        ----------
        filepath : Optional[str]
            Path to save file. If None, uses default from config.
        """
        if filepath is None:
            filepath = str(self.config.get_training_data_path("training_data.json"))

        data = [ex.to_dict() for ex in self.examples]

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Load examples from a file.

        Parameters
        ----------
        filepath : str
            Path to load file.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        self.examples = [TrainingExample.from_dict(ex) for ex in data]

    def record_game_outcome(
        self,
        game_id: str,
        scores: List[int],
        tricks_won_history: List[List[int]],
        winner: Optional[int],
    ) -> None:
        """
        Record the outcome of a game.

        Parameters
        ----------
        game_id : str
            Unique identifier for the game.
        scores : List[int]
            Final scores for each team [team0, team1].
        tricks_won_history : List[List[int]]
            History of tricks won per hand [[team0, team1], ...].
        winner : Optional[int]
            Winning team (0 or 1), or None if no winner yet.
        """
        outcome = {
            "game_id": game_id,
            "scores": scores,
            "tricks_won_history": tricks_won_history,
            "winner": winner,
        }
        self.game_outcomes.append(outcome)

        # Update all examples from this game with final outcomes
        if tricks_won_history:
            final_tricks = tricks_won_history[-1]
            for example in self.examples:
                if example.tricks_won_by_team is None:
                    example.tricks_won_by_team = final_tricks
                if example.points_scored is None:
                    example.points_scored = scores
                # Calculate outcome if not set
                if example.outcome is None:
                    if example.team == 0:
                        example.outcome = float(scores[0] - scores[1])
                    else:
                        example.outcome = float(scores[1] - scores[0])

    def save_data(self, save_csv: bool = True) -> None:
        """
        Save collected data to files.

        Parameters
        ----------
        save_csv : bool
            Whether to save CSV files (can be slow for large datasets).
        """
        # Save JSON data
        json_path = self.output_dir / "training_data.json"
        self.save(filepath=str(json_path))

        # Save game outcomes
        outcomes_path = self.output_dir / "game_outcomes.json"
        with open(outcomes_path, "w") as f:
            json.dump(self.game_outcomes, f, indent=2)

        if save_csv:
            # Save CSV files (simplified - would need proper DataFrame conversion)
            # For now, just save the JSON data
            pass

    def clear(self) -> None:
        """Clear all collected examples."""
        self.examples = []
        self.current_hand_examples = []
        self.current_game_id = None
        self.game_outcomes = []

