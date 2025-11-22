"""Data collection for training ML models."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from src.cards import Card, Suit
from src.ml_config import MLConfig


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

    def __init__(self, config: Optional[MLConfig] = None) -> None:
        """
        Initialize the data collector.

        Parameters
        ----------
        config : Optional[MLConfig]
            ML configuration. If None, creates a new config.
        """
        self.config = config or MLConfig()
        self.examples: List[TrainingExample] = []
        self.current_hand_examples: List[TrainingExample] = []

    def start_new_hand(self) -> None:
        """Start collecting data for a new hand."""
        self.current_hand_examples = []

    def record_order_up_decision(
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
        Record an order_up decision.

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

    def record_call_trump_decision(
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
        Record a call_trump decision.

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

    def record_card_play(
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
        Record a card play decision.

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

    def record_discard(
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
        Record a discard decision.

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

    def clear(self) -> None:
        """Clear all collected examples."""
        self.examples = []
        self.current_hand_examples = []

