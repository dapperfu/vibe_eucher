"""Data collection for training ML models."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.cards import Card, Suit
from src.ml_features import GameStateEncoder


class GameDataCollector:
    """Collects game state and decision data for training."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """
        Initialize the data collector.

        Parameters
        ----------
        output_dir : Optional[Path]
            Directory to save collected data. If None, uses default from MLConfig.
        """
        from src.ml_config import MLConfig

        self.config = MLConfig()
        if output_dir is None:
            output_dir = self.config.training_data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.encoder = GameStateEncoder()

        # Data storage
        self.order_up_data: List[Dict[str, Any]] = []
        self.call_trump_data: List[Dict[str, Any]] = []
        self.play_card_data: List[Dict[str, Any]] = []
        self.discard_data: List[Dict[str, Any]] = []

        # Game outcome tracking
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
        Record an order up decision.

        Parameters
        ----------
        player_id : int
            ID of the player making the decision.
        hand : List[Card]
            Player's hand.
        turned_card : Card
            The turned card.
        dealer_id : int
            ID of the dealer.
        decision : bool
            True to order up, False to pass.
        game_state : Dict[str, Any]
            Additional game state (trick_number, tricks_won, etc.).
        """
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

        self.order_up_data.append({
            "game_id": self.current_game_id,
            "player_id": player_id,
            "features": features.numpy().tolist(),
            "decision": int(decision),
            "turned_card": f"{turned_card.rank.value}_{turned_card.suit.value}",
        })

    def record_call_trump_decision(
        self,
        player_id: int,
        hand: List[Card],
        turned_card: Card,
        decision: Optional[Suit],
        game_state: Dict[str, Any],
    ) -> None:
        """
        Record a call trump decision.

        Parameters
        ----------
        player_id : int
            ID of the player making the decision.
        hand : List[Card]
            Player's hand.
        turned_card : Card
            The turned card (forbidden suit).
        decision : Optional[Suit]
            Suit to call, or None to pass.
        game_state : Dict[str, Any]
            Additional game state.
        """
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

        # Encode decision: 0 = pass, 1-4 = suit indices
        if decision is None:
            decision_class = 0
        else:
            suit_map = {Suit.HEARTS: 1, Suit.DIAMONDS: 2, Suit.CLUBS: 3, Suit.SPADES: 4}
            decision_class = suit_map.get(decision, 0)

        self.call_trump_data.append({
            "game_id": self.current_game_id,
            "player_id": player_id,
            "features": features.numpy().tolist(),
            "decision": decision_class,
            "turned_card": f"{turned_card.rank.value}_{turned_card.suit.value}",
        })

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
        Record a play card decision.

        Parameters
        ----------
        player_id : int
            ID of the player making the decision.
        hand : List[Card]
            Player's hand.
        led_suit : Optional[Suit]
            Suit that was led.
        trump_suit : Optional[Suit]
            Current trump suit.
        trick_cards : List[Card]
            Cards already played in the trick.
        decision : Card
            Card that was played.
        game_state : Dict[str, Any]
            Additional game state.
        """
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
        decision_idx = self.encoder.card_to_index.get(decision, -1)

        self.play_card_data.append({
            "game_id": self.current_game_id,
            "player_id": player_id,
            "features": features.numpy().tolist(),
            "decision": decision_idx,
            "led_suit": led_suit.value if led_suit else None,
            "trump_suit": trump_suit.value if trump_suit else None,
        })

    def record_discard_decision(
        self,
        player_id: int,
        hand: List[Card],
        decision: Card,
        game_state: Dict[str, Any],
    ) -> None:
        """
        Record a discard decision.

        Parameters
        ----------
        player_id : int
            ID of the player making the decision.
        hand : List[Card]
            Player's hand (6 cards after picking up).
        decision : Card
            Card that was discarded.
        game_state : Dict[str, Any]
            Additional game state.
        """
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
        decision_idx = self.encoder.card_to_index.get(decision, -1)

        self.discard_data.append({
            "game_id": self.current_game_id,
            "player_id": player_id,
            "features": features.numpy().tolist(),
            "decision": decision_idx,
        })

    def record_game_outcome(
        self,
        game_id: str,
        scores: tuple[int, int],
        tricks_won: List[List[int]],
        winner: Optional[int],
    ) -> None:
        """
        Record the outcome of a game.

        Parameters
        ----------
        game_id : str
            Unique identifier for the game.
        scores : tuple[int, int]
            Final scores (team0, team1).
        tricks_won : List[List[int]]
            Tricks won per hand for each team.
        winner : Optional[int]
            Winning team ID (0 or 1), or None if game not finished.
        """
        self.game_outcomes.append({
            "game_id": game_id,
            "scores": scores,
            "tricks_won": tricks_won,
            "winner": winner,
        })

    def save_data(self, prefix: str = "training", save_csv: bool = False) -> None:
        """
        Save collected data to files.

        Parameters
        ----------
        prefix : str
            Prefix for output filenames.
        save_csv : bool
            Whether to save CSV files. Default False - CSV files are not used by training
            and add significant overhead. Only enable if you need them for manual analysis.
        """
        # Save as JSON (faster than CSV for large datasets)
        order_up_file = self.output_dir / f"{prefix}_order_up.json"
        call_trump_file = self.output_dir / f"{prefix}_call_trump.json"
        play_card_file = self.output_dir / f"{prefix}_play_card.json"
        discard_file = self.output_dir / f"{prefix}_discard.json"
        outcomes_file = self.output_dir / f"{prefix}_outcomes.json"

        # Use compact JSON (no indentation) for faster writes
        with open(order_up_file, "w") as f:
            json.dump(self.order_up_data, f, separators=(',', ':'))
        with open(call_trump_file, "w") as f:
            json.dump(self.call_trump_data, f, separators=(',', ':'))
        with open(play_card_file, "w") as f:
            json.dump(self.play_card_data, f, separators=(',', ':'))
        with open(discard_file, "w") as f:
            json.dump(self.discard_data, f, separators=(',', ':'))
        with open(outcomes_file, "w") as f:
            json.dump(self.game_outcomes, f, separators=(',', ':'))

        # CSV files are not used by training pipeline - only save if explicitly requested
        # They add significant overhead (pandas DataFrame creation + CSV writing)
        if save_csv:
            if self.order_up_data:
                df = pd.DataFrame(self.order_up_data)
                df.to_csv(self.output_dir / f"{prefix}_order_up.csv", index=False)
            if self.call_trump_data:
                df = pd.DataFrame(self.call_trump_data)
                df.to_csv(self.output_dir / f"{prefix}_call_trump.csv", index=False)
            if self.play_card_data:
                df = pd.DataFrame(self.play_card_data)
                df.to_csv(self.output_dir / f"{prefix}_play_card.csv", index=False)
            if self.discard_data:
                df = pd.DataFrame(self.discard_data)
                df.to_csv(self.output_dir / f"{prefix}_discard.csv", index=False)

    def load_data(self, prefix: str = "training") -> Dict[str, List[Dict[str, Any]]]:
        """
        Load previously collected data.

        Parameters
        ----------
        prefix : str
            Prefix for input filenames.

        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Dictionary containing loaded data.
        """
        order_up_file = self.output_dir / f"{prefix}_order_up.json"
        call_trump_file = self.output_dir / f"{prefix}_call_trump.json"
        play_card_file = self.output_dir / f"{prefix}_play_card.json"
        discard_file = self.output_dir / f"{prefix}_discard.json"

        data = {}
        if order_up_file.exists():
            with open(order_up_file, "r") as f:
                data["order_up"] = json.load(f)
        if call_trump_file.exists():
            with open(call_trump_file, "r") as f:
                data["call_trump"] = json.load(f)
        if play_card_file.exists():
            with open(play_card_file, "r") as f:
                data["play_card"] = json.load(f)
        if discard_file.exists():
            with open(discard_file, "r") as f:
                data["discard"] = json.load(f)

        return data

    def clear_data(self) -> None:
        """Clear all collected data."""
        self.order_up_data.clear()
        self.call_trump_data.clear()
        self.play_card_data.clear()
        self.discard_data.clear()
        self.game_outcomes.clear()

