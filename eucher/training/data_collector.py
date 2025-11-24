"""Data collection for training ML models."""

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from eucher.cards import Card, Suit
from eucher.players.computer.ml.ml_features import GameStateEncoder

try:
    from eucher.training.system_info import (
        format_system_info,
        get_system_info,
    )
except ImportError:
    # Fallback if system_info not available
    def get_system_info() -> Dict[str, any]:
        """Fallback system info function."""
        return {"timestamp": "", "hostname": "unknown"}

    def format_system_info(info: Dict[str, any]) -> str:
        """Fallback system info formatter."""
        timestamp = info.get("timestamp", "unknown")
        hostname = info.get("hostname", "unknown")
        return f"Timestamp: {timestamp}\nHostname: {hostname}"


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
        from eucher.players.computer.ml.ml_config import MLConfig

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
        
        # Track saved game replays
        self.saved_game_replays: List[str] = []

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

        # Check if this decision resulted in a renege
        is_renege = game_state.get("is_renege", False)
        
        self.play_card_data.append({
            "game_id": self.current_game_id,
            "player_id": player_id,
            "features": features.numpy().tolist(),
            "decision": decision_idx,
            "led_suit": led_suit.value if led_suit else None,
            "trump_suit": trump_suit.value if trump_suit else None,
            "is_renege": is_renege,  # Mark renege for training
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
        renege_occurred: bool = False,
        renege_player_id: Optional[int] = None,
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
        renege_occurred : bool
            Whether a renege occurred in this game.
        renege_player_id : Optional[int]
            Player ID who reneged, if any.
        """
        self.game_outcomes.append({
            "game_id": game_id,
            "scores": scores,
            "tricks_won": tricks_won,
            "winner": winner,
            "renege_occurred": renege_occurred,
            "renege_player_id": renege_player_id,
        })

    def save_data(
        self,
        prefix: Optional[str] = None,
        use_uuid_naming: bool = True,
        append: bool = True,
        file_extension: str = "npz",
    ) -> List[Path]:
        """
        Save collected data to files.

        Uses UUID-based naming by default for easy rsync and accumulation across machines.
        Each dataset gets its own UUID and metadata file.

        Parameters
        ----------
        prefix : Optional[str]
            Prefix for output filenames. If None and use_uuid_naming is False, uses "training".
            Ignored if use_uuid_naming is True.
        use_uuid_naming : bool
            If True, uses UUID-based naming (<UUID>.npz with <UUID>.txt metadata).
            If False, uses traditional prefix-based naming.
        append : bool
            If True and use_uuid_naming is False, appends to existing files.
            If False, overwrites existing files.
        file_extension : str
            File extension to use (default: "npz").

        Returns
        -------
        List[Path]
            List of paths to saved files.
        """
        saved_files: List[Path] = []
        datasets = [
            ("order_up", self.order_up_data),
            ("call_trump", self.call_trump_data),
            ("play_card", self.play_card_data),
            ("discard", self.discard_data),
        ]

        # Get system info for metadata
        system_info = get_system_info()
        metadata_text = format_system_info(system_info)

        for name, data_list in datasets:
            if not data_list:
                continue

            # Extract features and decisions as numpy arrays
            features_list = [item["features"] for item in data_list]
            decisions_list = [item["decision"] for item in data_list]

            # Convert to numpy arrays (much faster than JSON)
            X_new = np.array(features_list, dtype=np.float32)
            y_new = np.array(decisions_list, dtype=np.int32)

            if use_uuid_naming:
                # Generate UUID for this dataset
                dataset_uuid = str(uuid.uuid4())
                # NumPy compressed format always uses .npz extension
                data_file = self.output_dir / f"{dataset_uuid}.npz"
                metadata_file = self.output_dir / f"{dataset_uuid}.txt"

                # Save data using numpy's compressed format
                np.savez_compressed(data_file, X=X_new, y=y_new)
                saved_files.append(data_file)

                # Save metadata with additional dataset info
                metadata_content = metadata_text
                metadata_content += f"\nDataset: {name}\n"
                metadata_content += f"UUID: {dataset_uuid}\n"
                metadata_content += f"File: {dataset_uuid}.npz\n"
                metadata_content += f"Records: {len(data_list)}\n"
                metadata_content += f"Features shape: {X_new.shape}\n"
                metadata_content += f"Decisions shape: {y_new.shape}\n"
                game_ids = set(item.get("game_id", "") for item in data_list)
                metadata_content += f"Game IDs: {len(game_ids)}\n"
                # Save list of game IDs for this dataset
                if game_ids:
                    metadata_content += f"Game ID List: {','.join(sorted(game_ids))}\n"

                with open(metadata_file, "w") as f:
                    f.write(metadata_content)
                saved_files.append(metadata_file)

            else:
                # Traditional prefix-based naming with optional append
                if prefix is None:
                    prefix = "training"
                npz_file = self.output_dir / f"{prefix}_{name}.{file_extension}"

                if append and npz_file.exists():
                    # Load existing data and append
                    existing = np.load(npz_file)
                    X_existing = existing["X"]
                    y_existing = existing["y"]

                    # Concatenate new data with existing
                    X_combined = np.concatenate(
                        [X_existing, X_new], axis=0
                    )
                    y_combined = np.concatenate(
                        [y_existing, y_new], axis=0
                    )

                    # Save combined data
                    np.savez_compressed(npz_file, X=X_combined, y=y_combined)
                else:
                    # Save new data (overwrite or create new)
                    np.savez_compressed(npz_file, X=X_new, y=y_new)

                saved_files.append(npz_file)

        # Add summary of saved game replays to return value
        # (game replays are saved individually during game execution)
        return saved_files

    def load_data(
        self,
        prefix: Optional[str] = None,
        use_uuid_naming: bool = True,
        dataset_type: Optional[str] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load previously collected data from .npz files.

        Parameters
        ----------
        prefix : Optional[str]
            Prefix for input filenames (used only if use_uuid_naming is False).
            If None and use_uuid_naming is False, uses "training".
        use_uuid_naming : bool
            If True, loads all UUID-based files in the directory.
            If False, loads prefix-based files.
        dataset_type : Optional[str]
            If provided and use_uuid_naming is True, only loads files matching this dataset type.
            Can be "order_up", "call_trump", "play_card", or "discard".

        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            Dictionary containing loaded data with 'X' (features) and 'y' (labels)
            for each dataset. If use_uuid_naming is True, keys are dataset types.
            Data is concatenated from all matching files.
        """
        data: Dict[str, Dict[str, np.ndarray]] = {}

        if use_uuid_naming:
            # Load all UUID-based files
            # Find all .npz files
            data_files = list(self.output_dir.glob("*.npz"))

            # Group by dataset type based on metadata
            dataset_arrays: Dict[str, List[np.ndarray]] = {}
            dataset_labels: Dict[str, List[np.ndarray]] = {}

            for data_file in data_files:
                # Try to find corresponding metadata file
                metadata_file = data_file.with_suffix(".txt")
                if metadata_file.exists():
                    # Read metadata to determine dataset type
                    with open(metadata_file) as f:
                        metadata = f.read()
                        # Extract dataset type from metadata
                        dataset_name = None
                        for line in metadata.split("\n"):
                            if line.startswith("Dataset:"):
                                dataset_name = line.split(":", 1)[1].strip()
                                break

                        if dataset_name and (dataset_type is None or dataset_name == dataset_type):
                            # Load the data - training data files are always .npz format
                            if data_file.suffix == ".npz":
                                loaded = np.load(data_file)
                            else:
                                continue  # Skip non-numpy files
                            
                            if dataset_name not in dataset_arrays:
                                dataset_arrays[dataset_name] = []
                                dataset_labels[dataset_name] = []
                            dataset_arrays[dataset_name].append(loaded["X"])
                            dataset_labels[dataset_name].append(loaded["y"])

            # Concatenate all data for each dataset type
            for dataset_name, X_list in dataset_arrays.items():
                if X_list:
                    X_combined = np.concatenate(X_list, axis=0)
                    y_combined = np.concatenate(dataset_labels[dataset_name], axis=0)
                    data[dataset_name] = {"X": X_combined, "y": y_combined}
        else:
            # Traditional prefix-based loading
            if prefix is None:
                prefix = "training"
            datasets = ["order_up", "call_trump", "play_card", "discard"]

            for dataset_name in datasets:
                # Load .npz file
                npz_file = self.output_dir / f"{prefix}_{dataset_name}.npz"
                if npz_file.exists():
                    loaded = np.load(npz_file)
                    data[dataset_name] = {"X": loaded["X"], "y": loaded["y"]}

        return data

    def clear_data(self) -> None:
        """Clear all collected data."""
        self.order_up_data.clear()
        self.call_trump_data.clear()
        self.play_card_data.clear()
        self.discard_data.clear()
        self.game_outcomes.clear()

