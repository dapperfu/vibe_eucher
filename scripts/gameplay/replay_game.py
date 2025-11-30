"""Script to replay a game from training data files.

This script loads training data files and reconstructs/displays a game replay
based on the game_id stored in the data.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eucher.cards import Card, Rank, Suit
from eucher.players.computer.ml.ml_config import MLConfig
from eucher.players.computer.ml.ml_features import GameStateEncoder


def parse_card_string(card_str: str) -> Optional[Card]:
    """Parse a card string like 'KING_Hearts' into a Card object."""
    try:
        parts = card_str.split("_")
        if len(parts) == 2:
            rank_str, suit_str = parts
            rank = Rank[rank_str.upper()]
            suit = Suit(suit_str.capitalize())
            return Card(suit, rank)
    except (KeyError, ValueError):
        pass
    return None


def load_training_data_files(data_dir: Path) -> Dict[str, Dict]:
    """
    Load all training data files and extract decisions with game_id.

    Parameters
    ----------
    data_dir : Path
        Directory containing training data files.

    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping game_id to lists of decisions.
    """
    games: Dict[str, Dict[str, List]] = {}
    
    # Find all .gz and .npz files
    data_files = list(data_dir.glob("*.gz")) + list(data_dir.glob("*.npz"))
    
    encoder = GameStateEncoder()
    
    for data_file in data_files:
        # Try to find corresponding metadata file
        metadata_file = data_file.with_suffix(".txt")
        if not metadata_file.exists():
            continue
            
        # Read metadata to determine dataset type
        with open(metadata_file) as f:
            metadata = f.read()
            dataset_name = None
            for line in metadata.split("\n"):
                if line.startswith("Dataset:"):
                    dataset_name = line.split(":", 1)[1].strip()
                    break
            
            if not dataset_name:
                continue
        
        # Load the data
        try:
            loaded = np.load(data_file)
            X = loaded["X"]
            y = loaded["y"]
            
            # Try to extract game_id information from metadata or reconstruct
            # Note: The current format doesn't preserve game_id in the saved arrays,
            # so we can only show decisions without full game context
            # This is a limitation of the current data format
            
            # For now, we'll create a synthetic game_id based on the file
            # In a future version, we should save game_id mapping separately
            file_uuid = data_file.stem
            
            # Store decisions by dataset type
            if file_uuid not in games:
                games[file_uuid] = {
                    "order_up": [],
                    "call_trump": [],
                    "play_card": [],
                    "discard": [],
                }
            
            # Store the decisions (we can't recover game_id from numpy arrays)
            games[file_uuid][dataset_name] = {
                "X": X,
                "y": y,
                "file": data_file.name,
            }
            
        except Exception as e:
            print(f"Warning: Failed to load {data_file}: {e}", file=sys.stderr)
            continue
    
    return games


def display_game_replay(
    game_id: str,
    data_dir: Path,
    encoder: GameStateEncoder,
) -> None:
    """
    Display a game replay from training data.

    Parameters
    ----------
    game_id : str
        Game ID or UUID of the data file to replay.
    data_dir : Path
        Directory containing training data files.
    encoder : GameStateEncoder
        Encoder for decoding card indices.
    """
    # Check if game_id is a file UUID
    game_file = data_dir / f"{game_id}.gz"
    if not game_file.exists():
        game_file = data_dir / f"{game_id}.npz"
    
    if not game_file.exists():
        print(f"Error: File not found: {game_id}.gz or {game_id}.npz")
        print(f"Looking in: {data_dir}")
        return
    
    # Load the file
    try:
        loaded = np.load(game_file)
        X = loaded["X"]
        y = loaded["y"]
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Load metadata
    metadata_file = game_file.with_suffix(".txt")
    dataset_type = "unknown"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = f.read()
            for line in metadata.split("\n"):
                if line.startswith("Dataset:"):
                    dataset_type = line.split(":", 1)[1].strip()
                    break
    
    print("=" * 80)
    print(f"Game Replay: {game_id}")
    print(f"Dataset Type: {dataset_type}")
    print(f"File: {game_file.name}")
    print("=" * 80)
    print()
    
    print(f"Total decisions: {len(y)}")
    print()
    
    # Display decisions
    for i, (features, decision) in enumerate(zip(X, y)):
        print(f"Decision {i + 1}:")
        print(f"  Decision value: {decision}")
        
        if dataset_type == "play_card" or dataset_type == "discard":
            card = encoder.decode_card_index(int(decision))
            if card:
                print(f"  Card: {card}")
            else:
                print(f"  Card: (invalid index {decision})")
        elif dataset_type == "call_trump":
            suit_map = {0: "Pass", 1: "Hearts", 2: "Diamonds", 3: "Clubs", 4: "Spades"}
            suit_name = suit_map.get(int(decision), f"Unknown ({decision})")
            print(f"  Trump suit: {suit_name}")
        elif dataset_type == "order_up":
            decision_text = "Order up" if int(decision) == 1 else "Pass"
            print(f"  Decision: {decision_text}")
        
        print()
    
    print("=" * 80)
    print("Note: Full game reconstruction requires game_id mapping in data files.")
    print("Current format only shows decisions from individual files.")
    print("=" * 80)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Replay a game from training data files"
    )
    parser.add_argument(
        "game_id",
        type=str,
        help="Game ID (UUID) or file UUID to replay",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing training data (default: from MLConfig)",
    )
    
    args = parser.parse_args()
    
    # Get data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        config = MLConfig()
        data_dir = config.training_data_dir
    
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    encoder = GameStateEncoder()
    display_game_replay(args.game_id, data_dir, encoder)


if __name__ == "__main__":
    main()

