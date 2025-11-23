"""Game file save/load functionality using UUID-based naming."""

import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from eucher.cards import Card, Deck, Rank, Suit
from eucher.game import Game


def save_game(game: Game, output_dir: Path) -> Path:
    """
    Save a game to a file using UUID-based naming.

    The game will be saved as `<uuid>.gz` where uuid is the game's UUID.
    The file contains all information needed to reconstruct the game state.

    Parameters
    ----------
    game : Game
        The game instance to save.
    output_dir : Path
        Directory where the game file should be saved.

    Returns
    -------
    Path
        Path to the saved game file.

    Raises
    ------
    ValueError
        If the game doesn't have a UUID.
    """
    if not hasattr(game, 'game_uuid') or game.game_uuid is None:
        raise ValueError("Game must have a UUID to save")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    game_file = output_dir / f"{game.game_uuid}.gz"

    # Collect game state
    game_data: Dict[str, Any] = {
        "uuid": game.game_uuid,
        "player_config": [(player.name, player.profile.__class__.__name__) for player in game.players],
        "dealer_id": game.dealer_id,
        "scores": game.scores,
        "trump_suit": game.trump_suit.value if game.trump_suit else None,
        "turned_card": {
            "suit": game.turned_card.suit.value if game.turned_card else None,
            "rank": game.turned_card.rank.value if game.turned_card else None,
        } if game.turned_card else None,
        "trump_selection_risk": game.trump_selection_risk,
        "gameplay_risk": game.gameplay_risk,
    }

    # Save as compressed pickle
    with gzip.open(game_file, 'wb') as f:
        pickle.dump(game_data, f)

    return game_file


def load_game(game_file: Union[str, Path], seed: Optional[Union[int, str]] = None) -> Game:
    """
    Load a game from a file.

    Parameters
    ----------
    game_file : Union[str, Path]
        Path to the game file (.gz) or UUID string.
        If a UUID string is provided, it will look for `<uuid>.gz` in the current directory.
    seed : Optional[Union[int, str]]
        Optional seed to override the saved seed. If None, uses the UUID from the file.

    Returns
    -------
    Game
        Loaded game instance.

    Raises
    ------
    FileNotFoundError
        If the game file doesn't exist.
    ValueError
        If the game file is invalid.
    """
    game_path = Path(game_file)

    # If it's not a file path, treat it as a UUID and look for the file
    if not game_path.exists():
        # Try treating it as a UUID
        if game_path.suffix == '':
            # No extension, might be a UUID
            uuid_str = str(game_path)
            # Look in current directory
            game_path = Path(f"{uuid_str}.gz")
            if not game_path.exists():
                # Try common game directories
                for common_dir in [Path("games"), Path("saved_games"), Path(".")]:
                    potential_path = common_dir / f"{uuid_str}.gz"
                    if potential_path.exists():
                        game_path = potential_path
                        break
                else:
                    raise FileNotFoundError(f"Game file not found for UUID: {uuid_str}")

    if not game_path.exists():
        raise FileNotFoundError(f"Game file not found: {game_path}")

    # Load game data
    try:
        with gzip.open(game_path, 'rb') as f:
            game_data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load game file: {e}") from e

    # Extract UUID
    game_uuid = game_data.get("uuid")
    if seed is None:
        seed = game_uuid

    # Reconstruct player config
    player_config = game_data.get("player_config", [])
    # Convert profile class names back to profile types
    profile_type_map = {
        "HumanProfile": "human",
        "HeuristicPlayer": "heuristic",
        "AIPlayer": "ai",
        "RandomPlayer": "random",
        "MLPlayer": "ml_sklearn",
        "MLBasedProfile": "ml_pytorch",
    }

    player_config_typed: List[Tuple[str, str]] = []
    for name, profile_class in player_config:
        profile_type = profile_type_map.get(profile_class, "heuristic")
        player_config_typed.append((name, profile_type))

    # Create game with seed
    game = Game(
        player_config=player_config_typed,
        trump_selection_risk=game_data.get("trump_selection_risk"),
        gameplay_risk=game_data.get("gameplay_risk"),
        seed=seed,
    )

    # Restore game state
    game.dealer_id = game_data.get("dealer_id", 0)
    game.scores = game_data.get("scores", [0, 0])

    trump_suit_str = game_data.get("trump_suit")
    if trump_suit_str:
        game.trump_suit = Suit(trump_suit_str)

    turned_card_data = game_data.get("turned_card")
    if turned_card_data and turned_card_data.get("suit") and turned_card_data.get("rank"):
        game.turned_card = Card(
            Suit(turned_card_data["suit"]),
            Rank(turned_card_data["rank"])
        )

    return game

