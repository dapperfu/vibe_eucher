#!/usr/bin/env python3
"""Script to display probability distributions for computer player decisions.

This script creates a game with different types of computer players (random, AI, ML, PyTorch)
and displays the probability distributions for each decision they make:
- Order up decisions (True/False)
- Call trump decisions (each suit + pass)
- Card discard decisions (each card in hand)
- Card play decisions (each valid card)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from eucher.game import Game
from eucher.players.computer.ai import AIDecisionMaker, AIPlayer
from eucher.players.computer import HeuristicPlayer, RandomPlayer
from eucher.players.base import PlayerProfile
from eucher.players.computer.ml.ml_config import MLConfig
from eucher.players.computer.ml.ml_features import GameStateEncoder
from eucher.players.computer.ml.ml_model import EucherMLModel
from eucher.players.computer.ml.player import MLPlayer
from src.probability_wrapper import ProbabilityAwareProfile

# Optional import for PyTorch players
try:
    from eucher.ai_players.pytorch_player import PyTorchStrategicPlayer
except ImportError:
    PyTorchStrategicPlayer = None  # type: ignore


def create_player_config_with_probabilities(
    player_types: List[str],
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Game:
    """
    Create a game with players wrapped in probability-aware profiles.

    Parameters
    ----------
    player_types : List[str]
        List of player types: "random", "weighted_heuristic", "heuristic", "ml_sklearn", "pytorch"
    model_path : Optional[str]
        Path to PyTorch model file (for pytorch players).
    device : Optional[str]
        Device to use: "cpu" or "cuda" (for pytorch players).

    Returns
    -------
    Game
        Created game instance with wrapped profiles.
    """
    if len(player_types) != 4:
        raise ValueError("Must provide exactly 4 player types")

    # Create base profiles
    base_profiles: List[PlayerProfile] = []
    for i, player_type in enumerate(player_types):
        if player_type == "random":
            profile = RandomPlayer()
        elif player_type == "weighted_heuristic" or player_type == "ai":
            profile = AIPlayer(AIDecisionMaker())
        elif player_type == "heuristic":
            profile = HeuristicPlayer()
        elif player_type == "ml_sklearn":
            # Create game state provider function
            def get_game_state() -> tuple[int, int, int]:
                """Get current game state for ML player."""
                return (0, 0, 0)  # Will be updated by game

            profile = MLPlayer(
                backend="supervised",
                model_type="random_forest",
                game_state_provider=get_game_state,
            )
        elif player_type == "pytorch":
            if PyTorchStrategicPlayer is None:
                raise ImportError("PyTorchStrategicPlayer not available. Install PyTorch dependencies.")
            profile = PyTorchStrategicPlayer(
                model_path=model_path,
                device=device,
                use_strategic_overrides=True,
            )
        else:
            raise ValueError(f"Unknown player type: {player_type}")

        base_profiles.append(profile)

    # Wrap profiles with probability-aware wrapper
    wrapped_profiles: List[PlayerProfile] = []
    for profile in base_profiles:
        wrapped_profile = ProbabilityAwareProfile(profile)
        wrapped_profiles.append(wrapped_profile)

    # Create game with placeholder config, then replace profiles
    player_config = [
        ("Player 1", "heuristic"),
        ("Player 2", "heuristic"),
        ("Player 3", "heuristic"),
        ("Player 4", "heuristic"),
    ]
    game = Game(player_config)

    # Replace profiles with wrapped ones
    for i, profile in enumerate(wrapped_profiles):
        game.players[i].profile = profile
        game.players[i].name = f"{player_types[i].title()} Player {i+1}"

    # Update game state provider for ML players and PyTorch players
    def create_game_state_provider(game_instance: Game):
        """Create game state provider function for ML players."""

        def get_game_state() -> tuple[int, int, int]:
            """Get current game state for ML player."""
            trick_num = getattr(game_instance, "_current_trick_number", 0)
            tricks_won = getattr(game_instance, "_current_tricks_won", [0, 0])
            return (trick_num, tricks_won[0], tricks_won[1])

        return get_game_state

    game_state_provider = create_game_state_provider(game)

    # Update ML players with game state provider
    # Also update PyTorch players with scores
    for i, player in enumerate(game.players):
        if isinstance(player.profile, ProbabilityAwareProfile):
            wrapped = player.profile.wrapped_profile
            if isinstance(wrapped, MLPlayer):
                wrapped.game_state_provider = game_state_provider
            elif PyTorchStrategicPlayer is not None and isinstance(wrapped, PyTorchStrategicPlayer):
                # Update scores for PyTorch players
                wrapped.update_scores(game.scores)
                wrapped.dealer_id = game.dealer_id

    return game


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Display probability distributions for computer player decisions"
    )
    parser.add_argument(
        "--players",
        type=str,
        default="random,weighted_heuristic,ml_sklearn,pytorch",
        help=(
            "Comma-separated list of 4 player types: "
            "random, weighted_heuristic, heuristic, ml_sklearn, pytorch "
            "(default: random,weighted_heuristic,ml_sklearn,pytorch)"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to PyTorch model file (for pytorch players)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cpu' or 'cuda' (for pytorch players, default: auto-detect)",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=1,
        help="Number of hands to play (default: 1)",
    )

    args = parser.parse_args()

    # Parse player types
    try:
        player_types = [pt.strip().lower() for pt in args.players.split(",")]
        if len(player_types) != 4:
            raise ValueError("Must provide exactly 4 player types")
        valid_types = {"random", "weighted_heuristic", "ai", "heuristic", "ml_sklearn", "pytorch"}
        for pt in player_types:
            if pt not in valid_types:
                raise ValueError(f"Invalid player type: {pt}. Must be one of {valid_types}")
    except ValueError as e:
        print(f"Error parsing player types: {e}")
        print("Expected format: player1,player2,player3,player4")
        print("Valid types: random, weighted_heuristic, heuristic, ml_sklearn, pytorch")
        return

    print("=" * 80)
    print("Decision Probability Display")
    print("=" * 80)
    print("")
    print("Player Types:")
    for i, pt in enumerate(player_types):
        print(f"  Player {i+1}: {pt.title()}")
    print("")

    if args.model_path:
        print(f"Using PyTorch model: {args.model_path}")
    print("")

    try:
        # Create game with wrapped profiles
        game = create_player_config_with_probabilities(
            player_types=player_types,
            model_path=args.model_path,
            device=args.device,
        )

        print("Game created successfully")
        print("")
        print("=" * 80)
        print("Starting game - probabilities will be displayed for each decision")
        print("=" * 80)
        print("")

        # Helper function to update player state
        def update_player_state() -> None:
            """Update state for all players (scores, trick history, etc.)."""
            for player in game.players:
                if isinstance(player.profile, ProbabilityAwareProfile):
                    wrapped = player.profile.wrapped_profile
                    # Update PyTorch players
                    if PyTorchStrategicPlayer is not None and isinstance(wrapped, PyTorchStrategicPlayer):
                        wrapped.update_scores(game.scores)
                        wrapped.dealer_id = game.dealer_id

        # Play specified number of hands
        for hand_num in range(1, args.hands + 1):
            print("=" * 80)
            print(f"Hand {hand_num}")
            print("=" * 80)
            print("")

            # Update player state before hand
            update_player_state()

            continue_game = game.play_hand()

            # Update player state after hand
            update_player_state()

            print("")
            print(f"Hand {hand_num} complete")
            print(f"Current scores: Team 0 = {game.scores[0]}, Team 1 = {game.scores[1]}")
            print("")

            # Check for game over
            winner = game.get_winner()
            if winner is not None:
                print("=" * 80)
                print("GAME OVER")
                print("=" * 80)
                print(f"Team {winner} wins!")
                print(f"Final scores: Team 0 = {game.scores[0]}, Team 1 = {game.scores[1]}")
                print("")
                break

            if not continue_game:
                print("Game ended (no continue)")
                break

        print("=" * 80)
        print("Game Summary")
        print("=" * 80)
        print(f"Total hands played: {hand_num}")
        print(f"Final scores: Team 0 = {game.scores[0]}, Team 1 = {game.scores[1]}")
        winner = game.get_winner()
        if winner is not None:
            print(f"Winner: Team {winner}")
        print("")

    except Exception as e:
        print(f"ERROR: Game failed with exception: {e}")
        import traceback

        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

