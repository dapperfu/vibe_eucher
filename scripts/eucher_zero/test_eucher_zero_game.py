#!/usr/bin/env python3
"""Test script to play a complete game with 2 EucherZero bots vs 2 baseline players."""

import argparse
from pathlib import Path

from eucher.game import Game
from eucher.players.computer.eucher_zero.config import EucherZeroConfig
from eucher.players.computer.eucher_zero.networks.model import EucherZeroModel


def main() -> None:
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test EucherZero with complete game")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda"],
        help="Device to run on (default: auto)",
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        import torch

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "gpu":
        device_str = "cuda"
    else:
        device_str = args.device

    print("=" * 80)
    print("EucherZero Test Game")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device_str}")
    print("=" * 80)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    config = EucherZeroConfig(device=device_str)
    model = EucherZeroModel(config)
    model.load_checkpoint(checkpoint_path)
    print("Model loaded successfully")

    # Create game with 2 EucherZero bots vs 2 baseline players
    player_config = [
        ("EucherZero_0", "eucher_zero"),  # Team 0
        ("Heuristic_1", "heuristic"),  # Team 1
        ("EucherZero_2", "eucher_zero"),  # Team 0
        ("Heuristic_3", "heuristic"),  # Team 1
    ]

    print("\nCreating game...")
    print("Players:")
    for i, (name, profile_type) in enumerate(player_config):
        print(f"  Player {i}: {name} ({profile_type})")

    game = Game(player_config)

    # Set game reference for EucherZero players
    for player in game.players:
        if hasattr(player.profile, "set_game"):
            player.profile.set_game(game)

    print("\nStarting game...")
    print("=" * 80)

    # Play game until completion
    hand_count = 0
    while True:
        hand_count += 1
        print(f"\n--- Hand {hand_count} ---")
        print(f"Dealer: {game.players[game.dealer_id].name}")

        continue_game = game.play_hand()

        scores = game.get_scores()
        print(f"Scores: Team 0 = {scores[0]}, Team 1 = {scores[1]}")

        # Check for game over
        winner = game.get_winner()
        if winner is not None:
            print("\n" + "=" * 80)
            print(f"GAME OVER! Team {winner} wins!")
            print(f"Final scores: Team 0 = {scores[0]}, Team 1 = {scores[1]}")
            print(f"Total hands played: {hand_count}")
            print("=" * 80)
            break

        if not continue_game:
            print("\nGame ended without winner")
            break

    print("\nTest complete!")


if __name__ == "__main__":
    main()

