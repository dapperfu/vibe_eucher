#!/usr/bin/env python3
"""Training script for EucherZero player.

EucherZero is a search-enhanced reinforcement learning architecture inspired by AlphaZero.
Supports cumulative training, multi-device support, and checkpoint management.
"""

import argparse
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from eucher.game import Game
from eucher.players.computer.eucher_zero.config import EucherZeroConfig
from eucher.players.computer.eucher_zero.networks.model import EucherZeroModel
from eucher.players.computer.eucher_zero.training.replay_buffer import ReplayBuffer
from eucher.players.computer.eucher_zero.training.self_play import generate_self_play_game
from eucher.players.computer.eucher_zero.training.trainer import EucherZeroTrainer


class TrainingInterrupt(Exception):
    """Exception raised when training is interrupted."""

    pass


def parse_duration(duration_str: str) -> Optional[float]:
    """
    Parse duration string to seconds.

    Parameters
    ----------
    duration_str : str
        Duration string in format like "1m", "10m", "1h", "2h", "1d".

    Returns
    -------
    Optional[float]
        Duration in seconds, or None if invalid format.
    """
    if not duration_str:
        return None

    duration_match = re.match(r"(\d+)([mhd])", duration_str.lower())
    if duration_match:
        value = int(duration_match.group(1))
        unit = duration_match.group(2)
        if unit == "m":
            return value * 60.0
        elif unit == "h":
            return value * 3600.0
        elif unit == "d":
            return value * 86400.0

    return None


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find latest checkpoint in directory.

    Parameters
    ----------
    checkpoint_dir : Path
        Checkpoint directory.

    Returns
    -------
    Optional[Path]
        Path to latest checkpoint, or None if not found.
    """
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train EucherZero player")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda"],
        help="Device to train on (default: auto). 'gpu' is alias for 'cuda'",
    )
    parser.add_argument(
        "--duration",
        type=str,
        default="1m",
        help="Training duration in format like '1m', '10m', '1h', '2h', '1d' (default: 1m)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N games (default: 5)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints/eucher_zero",
        help="Checkpoint directory (default: models/checkpoints/eucher_zero)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint to load (if not specified, loads latest)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from latest checkpoint (start from scratch)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=5,
        help="Number of games per iteration (default: 5 for 1-min training)",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=10,
        help="Number of MCTS simulations per decision (default: 10)",
    )

    args = parser.parse_args()

    # Parse duration
    max_duration_seconds = parse_duration(args.duration)
    if max_duration_seconds is None:
        print(f"Warning: Invalid duration format '{args.duration}', using 60 seconds")
        max_duration_seconds = 60.0

    print("=" * 80)
    print("EucherZero Training Script")
    print("=" * 80)
    print(f"Duration: {args.duration} ({max_duration_seconds:.0f} seconds)")
    print(f"Device: {args.device}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Games per iteration: {args.num_games}")
    print(f"MCTS simulations: {args.num_simulations}")
    print("=" * 80)

    # Setup device
    if args.device == "auto":
        import torch

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "gpu":
        device_str = "cuda"
    else:
        device_str = args.device

    # Create config
    config = EucherZeroConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        device=device_str,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Initialize model
    model = EucherZeroModel(config)

    # Load checkpoint if exists
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_resume:
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = find_latest_checkpoint(checkpoint_dir)

        if checkpoint_path and checkpoint_path.exists():
            print(f"\nLoading checkpoint: {checkpoint_path}")
            model.load_checkpoint(checkpoint_path)
            print("Checkpoint loaded successfully")
        else:
            print("\nNo checkpoint found, starting from scratch")

    # Initialize replay buffer and trainer
    replay_buffer = ReplayBuffer(max_size=1000)
    trainer = EucherZeroTrainer(model, config, replay_buffer)

    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    game_count = 0
    iteration = 0

    try:
        while True:
            iteration += 1
            elapsed = time.time() - start_time

            if elapsed >= max_duration_seconds:
                print(f"\nTraining duration reached ({args.duration})")
                break

            print(f"\nIteration {iteration} (elapsed: {elapsed:.1f}s)")

            # Generate self-play games
            print("Generating self-play games...")
            for game_num in range(config.num_games):
                if time.time() - start_time >= max_duration_seconds:
                    break

                try:
                    # Create game with 4 EucherZero players
                    player_config = [
                        ("EucherZero_0", "eucher_zero"),
                        ("EucherZero_1", "eucher_zero"),
                        ("EucherZero_2", "eucher_zero"),
                        ("EucherZero_3", "eucher_zero"),
                    ]

                    game = Game(player_config)

                    # Set game reference for all EucherZero players
                    for player in game.players:
                        if hasattr(player.profile, "set_game"):
                            player.profile.set_game(game)

                    # Play hand and collect training data
                    continue_game = game.play_hand()

                    # Collect training examples from self-play
                    examples = generate_self_play_game(model, config, game=game, risk_factor=0.0)

                    # Add to replay buffer
                    for example in examples:
                        replay_buffer.add(
                            example.state,
                            example.policy,
                            example.value,
                            example.reward,
                        )

                    game_count += 1
                    if len(examples) > 0:
                        print(f"  Game {game_num + 1}/{config.num_games} completed, collected {len(examples)} examples")
                    else:
                        print(f"  Game {game_num + 1}/{config.num_games} completed (no examples collected)")

                except Exception as e:
                    print(f"Error in self-play game: {e}")
                    continue

            # Train on replay buffer
            if len(replay_buffer) >= config.batch_size:
                print("Training on replay buffer...")
                losses = trainer.train_step()
                print(f"Losses: {losses}")

            # Save checkpoint
            if game_count % args.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{game_count}.pt"
                print(f"Saving checkpoint: {checkpoint_path}")
                model.save_checkpoint(checkpoint_path, metadata={"game_count": game_count, "iteration": iteration})

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback

        traceback.print_exc()

    # Save final checkpoint
    final_checkpoint = checkpoint_dir / "checkpoint_final.pt"
    print(f"\nSaving final checkpoint: {final_checkpoint}")
    model.save_checkpoint(final_checkpoint, metadata={"game_count": game_count, "iteration": iteration})

    print(f"\nTraining complete!")
    print(f"Total games: {game_count}")
    print(f"Total iterations: {iteration}")
    print(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
