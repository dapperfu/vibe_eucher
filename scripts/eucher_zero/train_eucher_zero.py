#!/usr/bin/env python3
"""Training script for EuchreZero player.

EuchreZero is a search-enhanced reinforcement learning architecture inspired by AlphaZero.
Supports cumulative training, multi-device support, and checkpoint management.
"""

import argparse
import re
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from eucher.game import Game
from eucher.players.computer.euchre_zero.config import EuchreZeroConfig
from eucher.players.computer.euchre_zero.networks.model import EuchreZeroModel
from eucher.players.computer.euchre_zero.training.dashboard import EuchreZeroDashboard
from eucher.players.computer.euchre_zero.training.replay_buffer import ReplayBuffer
from eucher.players.computer.euchre_zero.training.self_play import generate_self_play_game
from eucher.players.computer.euchre_zero.training.trainer import EuchreZeroTrainer


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
    parser = argparse.ArgumentParser(description="Train EuchreZero player")
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
    print("EuchreZero Training Script")
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
    config = EuchreZeroConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        device=device_str,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Initialize model
    model = EuchreZeroModel(config)

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
    trainer = EuchreZeroTrainer(model, config, replay_buffer)

    # Initialize dashboard
    dashboard = EuchreZeroDashboard(num_games=None, refresh_rate=2.0)

    # Training loop
    start_time = time.time()
    game_count = 0
    iteration = 0
    total_examples_collected = 0
    game_times: list[float] = []
    last_checkpoint_path: Optional[str] = None

    try:
        with dashboard.start_live_display(config) as live_display:
            while True:
                iteration += 1
                iteration_start = time.time()
                elapsed = time.time() - start_time

                if elapsed >= max_duration_seconds:
                    break

                # Generate self-play games
                for game_num in range(config.num_games):
                    if time.time() - start_time >= max_duration_seconds:
                        break

                    try:
                        game_start_time = time.time()

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
                        total_examples_collected += len(examples)
                        game_time = time.time() - game_start_time
                        game_times.append(game_time)

                    except Exception as e:
                        # Update dashboard even on error
                        dashboard.update(
                            game_count=game_count,
                            iteration=iteration,
                            examples_collected=total_examples_collected,
                            replay_buffer_size=len(replay_buffer),
                        )
                        dashboard.update_live_display(live_display, config)
                        continue

                # Train on replay buffer
                losses: Dict[str, float] = {}
                if len(replay_buffer) >= config.batch_size:
                    losses = trainer.train_step()

                # Calculate average game time
                avg_game_time = sum(game_times) / len(game_times) if game_times else None
                iteration_time = time.time() - iteration_start

                # Update dashboard
                dashboard.update(
                    game_count=game_count,
                    iteration=iteration,
                    examples_collected=total_examples_collected,
                    replay_buffer_size=len(replay_buffer),
                    policy_loss=losses.get("policy"),
                    value_loss=losses.get("value"),
                    risk_value_loss=losses.get("risk_value"),
                    dynamics_loss=losses.get("dynamics"),
                    reward_loss=losses.get("reward"),
                    total_loss=losses.get("total"),
                    game_time=avg_game_time,
                    simulations_per_move=config.num_simulations,
                    avg_iteration_time=iteration_time,
                )

                # Save checkpoint
                if game_count % args.checkpoint_interval == 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_{game_count}.pt"
                    model.save_checkpoint(checkpoint_path, metadata={"game_count": game_count, "iteration": iteration})
                    last_checkpoint_path = str(checkpoint_path)
                    dashboard.update(
                        game_count=game_count,
                        iteration=iteration,
                        examples_collected=total_examples_collected,
                        replay_buffer_size=len(replay_buffer),
                        checkpoint_game=game_count,
                        checkpoint_path=last_checkpoint_path,
                    )

                dashboard.update_live_display(live_display, config)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()

    # Save final checkpoint
    final_checkpoint = checkpoint_dir / "checkpoint_final.pt"
    model.save_checkpoint(final_checkpoint, metadata={"game_count": game_count, "iteration": iteration})

    # Print final summary
    dashboard.print_summary()


if __name__ == "__main__":
    main()
