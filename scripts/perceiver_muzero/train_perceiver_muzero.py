#!/usr/bin/env python3
"""Training script for EucherPerceiverMuZero player.

EucherPerceiverMuZero is a MuZero-style planning architecture with Perceiver-IO encoder.
Supports cumulative training, multi-device support, and checkpoint management.
"""

import argparse
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eucher.game import Game

from eucher.players.computer.perceiver_muzero.config import PerceiverMuZeroConfig
from eucher.players.computer.perceiver_muzero.networks.model import PerceiverMuZeroModel
from eucher.players.computer.perceiver_muzero.training.dashboard import (
    PerceiverMuZeroDashboard,
)
from eucher.players.computer.perceiver_muzero.training.replay_buffer import ReplayBuffer
from eucher.players.computer.perceiver_muzero.training.self_play import (
    generate_self_play_game,
)
from eucher.players.computer.perceiver_muzero.training.trainer import (
    PerceiverMuZeroTrainer,
)


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

    duration_match = re.match(r"(\d+)([smhd])", duration_str.lower())
    if duration_match:
        value = int(duration_match.group(1))
        unit = duration_match.group(2)
        if unit == "s":
            return float(value)
        elif unit == "m":
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
    parser = argparse.ArgumentParser(description="Train EucherPerceiverMuZero player")
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
        help="Training duration in format like '10s', '1m', '10m', '1h', '2h', '1d' (default: 1m)",
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
        default="models/checkpoints/perceiver_muzero",
        help="Checkpoint directory (default: models/checkpoints/perceiver_muzero)",
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
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=10,
        help="Number of games per iteration (default: 10)",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=128,
        help="Number of MCTS simulations per decision (default: 128)",
    )

    args = parser.parse_args()

    # Parse duration
    max_duration_seconds = parse_duration(args.duration)
    if max_duration_seconds is None:
        print(f"Warning: Invalid duration format '{args.duration}', using 60 seconds")
        max_duration_seconds = 60.0

    print("=" * 80)
    print("EucherPerceiverMuZero Training Script")
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
    config = PerceiverMuZeroConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        device=device_str,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Initialize model
    model = PerceiverMuZeroModel(config)

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
    replay_buffer = ReplayBuffer(config)
    trainer = PerceiverMuZeroTrainer(model, config, replay_buffer)

    # Initialize Rich dashboard
    # Estimate total games for duration-based training
    estimated_games = int(max_duration_seconds / 2.0) if max_duration_seconds else config.num_games * 100
    dashboard = PerceiverMuZeroDashboard(num_games=estimated_games, refresh_rate=2.0)
    live_display = dashboard.start_live_display(config=config)

    # Training loop
    start_time = time.time()
    game_count = 0
    iteration = 0
    total_examples = 0
    last_losses = None

    try:
        with live_display:
            while True:
                iteration += 1
                elapsed = time.time() - start_time

                if elapsed >= max_duration_seconds:
                    break

                # Generate self-play games
                iteration_start_time = time.time()
                iteration_examples = 0

                for game_num in range(config.num_games):
                    if time.time() - start_time >= max_duration_seconds:
                        break

                    try:
                        game_start_time = time.time()

                        # Create game with 4 PerceiverMuZero players
                        player_config = [
                            ("PerceiverMuZero_0", "perceiver_muzero"),
                            ("PerceiverMuZero_1", "perceiver_muzero"),
                            ("PerceiverMuZero_2", "perceiver_muzero"),
                            ("PerceiverMuZero_3", "perceiver_muzero"),
                        ]

                        game = Game(player_config)

                        # Set game reference for all players
                        for player in game.players:
                            if hasattr(player.profile, "set_game"):
                                player.profile.set_game(game)

                        # Play hand and collect training data
                        continue_game = game.play_hand()
                        game_time = time.time() - game_start_time

                        # Collect training examples from self-play
                        examples = generate_self_play_game(
                            model, config, game=game, risk_factor=0.0
                        )

                        # Add to replay buffer
                        for example in examples:
                            replay_buffer.add(
                                example.input_tokens,
                                example.action,
                                example.policy,
                                example.value,
                                example.reward,
                                example.next_input_tokens,
                            )

                        game_count += 1
                        iteration_examples += len(examples)
                        total_examples += len(examples)

                    except Exception as e:
                        # Log error but continue
                        import traceback

                        traceback.print_exc()
                        continue

                # Train on replay buffer
                if len(replay_buffer) >= config.batch_size:
                    losses = trainer.train_step()
                    last_losses = losses

                # Update dashboard
                dashboard.update(
                    game_count=game_count,
                    iteration=iteration,
                    examples_collected=total_examples,
                    replay_buffer_size=len(replay_buffer),
                    policy_loss=last_losses.get("policy") if last_losses else None,
                    value_loss=last_losses.get("value") if last_losses else None,
                    reward_loss=last_losses.get("reward") if last_losses else None,
                    entropy=last_losses.get("entropy") if last_losses else None,
                    total_loss=last_losses.get("total") if last_losses else None,
                    game_time=game_time if "game_time" in locals() else None,
                )
                dashboard.update_live_display(live_display, config=config)

                # Save checkpoint
                if game_count % args.checkpoint_interval == 0 and game_count > 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_{game_count}.pt"
                    model.save_checkpoint(
                        checkpoint_path,
                        metadata={"game_count": game_count, "iteration": iteration},
                    )

    except KeyboardInterrupt:
        pass  # Dashboard will handle display cleanup
    except Exception as e:
        import traceback

        traceback.print_exc()
    finally:
        # Close live display
        if "live_display" in locals():
            live_display.__exit__(None, None, None)

    # Save final checkpoint
    final_checkpoint = checkpoint_dir / "checkpoint_final.pt"
    model.save_checkpoint(
        final_checkpoint, metadata={"game_count": game_count, "iteration": iteration}
    )

    # Print final summary
    dashboard.print_summary()
    print(f"\nFinal checkpoint saved: {final_checkpoint}")


if __name__ == "__main__":
    main()

