#!/usr/bin/env python3
"""Training script for EucherGo player.

EucherGo is a hybrid Monte Carlo Tree Search and Deep Neural Network engine
for Euchre, patterned after AlphaZero/MuZero architecture.
"""

import argparse
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from eucher.players.computer.euchergo.config import EucherGoConfig
from eucher.players.computer.euchergo.networks.model import EucherGoModel
from eucher.players.computer.euchergo.training.dashboard import EucherGoDashboard


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
    parser = argparse.ArgumentParser(description="Train EucherGo player")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda"],
        help="Device to train on (default: auto)",
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
        help="Save checkpoint every N iterations (default: 5)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints/euchergo",
        help="Checkpoint directory (default: models/checkpoints/euchergo)",
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
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=10,
        help="Number of games per training iteration (default: 10)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Maximum number of training iterations (default: unlimited)",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=100,
        help="Number of MCTS simulations per decision (default: 100)",
    )
    parser.add_argument(
        "--evaluate-every",
        type=int,
        default=10,
        help="Evaluate model every N iterations (default: 10)",
    )

    args = parser.parse_args()

    # Parse duration
    max_duration_seconds = parse_duration(args.duration)
    if max_duration_seconds is None:
        print(f"Warning: Invalid duration format '{args.duration}', using 60 seconds")
        max_duration_seconds = 60.0

    print("=" * 80)
    print("EucherGo Training Script")
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
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "gpu":
        device_str = "cuda"
    else:
        device_str = args.device

    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Create config
    config = EucherGoConfig(
        num_simulations=args.num_simulations,
        device=device,
    )
    # Add batch_size and learning_rate to config for dashboard display
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate

    # Initialize model
    model = EucherGoModel(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        use_conv1d=config.use_conv1d,
    )
    model.to(device)

    # Load checkpoint if exists
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    iteration = 0
    if not args.no_resume:
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = find_latest_checkpoint(checkpoint_dir)

        if checkpoint_path and checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            model.load_checkpoint(checkpoint_path)
            # Extract iteration number from filename if possible
            match = re.search(r"iter_(\d+)", checkpoint_path.stem)
            if match:
                iteration = int(match.group(1))
            print(f"Resuming from iteration {iteration}")

    # Setup signal handler for graceful shutdown
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\nTraining interrupted. Saving checkpoint...")

    signal.signal(signal.SIGINT, signal_handler)

    # Create dashboard
    dashboard = EucherGoDashboard(
        num_iterations=args.num_iterations,
        num_games_per_iteration=args.num_games,
        refresh_rate=2.0,
    )

    # Training loop
    start_time = time.time()
    games_played = 0
    last_checkpoint_iteration = None
    last_checkpoint_path = None

    # Track metrics for dashboard
    iteration_times = []
    game_times = []
    mcts_times = []

    print("\nStarting training...")

    try:
        with dashboard.start_live_display(config=config) as live_display:
            while True:
                if interrupted:
                    break

                elapsed = time.time() - start_time
                if max_duration_seconds and elapsed >= max_duration_seconds:
                    dashboard.console.print(f"\n[bold yellow]Training duration ({args.duration}) reached.[/bold yellow]")
                    break

                if args.num_iterations and iteration >= args.num_iterations:
                    dashboard.console.print(f"\n[bold yellow]Maximum iterations ({args.num_iterations}) reached.[/bold yellow]")
                    break

                iteration_start_time = time.time()

                # TODO: Implement self-play generation
                # TODO: Implement training on replay buffer
                # TODO: Implement evaluation
                
                # Placeholder: simulate some work for demonstration
                # In real implementation, this would be:
                # - Generate self-play games
                # - Train on replay buffer
                # - Calculate losses
                # - Evaluate model
                time.sleep(0.1)  # Placeholder delay

                # Update games played (placeholder - will be actual count in real implementation)
                games_played += args.num_games

                # Calculate iteration time
                iteration_time = time.time() - iteration_start_time
                iteration_times.append(iteration_time)
                avg_iteration_time = sum(iteration_times) / len(iteration_times) if iteration_times else None

                # Placeholder metrics (will be actual values in real implementation)
                policy_loss = None  # TODO: Calculate from training
                value_loss = None  # TODO: Calculate from training
                total_loss = None  # TODO: Calculate from training
                avg_mcts_time = None  # TODO: Track MCTS time
                avg_simulations_per_move = args.num_simulations  # From config
                avg_game_time = None  # TODO: Track game time

                # Save checkpoint periodically
                checkpoint_path = None
                if iteration % args.checkpoint_interval == 0:
                    checkpoint_path = checkpoint_dir / f"euchergo_iter_{iteration}.pt"
                    model.save_checkpoint(checkpoint_path)
                    last_checkpoint_iteration = iteration
                    last_checkpoint_path = str(checkpoint_path)

                # Update dashboard
                dashboard.update(
                    iteration=iteration,
                    games_played=games_played,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    total_loss=total_loss,
                    avg_mcts_time=avg_mcts_time,
                    avg_simulations_per_move=avg_simulations_per_move,
                    avg_game_time=avg_game_time,
                    avg_iteration_time=avg_iteration_time,
                    checkpoint_iteration=last_checkpoint_iteration,
                    checkpoint_path=last_checkpoint_path,
                )

                # Update live display
                dashboard.update_live_display(live_display, config=config)

                iteration += 1

    except TrainingInterrupt:
        dashboard.console.print("\n[bold red]Training interrupted by user.[/bold red]")

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / f"euchergo_iter_{iteration}_final.pt"
    model.save_checkpoint(final_checkpoint_path)
    
    # Final dashboard update
    dashboard.update(
        iteration=iteration,
        games_played=games_played,
        checkpoint_iteration=iteration,
        checkpoint_path=str(final_checkpoint_path),
    )

    # Print summary
    dashboard.print_summary()
    dashboard.console.print(f"\n[bold green]Final checkpoint saved: {final_checkpoint_path}[/bold green]")


if __name__ == "__main__":
    main()

