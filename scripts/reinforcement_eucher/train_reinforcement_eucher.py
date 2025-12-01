#!/usr/bin/env python3
"""Training script for ReinforcementEucher player.

ReinforcementEucher is a pure reinforcement learning Euchre agent trained entirely
through self-play without supervised data.
"""

import argparse
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from eucher.players.computer.reinforcement_eucher.config import ReinforcementEucherConfig
from eucher.players.computer.reinforcement_eucher.networks.model import ReinforcementEucherModel
from eucher.players.computer.reinforcement_eucher.state_encoder import StateEncoder
from eucher.players.computer.reinforcement_eucher.training.trainer import ReinforcementEucherTrainer


class TrainingInterrupt(Exception):
    """Exception raised when training is interrupted."""

    pass


def parse_duration(duration_str: str) -> Optional[float]:
    """Parse duration string to seconds.

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
    """Find latest checkpoint in directory.

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

    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ReinforcementEucher player")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda"],
        help="Device to train on (default: auto). 'gpu' is alias for 'cuda'",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="auto",
        choices=["trick_only", "full_hand", "auto"],
        help="Training stage (default: auto - uses config or trick_only)",
    )
    parser.add_argument(
        "--num-tricks",
        type=int,
        default=None,
        help="Number of trick scenarios for Stage 1 (default: 1000)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=None,
        help="Number of games for Stage 2 (default: 100)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N games/tricks (default: 100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from latest checkpoint",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (default: models/checkpoints/reinforcement_eucher)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Specific checkpoint path to load",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable Rich dashboard display",
    )

    args = parser.parse_args()

    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "gpu":
        device = "cuda"
    else:
        device = args.device

    print(f"Training on device: {device}")

    # Configuration
    config = ReinforcementEucherConfig(device=device)
    if args.checkpoint_dir:
        config.checkpoint_dir = Path(args.checkpoint_dir)

    # Update state_dim based on encoder
    state_encoder = StateEncoder()
    config.state_dim = state_encoder.get_state_dim()

    # Determine training stage
    if args.stage != "auto":
        config.training_stage = args.stage
    elif args.num_tricks and not args.num_games:
        config.training_stage = "trick_only"
    elif args.num_games and not args.num_tricks:
        config.training_stage = "full_hand"

    # Create model
    model = ReinforcementEucherModel(config)

    # Load checkpoint if requested
    if args.resume or args.checkpoint_path:
        checkpoint_path = None
        if args.checkpoint_path:
            checkpoint_path = Path(args.checkpoint_path)
        else:
            checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)

        if checkpoint_path and checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "config" in checkpoint:
                config = ReinforcementEucherConfig.from_dict(checkpoint["config"])
                config.device = device
                config.torch_device = torch.device(device)
            print("Checkpoint loaded successfully")
        else:
            print("No checkpoint found, starting fresh")

    # Create trainer
    trainer = ReinforcementEucherTrainer(model, config)

    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        """Handle interrupt signal."""
        print("\nTraining interrupted, saving checkpoint...")
        trainer.save_checkpoint(suffix="interrupted")
        print("Checkpoint saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Training
    print(f"Starting training: stage={config.training_stage}")
    start_time = time.time()

    try:
        # Disable dashboard if requested
        use_dashboard = not args.no_dashboard
        
        results = trainer.train(
            num_tricks=args.num_tricks or 1000,
            num_games=args.num_games or 100,
            auto_resume=args.resume,
            use_dashboard=use_dashboard,
        )

        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time:.2f} seconds")
        print(f"Games trained: {results['game_count']}")
        print(f"Tricks trained: {results['trick_count']}")
        print(f"Final stage: {results['current_stage']}")

    except KeyboardInterrupt:
        print("\nTraining interrupted, saving checkpoint...")
        trainer.save_checkpoint(suffix="interrupted")
        print("Checkpoint saved. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback

        traceback.print_exc()
        trainer.save_checkpoint(suffix="error")
        sys.exit(1)


if __name__ == "__main__":
    main()

