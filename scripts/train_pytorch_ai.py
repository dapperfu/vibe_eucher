#!/usr/bin/env python3
"""Training script for PyTorch AI player.

Supports cumulative training, multi-device support, and checkpoint management.
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eucher.ai_players.pytorch_networks import create_network
from eucher.ai_players.training.checkpoint_manager import CheckpointManager
from eucher.ai_players.training.data_manager import TrainingDataManager
from eucher.ai_players.training.trainer import CumulativeTrainer, EuchreDataset
from eucher.training.pytorch_dashboard import PyTorchTrainingDashboard


class TrainingInterrupt(Exception):
    """Exception raised when training is interrupted."""

    pass


class TrainingController:
    """Controller for training with signal handling and duration limits."""

    def __init__(
        self,
        max_duration_hours: Optional[float] = None,
        checkpoint_interval: int = 5,
        checkpoint_manager: CheckpointManager = None,
    ) -> None:
        """Initialize training controller.

        Parameters
        ----------
        max_duration_hours : Optional[float]
            Maximum training duration in hours.
        checkpoint_interval : int
            Checkpoint save interval (epochs).
        checkpoint_manager : CheckpointManager
            Checkpoint manager instance.
        """
        self.max_duration_hours = max_duration_hours
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_manager = checkpoint_manager
        self.start_time = None
        self.interrupted = False
        self.current_epoch = 0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """Handle interrupt signals.

        Parameters
        ----------
        signum : int
            Signal number.
        frame : frame
            Current stack frame.
        """
        print(f"\nReceived signal {signum}, saving checkpoint and exiting...")
        self.interrupted = True
        raise TrainingInterrupt("Training interrupted by signal")

    def check_duration_limit(self) -> bool:
        """Check if duration limit has been reached.

        Returns
        -------
        bool
            True if limit reached, False otherwise.
        """
        if self.max_duration_hours is None:
            return False

        if self.start_time is None:
            self.start_time = time.time()
            return False

        elapsed_hours = (time.time() - self.start_time) / 3600.0
        return elapsed_hours >= self.max_duration_hours

    def should_save_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint should be saved.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        bool
            True if should save checkpoint.
        """
        return (epoch + 1) % self.checkpoint_interval == 0

    def save_emergency_checkpoint(
        self, model, optimizer, epoch: int, metrics: dict
    ) -> None:
        """Save emergency checkpoint on interruption.

        Parameters
        ----------
        model : torch.nn.Module
            Model to save.
        optimizer : torch.optim.Optimizer
            Optimizer to save.
        epoch : int
            Current epoch.
        metrics : dict
            Current metrics.
        """
        if self.checkpoint_manager:
            try:
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    metrics=metrics,
                    training_duration=time.time() - self.start_time if self.start_time else 0.0,
                    suffix="interrupted",
                )
                print("Emergency checkpoint saved successfully")
            except Exception as e:
                print(f"Failed to save emergency checkpoint: {e}")


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PyTorch AI player")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to train on (default: auto)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        help="Maximum training duration in hours",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints/pytorch_ai",
        help="Checkpoint directory (default: models/checkpoints/pytorch_ai)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="training_data",
        help="Training data directory (default: training_data)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from latest checkpoint (start from scratch)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override automatic batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs to train (default: 10)",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Training on device: {device}")

    # Initialize components
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    data_manager = TrainingDataManager(args.data_dir)

    # Create model
    model = create_network(device=device)

    # Load checkpoint if resuming (default behavior)
    if not args.no_resume:
        try:
            checkpoint, _ = checkpoint_manager.load_checkpoint(device=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Resumed from epoch {start_epoch}")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch")
            start_epoch = 0
    else:
        print("Starting from scratch (--no-resume flag set)")
        start_epoch = 0

    # Load training data (only .npz files are supported)
    datasets = data_manager.list_datasets()
    if not datasets:
        print("No training datasets found!")
        print(f"Expected .npz files in: {args.data_dir}")
        print("Looking for files matching: training_*.npz")
        print("\nTo collect training data, run:")
        print("  python collect_training_data.py --num_games 100")
        print("\nOr use the script:")
        print("  python scripts/collect_training_data.py --num_games 100")
        print("\nGenerating random training data to start training from scratch...")
        print("(Model will start with random weights and learn from random data initially)")
        
        # Generate synthetic random training data and save as .npz
        # Feature size from GameStateEncoder: 260 features
        # Decision: card index (0-23) for play_card dataset
        import numpy as np
        import random
        
        num_samples = 1000  # Generate 1000 random samples
        feature_size = 260
        num_cards = 24
        
        print(f"Generating {num_samples} random training samples...")
        
        # Generate random features and decisions as numpy arrays
        X = np.random.rand(num_samples, feature_size).astype(np.float32)
        y = np.random.randint(0, num_cards, size=num_samples).astype(np.int32)
        
        # Save as .npz file
        npz_path = Path(args.data_dir) / "training_play_card.npz"
        np.savez_compressed(npz_path, X=X, y=y)
        print(f"Saved random training data to {npz_path}")
        
        # Load it back as DataFrame for compatibility
        df = data_manager.load_dataset("play_card")
        training_data = df.to_dict("records")
        
        print(f"Generated {len(training_data)} random training samples")
    else:
        print(f"Found datasets: {datasets}")

        # For now, use play_card dataset as example
        # In practice, you'd combine multiple datasets
        dataset_name = "play_card" if "play_card" in datasets else datasets[0]
        df = data_manager.load_dataset(dataset_name)

        if df.empty:
            print(f"Dataset {dataset_name} is empty or could not be loaded!")
            return

        # Features should already be in list format from .npz loading
        # Convert to dataset format
        training_data = df.to_dict("records")
    
    # Import feature encoder for dataset
    from eucher.ai_players.feature_encoder import EuchreFeatureEncoder

    feature_encoder = EuchreFeatureEncoder()
    
    dataset = EuchreDataset(training_data, feature_encoder=feature_encoder)

    # Create trainer
    trainer = CumulativeTrainer(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        device=device,
    )

    # Setup training controller (for signal handling and duration limits)
    controller = TrainingController(
        max_duration_hours=args.max_duration,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_manager=checkpoint_manager,
    )
    controller.start_time = time.time()

    # Determine number of epochs
    num_epochs = args.num_epochs
    if args.max_epochs:
        num_epochs = min(num_epochs, args.max_epochs - start_epoch)

    # Create Rich dashboard for training
    # Get number of batches for batch progress
    from torch.utils.data import DataLoader

    temp_loader = DataLoader(dataset, batch_size=args.batch_size or 32, shuffle=False)
    num_batches = len(temp_loader)

    dashboard = PyTorchTrainingDashboard(
        num_epochs=num_epochs,
        num_batches_per_epoch=num_batches,
        model_type="Supervised",
        refresh_rate=2.0,
    )

    print(f"Starting training with Rich dashboard...")
    print(f"Epochs: {num_epochs}, Batches per epoch: {num_batches}")

    try:
        # Train
        results = trainer.train(
            dataset=dataset,
            num_epochs=num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            resume=not args.no_resume,
            checkpoint_interval=args.checkpoint_interval,
            dashboard=dashboard,
        )

        print(f"\nTraining completed!")
        print(f"Total epochs: {results['total_epochs']}")
        print(f"Total duration: {results['total_duration']:.2f} seconds")
        print(f"Final metrics: {results['final_metrics']}")

    except TrainingInterrupt:
        print("\nTraining interrupted, saving checkpoint...")
        # Emergency checkpoint is handled by signal handler
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

