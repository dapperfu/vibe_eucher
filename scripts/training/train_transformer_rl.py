#!/usr/bin/env python3
"""Training script for Transformer RL Euchre agent.

Supports curriculum-based training, checkpointing, and evaluation.
"""

import argparse
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ai_players.transformer_rl.actor_critic_agent import ActorCriticAgent
from src.ai_players.transformer_rl.checkpoint_manager import CumulativeCheckpointManager
from src.training.transformer_rl.curriculum_trainer import CurriculumTrainer
from src.training.transformer_rl.train_transformer_rl import train_transformer_rl
from src.training.transformer_rl.evaluation import EvaluationMetrics, evaluate_agent
from eucher.training.training_progress import TrainingProgressDisplay
from eucher.training.training_orchestrator import TrainingOrchestrator
from eucher.training.convergence_tracker import ConvergenceTracker


class TrainingInterrupt(Exception):
    """Exception raised when training is interrupted."""

    pass


class TransformerRLTrainingController:
    """Controller for transformer RL training with signal handling."""

    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_interval: int = 100,
    ) -> None:
        """Initialize training controller.

        Parameters
        ----------
        checkpoint_dir : Path
            Directory for checkpoints.
        checkpoint_interval : int
            Checkpoint save interval (hands).
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.interrupted = False

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


def parse_duration(duration_str: str) -> Optional[float]:
    """Parse duration string to hours.

    Parameters
    ----------
    duration_str : str
        Duration string in format like "1m", "10m", "1h", "2h", "1d".

    Returns
    -------
    Optional[float]
        Duration in hours, or None if invalid format.
    """
    if not duration_str:
        return None

    duration_match = re.match(r"(\d+)([mhd])", duration_str.lower())
    if duration_match:
        value = int(duration_match.group(1))
        unit = duration_match.group(2)
        if unit == "m":
            return value / 60.0
        elif unit == "h":
            return float(value)
        elif unit == "d":
            return value * 24.0

    return None


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train Transformer RL agent for Euchre",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    parser.add_argument(
        "--num-hands",
        type=int,
        default=10000,
        help="Number of hands to train on (alternative to --duration)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (alternative to --num-hands or --duration)",
    )
    parser.add_argument(
        "--duration",
        type=str,
        help="Training duration in format like '1m', '10m', '1h', '2h', '1d' (alternative to --num-hands)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--update-frequency",
        type=int,
        default=10,
        help="Update agent every N hands",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )

    # Device and model
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda"],
        help="Device to use (default: auto). 'gpu' is alias for 'cuda'",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load existing model from",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints/transformer_rl",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N hands",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from latest checkpoint (start from scratch)",
    )

    # Curriculum
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        help="Use curriculum training with progressive stages",
    )
    parser.add_argument(
        "--stage-threshold",
        type=float,
        default=0.7,
        help="Performance threshold for stage progression",
    )

    # Evaluation
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Run evaluation every N hands",
    )
    parser.add_argument(
        "--eval-hands",
        type=int,
        default=100,
        help="Number of hands for evaluation",
    )

    # Risk factors
    parser.add_argument(
        "--risk-factor",
        type=float,
        default=0.5,
        help="Risk factor for decision-making (0.0-1.0)",
    )

    # Training orchestrator
    parser.add_argument(
        "--use-orchestrator",
        action="store_true",
        help="Use training orchestrator for convergence tracking",
    )
    parser.add_argument(
        "--target-win-rate",
        type=float,
        default=0.6,
        help="Target win rate for convergence (0.0-1.0)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games (overrides num-hands if set)",
    )

    args = parser.parse_args()

    # Setup device
    device_str = args.device
    if device_str == "gpu":
        device_str = "cuda"
    
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Parse duration if provided
    max_duration_hours = None
    if args.duration:
        max_duration_hours = parse_duration(args.duration)
        if max_duration_hours is None:
            print(f"Warning: Invalid duration format '{args.duration}'. Expected format: '1m', '10m', '1h', or '1d'")
            print("Continuing with hand-based training...")
        else:
            print(f"Training for maximum duration: {max_duration_hours:.2f} hours ({args.duration})")
            # For duration-based training, use a large number of hands
            # The training loop should check duration periodically
            if args.num_hands == 10000:  # Only override if using default
                args.num_hands = 1000000  # Large number, will be limited by duration

    # Adjust num_hands based on epochs if provided
    if args.epochs:
        # Rough estimate: 1 epoch ≈ 1000 hands (adjust based on your training setup)
        args.num_hands = args.epochs * 1000
        print(f"Training for {args.epochs} epochs (≈ {args.num_hands} hands)")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint manager
    checkpoint_manager = CumulativeCheckpointManager(checkpoint_dir)

    # Create agent
    print("\nInitializing Transformer RL agent...")
    agent = ActorCriticAgent(
        device=device,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
    )

    # Load existing model if provided
    if args.model_path:
        # Check if it's a UUID or file path
        if Path(args.model_path).exists():
            # Legacy file path
            print(f"Loading model from {args.model_path}")
            agent.load(Path(args.model_path))
        else:
            # Try as UUID
            print(f"Loading checkpoint UUID: {args.model_path}")
            agent.load(checkpoint_uuid=args.model_path, checkpoint_manager=checkpoint_manager)
    elif not args.no_resume:
        # Try to load latest checkpoint (cumulative training)
        latest_uuid = checkpoint_manager.get_latest_checkpoint_uuid()
        if latest_uuid:
            print(f"Resuming from latest checkpoint: {latest_uuid}")
            agent.load(checkpoint_uuid=latest_uuid, checkpoint_manager=checkpoint_manager)
        else:
            print("No checkpoint found, starting from scratch")
    else:
        print("Starting from scratch (--no-resume flag set)")

    # Create curriculum trainer
    curriculum = None
    if args.use_curriculum:
        print("\nInitializing curriculum trainer...")
        curriculum = CurriculumTrainer(
            agent=agent,
            stage_progression_threshold=args.stage_threshold,
        )
        print(f"Starting at: {curriculum.get_stage_name()}")

    # Create training orchestrator if requested
    orchestrator = None
    if args.use_orchestrator:
        convergence_tracker = ConvergenceTracker(
            target_win_rate=args.target_win_rate,
            window_size=100,
        )
        orchestrator = TrainingOrchestrator(
            convergence_tracker=convergence_tracker,
            max_games=args.max_games if args.max_games else args.num_hands,
        )

    # Create training controller
    controller = TransformerRLTrainingController(
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Training progress display
    progress_display = None
    if orchestrator:
        progress_display = TrainingProgressDisplay(
            orchestrator=orchestrator,
            trump_selection_risk=args.risk_factor,
            gameplay_risk=args.risk_factor,
        )

    print("\n" + "=" * 80)
    print("Starting Transformer RL Training")
    print("=" * 80)
    print(f"Number of hands: {args.num_hands}")
    print(f"Batch size: {args.batch_size}")
    print(f"Update frequency: {args.update_frequency}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Risk factor: {args.risk_factor}")
    if max_duration_hours:
        print(f"Maximum duration: {max_duration_hours:.2f} hours")
    if curriculum:
        print(f"Curriculum: Enabled (starting at {curriculum.get_stage_name()})")
    print("=" * 80 + "\n")

    # Create dashboard for progress display
    from src.training.transformer_rl.transformer_rl_dashboard import TransformerRLDashboard

    dashboard = TransformerRLDashboard(
        num_hands=args.num_hands,
        refresh_rate=2.0,
    )

    # Training loop
    # Note: Duration-based training is handled by setting a large num_hands
    # and monitoring time in the training loop (if max_duration_hours is set)
    try:
        training_stats = train_transformer_rl(
            agent=agent,
            curriculum=curriculum if curriculum else CurriculumTrainer(agent=agent),
            num_hands=args.num_hands,
            batch_size=args.batch_size,
            update_frequency=args.update_frequency,
            checkpoint_dir=checkpoint_dir,
            checkpoint_manager=checkpoint_manager,
            device=device,
            dashboard=dashboard,
        )

        # Final evaluation
        print("\n" + "=" * 80)
        print("Running final evaluation...")
        print("=" * 80)

        from src.ai_players.transformer_rl.transformer_rl_player import TransformerRLPlayer

        eval_agent = TransformerRLPlayer(
            model_path=str(checkpoint_dir / "final_model.pth") if (checkpoint_dir / "final_model.pth").exists() else None,
            device=str(device),
            risk_factor=args.risk_factor,
            use_deduction=True,
        )

        metrics = evaluate_agent(
            agent=eval_agent,
            num_hands=args.eval_hands,
            opponent_type="random",
        )

        eval_stats = metrics.get_stats()
        print("\nFinal Evaluation Results:")
        print(f"  Win Rate: {eval_stats['win_rate']:.2%}")
        print(f"  Average Tricks per Hand: {eval_stats['avg_tricks_per_hand']:.2f}")
        print(f"  Sweep Rate: {eval_stats['sweep_rate']:.2%}")
        print(f"  Set Rate: {eval_stats['set_rate']:.2%}")
        print(f"  Bidding Accuracy: {eval_stats['bidding_accuracy']:.2%}")
        print(f"  Play Accuracy: {eval_stats['play_accuracy']:.2%}")

        # Save final model (UUID-based)
        final_uuid = agent.save(training_stats=training_stats, checkpoint_manager=checkpoint_manager)
        print(f"\nFinal model saved with UUID: {final_uuid}")
        
        # Also save as "final" reference
        final_checkpoint_path = checkpoint_dir / "final_checkpoint.npz"
        if final_uuid:
            import shutil
            shutil.copy2(checkpoint_dir / f"{final_uuid}.npz", final_checkpoint_path)
            print(f"Final checkpoint also saved to: {final_checkpoint_path}")

        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)

    except TrainingInterrupt:
        print("\nTraining interrupted, saving checkpoint...")
        interrupt_uuid = agent.save(
            training_stats=training_stats if 'training_stats' in locals() else {},
            checkpoint_manager=checkpoint_manager
        )
        print(f"Checkpoint saved with UUID: {interrupt_uuid}")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

