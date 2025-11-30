"""Main training script for ML models."""

import argparse
import re
from pathlib import Path

from eucher.players.computer.ml.ml_config import MLConfig
from eucher.training.convergence_tracker import ConvergenceTracker
from eucher.training.self_play import SelfPlayTrainer
from eucher.training.train_gan import train_gan_for_decision_type
from eucher.training.train_rl import train_rl_through_self_play
from eucher.training.train_supervised import train_all_models
from eucher.training.training_orchestrator import TrainingOrchestrator
from eucher.training.training_scheduler import TrainingScheduler


def main() -> None:
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train ML models for Euchre")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["self_play", "train", "train_gan", "train_rl", "both"],
        default="both",
        help="Training mode: self_play (collect data), train (train sklearn), train_gan, train_rl, or both",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["random_forest", "gradient_boosting", "neural_network"],
        default="random_forest",
        help="Type of sklearn model to use",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=100,
        help="Number of games to play for self-play training",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory for training data (default: training_data/)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory for trained models (default: models/)",
    )
    parser.add_argument(
        "--data_prefix",
        type=str,
        default="training",
        help="Prefix for data filenames",
    )
    parser.add_argument(
        "--duration",
        type=str,
        default=None,
        help='Training duration (e.g., "30m", "2h", "1d"). If not specified, uses num_games.',
    )
    parser.add_argument(
        "--until-converged",
        action="store_true",
        help="Train until convergence (ignores duration and num_games)",
    )
    parser.add_argument(
        "--target-win-rate",
        type=float,
        default=0.90,
        help="Target win rate for convergence (default: 0.90 = 90%%)",
    )
    parser.add_argument(
        "--trump-selection-risk",
        type=float,
        default=None,
        help="Risk factor for trump selection decisions (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--gameplay-risk",
        type=float,
        default=None,
        help="Risk factor for gameplay decisions (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Number of games for convergence window (default: 100)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N games (default: 100)",
    )

    args = parser.parse_args()

    config = MLConfig()
    data_dir = Path(args.data_dir) if args.data_dir else config.training_data_dir
    model_dir = Path(args.model_dir) if args.model_dir else config.models_dir

    # Set risk factors if provided
    if args.trump_selection_risk is not None:
        config.trump_selection_risk = args.trump_selection_risk
    if args.gameplay_risk is not None:
        config.gameplay_risk = args.gameplay_risk

    # Parse duration if provided
    duration_minutes = None
    duration_hours = None
    duration_days = None

    if args.duration:
        # Parse duration string (e.g., "30m", "2h", "1d")
        duration_match = re.match(r"(\d+)([mhd])", args.duration.lower())
        if duration_match:
            value = int(duration_match.group(1))
            unit = duration_match.group(2)
            if unit == "m":
                duration_minutes = value
            elif unit == "h":
                duration_hours = value
            elif unit == "d":
                duration_days = value
        else:
            print(f"Warning: Invalid duration format '{args.duration}'. Expected format: '30m', '2h', or '1d'")

    # Create training scheduler
    scheduler = TrainingScheduler(
        duration_minutes=duration_minutes,
        duration_hours=duration_hours,
        duration_days=duration_days,
        until_converged=args.until_converged,
    )

    # Create convergence tracker if needed
    convergence_tracker = None
    if args.until_converged:
        convergence_tracker = ConvergenceTracker(
            target_win_rate=args.target_win_rate, window_size=args.window_size
        )

    # Create training orchestrator
    orchestrator = TrainingOrchestrator(
        scheduler=scheduler,
        convergence_tracker=convergence_tracker,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=model_dir / "checkpoints",
    )

    if args.mode in ["self_play", "both"]:
        print("=" * 60)
        print("Self-Play Training")
        print("=" * 60)

        # Setup progress display if using orchestrator
        from eucher.training.training_progress import TrainingProgressDisplay

        progress_display = None
        live_display = None

        if args.until_converged or args.duration:
            orchestrator.start_training()
            progress_display = TrainingProgressDisplay(
                orchestrator,
                trump_selection_risk=config.trump_selection_risk,
                gameplay_risk=config.gameplay_risk,
            )
            live_display = progress_display.start_live_display()
            live_display.__enter__()

        trainer = SelfPlayTrainer(output_dir=data_dir)

        game_count = 0
        try:
            while True:
                # Check if should continue
                if args.until_converged or args.duration:
                    if not orchestrator.should_continue():
                        break
                else:
                    if game_count >= args.num_games:
                        break

                # Run training round
                trainer.run_training_round(num_games=1)
                game_count += 1

                # Record game result (simplified - would get actual result from game)
                if args.until_converged or args.duration:
                    orchestrator.record_game_result(won=(game_count % 2 == 0))

                    # Update progress display
                    if progress_display:
                        progress_display.update()
                        if game_count % 10 == 0:
                            progress_display.update_live_display(live_display)

                    if orchestrator.should_checkpoint():
                        if progress_display:
                            progress_display.console.print(f"[yellow]Checkpoint saved at game {game_count}[/yellow]")

        finally:
            if live_display:
                live_display.__exit__(None, None, None)
                if progress_display:
                    progress_display.print_summary()

        print("\nSelf-play data collection complete!")

    if args.mode in ["train", "both"]:
        print("\n" + "=" * 60)
        print("Training Supervised Models (sklearn)")
        print("=" * 60)
        train_all_models(
            data_dir=data_dir,
            model_type=args.model_type,
            output_dir=model_dir,
            prefix=args.data_prefix,
        )
        print("\nSupervised model training complete!")

    if args.mode in ["train_gan", "both"]:
        print("\n" + "=" * 60)
        print("Training GAN Models")
        print("=" * 60)
        train_gan_for_decision_type(
            data_dir=data_dir,
            decision_type="play_card",
            config=config,
            num_epochs=50,
            output_dir=model_dir,
            prefix=args.data_prefix,
        )
        print("\nGAN model training complete!")

    if args.mode in ["train_rl", "both"]:
        print("\n" + "=" * 60)
        print("Training RL Agents")
        print("=" * 60)

        # Use num_games if not using duration/convergence
        num_games = args.num_games if not args.until_converged and not args.duration else 10000

        train_rl_through_self_play(
            num_games=num_games,
            config=config,
            output_dir=model_dir,
            checkpoint_interval=args.checkpoint_interval,
            convergence_tracker=convergence_tracker,
            orchestrator=orchestrator if (args.until_converged or args.duration) else None,
        )
        print("\nRL agent training complete!")


if __name__ == "__main__":
    main()

