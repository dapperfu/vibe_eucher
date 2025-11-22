"""Main training script for ML models."""

import argparse
import re
from pathlib import Path

from src.ml_config import MLConfig
from src.training.convergence_tracker import ConvergenceTracker
from src.training.profiling import Profiler, get_timing_stats
from src.training.self_play import SelfPlayTrainer
from src.training.train_gan import train_gan_for_decision_type
from src.training.train_rl import train_rl_through_self_play
from src.training.train_supervised import train_all_models
from src.training.training_orchestrator import TrainingOrchestrator
from src.training.training_scheduler import TrainingScheduler


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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling (cProfile). Auto-enabled for self_play mode with duration.",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default=None,
        help="Output file for profile results (default: profiles/train_profile.txt)",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Enable timing statistics collection",
    )
    parser.add_argument(
        "--timing-output",
        type=str,
        default=None,
        help="Output file for timing statistics (default: profiles/timing_stats.txt)",
    )
    parser.add_argument(
        "--disable-live-display",
        action="store_true",
        help="Disable Rich live display to reduce overhead (useful for profiling)",
    )
    parser.add_argument(
        "--display-refresh-rate",
        type=float,
        default=None,
        help="Refresh rate for live display in updates per second (default: 2.0, lower for profiling)",
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

    # Setup profiling
    # Auto-enable profiling for self_play mode with duration (for performance analysis)
    enable_profiling = args.profile
    if args.mode == "self_play" and args.duration and not args.profile:
        enable_profiling = True
        print("Note: Profiling auto-enabled for self_play mode with duration. Use --profile to control explicitly.")

    profile_output = None
    if enable_profiling:
        if args.profile_output:
            profile_output = Path(args.profile_output)
        else:
            profile_output = Path("profiles") / "train_profile.txt"

    # Auto-enable timing for self_play mode with duration
    enable_timing = args.timing
    if args.mode == "self_play" and args.duration and not args.timing:
        enable_timing = True
        print("Note: Timing statistics auto-enabled for self_play mode with duration. Use --timing to control explicitly.")

    timing_output = None
    if enable_timing:
        if args.timing_output:
            timing_output = Path(args.timing_output)
        else:
            timing_output = Path("profiles") / "timing_stats.txt"

    # Context manager for when profiling is disabled
    class nullcontext:
        """Null context manager for when profiling is disabled."""
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    # Main training with profiling
    profiler_context = Profiler(output_file=profile_output) if enable_profiling else nullcontext()
    with profiler_context:
        if args.mode in ["self_play", "both"]:
            print("=" * 60)
            print("Self-Play Training")
            print("=" * 60)

            # Setup progress display if using orchestrator
            from src.training.training_progress import TrainingProgressDisplay

            progress_display = None
            live_display = None

            if args.until_converged or args.duration:
                orchestrator.start_training()
                # Disable live display if requested or when profiling (to reduce overhead)
                use_live_display = not args.disable_live_display and not enable_profiling
                refresh_rate = args.display_refresh_rate if args.display_refresh_rate else (0.5 if enable_profiling else 2.0)
                
                progress_display = TrainingProgressDisplay(
                    orchestrator,
                    trump_selection_risk=config.trump_selection_risk,
                    gameplay_risk=config.gameplay_risk,
                )
                if use_live_display:
                    live_display = progress_display.start_live_display(refresh_rate=refresh_rate)
                    live_display.__enter__()
                else:
                    live_display = None
                    if enable_profiling:
                        print("Note: Live display disabled during profiling to reduce overhead.")

            trainer = SelfPlayTrainer(output_dir=data_dir)

            game_count = 0
            # Batch save interval - save less frequently to reduce I/O overhead
            # JSON saves are faster than CSV, so we can save JSON more often
            save_interval = 100 if enable_profiling else 50
            csv_save_interval = 500 if enable_profiling else 200
            try:
                while True:
                    # Check if should continue
                    if args.until_converged or args.duration:
                        if not orchestrator.should_continue():
                            break
                    else:
                        if game_count >= args.num_games:
                            break

                    # Determine if we should save data
                    is_final = (
                        (args.until_converged or args.duration) and not orchestrator.should_continue()
                    ) or (not args.until_converged and not args.duration and game_count >= args.num_games - 1)
                    
                    should_save_json = is_final or (game_count % save_interval == 0)
                    should_save_csv = is_final or (game_count % csv_save_interval == 0)
                    
                    # Run training round (disable auto-save, we'll batch it)
                    trainer.run_training_round(
                        num_games=1,
                        save_data=should_save_json,
                        save_csv=should_save_csv and not enable_profiling,  # Skip CSV during profiling
                    )
                    
                    game_count += 1

                    # Record game result (simplified - would get actual result from game)
                    if args.until_converged or args.duration:
                        orchestrator.record_game_result(won=(game_count % 2 == 0))

                        # Update progress display (less frequently when profiling)
                        if progress_display:
                            progress_display.update()
                            # Update live display less frequently to reduce overhead
                            update_interval = 50 if enable_profiling else 10
                            if live_display and game_count % update_interval == 0:
                                progress_display.update_live_display(live_display)

                        if orchestrator.should_checkpoint():
                            if progress_display:
                                progress_display.console.print(f"[yellow]Checkpoint saved at game {game_count}[/yellow]")

            finally:
                # Final save of all collected data
                print("\nSaving final collected data...")
                trainer.data_collector.save_data(save_csv=not enable_profiling)
                
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

    # Print timing statistics if enabled
    if enable_timing:
        timing_stats = get_timing_stats()
        timing_stats.print_summary()
        if timing_output:
            timing_stats.save_summary(timing_output)


if __name__ == "__main__":
    main()

