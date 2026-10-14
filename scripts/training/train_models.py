"""Main training script for ML models."""

import argparse
import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from eucher.game import parse_seed_text
from eucher.training.convergence_tracker import ConvergenceTracker
from eucher.training.self_play import SelfPlayTrainer
from eucher.training.system_info import get_system_info
from eucher.training.train_gan import train_gan_for_decision_type
from eucher.training.train_rl import train_rl_through_self_play
from eucher.training.train_supervised import train_all_models
from eucher.training.training_orchestrator import TrainingOrchestrator
from eucher.training.training_scheduler import TrainingScheduler
from plugins.ml.ml_config import MLConfig


def _git_commit() -> str:
    """
    Return the current git commit, or ``unknown`` when git is unavailable.

    Returns
    -------
    str
        Commit hash from ``git rev-parse HEAD``.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_seeded_random_log(
    args: argparse.Namespace,
    data_dir: Path,
    trainer: SelfPlayTrainer,
    parsed_seed: object,
) -> None:
    """
    Write the experiment log for one seeded random-opponent run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    data_dir : Path
        Directory that received the training data.
    trainer : SelfPlayTrainer
        Trainer that just finished the round.
    parsed_seed : object
        Deal seed used for every game.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    log_dir = Path("logs") / "seeded_random" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    collector = trainer.data_collector
    config = {
        "mode": "seeded_random",
        "seed": str(parsed_seed),
        "num_games": args.num_games,
        "learner": args.learner,
        "data_dir": str(data_dir),
        "data_prefix": args.data_prefix,
        "git_commit": _git_commit(),
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    metrics = {
        "games": len(collector.game_outcomes),
        "order_up": len(collector.order_up_data),
        "call_trump": len(collector.call_trump_data),
        "play_card": len(collector.play_card_data),
        "discard": len(collector.discard_data),
    }
    (log_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (log_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    system_metrics = json.dumps(get_system_info(), indent=2, default=str)
    (log_dir / "system_metrics.json").write_text(system_metrics + "\n")
    print(f"Experiment log: {log_dir}")


def _run_seeded_random(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    data_dir: Path,
) -> None:
    """
    Replay one played-game seed against random AI and record the learner.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    parser : argparse.ArgumentParser
        Parser used to report a missing seed.
    data_dir : Path
        Directory for the collected decisions.
    """
    seed_text = args.seed if isinstance(args.seed, str) else None
    parsed_seed = parse_seed_text(seed_text)
    if parsed_seed is None:
        parser.error("--seed is required for --mode seeded_random")
    if args.num_games < 1:
        parser.error("--num_games must be >= 1")

    print("=" * 60)
    print("Seeded training against random AI")
    print("=" * 60)
    print(f"Deal seed: {parsed_seed}")
    print(f"Learner seat: {args.learner} (seat 0, same cards as the human)")
    print("Opponents: random, random, random")
    print(f"Games: {args.num_games}")
    print(f"Output directory: {data_dir}")
    print()

    trainer = SelfPlayTrainer(output_dir=data_dir)
    trainer.run_seeded_random_round(
        seed=parsed_seed,
        num_games=args.num_games,
        learner_type=args.learner,
        show_progress=True,
        data_prefix=args.data_prefix,
    )
    _write_seeded_random_log(args, data_dir, trainer, parsed_seed)
    print("\nSeeded random-opponent data collection complete!")
    print("Train a model on this data with:")
    print("  python scripts/training/train_models.py --mode train")


def main() -> None:
    """Train ML models for Euchre."""
    parser = argparse.ArgumentParser(description="Train ML models for Euchre")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["self_play", "train", "train_gan", "train_rl", "both", "seeded_random"],
        default="both",
        help=(
            "Training mode: self_play (collect data), train (train sklearn), "
            "train_gan, train_rl, both, or seeded_random (replay one seed "
            "against random opponents)"
        ),
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
        "--seed",
        type=str,
        default=None,
        help=(
            "Deal seed for --mode seeded_random (integer or UUID). "
            "Use the seed from a game you played. Every training game is "
            "dealt from this seed. Random opponents make different decisions."
        ),
    )
    parser.add_argument(
        "--learner",
        type=str,
        default="ml_sklearn",
        help=(
            "Personality recorded in the human seat for --mode seeded_random. "
            "The other three players are random (default: ml_sklearn)."
        ),
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
            print(
                f"Warning: Invalid duration format '{args.duration}'. "
                "Expected format: '30m', '2h', or '1d'"
            )

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

    if args.mode == "seeded_random":
        _run_seeded_random(args, parser, data_dir)
        return

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
            if args.until_converged or args.duration:
                while orchestrator.should_continue():
                    trainer.run_training_round(num_games=1, show_progress=False)
                    game_count += 1
                    orchestrator.record_game_result(won=(game_count % 2 == 0))

                    if progress_display:
                        progress_display.update()
                        if game_count % 10 == 0:
                            progress_display.update_live_display(live_display)

                    if orchestrator.should_checkpoint() and progress_display:
                        progress_display.console.print(
                            f"[yellow]Checkpoint saved at game {game_count}[/yellow]"
                        )
            else:
                trainer.run_training_round(num_games=args.num_games, show_progress=True)

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
