#!/usr/bin/env python3
"""Training script for EucherGo player.

EucherGo is a hybrid Monte Carlo Tree Search and Deep Neural Network engine
for Euchre, patterned after AlphaZero/MuZero architecture.
"""

import argparse
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from eucher.game import Game
from eucher.plugins import get_registry
from plugins.euchergo.config import EucherGoConfig
from plugins.euchergo.networks.model import EucherGoModel
from plugins.euchergo.training.dashboard import EucherGoDashboard
from plugins.euchergo.training.replay_buffer import ReplayBuffer
from plugins.euchergo.training.self_play import generate_self_play_game
from plugins.euchergo.training.trainer import EucherGoTrainer


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


def list_available_plugins() -> None:
    """
    List all available player plugins.
    """
    registry = get_registry()
    # Plugins are auto-discovered via entry points on import
    plugins = registry.list_plugins()
    
    print("Available player plugins:")
    print("=" * 80)
    for plugin in sorted(plugins):
        metadata = registry.get(plugin)
        if metadata:
            desc = metadata.description or "No description"
            print(f"  {plugin:20s} - {desc}")
        else:
            print(f"  {plugin:20s}")
    print("=" * 80)
    print(f"\nTotal: {len(plugins)} plugins")
    print("\nUse --opponent <plugin_name> to train against a specific opponent.")
    print("Example: --opponent heuristic")


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
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        help="Opponent plugin to train against. Use 'list' to enumerate available plugins. "
             "If not specified, trains against itself (self-play).",
    )

    args = parser.parse_args()
    
    # Configure CPU threading for optimal performance
    # Set PyTorch to use all available CPU threads
    num_threads = os.cpu_count() or 16
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    
    # Set environment variables for BLAS/MKL libraries
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    
    print(f"CPU Threading Configuration:")
    print(f"  Available CPU cores: {num_threads}")
    print(f"  PyTorch threads: {torch.get_num_threads()}")
    print(f"  PyTorch interop threads: {torch.get_num_interop_threads()}")
    
    # Handle --opponent list
    if args.opponent == "list":
        list_available_plugins()
        sys.exit(0)

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
    if args.opponent:
        print(f"Opponent: {args.opponent}")
    else:
        print("Opponent: self-play (EucherGo vs EucherGo)")
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
    config.going_alone_threshold = 0.75  # Use conservative threshold

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

    # Create replay buffer and trainer
    replay_buffer = ReplayBuffer(max_size=10000)
    trainer = EucherGoTrainer(model, config, replay_buffer)

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

                # Generate self-play games
                for game_num in range(args.num_games):
                    try:
                        game_start_time = time.time()

                        # Create game configuration
                        # Team 0 (players 0, 2): EucherGo (training player)
                        # Team 1 (players 1, 3): Opponent or EucherGo (self-play)
                        opponent_type = args.opponent if args.opponent else "euchergo"
                        
                        # Validate opponent plugin exists
                        if args.opponent:
                            registry = get_registry()
                            # Plugins are auto-discovered via entry points on import
                            if not registry.has(args.opponent):
                                print(f"Error: Opponent plugin '{args.opponent}' not found.")
                                print("Use --opponent list to see available plugins.")
                                sys.exit(1)
                        
                        player_config = [
                            ("EucherGo_0", "euchergo"),
                            (f"Opponent_1", opponent_type),
                            ("EucherGo_2", "euchergo"),
                            (f"Opponent_3", opponent_type),
                        ]

                        game = Game(player_config)

                        # Set game reference and model for all EucherGo players
                        from plugins.euchergo.player import EucherGoPlayer
                        from plugins.euchergo.mcts.search import EucherGoMCTSSearch
                        from plugins.euchergo.state_encoder import EucherGoStateEncoder

                        state_encoder = EucherGoStateEncoder()
                        mcts = EucherGoMCTSSearch(model, config.num_simulations, config.exploration_constant, config.device)

                        for player in game.players:
                            if hasattr(player.profile, "set_game"):
                                player.profile.set_game(game)
                            if isinstance(player.profile, EucherGoPlayer):
                                player.profile.model = model
                                player.profile.mcts = mcts
                                player.profile.state_encoder = state_encoder

                        # Play hand
                        continue_game = game.play_hand()

                        # Collect training examples from self-play
                        examples = generate_self_play_game(model, config, game=game, risk_factor=0.0)

                        # Add to replay buffer
                        for example in examples:
                            replay_buffer.add(
                                example.state,
                                example.policy,
                                example.value,
                            )

                        games_played += 1
                        game_time = time.time() - game_start_time
                        game_times.append(game_time)

                    except Exception as e:
                        print(f"Error in self-play game: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                # Train on replay buffer
                losses: Dict[str, float] = {}
                if len(replay_buffer) >= args.batch_size:
                    losses = trainer.train_step()

                # Calculate iteration time
                iteration_time = time.time() - iteration_start_time
                iteration_times.append(iteration_time)
                avg_iteration_time = sum(iteration_times) / len(iteration_times) if iteration_times else None
                avg_game_time = sum(game_times) / len(game_times) if game_times else None

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
                    policy_loss=losses.get("policy"),
                    value_loss=losses.get("value"),
                    total_loss=losses.get("total"),
                    avg_mcts_time=None,  # Could track this if needed
                    avg_simulations_per_move=args.num_simulations,
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

