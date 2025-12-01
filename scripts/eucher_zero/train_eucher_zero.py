#!/usr/bin/env python3
"""Training script for EuchreZero player.

EuchreZero is a search-enhanced reinforcement learning architecture inspired by AlphaZero.
Supports cumulative training, multi-device support, and checkpoint management.
"""

import argparse
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

from eucher.game import Game
from eucher.plugins import get_registry
from plugins.eucher_zero.config import EucherZeroConfig
from plugins.eucher_zero.networks.model import EucherZeroModel
from plugins.eucher_zero.training.dashboard import EuchreZeroDashboard
from plugins.eucher_zero.training.replay_buffer import ReplayBuffer
from plugins.eucher_zero.training.self_play import generate_self_play_game
from plugins.eucher_zero.training.trainer import EucherZeroTrainer


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
    print("EuchreZero Training Script")
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
        print("Opponent: self-play (EucherZero vs EucherZero)")
    print("=" * 80)

    # Setup device
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "gpu":
        device_str = "cuda"
    else:
        device_str = args.device

    # Create config
    config = EucherZeroConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        device=device_str,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Initialize model
    model = EucherZeroModel(config)

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
    trainer = EucherZeroTrainer(model, config, replay_buffer)

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

                # Generate self-play games in parallel
                # Use threading to parallelize game generation (CPU-bound operations)
                num_workers = min(config.num_games, max(1, os.cpu_count() or 4))
                
                def generate_single_game(game_num: int) -> Optional[List]:
                    """Generate a single self-play game."""
                    try:
                        # Create game configuration
                        opponent_type = args.opponent if args.opponent else "eucher_zero"
                        
                        player_config = [
                            ("EucherZero_0", "eucher_zero"),
                            (f"Opponent_1", opponent_type),
                            ("EucherZero_2", "eucher_zero"),
                            (f"Opponent_3", opponent_type),
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
                        return examples
                    except Exception as e:
                        print(f"Error generating game {game_num}: {e}")
                        return None

                # Generate games in parallel using ThreadPoolExecutor
                games_to_generate = config.num_games
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(generate_single_game, game_num) for game_num in range(games_to_generate)]
                    
                    for future in as_completed(futures):
                        if time.time() - start_time >= max_duration_seconds:
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break
                            
                        try:
                            examples = future.result()
                            if examples is not None:
                                # Add to replay buffer (thread-safe append operations)
                                for example in examples:
                                    replay_buffer.add(
                                        example.state,
                                        example.policy,
                                        example.value,
                                        example.reward,
                                    )
                                game_count += 1
                                total_examples_collected += len(examples)
                        except Exception as e:
                            print(f"Error processing game result: {e}")
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
