#!/usr/bin/env python3
"""Self-play training script for PyTorch AI player.

Trains PyTorch AI by playing live games against itself, collecting data,
and training incrementally on that data.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eucher.plugins import get_registry
from plugins.ml.pytorch.pytorch_networks import create_network
from plugins.ml.pytorch.training.checkpoint_manager import CheckpointManager
from plugins.ml.pytorch.training.self_play_trainer import SelfPlayPyTorchTrainer


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


def parse_player_config(config_str: str) -> List[Tuple[str, str]]:
    """Parse player configuration string.

    Parameters
    ----------
    config_str : str
        Configuration string like "all_pytorch" or "pytorch,heuristic,pytorch,random"

    Returns
    -------
    List[Tuple[str, str]]
        List of (name, profile_type) tuples.
    """
    # Predefined configurations
    predefined = {
        "all_pytorch": [
            ("PyTorch1", "pytorch_ai"),
            ("PyTorch2", "pytorch_ai"),
            ("PyTorch3", "pytorch_ai"),
            ("PyTorch4", "pytorch_ai"),
        ],
        "pytorch_vs_heuristic": [
            ("PyTorch1", "pytorch_ai"),
            ("Heuristic1", "heuristic"),
            ("PyTorch2", "pytorch_ai"),
            ("Heuristic2", "heuristic"),
        ],
        "pytorch_vs_random": [
            ("PyTorch1", "pytorch_ai"),
            ("Random1", "random"),
            ("PyTorch2", "pytorch_ai"),
            ("Random2", "random"),
        ],
        "pytorch_vs_weighted_heuristic": [
            ("PyTorch1", "pytorch_ai"),
            ("WeightedHeuristic1", "weighted_heuristic"),
            ("PyTorch2", "pytorch_ai"),
            ("WeightedHeuristic2", "weighted_heuristic"),
        ],
        "pytorch_vs_ai": [  # Backward compatibility alias
            ("PyTorch1", "pytorch_ai"),
            ("AI1", "ai"),
            ("PyTorch2", "pytorch_ai"),
            ("AI2", "ai"),
        ],
    }

    if config_str in predefined:
        return predefined[config_str]

    # Parse comma-separated list
    types = [t.strip() for t in config_str.split(",")]
    if len(types) != 4:
        raise ValueError(f"Player config must have exactly 4 players, got {len(types)}")

    return [
        (f"Player{i+1}", types[i])
        for i in range(4)
    ]


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PyTorch AI through self-play")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda"],
        help="Device to train on (default: auto)",
    )
    parser.add_argument(
        "--games-per-training",
        type=int,
        default=50,
        help="Number of games to play before each training step (default: 50)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration probability for epsilon-greedy (default: 0.1)",
    )
    parser.add_argument(
        "--player-config",
        type=str,
        default="all_pytorch",
        help="Player configuration: 'all_pytorch', 'pytorch_vs_heuristic', 'pytorch_vs_weighted_heuristic', 'pytorch_vs_random', "
             "or comma-separated list like 'pytorch,heuristic,pytorch,random' (default: all_pytorch)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Maximum training cycles (default: unlimited)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N cycles (default: 5)",
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
        "--tune-batch",
        action="store_true",
        help="Automatically tune batch size to fit GPU memory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Manual batch size (ignored if --tune-batch is used)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from checkpoint (start fresh)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        help="Opponent plugin to train against. Use 'list' to enumerate available plugins. "
             "If specified, overrides --player-config. Team 0 (players 0,2) will be pytorch_ai, "
             "Team 1 (players 1,3) will be the specified opponent.",
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

    # Determine device
    device_str = args.device
    if device_str == "gpu":
        device_str = "cuda"

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 80)
    print("PyTorch AI Self-Play Training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Games per training: {args.games_per_training}")
    print(f"Epsilon (exploration): {args.epsilon}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print(f"Batch tuning: {args.tune_batch}")
    
    # Handle opponent configuration
    if args.opponent:
        # Validate opponent plugin exists
        registry = get_registry()
        # Plugins are auto-discovered via entry points on import
        if not registry.has(args.opponent):
            print(f"Error: Opponent plugin '{args.opponent}' not found.")
            print("Use --opponent list to see available plugins.")
            sys.exit(1)
        
        # Override player config with opponent setup
        # Team 0 (players 0, 2): pytorch_ai (training player)
        # Team 1 (players 1, 3): opponent
        player_config = [
            ("PyTorch_0", "pytorch_ai"),
            (f"Opponent_1", args.opponent),
            ("PyTorch_2", "pytorch_ai"),
            (f"Opponent_3", args.opponent),
        ]
        print(f"Opponent: {args.opponent}")
    else:
        # Parse player configuration
        try:
            player_config = parse_player_config(args.player_config)
        except ValueError as e:
            print(f"Error parsing player config: {e}")
            sys.exit(1)
    
    print(f"Player configuration: {player_config}")
    print()

    print()

    # Create model
    model = create_network(device=device)
    print(f"Model created with {model.get_parameter_count():,} parameters")
    print()

    # Create trainer
    trainer = SelfPlayPyTorchTrainer(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        device=device,
    )

    # Run training loop
    try:
        trainer.run_training_loop(
            games_per_training=args.games_per_training,
            epsilon=args.epsilon,
            player_config=player_config,
            max_cycles=args.max_cycles,
            checkpoint_interval=args.checkpoint_interval,
            tune_batch=args.tune_batch,
            resume=not args.no_resume,
            learning_rate=args.learning_rate,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer.save_checkpoint()
        sys.exit(0)
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint()
        sys.exit(1)


if __name__ == "__main__":
    main()

