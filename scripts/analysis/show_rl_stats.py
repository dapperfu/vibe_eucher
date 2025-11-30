"""Utility script to display training statistics from a saved RL model.

This script loads a saved RL model and displays the training statistics
that were saved with it (loss, rewards, epsilon, etc.).
"""

from pathlib import Path
from typing import Optional

from eucher.players.computer.ml.ml_config import MLConfig
from eucher.players.computer.ml.ml_models_rl import RLAgent


def show_rl_stats(model_path: Optional[Path] = None) -> None:
    """
    Display training statistics from a saved RL model.

    Parameters
    ----------
    model_path : Optional[Path]
        Path to the RL model file. If None, uses default from MLConfig.
    """
    if model_path is None:
        config = MLConfig()
        model_path = config.models_dir / "rl_model.pth"

    if not model_path.exists():
        print(f"Error: RL model not found at {model_path}")
        return

    print(f"Loading RL model from: {model_path}")
    print()

    try:
        agent = RLAgent()
        stats = agent.load(model_path)

        if stats is None:
            print("No training statistics found in model file.")
            print("This model was likely saved before statistics tracking was added.")
            print()
            print("Model information:")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            return

        print("=" * 60)
        print("RL Model Training Statistics")
        print("=" * 60)
        print()

        # Basic statistics
        if "games_played" in stats:
            print(f"Games played: {stats['games_played']}")
        if "final_epsilon" in stats:
            print(f"Final epsilon: {stats['final_epsilon']:.4f}")
        print(f"Current epsilon: {agent.epsilon:.4f}")
        print()

        # Reward statistics
        if "avg_reward" in stats:
            print(f"Average reward: {stats['avg_reward']:.2f}")
        if "total_rewards" in stats and stats["total_rewards"]:
            rewards = stats["total_rewards"]
            if isinstance(rewards, list) and len(rewards) > 0:
                print(f"Total rewards recorded: {len(rewards)}")
                print(f"  Min reward: {min(rewards):.2f}")
                print(f"  Max reward: {max(rewards):.2f}")
                print(f"  Last 100 avg: {sum(rewards[-100:]) / len(rewards[-100:]):.2f}" if len(rewards) >= 100 else "")
        print()

        # Loss statistics
        if "avg_loss" in stats:
            print(f"Average training loss: {stats['avg_loss']:.4f}")
        if "training_losses" in stats and stats["training_losses"]:
            losses = stats["training_losses"]
            if isinstance(losses, list) and len(losses) > 0:
                print(f"Training losses recorded: {len(losses)}")
                print(f"  Min loss: {min(losses):.4f}")
                print(f"  Max loss: {max(losses):.4f}")
                print(f"  Last 100 avg: {sum(losses[-100:]) / len(losses[-100:]):.4f}" if len(losses) >= 100 else "")
        print()

        # Epsilon history
        if "epsilon_history" in stats and stats["epsilon_history"]:
            eps_history = stats["epsilon_history"]
            if isinstance(eps_history, list) and len(eps_history) > 0:
                print(f"Epsilon history: {len(eps_history)} entries")
                print(f"  Initial: {eps_history[0]:.4f}")
                print(f"  Final: {eps_history[-1]:.4f}")

        print("=" * 60)

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Display training statistics from a saved RL model")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to RL model file (default: models/rl_model.pth)",
    )

    args = parser.parse_args()
    show_rl_stats(model_path=args.model_path)


if __name__ == "__main__":
    main()

