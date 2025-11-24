"""Example script for training RL agents."""

from src.ml_config import MLConfig
from src.training.train_rl import train_rl_through_self_play


def main() -> None:
    """Example: Train RL agent through self-play."""
    config = MLConfig()

    print("Training RL agent through self-play...")
    print("=" * 60)

    agent = train_rl_through_self_play(
        num_games=1000,
        config=config,
        output_dir=config.models_dir,
        checkpoint_interval=100,
    )

    print("\nRL training complete!")
    print(f"Model saved to: {config.models_dir / 'rl_model.pth'}")


if __name__ == "__main__":
    main()

