"""Script to collect training data by running games."""

import argparse
from pathlib import Path

from eucher.players.computer.ml.ml_config import MLConfig
from eucher.training.self_play import SelfPlayTrainer


def main() -> None:
    """Collect training data by running games."""
    parser = argparse.ArgumentParser(description="Collect training data for ML models")
    parser.add_argument(
        "--num_games",
        type=int,
        default=100,
        help="Number of games to play",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save training data (default: training_data/)",
    )
    parser.add_argument(
        "--data_prefix",
        type=str,
        default="training",
        help="Prefix for data filenames",
    )

    args = parser.parse_args()

    config = MLConfig()
    output_dir = Path(args.output_dir) if args.output_dir else config.training_data_dir

    print("=" * 60)
    print("Collecting Training Data")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Number of games: {args.num_games}")
    print()

    trainer = SelfPlayTrainer(output_dir=output_dir)
    trainer.run_training_round(num_games=args.num_games)

    print("\n" + "=" * 60)
    print("Data Collection Complete!")
    print("=" * 60)
    print(f"Data saved to: {output_dir}")
    print(f"Files: {args.data_prefix}_*.json and {args.data_prefix}_*.csv")


if __name__ == "__main__":
    main()

