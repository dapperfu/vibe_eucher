"""Example script for reviewing AI decisions."""

from pathlib import Path

from eucher.players.computer.ml.ml_config import MLConfig
from eucher.training.decision_reviewer import DecisionReviewer


def main() -> None:
    """Example: Review AI decisions interactively."""
    config = MLConfig()

    # Create decisions file path (would normally be generated from game data)
    decisions_file = config.get_training_data_path("decisions_to_review.json")
    reviews_file = config.get_training_data_path("reviews.json")

    print("Decision Review Example")
    print("=" * 60)
    print(f"Decisions file: {decisions_file}")
    print(f"Reviews file: {reviews_file}")
    print()

    reviewer = DecisionReviewer(decisions_file=decisions_file, reviews_file=reviews_file)

    # Load decisions
    decisions = reviewer.load_decisions()
    if not decisions:
        print("No decisions found to review.")
        print("Generate decisions first by running games with ML players.")
        return

    print(f"Found {len(decisions)} decisions to review")
    print()

    # Review decisions
    reviewer.review_batch(reviewer_id="example_reviewer")

    # Show statistics
    stats = reviewer.get_statistics()
    print("\nReview Statistics:")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

