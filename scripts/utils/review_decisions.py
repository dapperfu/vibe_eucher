"""Command-line interface for reviewing AI decisions."""

import argparse
from pathlib import Path

from plugins.ml.ml_config import MLConfig
from eucher.training.decision_format import DecisionType
from eucher.training.decision_reviewer import DecisionReviewer


def main() -> None:
    """Main review script."""
    parser = argparse.ArgumentParser(description="Review AI decisions")
    parser.add_argument(
        "--decisions_file",
        type=str,
        default=None,
        help="File containing decisions to review (default: training_data/decisions_to_review.json)",
    )
    parser.add_argument(
        "--reviews_file",
        type=str,
        default=None,
        help="File to save reviews (default: training_data/reviews.json)",
    )
    parser.add_argument(
        "--filter_type",
        type=str,
        choices=["order_up", "call_trump", "play_card", "discard"],
        default=None,
        help="Filter by decision type",
    )
    parser.add_argument(
        "--reviewer_id",
        type=str,
        default=None,
        help="ID of the reviewer",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show review statistics",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export reviews to file (specify path)",
    )
    parser.add_argument(
        "--export_format",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Export format (default: json)",
    )

    args = parser.parse_args()

    config = MLConfig()
    decisions_file = Path(args.decisions_file) if args.decisions_file else None
    reviews_file = Path(args.reviews_file) if args.reviews_file else None

    reviewer = DecisionReviewer(decisions_file=decisions_file, reviews_file=reviews_file)

    if args.stats:
        stats = reviewer.get_statistics()
        print("\nReview Statistics:")
        print("=" * 60)
        for key, value in stats.items():
            print(f"{key}: {value}")
        return

    if args.export:
        reviewer.export_reviews(Path(args.export), format=args.export_format)
        print(f"Exported reviews to {args.export}")
        return

    # Interactive review
    filter_type = None
    if args.filter_type:
        filter_type = DecisionType(args.filter_type)

    print("Starting interactive review session...")
    print("Press Ctrl+C to stop and save progress\n")

    try:
        reviewer.review_batch(filter_type=filter_type, reviewer_id=args.reviewer_id)
    except KeyboardInterrupt:
        print("\n\nReview session interrupted. Saving progress...")
        reviewer.save_reviews()
        stats = reviewer.get_statistics()
        print("\nReview Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

