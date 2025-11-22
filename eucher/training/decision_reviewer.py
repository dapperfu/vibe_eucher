"""Interactive review system for AI decisions."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from eucher.cards import Card, Rank, Suit
from eucher.players.computer.ml.ml_features import GameStateEncoder
from eucher.training.decision_format import DecisionRecord, DecisionType, load_decisions
from eucher.training.review_data import (
    DecisionReview,
    ReviewStatus,
    get_review_statistics,
    load_reviews,
    save_reviews,
)


class DecisionReviewer:
    """Interactive system for reviewing AI decisions."""

    def __init__(self, decisions_file: Optional[Path] = None, reviews_file: Optional[Path] = None) -> None:
        """
        Initialize the decision reviewer.

        Parameters
        ----------
        decisions_file : Optional[Path]
            File containing decisions to review.
        reviews_file : Optional[Path]
            File to save reviews.
        """
        from eucher.players.computer.ml.ml_config import MLConfig

        self.config = MLConfig()
        self.encoder = GameStateEncoder()

        if decisions_file is None:
            decisions_file = self.config.get_training_data_path("decisions_to_review.json")
        self.decisions_file = Path(decisions_file)

        if reviews_file is None:
            reviews_file = self.config.get_training_data_path("reviews.json")
        self.reviews_file = Path(reviews_file)

        self.decisions: List[DecisionRecord] = []
        self.reviews: List[DecisionReview] = []
        self.load_existing_reviews()

    def load_decisions(self) -> List[DecisionRecord]:
        """
        Load decisions to review.

        Returns
        -------
        List[DecisionRecord]
            List of decisions.
        """
        if self.decisions_file.exists():
            self.decisions = load_decisions(str(self.decisions_file))
        return self.decisions

    def load_existing_reviews(self) -> None:
        """Load existing reviews from file."""
        if self.reviews_file.exists():
            self.reviews = load_reviews(str(self.reviews_file))

    def save_reviews(self) -> None:
        """Save reviews to file."""
        self.reviews_file.parent.mkdir(parents=True, exist_ok=True)
        save_reviews(self.reviews, str(self.reviews_file))
        print(f"Saved {len(self.reviews)} reviews to {self.reviews_file}")

    def display_game_state(self, decision: DecisionRecord) -> str:
        """
        Display game state in human-readable format.

        Parameters
        ----------
        decision : DecisionRecord
            Decision record to display.

        Returns
        -------
        str
            Human-readable game state description.
        """
        # Extract information from game state
        features = decision.game_state.get("features", [])
        metadata = decision.metadata

        lines = []
        lines.append(f"Game ID: {decision.game_id}")
        lines.append(f"Player ID: {decision.player_id}")
        lines.append(f"Decision Type: {decision.decision_type.value}")

        if "turned_card" in metadata:
            lines.append(f"Turned Card: {metadata['turned_card']}")
        if "led_suit" in metadata:
            lines.append(f"Led Suit: {metadata.get('led_suit', 'None')}")
        if "trump_suit" in metadata:
            lines.append(f"Trump Suit: {metadata.get('trump_suit', 'None')}")

        return "\n".join(lines)

    def display_ai_decision(self, decision: DecisionRecord) -> str:
        """
        Display AI decision in human-readable format.

        Parameters
        ----------
        decision : DecisionRecord
            Decision record.

        Returns
        -------
        str
            Human-readable decision description.
        """
        decision_value = decision.decision_value

        if decision.decision_type == DecisionType.ORDER_UP:
            return f"Order Up: {decision_value}"
        elif decision.decision_type == DecisionType.CALL_TRUMP:
            if decision_value is None:
                return "Call Trump: Pass"
            return f"Call Trump: {decision_value}"
        elif decision.decision_type == DecisionType.PLAY_CARD:
            if isinstance(decision_value, int):
                card = self.encoder.decode_card_index(decision_value)
                if card:
                    return f"Play Card: {card}"
            return f"Play Card: {decision_value}"
        elif decision.decision_type == DecisionType.DISCARD:
            if isinstance(decision_value, int):
                card = self.encoder.decode_card_index(decision_value)
                if card:
                    return f"Discard: {card}"
            return f"Discard: {decision_value}"

        return str(decision_value)

    def review_decision_interactive(
        self, decision: DecisionRecord, reviewer_id: Optional[str] = None
    ) -> DecisionReview:
        """
        Interactively review a decision.

        Parameters
        ----------
        decision : DecisionRecord
            Decision to review.
        reviewer_id : Optional[str]
            ID of the reviewer.

        Returns
        -------
        DecisionReview
            Review record.
        """
        print("\n" + "=" * 60)
        print("Reviewing Decision")
        print("=" * 60)
        print(self.display_game_state(decision))
        print(f"\nAI Decision: {self.display_ai_decision(decision)}")

        # Check if already reviewed
        existing_review = next(
            (r for r in self.reviews if r.decision_id == decision.decision_id), None
        )
        if existing_review:
            print(f"\nExisting review: {existing_review.review_status.value}")
            if existing_review.human_rating:
                print(f"Rating: {existing_review.human_rating}/5")
            if existing_review.feedback_notes:
                print(f"Notes: {existing_review.feedback_notes}")

        # Get review input
        print("\nReview options:")
        print("1. Approve")
        print("2. Reject")
        print("3. Rate (1-5)")
        print("4. Add notes")
        print("5. Skip")

        choice = input("Your choice (1-5): ").strip()

        review_status = ReviewStatus.PENDING
        rating = None
        correct_decision = None
        notes = None

        if choice == "1":
            review_status = ReviewStatus.APPROVED
        elif choice == "2":
            review_status = ReviewStatus.REJECTED
            correct_input = input("What should the correct decision be? ").strip()
            # Parse correct decision based on decision type
            correct_decision = correct_input
        elif choice == "3":
            rating_input = input("Rating (1-5): ").strip()
            try:
                rating = int(rating_input)
                if 1 <= rating <= 5:
                    review_status = ReviewStatus.APPROVED if rating >= 3 else ReviewStatus.REJECTED
            except ValueError:
                print("Invalid rating")
        elif choice == "4":
            notes = input("Enter notes: ").strip()
        elif choice == "5":
            review_status = ReviewStatus.PENDING

        review_id = f"review_{decision.decision_id}_{datetime.now().isoformat()}"

        review = DecisionReview(
            review_id=review_id,
            decision_id=decision.decision_id,
            game_state_snapshot=decision.game_state,
            ai_decision=decision.decision_value,
            review_status=review_status,
            human_rating=rating,
            correct_decision=correct_decision,
            feedback_notes=notes,
            reviewer_id=reviewer_id,
        )

        # Update or add review
        existing_idx = next(
            (i for i, r in enumerate(self.reviews) if r.decision_id == decision.decision_id), None
        )
        if existing_idx is not None:
            self.reviews[existing_idx] = review
        else:
            self.reviews.append(review)

        return review

    def review_batch(
        self,
        decisions: Optional[List[DecisionRecord]] = None,
        filter_type: Optional[DecisionType] = None,
        reviewer_id: Optional[str] = None,
    ) -> List[DecisionReview]:
        """
        Review a batch of decisions.

        Parameters
        ----------
        decisions : Optional[List[DecisionRecord]]
            Decisions to review. If None, uses loaded decisions.
        filter_type : Optional[DecisionType]
            Filter by decision type.
        reviewer_id : Optional[str]
            ID of the reviewer.

        Returns
        -------
        List[DecisionReview]
            List of reviews.
        """
        if decisions is None:
            decisions = self.load_decisions()

        if filter_type:
            decisions = [d for d in decisions if d.decision_type == filter_type]

        reviews = []
        for i, decision in enumerate(decisions):
            print(f"\nReviewing decision {i + 1}/{len(decisions)}")
            review = self.review_decision_interactive(decision, reviewer_id)
            reviews.append(review)

            # Auto-save periodically
            if (i + 1) % 10 == 0:
                self.save_reviews()

        self.save_reviews()
        return reviews

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get review statistics.

        Returns
        -------
        Dict[str, Any]
            Statistics dictionary.
        """
        return get_review_statistics(self.reviews)

    def export_reviews(self, output_file: Path, format: str = "json") -> None:
        """
        Export reviews to file.

        Parameters
        ----------
        output_file : Path
            Path to export file.
        format : str
            Export format: "json" or "csv".
        """
        if format == "json":
            save_reviews(self.reviews, str(output_file))
        elif format == "csv":
            import csv

            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "review_id",
                        "decision_id",
                        "review_status",
                        "human_rating",
                        "feedback_notes",
                        "reviewer_id",
                        "timestamp",
                    ],
                )
                writer.writeheader()
                for review in self.reviews:
                    writer.writerow(review.to_dict())


if __name__ == "__main__":
    reviewer = DecisionReviewer()
    reviewer.review_batch()

