"""Tests for decision review system."""

import tempfile
from pathlib import Path

import pytest

from eucher.training.decision_format import DecisionRecord, DecisionType
from eucher.training.review_data import (
    DecisionReview,
    ReviewStatus,
    get_review_statistics,
    load_reviews,
    save_reviews,
)
from eucher.training.decision_reviewer import DecisionReviewer


class TestReviewData:
    """Tests for review data structures."""

    def test_decision_review_creation(self) -> None:
        """Test creating a decision review."""
        review = DecisionReview(
            review_id="test_review_1",
            decision_id="test_decision_1",
            game_state_snapshot={"features": [1, 2, 3]},
            ai_decision=5,
            review_status=ReviewStatus.APPROVED,
            human_rating=4,
            feedback_notes="Good decision",
            reviewer_id="reviewer1",
        )

        assert review.review_id == "test_review_1"
        assert review.review_status == ReviewStatus.APPROVED
        assert review.human_rating == 4

    def test_save_and_load_reviews(self) -> None:
        """Test saving and loading reviews."""
        reviews = [
            DecisionReview(
                review_id="review1",
                decision_id="decision1",
                game_state_snapshot={"features": [1, 2, 3]},
                ai_decision=5,
                review_status=ReviewStatus.APPROVED,
            ),
            DecisionReview(
                review_id="review2",
                decision_id="decision2",
                game_state_snapshot={"features": [4, 5, 6]},
                ai_decision=10,
                review_status=ReviewStatus.REJECTED,
                correct_decision=12,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "reviews.json"
            save_reviews(reviews, str(filepath))
            assert filepath.exists()

            loaded_reviews = load_reviews(str(filepath))
            assert len(loaded_reviews) == 2
            assert loaded_reviews[0].review_id == "review1"
            assert loaded_reviews[1].review_status == ReviewStatus.REJECTED

    def test_review_statistics(self) -> None:
        """Test review statistics calculation."""
        reviews = [
            DecisionReview(
                review_id="review1",
                decision_id="decision1",
                game_state_snapshot={},
                ai_decision=5,
                review_status=ReviewStatus.APPROVED,
                human_rating=5,
            ),
            DecisionReview(
                review_id="review2",
                decision_id="decision2",
                game_state_snapshot={},
                ai_decision=10,
                review_status=ReviewStatus.APPROVED,
                human_rating=4,
            ),
            DecisionReview(
                review_id="review3",
                decision_id="decision3",
                game_state_snapshot={},
                ai_decision=15,
                review_status=ReviewStatus.REJECTED,
                human_rating=2,
            ),
        ]

        stats = get_review_statistics(reviews)
        assert stats["total_reviews"] == 3
        assert stats["approved"] == 2
        assert stats["rejected"] == 1
        assert stats["approval_rate"] == 2 / 3
        assert stats["average_rating"] == (5 + 4 + 2) / 3


class TestDecisionReviewer:
    """Tests for decision reviewer."""

    def test_load_decisions(self) -> None:
        """Test loading decisions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            decisions_file = Path(tmpdir) / "decisions.json"
            reviews_file = Path(tmpdir) / "reviews.json"

            # Create test decisions file
            from eucher.training.decision_format import save_decisions

            decisions = [
                DecisionRecord(
                    decision_id="decision1",
                    game_id="game1",
                    player_id=0,
                    decision_type=DecisionType.PLAY_CARD,
                    game_state={"features": [1, 2, 3]},
                    decision_value=5,
                    metadata={},
                )
            ]
            save_decisions(decisions, str(decisions_file))

            reviewer = DecisionReviewer(decisions_file=decisions_file, reviews_file=reviews_file)
            loaded = reviewer.load_decisions()
            assert len(loaded) == 1
            assert loaded[0].decision_id == "decision1"

    def test_review_statistics(self) -> None:
        """Test getting review statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reviewer = DecisionReviewer(
                decisions_file=Path(tmpdir) / "decisions.json",
                reviews_file=Path(tmpdir) / "reviews.json",
            )

            # Add some reviews
            reviewer.reviews = [
                DecisionReview(
                    review_id="review1",
                    decision_id="decision1",
                    game_state_snapshot={},
                    ai_decision=5,
                    review_status=ReviewStatus.APPROVED,
                )
            ]

            stats = reviewer.get_statistics()
            assert stats["total_reviews"] == 1
            assert stats["approved"] == 1

