"""Review data structures for human review of AI decisions."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from eucher.cards import Card, Suit


class ReviewStatus(Enum):
    """Status of a review."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


@dataclass
class DecisionReview:
    """Review of an AI decision."""

    review_id: str
    decision_id: str
    game_state_snapshot: Dict[str, Any]
    ai_decision: Any
    review_status: ReviewStatus
    human_rating: Optional[int] = None  # 1-5 scale
    correct_decision: Optional[Any] = None  # If rejected, what should it be?
    feedback_notes: Optional[str] = None
    reviewer_id: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation.
        """
        data = asdict(self)
        data["review_status"] = self.review_status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionReview":
        """
        Create from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary representation.

        Returns
        -------
        DecisionReview
            Decision review instance.
        """
        data = data.copy()
        data["review_status"] = ReviewStatus(data["review_status"])
        return cls(**data)


def save_reviews(reviews: List[DecisionReview], filepath: str) -> None:
    """
    Save reviews to JSON file.

    Parameters
    ----------
    reviews : List[DecisionReview]
        List of reviews.
    filepath : str
        Path to save file.
    """
    data = [review.to_dict() for review in reviews]
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_reviews(filepath: str) -> List[DecisionReview]:
    """
    Load reviews from JSON file.

    Parameters
    ----------
    filepath : str
        Path to load file from.

    Returns
    -------
    List[DecisionReview]
        List of reviews.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    return [DecisionReview.from_dict(item) for item in data]


def get_review_statistics(reviews: List[DecisionReview]) -> Dict[str, Any]:
    """
    Get statistics about reviews.

    Parameters
    ----------
    reviews : List[DecisionReview]
        List of reviews.

    Returns
    -------
    Dict[str, Any]
        Statistics dictionary.
    """
    total = len(reviews)
    approved = sum(1 for r in reviews if r.review_status == ReviewStatus.APPROVED)
    rejected = sum(1 for r in reviews if r.review_status == ReviewStatus.REJECTED)
    pending = sum(1 for r in reviews if r.review_status == ReviewStatus.PENDING)

    ratings = [r.human_rating for r in reviews if r.human_rating is not None]
    avg_rating = sum(ratings) / len(ratings) if ratings else None

    return {
        "total_reviews": total,
        "approved": approved,
        "rejected": rejected,
        "pending": pending,
        "approval_rate": approved / total if total > 0 else 0.0,
        "average_rating": avg_rating,
        "reviews_with_rating": len(ratings),
    }

