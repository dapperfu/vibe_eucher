"""Game state tracking for PyTorch AI player.

Tracks played cards from won tricks and estimates probabilities of remaining cards.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from src.cards import Card, Rank, Suit


class TrickHistoryTracker:
    """Maintains history of completed tricks and tracks played cards.

    Parameters
    ----------
    None

    Attributes
    ----------
    won_tricks : List[Dict]
        List of completed tricks, each containing cards played and winner info.
    cards_seen : Set[Card]
        Set of all cards that have been played in won tricks.
    suit_distribution : Dict[Suit, int]
        Count of cards played per suit.
    team_tricks_won : Dict[int, int]
        Number of tricks won by each team.
    """

    def __init__(self) -> None:
        """Initialize an empty trick history tracker."""
        self.won_tricks: List[Dict] = []
        self.cards_seen: Set[Card] = set()
        self.suit_distribution: Dict[Suit, int] = defaultdict(int)
        self.team_tricks_won: Dict[int, int] = defaultdict(int)

    def record_trick(
        self, played_cards: List[Card], player_ids: List[int], winner_team: int, winner_player_id: int
    ) -> None:
        """Record a completed trick.

        Parameters
        ----------
        played_cards : List[Card]
            Cards played in the trick (in order).
        player_ids : List[int]
            Player IDs who played each card (same order).
        winner_team : int
            Team that won the trick (0 or 1).
        winner_player_id : int
            Player ID of the trick winner.
        """
        trick_info = {
            "cards": played_cards.copy(),
            "player_ids": player_ids.copy(),
            "winner_team": winner_team,
            "winner_player_id": winner_player_id,
        }
        self.won_tricks.append(trick_info)

        # Update tracking data
        for card in played_cards:
            self.cards_seen.add(card)
            self.suit_distribution[card.suit] += 1

        self.team_tricks_won[winner_team] += 1

    def get_cards_seen(self) -> Set[Card]:
        """Get all cards that have been seen in won tricks.

        Returns
        -------
        Set[Card]
            Set of cards that have been played.
        """
        return self.cards_seen.copy()

    def get_suit_distribution(self) -> Dict[Suit, int]:
        """Get distribution of suits in played cards.

        Returns
        -------
        Dict[Suit, int]
            Count of cards played per suit.
        """
        return self.suit_distribution.copy()

    def get_team_tricks_won(self) -> Dict[int, int]:
        """Get number of tricks won by each team.

        Returns
        -------
        Dict[int, int]
            Dictionary mapping team ID to number of tricks won.
        """
        return self.team_tricks_won.copy()

    def estimate_remaining_cards(self, known_hand: List[Card], trump_suit: Optional[Suit] = None) -> Dict[Card, float]:
        """Estimate probability distribution of remaining cards.

        Parameters
        ----------
        known_hand : List[Card]
            Cards known to be in the player's hand.
        trump_suit : Optional[Suit]
            Current trump suit, if any.

        Returns
        -------
        Dict[Card, float]
            Dictionary mapping cards to estimated probability of still being in play.
        """
        # Create full deck
        all_cards: Set[Card] = set()
        for suit in Suit:
            for rank in [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]:
                all_cards.add(Card(suit, rank))

        # Remove known cards
        remaining = all_cards - self.cards_seen - set(known_hand)

        # Calculate probabilities (uniform distribution among remaining cards)
        num_remaining = len(remaining)
        if num_remaining == 0:
            return {}

        prob_per_card = 1.0 / num_remaining if num_remaining > 0 else 0.0
        return {card: prob_per_card for card in remaining}

    def reset(self) -> None:
        """Reset the tracker for a new hand."""
        self.won_tricks.clear()
        self.cards_seen.clear()
        self.suit_distribution.clear()
        self.team_tricks_won.clear()


class BowerProbabilityEstimator:
    """Estimates likelihood that opponents hold bowers.

    Parameters
    ----------
    None

    Attributes
    ----------
    tracker : TrickHistoryTracker
        Reference to the trick history tracker.
    """

    def __init__(self, tracker: TrickHistoryTracker) -> None:
        """Initialize the bower probability estimator.

        Parameters
        ----------
        tracker : TrickHistoryTracker
            The trick history tracker to use for card tracking.
        """
        self.tracker = tracker

    def estimate_bower_probability(
        self, trump_suit: Suit, known_hand: List[Card], player_position: int
    ) -> Dict[str, float]:
        """Estimate probability that opponents hold bowers.

        Parameters
        ----------
        trump_suit : Suit
            Current trump suit.
        known_hand : List[Card]
            Cards in the player's hand.
        player_position : int
            Position of the player (0-3).

        Returns
        -------
        Dict[str, float]
            Dictionary with probabilities for:
            - "right_bower_held": Probability right bower is held by opponents
            - "left_bower_held": Probability left bower is held by opponents
            - "both_bowers_held": Probability both bowers are held by opponents
        """
        # Right bower: Jack of trump suit
        right_bower = Card(trump_suit, Rank.JACK)

        # Left bower: Jack of same color as trump
        left_bower_suit = self._get_left_bower_suit(trump_suit)
        left_bower = Card(left_bower_suit, Rank.JACK)

        # Check if bowers are in known hand or already played
        right_bower_seen = right_bower in self.tracker.cards_seen or right_bower in known_hand
        left_bower_seen = left_bower in self.tracker.cards_seen or left_bower in known_hand

        # Get remaining cards distribution
        remaining_probs = self.tracker.estimate_remaining_cards(known_hand, trump_suit)

        # Calculate probabilities
        if right_bower_seen:
            right_prob = 0.0
        else:
            right_prob = remaining_probs.get(right_bower, 0.0)
            # Adjust for number of opponents (3 opponents, but cards distributed)
            # Rough estimate: if card is in remaining set, probability it's with opponents
            # is proportional to number of opponent cards vs total remaining
            num_opponent_cards = 18 - len(known_hand)  # 3 opponents × 6 cards
            num_remaining = len(remaining_probs)
            if num_remaining > 0:
                right_prob = min(1.0, right_prob * (num_opponent_cards / num_remaining))

        if left_bower_seen:
            left_prob = 0.0
        else:
            left_prob = remaining_probs.get(left_bower, 0.0)
            num_opponent_cards = 18 - len(known_hand)
            num_remaining = len(remaining_probs)
            if num_remaining > 0:
                left_prob = min(1.0, left_prob * (num_opponent_cards / num_remaining))

        # Both bowers held (independent events, approximate)
        both_prob = right_prob * left_prob

        return {
            "right_bower_held": right_prob,
            "left_bower_held": left_prob,
            "both_bowers_held": both_prob,
        }

    def _get_left_bower_suit(self, trump_suit: Suit) -> Suit:
        """Get the suit of the left bower for a given trump suit.

        Parameters
        ----------
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        Suit
            The suit of the left bower (same color as trump).
        """
        red_suits = {Suit.HEARTS, Suit.DIAMONDS}
        black_suits = {Suit.CLUBS, Suit.SPADES}

        if trump_suit in red_suits:
            # If trump is red, left bower is the other red suit
            return Suit.DIAMONDS if trump_suit == Suit.HEARTS else Suit.HEARTS
        else:
            # If trump is black, left bower is the other black suit
            return Suit.SPADES if trump_suit == Suit.CLUBS else Suit.CLUBS

    def estimate_bower_from_play_pattern(
        self, trump_suit: Suit, trick_history: List[Dict], current_trick: List[Card]
    ) -> float:
        """Estimate bower probability based on play patterns.

        If opponents avoid playing trump when they could, they likely hold high trump.

        Parameters
        ----------
        trump_suit : Suit
            Current trump suit.
        trick_history : List[Dict]
            History of completed tricks.
        current_trick : List[Card]
            Cards played in current trick so far.

        Returns
        -------
        float
            Estimated probability (0.0-1.0) that opponents hold bowers based on play patterns.
        """
        # Count times opponents could have played trump but didn't
        trump_avoidance_count = 0
        total_opportunities = 0

        for trick in trick_history:
            cards = trick["cards"]
            # If trick was led with non-trump and opponents didn't trump when they could
            if cards and cards[0].suit != trump_suit:
                # Check if any opponent played non-trump when they might have had trump
                # This is a heuristic - in reality we'd need to know what cards they had
                total_opportunities += 1

        # Simple heuristic: more avoidance = higher probability of holding bowers
        if total_opportunities == 0:
            return 0.5  # Neutral estimate

        avoidance_ratio = trump_avoidance_count / total_opportunities
        return min(1.0, 0.3 + (avoidance_ratio * 0.4))  # Range: 0.3 to 0.7

