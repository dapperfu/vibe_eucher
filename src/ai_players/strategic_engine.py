"""Strategic decision engine for PyTorch AI player.

Implements bower drawing strategy and card evaluation system.
"""

from typing import List, Optional, Tuple

from src.cards import Card, Rank, Suit
from src.rules import RulesEngine

from src.ai_players.game_state_tracker import BowerProbabilityEstimator, TrickHistoryTracker


class CardEvaluationSystem:
    """Evaluates card strength in various game contexts.

    Parameters
    ----------
    rules_engine : RulesEngine
        Rules engine for validating plays.
    """

    def __init__(self, rules_engine: RulesEngine) -> None:
        """Initialize the card evaluation system.

        Parameters
        ----------
        rules_engine : RulesEngine
            Rules engine instance.
        """
        self.rules = rules_engine

    def evaluate_trump_value(self, card: Card, trump_suit: Suit) -> float:
        """Evaluate the value of a card as trump.

        Parameters
        ----------
        card : Card
            The card to evaluate.
        trump_suit : Suit
            Current trump suit.

        Returns
        -------
        float
            Trump value (0.0-1.0, higher is better).
        """
        if not card._is_trump(trump_suit):
            return 0.0

        # Get trump rank (1-7, where 7 is right bower)
        trump_rank = card._get_trump_rank(trump_suit)
        return trump_rank / 7.0

    def evaluate_off_suit_value(self, card: Card, trump_suit: Optional[Suit]) -> float:
        """Evaluate the value of a card as off-suit.

        Parameters
        ----------
        card : Card
            The card to evaluate.
        trump_suit : Optional[Suit]
            Current trump suit, if any.

        Returns
        -------
        float
            Off-suit value (0.0-1.0, higher is better).
        """
        if trump_suit is not None and card._is_trump(trump_suit):
            return 0.0  # Not an off-suit card

        # Standard rank value (9=1, 10=2, J=3, Q=4, K=5, A=6)
        rank_values = {
            Rank.NINE: 1,
            Rank.TEN: 2,
            Rank.JACK: 3,
            Rank.QUEEN: 4,
            Rank.KING: 5,
            Rank.ACE: 6,
        }
        rank_value = rank_values.get(card.rank, 0)
        return rank_value / 6.0

    def evaluate_trick_winning_probability(
        self,
        card: Card,
        hand: List[Card],
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
    ) -> float:
        """Estimate probability that playing this card will win the trick.

        Parameters
        ----------
        card : Card
            The card to evaluate.
        hand : List[Card]
            Player's current hand.
        led_suit : Optional[Suit]
            Suit that was led, if any.
        trump_suit : Optional[Suit]
            Current trump suit, if any.
        trick_cards : List[Card]
            Cards already played in the trick.

        Returns
        -------
        float
            Estimated probability (0.0-1.0) of winning the trick.
        """
        if not trick_cards:
            # Leading - estimate based on card strength
            if trump_suit and card._is_trump(trump_suit):
                return self.evaluate_trump_value(card, trump_suit)
            else:
                return self.evaluate_off_suit_value(card, trump_suit)

        # Following - compare to cards already played
        current_winner = trick_cards[0]
        for played_card in trick_cards[1:]:
            if played_card.compare_to(current_winner, trump_suit, led_suit) > 0:
                current_winner = played_card

        # Compare our card to current winner
        comparison = card.compare_to(current_winner, trump_suit, led_suit)
        if comparison > 0:
            # Our card beats current winner
            # Estimate probability based on remaining cards
            return 0.7  # High probability if we beat current winner
        elif comparison < 0:
            # Our card loses to current winner
            return 0.1  # Low probability
        else:
            # Equal or can't compare
            return 0.5  # Neutral

    def get_card_strength(self, card: Card, trump_suit: Optional[Suit], context: str = "general") -> float:
        """Get overall card strength in given context.

        Parameters
        ----------
        card : Card
            The card to evaluate.
        trump_suit : Optional[Suit]
            Current trump suit, if any.
        context : str
            Context for evaluation ("trump", "off_suit", "general").

        Returns
        -------
        float
            Card strength (0.0-1.0).
        """
        if context == "trump" and trump_suit:
            return self.evaluate_trump_value(card, trump_suit)
        elif context == "off_suit":
            return self.evaluate_off_suit_value(card, trump_suit)
        else:
            # General: combine both values
            trump_val = self.evaluate_trump_value(card, trump_suit) if trump_suit else 0.0
            off_suit_val = self.evaluate_off_suit_value(card, trump_suit)
            return max(trump_val, off_suit_val)


class BowerDrawingStrategy:
    """Implements logic to draw out bowers using low trump cards.

    Parameters
    ----------
    tracker : TrickHistoryTracker
        Trick history tracker.
    bower_estimator : BowerProbabilityEstimator
        Bower probability estimator.
    card_evaluator : CardEvaluationSystem
        Card evaluation system.
    """

    def __init__(
        self,
        tracker: TrickHistoryTracker,
        bower_estimator: BowerProbabilityEstimator,
        card_evaluator: CardEvaluationSystem,
    ) -> None:
        """Initialize the bower drawing strategy.

        Parameters
        ----------
        tracker : TrickHistoryTracker
            Trick history tracker.
        bower_estimator : BowerProbabilityEstimator
            Bower probability estimator.
        card_evaluator : CardEvaluationSystem
            Card evaluation system.
        """
        self.tracker = tracker
        self.bower_estimator = bower_estimator
        self.card_evaluator = card_evaluator

    def should_draw_bowers(
        self,
        hand: List[Card],
        trump_suit: Suit,
        led_suit: Optional[Suit],
        trick_number: int,
        team_score: int,
        opponent_score: int,
        player_position: int,
        is_leading: bool,
    ) -> Tuple[bool, Optional[Card]]:
        """Determine if we should play a low trump to draw out bowers.

        Parameters
        ----------
        hand : List[Card]
            Player's current hand.
        trump_suit : Suit
            Current trump suit.
        led_suit : Optional[Suit]
            Suit that was led, if any.
        trick_number : int
            Current trick number (0-4).
        team_score : int
            Current team score.
        opponent_score : int
            Opponent team score.
        player_position : int
            Player position (0-3).
        is_leading : bool
            Whether player is leading the trick.

        Returns
        -------
        Tuple[bool, Optional[Card]]
            (should_draw, card_to_play) - True if should draw, and card to play if so.
        """
        if not is_leading:
            # Only draw when leading
            return False, None

        if led_suit is not None:
            # Already following suit
            return False, None

        # Get low trump cards (9 and 10 of trump suit)
        low_trump_cards = [
            card
            for card in hand
            if card.suit == trump_suit and card.rank in [Rank.NINE, Rank.TEN] and card._is_trump(trump_suit)
        ]

        if not low_trump_cards:
            return False, None

        # Estimate bower probability
        bower_probs = self.bower_estimator.estimate_bower_probability(trump_suit, hand, player_position)

        # Decision factors
        # 1. Early tricks are more valuable for drawing
        early_trick_bonus = 1.0 - (trick_number / 5.0)  # 1.0 for trick 0, 0.0 for trick 4

        # 2. Score position matters
        score_differential = team_score - opponent_score
        if score_differential < 0:
            # Behind - more aggressive
            score_factor = 1.2
        elif score_differential > 5:
            # Way ahead - less need to draw
            score_factor = 0.7
        else:
            score_factor = 1.0

        # 3. High bower probability increases value of drawing
        bower_factor = bower_probs["right_bower_held"] + bower_probs["left_bower_held"]

        # Combined decision threshold
        draw_value = early_trick_bonus * score_factor * bower_factor

        # Threshold: draw if value > 0.5
        if draw_value > 0.5:
            # Choose the lowest trump card (9 before 10)
            low_trump_cards.sort(key=lambda c: c.rank.value)
            return True, low_trump_cards[0]

        return False, None

    def evaluate_drawing_opportunity(
        self,
        hand: List[Card],
        trump_suit: Suit,
        trick_number: int,
        team_tricks_won: int,
        opponent_tricks_won: int,
    ) -> float:
        """Evaluate the value of a bower drawing opportunity.

        Parameters
        ----------
        hand : List[Card]
            Player's current hand.
        trump_suit : Suit
            Current trump suit.
        trick_number : int
            Current trick number (0-4).
        team_tricks_won : int
            Tricks won by player's team so far.
        opponent_tricks_won : int
            Tricks won by opponents so far.

        Returns
        -------
        float
            Value of drawing opportunity (0.0-1.0).
        """
        # Check if we have low trump
        has_low_trump = any(
            card.suit == trump_suit and card.rank in [Rank.NINE, Rank.TEN] for card in hand
        )

        if not has_low_trump:
            return 0.0

        # Early tricks are more valuable
        trick_value = 1.0 - (trick_number / 5.0)

        # If behind in tricks, drawing is more valuable
        trick_differential = team_tricks_won - opponent_tricks_won
        if trick_differential < 0:
            trick_factor = 1.3
        else:
            trick_factor = 1.0

        # Estimate bower probability
        bower_probs = self.bower_estimator.estimate_bower_probability(trump_suit, hand, 0)
        bower_prob = (bower_probs["right_bower_held"] + bower_probs["left_bower_held"]) / 2.0

        return trick_value * trick_factor * bower_prob

