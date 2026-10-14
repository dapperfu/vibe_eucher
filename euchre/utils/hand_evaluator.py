"""
Hand evaluation utilities for Euchre.

This module provides sophisticated hand strength evaluation that considers
multiple trump suit scenarios and provides detailed strength analysis.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ..models import Card, Suit, Rank


class HandStrengthCategory(Enum):
    """Categories for hand strength assessment."""
    EXCELLENT = "excellent"       # Very strong hand (likely to take 3+ tricks)
    STRONG = "strong"             # Strong hand (likely to take 2-3 tricks)
    GOOD = "good"                 # Good hand (likely to take 1-2 tricks)
    AVERAGE = "average"           # Average hand (likely to take 0-1 tricks)
    WEAK = "weak"                 # Weak hand (unlikely to take tricks)
    POOR = "poor"                 # Very weak hand (very unlikely to take tricks)


@dataclass
class TrumpAnalysis:
    """Analysis of hand strength for a specific trump suit."""
    trump_suit: Suit
    total_score: float
    trump_cards: List[Card]
    left_bower_cards: List[Card]
    high_cards: List[Card]
    off_suit_cards: List[Card]
    strength_category: HandStrengthCategory
    estimated_tricks: float
    confidence: float


@dataclass
class HandEvaluation:
    """Comprehensive hand evaluation results."""
    hand: List[Card]
    trump_analyses: Dict[Suit, TrumpAnalysis]
    best_trump_suit: Suit
    worst_trump_suit: Suit
    overall_strength: HandStrengthCategory
    recommendations: List[str]


class HandEvaluator:
    """Advanced hand evaluator for Euchre hands."""
    
    # Card values in trump context (higher is better)
    TRUMP_VALUES = {
        Rank.JACK: 100,    # Right bower
        Rank.ACE: 85,
        Rank.KING: 80,
        Rank.QUEEN: 75,
        Rank.TEN: 70,
        Rank.NINE: 65
    }
    
    # Left bower values (slightly lower than right bower)
    LEFT_BOWER_VALUES = {
        Rank.JACK: 95,     # Left bower
        Rank.ACE: 80,
        Rank.KING: 75,
        Rank.QUEEN: 70,
        Rank.TEN: 65,
        Rank.NINE: 60
    }
    
    # Off-suit values (much lower)
    OFF_SUIT_VALUES = {
        Rank.ACE: 40,
        Rank.KING: 35,
        Rank.QUEEN: 30,
        Rank.JACK: 25,
        Rank.TEN: 20,
        Rank.NINE: 15
    }
    
    def __init__(self):
        """Initialize the hand evaluator."""
        pass
    
    def evaluate_hand_comprehensive(self, hand: List[Card]) -> HandEvaluation:
        """
        Evaluate hand strength for all possible trump suits.
        
        Parameters
        ----------
        hand : List[Card]
            The hand to evaluate
            
        Returns
        -------
        HandEvaluation
            Comprehensive evaluation results
        """
        if not hand:
            return self._create_empty_evaluation(hand)
        
        # Evaluate for each possible trump suit
        trump_analyses = {}
        for suit in Suit:
            trump_analyses[suit] = self._evaluate_for_trump_suit(hand, suit)
        
        # Find best and worst trump suits
        best_trump = max(trump_analyses.values(), key=lambda x: x.total_score)
        worst_trump = min(trump_analyses.values(), key=lambda x: x.total_score)
        
        # Determine overall strength
        overall_strength = self._determine_overall_strength(trump_analyses)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(trump_analyses, best_trump)
        
        return HandEvaluation(
            hand=hand,
            trump_analyses=trump_analyses,
            best_trump_suit=best_trump.trump_suit,
            worst_trump_suit=worst_trump.trump_suit,
            overall_strength=overall_strength,
            recommendations=recommendations
        )
    
    def evaluate_hand_for_trump(self, hand: List[Card], trump_suit: Suit) -> TrumpAnalysis:
        """
        Evaluate hand strength for a specific trump suit.
        
        Parameters
        ----------
        hand : List[Card]
            The hand to evaluate
        trump_suit : Suit
            The trump suit to evaluate for
            
        Returns
        -------
        TrumpAnalysis
            Analysis for the specific trump suit
        """
        return self._evaluate_for_trump_suit(hand, trump_suit)
    
    def _evaluate_for_trump_suit(self, hand: List[Card], trump_suit: Suit) -> TrumpAnalysis:
        """Internal method to evaluate hand for a specific trump suit."""
        # Get left bower suit
        left_bower_suit = self._get_left_bower_suit(trump_suit)
        
        # Categorize cards
        trump_cards = [card for card in hand if card.suit == trump_suit]
        left_bower_cards = [card for card in hand if card.suit == left_bower_suit and card.rank == Rank.JACK]
        off_suit_cards = [card for card in hand if card.suit not in [trump_suit, left_bower_suit]]
        
        # High cards are 10, J, Q, K, A
        high_cards = [card for card in hand if card.rank.value >= 10]
        
        # Calculate scores
        trump_score = self._calculate_trump_score(trump_cards, left_bower_cards)
        off_suit_score = self._calculate_off_suit_score(off_suit_cards)
        total_score = trump_score + off_suit_score
        
        # Determine strength category and estimated tricks
        strength_category = self._categorize_strength(total_score)
        estimated_tricks = self._estimate_tricks(total_score, len(trump_cards) + len(left_bower_cards))
        confidence = self._calculate_confidence(hand, trump_suit)
        
        return TrumpAnalysis(
            trump_suit=trump_suit,
            total_score=total_score,
            trump_cards=trump_cards,
            left_bower_cards=left_bower_cards,
            high_cards=high_cards,
            off_suit_cards=off_suit_cards,
            strength_category=strength_category,
            estimated_tricks=estimated_tricks,
            confidence=confidence
        )
    
    def _calculate_trump_score(self, trump_cards: List[Card], left_bower_cards: List[Card]) -> float:
        """Calculate score for trump cards."""
        score = 0.0
        
        # Score trump cards
        for card in trump_cards:
            if card.rank == Rank.JACK:
                score += self.TRUMP_VALUES[Rank.JACK]  # Right bower
            else:
                score += self.TRUMP_VALUES.get(card.rank, card.rank.value)
        
        # Score left bower cards
        for card in left_bower_cards:
            score += self.LEFT_BOWER_VALUES.get(card.rank, card.rank.value)
        
        # Bonus for multiple trump cards
        total_trump = len(trump_cards) + len(left_bower_cards)
        if total_trump >= 3:
            score += 50.0  # Strong trump holding
        elif total_trump >= 2:
            score += 25.0  # Good trump holding
        
        return score
    
    def _calculate_off_suit_score(self, off_suit_cards: List[Card]) -> float:
        """Calculate score for off-suit cards."""
        score = 0.0
        
        for card in off_suit_cards:
            score += self.OFF_SUIT_VALUES.get(card.rank, card.rank.value)
        
        # Bonus for having off-suit aces (can win tricks when trump is led)
        aces = [card for card in off_suit_cards if card.rank == Rank.ACE]
        score += len(aces) * 15.0
        
        return score
    
    def _get_left_bower_suit(self, trump_suit: Suit) -> Suit:
        """Get the left bower suit for a given trump suit."""
        if trump_suit in [Suit.HEARTS, Suit.DIAMONDS]:
            return Suit.DIAMONDS if trump_suit == Suit.HEARTS else Suit.HEARTS
        else:  # CLUBS or SPADES
            return Suit.SPADES if trump_suit == Suit.CLUBS else Suit.CLUBS
    
    def _categorize_strength(self, total_score: float) -> HandStrengthCategory:
        """Categorize hand strength based on total score."""
        if total_score >= 300:
            return HandStrengthCategory.EXCELLENT
        elif total_score >= 250:
            return HandStrengthCategory.STRONG
        elif total_score >= 200:
            return HandStrengthCategory.GOOD
        elif total_score >= 150:
            return HandStrengthCategory.AVERAGE
        elif total_score >= 100:
            return HandStrengthCategory.WEAK
        else:
            return HandStrengthCategory.POOR
    
    def _estimate_tricks(self, total_score: float, trump_count: int) -> float:
        """Estimate number of tricks this hand can take."""
        # Base estimation on score
        if total_score >= 300:
            base_tricks = 3.5
        elif total_score >= 250:
            base_tricks = 2.8
        elif total_score >= 200:
            base_tricks = 2.0
        elif total_score >= 150:
            base_tricks = 1.2
        elif total_score >= 100:
            base_tricks = 0.5
        else:
            base_tricks = 0.0
        
        # Adjust based on trump count
        if trump_count >= 4:
            base_tricks += 0.5
        elif trump_count <= 1:
            base_tricks -= 0.3
        
        return max(0.0, min(5.0, base_tricks))
    
    def _calculate_confidence(self, hand: List[Card], trump_suit: Suit) -> float:
        """Calculate confidence level in the evaluation."""
        # Higher confidence with more trump cards
        left_bower_suit = self._get_left_bower_suit(trump_suit)
        trump_count = len([card for card in hand if card.suit in [trump_suit, left_bower_suit]])
        
        # Base confidence on trump count
        if trump_count >= 4:
            base_confidence = 0.9
        elif trump_count >= 3:
            base_confidence = 0.8
        elif trump_count >= 2:
            base_confidence = 0.7
        elif trump_count >= 1:
            base_confidence = 0.6
        else:
            base_confidence = 0.4
        
        # Adjust for hand consistency
        suits = [card.suit for card in hand]
        unique_suits = len(set(suits))
        if unique_suits <= 2:
            base_confidence += 0.1  # More consistent hand
        elif unique_suits >= 4:
            base_confidence -= 0.1  # Less consistent hand
        
        return max(0.1, min(1.0, base_confidence))
    
    def _determine_overall_strength(self, trump_analyses: Dict[Suit, TrumpAnalysis]) -> HandStrengthCategory:
        """Determine overall hand strength across all trump suits."""
        scores = [analysis.total_score for analysis in trump_analyses.values()]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        
        # Consider both average and maximum potential
        if max_score >= 300 and avg_score >= 200:
            return HandStrengthCategory.EXCELLENT
        elif max_score >= 250 and avg_score >= 180:
            return HandStrengthCategory.STRONG
        elif max_score >= 200 and avg_score >= 150:
            return HandStrengthCategory.GOOD
        elif max_score >= 150 and avg_score >= 120:
            return HandStrengthCategory.AVERAGE
        elif max_score >= 100 and avg_score >= 80:
            return HandStrengthCategory.WEAK
        else:
            return HandStrengthCategory.POOR
    
    def _generate_recommendations(self, trump_analyses: Dict[Suit, TrumpAnalysis], 
                                best_trump: TrumpAnalysis) -> List[str]:
        """Generate strategic recommendations based on hand analysis."""
        recommendations = []
        
        # Trump calling recommendations
        if best_trump.strength_category in [HandStrengthCategory.EXCELLENT, HandStrengthCategory.STRONG]:
            recommendations.append(f"Strongly consider calling {best_trump.trump_suit.name.title()} as trump")
        elif best_trump.strength_category == HandStrengthCategory.GOOD:
            recommendations.append(f"Consider calling {best_trump.trump_suit.name.title()} as trump if partner is dealing")
        
        # Specific hand advice
        trump_count = len(best_trump.trump_cards) + len(best_trump.left_bower_cards)
        if trump_count >= 4:
            recommendations.append("Excellent trump holding - lead trump early to establish control")
        elif trump_count >= 2:
            recommendations.append("Good trump holding - use strategically to win key tricks")
        elif trump_count == 0:
            recommendations.append("No trump cards - play defensively and try to avoid leading")
        
        # Off-suit advice
        if len(best_trump.off_suit_cards) >= 3:
            recommendations.append("Multiple off-suit cards - consider leading off-suit to establish long suits")
        
        # High card advice
        high_card_count = len(best_trump.high_cards)
        if high_card_count >= 4:
            recommendations.append("Strong high card holding - can win tricks even without trump")
        elif high_card_count <= 1:
            recommendations.append("Weak high card holding - rely on trump cards for tricks")
        
        return recommendations
    
    def _create_empty_evaluation(self, hand: List[Card]) -> HandEvaluation:
        """Create evaluation for empty hand."""
        empty_analysis = TrumpAnalysis(
            trump_suit=Suit.HEARTS,  # Default
            total_score=0.0,
            trump_cards=[],
            left_bower_cards=[],
            high_cards=[],
            off_suit_cards=[],
            strength_category=HandStrengthCategory.POOR,
            estimated_tricks=0.0,
            confidence=0.0
        )
        
        return HandEvaluation(
            hand=hand,
            trump_analyses={suit: empty_analysis for suit in Suit},
            best_trump_suit=Suit.HEARTS,
            worst_trump_suit=Suit.HEARTS,
            overall_strength=HandStrengthCategory.POOR,
            recommendations=["Empty hand - no recommendations available"]
        )
    
    def get_hand_summary(self, evaluation: HandEvaluation) -> str:
        """
        Get a human-readable summary of the hand evaluation.
        
        Parameters
        ----------
        evaluation : HandEvaluation
            The hand evaluation results
            
        Returns
        -------
        str
            Formatted summary string
        """
        if not evaluation.hand:
            return "Empty hand"
        
        summary = f"Hand: {len(evaluation.hand)} cards\n"
        summary += f"Overall strength: {evaluation.overall_strength.value.title()}\n"
        summary += f"Best trump suit: {evaluation.best_trump_suit.name.title()}\n"
        summary += f"Worst trump suit: {evaluation.worst_trump_suit.name.title()}\n\n"
        
        summary += "Trump suit analysis:\n"
        for suit, analysis in evaluation.trump_analyses.items():
            summary += f"  {suit.name.title()}: {analysis.strength_category.value.title()} "
            summary += f"({analysis.total_score:.1f} pts, {analysis.estimated_tricks:.1f} tricks)\n"
        
        summary += "\nRecommendations:\n"
        for rec in evaluation.recommendations:
            summary += f"  • {rec}\n"
        
        return summary 