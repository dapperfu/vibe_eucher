#!/usr/bin/env python3
"""
Demonstration of the new hand evaluator functionality.

This script shows how the HandEvaluator can analyze hand strength
for different trump suit scenarios.
"""

import sys
import os

# Add the euchre package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from euchre.models import Card, Suit, Rank, Player
from euchre.utils.hand_evaluator import HandEvaluator


def create_sample_hand() -> list[Card]:
    """Create a sample hand for demonstration."""
    return [
        Card(Rank.JACK, Suit.HEARTS),    # Right bower if hearts is trump
        Card(Rank.JACK, Suit.DIAMONDS),  # Left bower if hearts is trump
        Card(Rank.ACE, Suit.DIAMONDS),   # Strong diamond
        Card(Rank.KING, Suit.HEARTS),    # Strong heart
        Card(Rank.QUEEN, Suit.CLUBS),    # Off-suit card
    ]


def create_weak_hand() -> list[Card]:
    """Create a weak hand for comparison."""
    return [
        Card(Rank.NINE, Suit.CLUBS),
        Card(Rank.TEN, Suit.SPADES),
        Card(Rank.NINE, Suit.DIAMONDS),
        Card(Rank.TEN, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.SPADES),
    ]


def create_balanced_hand() -> list[Card]:
    """Create a balanced hand for comparison."""
    return [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.KING, Suit.DIAMONDS),
        Card(Rank.JACK, Suit.CLUBS),
        Card(Rank.QUEEN, Suit.SPADES),
        Card(Rank.TEN, Suit.HEARTS),
    ]


def demonstrate_hand_evaluation():
    """Demonstrate the hand evaluator with different hands."""
    evaluator = HandEvaluator()
    
    print("🎴 Euchre Hand Evaluator Demonstration\n")
    print("=" * 60)
    
    # Test the strong hand (Jack of hearts, Jack of diamonds, Ace of diamonds, etc.)
    print("\n1. STRONG HAND ANALYSIS")
    print("-" * 30)
    strong_hand = create_sample_hand()
    print(f"Hand: {[card.unicode_str() for card in strong_hand]}")
    
    strong_eval = evaluator.evaluate_hand_comprehensive(strong_hand)
    print(f"\nOverall strength: {strong_eval.overall_strength.value.title()}")
    print(f"Best trump suit: {strong_eval.best_trump_suit.name.title()}")
    print(f"Worst trump suit: {strong_eval.worst_trump_suit.name.title()}")
    
    print("\nTrump suit analysis:")
    for suit, analysis in strong_eval.trump_analyses.items():
        print(f"  {suit.name.title()}: {analysis.strength_category.value.title()} "
              f"({analysis.total_score:.1f} pts, {analysis.estimated_tricks:.1f} tricks)")
    
    print("\nRecommendations:")
    for rec in strong_eval.recommendations:
        print(f"  • {rec}")
    
    # Test the weak hand
    print("\n\n2. WEAK HAND ANALYSIS")
    print("-" * 30)
    weak_hand = create_weak_hand()
    print(f"Hand: {[card.unicode_str() for card in weak_hand]}")
    
    weak_eval = evaluator.evaluate_hand_comprehensive(weak_hand)
    print(f"\nOverall strength: {weak_eval.overall_strength.value.title()}")
    print(f"Best trump suit: {weak_eval.best_trump_suit.name.title()}")
    
    # Test the balanced hand
    print("\n\n3. BALANCED HAND ANALYSIS")
    print("-" * 30)
    balanced_hand = create_balanced_hand()
    print(f"Hand: {[card.unicode_str() for card in balanced_hand]}")
    
    balanced_eval = evaluator.evaluate_hand_comprehensive(balanced_hand)
    print(f"\nOverall strength: {balanced_eval.overall_strength.value.title()}")
    print(f"Best trump suit: {balanced_eval.best_trump_suit.name.title()}")
    
    # Show detailed analysis for the strong hand with hearts as trump
    print("\n\n4. DETAILED ANALYSIS: Strong Hand with HEARTS as Trump")
    print("-" * 60)
    hearts_analysis = strong_eval.trump_analyses[Suit.HEARTS]
    print(f"Trump cards: {[card.unicode_str() for card in hearts_analysis.trump_cards]}")
    print(f"Left bower cards: {[card.unicode_str() for card in hearts_analysis.left_bower_cards]}")
    print(f"Off-suit cards: {[card.unicode_str() for card in hearts_analysis.off_suit_cards]}")
    print(f"Total score: {hearts_analysis.total_score:.1f}")
    print(f"Estimated tricks: {hearts_analysis.estimated_tricks:.1f}")
    print(f"Confidence: {hearts_analysis.confidence:.1f}")
    
    # Show detailed analysis for the strong hand with diamonds as trump
    print("\n\n5. DETAILED ANALYSIS: Strong Hand with DIAMONDS as Trump")
    print("-" * 60)
    diamonds_analysis = strong_eval.trump_analyses[Suit.DIAMONDS]
    print(f"Trump cards: {[card.unicode_str() for card in diamonds_analysis.trump_cards]}")
    print(f"Left bower cards: {[card.unicode_str() for card in diamonds_analysis.left_bower_cards]}")
    print(f"Off-suit cards: {[card.unicode_str() for card in diamonds_analysis.off_suit_cards]}")
    print(f"Total score: {diamonds_analysis.total_score:.1f}")
    print(f"Estimated tricks: {diamonds_analysis.estimated_tricks:.1f}")
    print(f"Confidence: {diamonds_analysis.confidence:.1f}")
    
    # Show the complete summary
    print("\n\n6. COMPLETE HAND SUMMARY")
    print("-" * 60)
    print(evaluator.get_hand_summary(strong_eval))


def demonstrate_specific_scenarios():
    """Demonstrate specific trump suit scenarios."""
    evaluator = HandEvaluator()
    
    print("\n\n🎯 SPECIFIC SCENARIO ANALYSIS")
    print("=" * 60)
    
    # Scenario 1: All red hand with Jack of hearts and diamonds, plus Ace of diamonds
    print("\nScenario 1: All Red Hand (Hearts/Diamonds focus)")
    print("-" * 50)
    red_hand = [
        Card(Rank.JACK, Suit.HEARTS),     # Right bower if hearts trump
        Card(Rank.JACK, Suit.DIAMONDS),   # Left bower if hearts trump, Right if diamonds trump
        Card(Rank.ACE, Suit.DIAMONDS),    # Strong diamond
        Card(Rank.KING, Suit.HEARTS),     # Strong heart
        Card(Rank.QUEEN, Suit.HEARTS),    # Another strong heart
    ]
    
    print(f"Hand: {[card.unicode_str() for card in red_hand]}")
    
    # Evaluate for hearts as trump
    hearts_eval = evaluator.evaluate_hand_for_trump(red_hand, Suit.HEARTS)
    print(f"\nWith HEARTS as trump:")
    print(f"  Strength: {hearts_eval.strength_category.value.title()}")
    print(f"  Score: {hearts_eval.total_score:.1f}")
    print(f"  Estimated tricks: {hearts_eval.estimated_tricks:.1f}")
    print(f"  Trump cards: {len(hearts_eval.trump_cards)}")
    print(f"  Left bower cards: {len(hearts_eval.left_bower_cards)}")
    
    # Evaluate for diamonds as trump
    diamonds_eval = evaluator.evaluate_hand_for_trump(red_hand, Suit.DIAMONDS)
    print(f"\nWith DIAMONDS as trump:")
    print(f"  Strength: {diamonds_eval.strength_category.value.title()}")
    print(f"  Score: {diamonds_eval.total_score:.1f}")
    print(f"  Estimated tricks: {diamonds_eval.estimated_tricks:.1f}")
    print(f"  Trump cards: {len(diamonds_eval.trump_cards)}")
    print(f"  Left bower cards: {len(diamonds_eval.left_bower_cards)}")
    
    # Evaluate for clubs as trump (off-suit)
    clubs_eval = evaluator.evaluate_hand_for_trump(red_hand, Suit.CLUBS)
    print(f"\nWith CLUBS as trump:")
    print(f"  Strength: {clubs_eval.strength_category.value.title()}")
    print(f"  Score: {clubs_eval.total_score:.1f}")
    print(f"  Estimated tricks: {clubs_eval.estimated_tricks:.1f}")
    print(f"  Trump cards: {len(clubs_eval.trump_cards)}")
    print(f"  Left bower cards: {len(clubs_eval.left_bower_cards)}")
    
    print(f"\nAnalysis: This hand is {'VERY strong' if hearts_eval.strength_category.value == 'excellent' else 'strong'} "
          f"with hearts as trump, {'strong' if diamonds_eval.strength_category.value in ['excellent', 'strong'] else 'moderate'} "
          f"with diamonds as trump, and {'weak' if clubs_eval.strength_category.value in ['weak', 'poor'] else 'moderate'} "
          f"with clubs/spades as trump.")


if __name__ == "__main__":
    try:
        demonstrate_hand_evaluation()
        demonstrate_specific_scenarios()
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc() 