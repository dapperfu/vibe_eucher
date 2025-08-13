#!/usr/bin/env python3
"""
Simple test for the hand evaluator functionality.
"""

import sys
import os

# Add the euchre package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from euchre.models import Card, Suit, Rank
from euchre.utils.hand_evaluator import HandEvaluator


def test_basic_functionality():
    """Test basic hand evaluator functionality."""
    print("Testing basic hand evaluator functionality...")
    
    evaluator = HandEvaluator()
    
    # Create a simple hand
    hand = [
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.ACE, Suit.DIAMONDS),
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.QUEEN, Suit.SPADES),
        Card(Rank.TEN, Suit.HEARTS),
    ]
    
    # Test comprehensive evaluation
    eval_result = evaluator.evaluate_hand_comprehensive(hand)
    
    assert eval_result.hand == hand
    assert len(eval_result.trump_analyses) == 4  # 4 suits
    assert eval_result.best_trump_suit in Suit
    assert eval_result.worst_trump_suit in Suit
    
    print("✓ Basic functionality test passed")
    return True


def test_trump_analysis():
    """Test trump-specific analysis."""
    print("Testing trump-specific analysis...")
    
    evaluator = HandEvaluator()
    
    # Create a hand with strong hearts
    hand = [
        Card(Rank.JACK, Suit.HEARTS),     # Right bower
        Card(Rank.JACK, Suit.DIAMONDS),   # Left bower if hearts is trump
        Card(Rank.ACE, Suit.HEARTS),      # Strong heart
        Card(Rank.KING, Suit.CLUBS),      # Off-suit
        Card(Rank.QUEEN, Suit.SPADES),    # Off-suit
    ]
    
    # Test hearts as trump
    hearts_analysis = evaluator.evaluate_hand_for_trump(hand, Suit.HEARTS)
    
    assert hearts_analysis.trump_suit == Suit.HEARTS
    assert len(hearts_analysis.trump_cards) == 2  # J♥, A♥
    assert len(hearts_analysis.left_bower_cards) == 1  # J♦
    assert hearts_analysis.strength_category.value in ['excellent', 'strong']
    
    print("✓ Trump analysis test passed")
    return True


def test_hand_categories():
    """Test hand strength categorization."""
    print("Testing hand strength categorization...")
    
    evaluator = HandEvaluator()
    
    # Test strong hand
    strong_hand = [
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.JACK, Suit.DIAMONDS),
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.ACE, Suit.DIAMONDS),
        Card(Rank.KING, Suit.HEARTS),
    ]
    
    strong_eval = evaluator.evaluate_hand_for_trump(strong_hand, Suit.HEARTS)
    assert strong_eval.strength_category.value in ['excellent', 'strong']
    
    # Test weak hand
    weak_hand = [
        Card(Rank.NINE, Suit.CLUBS),
        Card(Rank.NINE, Suit.SPADES),
        Card(Rank.TEN, Suit.DIAMONDS),
        Card(Rank.TEN, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.CLUBS),
    ]
    
    weak_eval = evaluator.evaluate_hand_for_trump(weak_hand, Suit.HEARTS)
    assert weak_eval.strength_category.value in ['weak', 'poor']
    
    print("✓ Hand categories test passed")
    return True


def test_left_bower_logic():
    """Test left bower identification logic."""
    print("Testing left bower logic...")
    
    evaluator = HandEvaluator()
    
    # Test that left bower is correctly identified
    hand = [
        Card(Rank.JACK, Suit.HEARTS),     # Right bower if hearts is trump
        Card(Rank.JACK, Suit.DIAMONDS),   # Left bower if hearts is trump
    ]
    
    # With hearts as trump
    hearts_analysis = evaluator.evaluate_hand_for_trump(hand, Suit.HEARTS)
    assert len(hearts_analysis.trump_cards) == 1  # J♥
    assert len(hearts_analysis.left_bower_cards) == 1  # J♦
    
    # With diamonds as trump
    diamonds_analysis = evaluator.evaluate_hand_for_trump(hand, Suit.DIAMONDS)
    assert len(diamonds_analysis.trump_cards) == 1  # J♦
    assert len(diamonds_analysis.left_bower_cards) == 1  # J♥
    
    print("✓ Left bower logic test passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("🧪 Running Hand Evaluator Tests\n")
    
    tests = [
        test_basic_functionality,
        test_trump_analysis,
        test_hand_categories,
        test_left_bower_logic,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 