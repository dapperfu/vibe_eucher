#!/usr/bin/env python3
"""Quick benchmark to compare HeuristicPlayer performance."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eucher.cards import Card, Rank, Suit
from eucher.players.computer.heuristic import HeuristicPlayer
from eucher.players import Player


def create_random_hand() -> list[Card]:
    """Create a random hand for testing."""
    all_cards = []
    for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
        for rank in [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]:
            all_cards.append(Card(suit, rank))
    
    random.shuffle(all_cards)
    return all_cards[:5]


def test_decision_speed() -> None:
    """Test that decisions are made quickly."""
    print("Testing decision speed...")
    player = HeuristicPlayer()
    test_player = Player("Test", 0, player)
    
    import time
    
    # Test order up decisions
    start = time.time()
    for _ in range(100):
        test_player.hand = create_random_hand()
        turned_card = Card(Suit.HEARTS, Rank.TEN)
        player.decide_order_up(test_player, turned_card, dealer_id=2, trump_suit=None)
    elapsed = time.time() - start
    print(f"  100 order_up decisions: {elapsed*1000:.2f}ms ({elapsed/100*1000:.3f}ms per decision)")
    
    # Test call trump decisions
    start = time.time()
    for _ in range(100):
        test_player.hand = create_random_hand()
        turned_card = Card(Suit.HEARTS, Rank.TEN)
        player.decide_call_trump(test_player, turned_card, trump_suit=None, must_choose=False)
    elapsed = time.time() - start
    print(f"  100 call_trump decisions: {elapsed*1000:.2f}ms ({elapsed/100*1000:.3f}ms per decision)")
    
    # Test card play decisions
    start = time.time()
    for _ in range(100):
        test_player.hand = create_random_hand()
        trump_suit = Suit.HEARTS
        valid_cards = test_player.hand.copy()
        player._decide_lead(test_player, valid_cards, trump_suit)
    elapsed = time.time() - start
    print(f"  100 lead decisions: {elapsed*1000:.2f}ms ({elapsed/100*1000:.3f}ms per decision)")


def test_decision_consistency() -> None:
    """Test that decisions are consistent for the same inputs."""
    print("\nTesting decision consistency...")
    player = HeuristicPlayer()
    test_player = Player("Test", 0, player)
    
    # Create a fixed hand
    test_player.hand = [
        Card(Suit.HEARTS, Rank.JACK),
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.DIAMONDS, Rank.ACE),
        Card(Suit.CLUBS, Rank.KING),
    ]
    
    turned_card = Card(Suit.HEARTS, Rank.TEN)
    
    # Run same decision multiple times
    decisions = []
    for _ in range(10):
        decision = player.decide_order_up(test_player, turned_card, dealer_id=2, trump_suit=None)
        decisions.append(decision)
    
    # All decisions should be the same
    assert all(d == decisions[0] for d in decisions), "Decisions should be consistent"
    print(f"  ✓ Consistent decisions: {decisions[0]}")


def test_hand_strength_range() -> None:
    """Test that hand strength values are reasonable."""
    print("\nTesting hand strength range...")
    player = HeuristicPlayer()
    
    # Test various hands
    test_hands = [
        # Very weak hand
        [
            Card(Suit.HEARTS, Rank.NINE),
            Card(Suit.DIAMONDS, Rank.NINE),
            Card(Suit.CLUBS, Rank.TEN),
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.HEARTS, Rank.TEN),
        ],
        # Medium hand
        [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.DIAMONDS, Rank.ACE),
            Card(Suit.CLUBS, Rank.NINE),
            Card(Suit.SPADES, Rank.NINE),
        ],
        # Strong hand
        [
            Card(Suit.HEARTS, Rank.JACK),  # Right bower
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.HEARTS, Rank.QUEEN),
            Card(Suit.DIAMONDS, Rank.ACE),
        ],
    ]
    
    for i, hand in enumerate(test_hands):
        strength = player._evaluate_hand_strength(hand, Suit.HEARTS)
        print(f"  Hand {i+1} strength: {strength:.1f}")
        assert strength > 0, "Hand strength should be positive"
    
    # Strong hand should be stronger than weak hand
    weak_strength = player._evaluate_hand_strength(test_hands[0], Suit.HEARTS)
    strong_strength = player._evaluate_hand_strength(test_hands[2], Suit.HEARTS)
    assert strong_strength > weak_strength, "Strong hand should have higher strength"
    print("  ✓ Hand strength values are reasonable")


def main() -> None:
    """Run benchmark tests."""
    print("=" * 80)
    print("HeuristicPlayer Quick Benchmark")
    print("=" * 80)
    
    random.seed(42)  # For reproducibility
    
    test_decision_speed()
    test_decision_consistency()
    test_hand_strength_range()
    
    print("\n" + "=" * 80)
    print("Benchmark tests completed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()

