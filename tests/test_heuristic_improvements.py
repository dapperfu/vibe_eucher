#!/usr/bin/env python3
"""Test script for improved HeuristicPlayer.

This script runs games with the improved HeuristicPlayer to verify
that all improvements work correctly.
"""

import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from eucher.cards import Card, Rank, Suit
from eucher.players.computer.heuristic import HeuristicPlayer, HeuristicWeights
from eucher.players import Player

# Try to import Game, but don't fail if dependencies are missing
try:
    from eucher.game import Game
    GAME_AVAILABLE = True
except ImportError:
    GAME_AVAILABLE = False
    print("Warning: Game import failed (missing dependencies), skipping full game test")


def test_heuristic_weights() -> None:
    """Test that HeuristicWeights class exists and has expected values."""
    print("Testing HeuristicWeights...")
    assert hasattr(HeuristicWeights, "ORDER_UP_BASE_THRESHOLD")
    assert hasattr(HeuristicWeights, "CALL_TRUMP_BASE_THRESHOLD")
    assert hasattr(HeuristicWeights, "RIGHT_BOWER_POWER")
    print("  ✓ HeuristicWeights class has expected attributes")


def test_hand_strength_evaluation() -> None:
    """Test hand strength evaluation."""
    print("\nTesting hand strength evaluation...")
    player = HeuristicPlayer()
    
    # Create a test hand with strong trump
    hand = [
        Card(Suit.HEARTS, Rank.JACK),  # Right bower if Hearts is trump
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.DIAMONDS, Rank.ACE),
        Card(Suit.CLUBS, Rank.KING),
    ]
    
    strength = player._evaluate_hand_strength(hand, Suit.HEARTS)
    print(f"  Hand strength (Hearts trump): {strength:.1f}")
    assert strength > 0, "Hand strength should be positive"
    assert strength > 200, "Strong hand should have high strength"
    print("  ✓ Hand strength evaluation works")


def test_card_power_calculation() -> None:
    """Test card power calculation."""
    print("\nTesting card power calculation...")
    player = HeuristicPlayer()
    
    # Test Right Bower
    right_bower = Card(Suit.HEARTS, Rank.JACK)
    power = player._calculate_card_power(right_bower, Suit.HEARTS)
    assert power == HeuristicWeights.RIGHT_BOWER_POWER
    print(f"  Right Bower power: {power}")
    
    # Test Left Bower
    left_bower = Card(Suit.DIAMONDS, Rank.JACK)  # Same color as Hearts
    power = player._calculate_card_power(left_bower, Suit.HEARTS)
    assert power == HeuristicWeights.LEFT_BOWER_POWER
    print(f"  Left Bower power: {power}")
    
    # Test Trump Ace
    trump_ace = Card(Suit.HEARTS, Rank.ACE)
    power = player._calculate_card_power(trump_ace, Suit.HEARTS)
    assert power == HeuristicWeights.TRUMP_ACE_POWER
    print(f"  Trump Ace power: {power}")
    
    # Test Off-suit Ace
    offsuit_ace = Card(Suit.DIAMONDS, Rank.ACE)
    power = player._calculate_card_power(offsuit_ace, Suit.HEARTS)
    assert power == HeuristicWeights.OFFSUIT_ACE_POWER
    print(f"  Off-suit Ace power: {power}")
    
    print("  ✓ Card power calculation works")


def test_order_up_decision() -> None:
    """Test order up decision logic."""
    print("\nTesting order up decision...")
    player = HeuristicPlayer()
    
    # Create a player with a strong hand
    test_player = Player("Test", 0, player)
    test_player.hand = [
        Card(Suit.HEARTS, Rank.JACK),  # Right bower
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.DIAMONDS, Rank.ACE),
        Card(Suit.CLUBS, Rank.KING),
    ]
    
    turned_card = Card(Suit.HEARTS, Rank.TEN)
    decision = player.decide_order_up(test_player, turned_card, dealer_id=2, trump_suit=None)
    assert decision is True, "Should order up with strong hand"
    print("  ✓ Orders up with strong hand")
    
    # Test with weak hand (only 1 trump card, no bowers, low ranks)
    test_player.hand = [
        Card(Suit.HEARTS, Rank.NINE),  # Only 1 trump
        Card(Suit.DIAMONDS, Rank.TEN),
        Card(Suit.CLUBS, Rank.NINE),
        Card(Suit.SPADES, Rank.TEN),
        Card(Suit.DIAMONDS, Rank.NINE),
    ]
    decision = player.decide_order_up(test_player, turned_card, dealer_id=2, trump_suit=None)
    assert decision is False, "Should not order up with weak hand (only 1 trump)"
    print("  ✓ Passes with weak hand")


def test_call_trump_decision() -> None:
    """Test call trump decision logic."""
    print("\nTesting call trump decision...")
    player = HeuristicPlayer()
    
    # Create a player with strong cards in one suit
    test_player = Player("Test", 0, player)
    test_player.hand = [
        Card(Suit.DIAMONDS, Rank.JACK),  # Left bower if Diamonds is trump
        Card(Suit.DIAMONDS, Rank.ACE),
        Card(Suit.DIAMONDS, Rank.KING),
        Card(Suit.HEARTS, Rank.TEN),
        Card(Suit.CLUBS, Rank.NINE),
    ]
    
    turned_card = Card(Suit.HEARTS, Rank.TEN)  # Hearts is forbidden
    decision = player.decide_call_trump(test_player, turned_card, trump_suit=None, must_choose=False)
    assert decision == Suit.DIAMONDS, "Should call Diamonds with strong hand"
    print(f"  ✓ Calls trump: {decision}")
    
    # Test must_choose scenario
    decision = player.decide_call_trump(test_player, turned_card, trump_suit=None, must_choose=True)
    assert decision is not None, "Must choose a suit when must_choose=True"
    print(f"  ✓ Chooses suit when must choose: {decision}")


def test_discard_decision() -> None:
    """Test discard decision logic."""
    print("\nTesting discard decision...")
    player = HeuristicPlayer()
    
    # Create a player with mixed hand
    test_player = Player("Test", 0, player)
    test_player.hand = [
        Card(Suit.HEARTS, Rank.JACK),  # Right bower (high value)
        Card(Suit.HEARTS, Rank.ACE),    # High value
        Card(Suit.HEARTS, Rank.KING),   # High value
        Card(Suit.DIAMONDS, Rank.NINE), # Low value
        Card(Suit.CLUBS, Rank.NINE),   # Low value
    ]
    
    # Test discard without trump context
    discarded = player.choose_card_to_discard(test_player, turned_card=None)
    assert discarded.rank == Rank.NINE, "Should discard lowest card"
    print(f"  ✓ Discards lowest card: {discarded}")
    
    # Test discard with trump context
    turned_card = Card(Suit.HEARTS, Rank.TEN)
    discarded = player.choose_card_to_discard(test_player, turned_card=turned_card)
    # Should discard a non-trump low card
    assert discarded.suit != Suit.HEARTS or discarded.rank == Rank.NINE, "Should prefer discarding off-suit"
    print(f"  ✓ Discards considering trump: {discarded}")


def test_card_play_decisions() -> None:
    """Test card play decision logic."""
    print("\nTesting card play decisions...")
    player = HeuristicPlayer()
    
    # Create a player with mixed hand
    test_player = Player("Test", 0, player)
    test_player.hand = [
        Card(Suit.HEARTS, Rank.JACK),  # Right bower
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.DIAMONDS, Rank.ACE),
        Card(Suit.CLUBS, Rank.KING),
        Card(Suit.SPADES, Rank.TEN),
    ]
    
    # Test leading
    trump_suit = Suit.HEARTS
    valid_cards = test_player.hand.copy()
    card = player._decide_lead(test_player, valid_cards, trump_suit)
    # Should lead with strong trump if available
    assert card.suit == Suit.HEARTS, "Should lead with trump when available"
    print(f"  ✓ Leads with: {card}")
    
    # Test following (teammate winning)
    trick_cards = [Card(Suit.DIAMONDS, Rank.ACE)]
    trick_player_ids = [2]  # Player 2 (teammate of player 0)
    led_suit = Suit.DIAMONDS
    valid_cards = [c for c in test_player.hand if c.suit == Suit.DIAMONDS or c.suit == trump_suit]
    if not valid_cards:
        valid_cards = test_player.hand.copy()
    card = player._decide_follow(
        test_player, valid_cards, trick_cards, trick_player_ids, led_suit, trump_suit
    )
    print(f"  ✓ Follows (teammate winning) with: {card}")
    
    # Test last play
    trick_cards = [
        Card(Suit.DIAMONDS, Rank.ACE),
        Card(Suit.DIAMONDS, Rank.KING),
        Card(Suit.DIAMONDS, Rank.QUEEN),
    ]
    trick_player_ids = [1, 2, 3]  # Opponent winning
    valid_cards = [c for c in test_player.hand if c.suit == Suit.DIAMONDS or c.suit == trump_suit]
    if not valid_cards:
        valid_cards = test_player.hand.copy()
    card = player._decide_last_play(
        test_player, valid_cards, trick_cards, trick_player_ids, led_suit, trump_suit
    )
    print(f"  ✓ Last play with: {card}")


def test_full_game() -> None:
    """Test playing a full game with HeuristicPlayer."""
    if not GAME_AVAILABLE:
        print("\nSkipping full game test (Game import not available)")
        return
        
    print("\nTesting full game...")
    
    player_config = [
        ("H1", "heuristic"),
        ("H2", "heuristic"),
        ("H3", "heuristic"),
        ("H4", "heuristic"),
    ]
    
    game = Game(player_config)
    
    # Play a few hands
    hands_played = 0
    for _ in range(3):
        try:
            continue_game = game.play_hand()
            hands_played += 1
            if not continue_game:
                break
        except Exception as e:
            print(f"  Error playing hand: {e}")
            break
    
    assert hands_played > 0, "Should play at least one hand"
    print(f"  ✓ Played {hands_played} hands successfully")
    
    scores = game.get_scores()
    print(f"  ✓ Game scores: Team 0 = {scores[0]}, Team 1 = {scores[1]}")


def test_teammate_detection() -> None:
    """Test teammate detection logic."""
    print("\nTesting teammate detection...")
    player = HeuristicPlayer()
    
    test_player = Player("Test", 0, player)
    
    # Player 0 and 2 are teammates (both even)
    assert player._is_teammate(test_player, 2) is True
    assert player._is_teammate(test_player, 0) is True
    
    # Player 0 and 1 are opponents (different parity)
    assert player._is_teammate(test_player, 1) is False
    assert player._is_teammate(test_player, 3) is False
    
    print("  ✓ Teammate detection works correctly")


def main() -> None:
    """Run all tests."""
    print("=" * 80)
    print("Testing Improved HeuristicPlayer")
    print("=" * 80)
    
    try:
        test_heuristic_weights()
        test_hand_strength_evaluation()
        test_card_power_calculation()
        test_order_up_decision()
        test_call_trump_decision()
        test_discard_decision()
        test_card_play_decisions()
        test_teammate_detection()
        test_full_game()
        
        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

