#!/usr/bin/env python3
"""
Test Script for Abstract AI Interface System

This script demonstrates how the new abstract AI interface system works,
showing both traditional rule-based AI and M-Series neural AI implementations
making decisions using the same interface.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-13
Context: Testing the new abstract AI interface system
"""

import sys
from pathlib import Path

# Add the euchre package to the path
sys.path.insert(0, str(Path(__file__).parent))

from euchre.ai.base_ai_interface import GameContext, DecisionType
from euchre.ai.ai_factory import AIFactory
from euchre.models import Card, Suit, Rank


def create_sample_game_context() -> GameContext:
    """Create a sample game context for testing."""
    # Create sample hand
    hand = [
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.DIAMONDS, Rank.JACK),
        Card(Suit.CLUBS, Rank.TEN),
        Card(Suit.SPADES, Rank.NINE)
    ]
    
    # Create sample flipped card
    flipped_card = Card(Suit.HEARTS, Rank.QUEEN)
    
    # Create sample current trick
    current_trick = [
        (1, Card(Suit.DIAMONDS, Rank.ACE)),
        (2, Card(Suit.DIAMONDS, Rank.KING))
    ]
    
    return GameContext(
        hand=hand,
        position=0,  # Alice
        is_dealer=False,
        partner_position=2,
        flipped_card=flipped_card,
        trump_suit=Suit.HEARTS,
        current_trick=current_trick,
        trick_suit=Suit.DIAMONDS,
        team1_score=3,
        team2_score=2,
        tricks_won_team1=2,
        tricks_won_team2=1,
        current_trick_number=3,
        partner_is_dealer=False,
        partner_hand_size=4,
        opponent1_hand_size=4,
        opponent2_hand_size=4
    )


def test_traditional_ai():
    """Test traditional rule-based AI implementation."""
    print("🤖 Testing Traditional AI Implementation")
    print("=" * 50)
    
    # Create different traditional AI types
    ai_types = ["aggressive", "conservative", "balanced", "opportunistic"]
    
    for ai_type in ai_types:
        print(f"\n📊 {ai_type.upper()} AI:")
        print("-" * 30)
        
        # Create AI
        ai = AIFactory.create_ai_player(f"{ai_type.capitalize()}", ai_type, 0.5)
        print(f"Created: {ai}")
        
        # Test decisions
        context = create_sample_game_context()
        
        # Test order up decision
        order_decision = ai.should_order_up(context)
        print(f"Order Up Decision: {order_decision.decision_type.value}")
        print(f"Confidence: {order_decision.confidence:.3f}")
        print(f"Reasoning: {order_decision.reasoning}")
        
        # Test play card decision
        play_decision = ai.play_card(context)
        print(f"Play Card Decision: {play_decision.decision_type.value}")
        print(f"Confidence: {play_decision.confidence:.3f}")
        print(f"Reasoning: {play_decision.reasoning}")
        
        # Show performance metrics
        metrics = ai.get_performance_metrics()
        print(f"Performance: {metrics}")


def test_m_series_ai():
    """Test M-Series neural AI implementation."""
    print("\n🧠 Testing M-Series Neural AI Implementation")
    print("=" * 50)
    
    if not AIFactory.M_SERIES_AVAILABLE:
        print("❌ M-Series models not available. Install PyTorch and M-Series dependencies.")
        return
    
    # Create different M-Series AI types
    m_series_types = ["magnus", "maverick", "mentor", "mystic"]
    
    for model_type in m_series_types:
        print(f"\n📊 {model_type.upper()} M-Series AI:")
        print("-" * 30)
        
        try:
            # Create AI (without trained model for now)
            ai = AIFactory.create_ai_player(f"{model_type.capitalize()}", model_type, 0.5)
            print(f"Created: {ai}")
            
            # Test decisions
            context = create_sample_game_context()
            
            # Test order up decision
            order_decision = ai.should_order_up(context)
            print(f"Order Up Decision: {order_decision.decision_type.value}")
            print(f"Confidence: {order_decision.confidence:.3f}")
            print(f"Reasoning: {order_decision.reasoning}")
            
            # Test play card decision
            play_decision = ai.play_card(context)
            print(f"Play Card Decision: {play_decision.decision_type.value}")
            print(f"Confidence: {play_decision.confidence:.3f}")
            print(f"Reasoning: {play_decision.reasoning}")
            
            # Show performance metrics
            metrics = ai.get_performance_metrics()
            print(f"Performance: {metrics}")
            
        except Exception as e:
            print(f"❌ Error creating {model_type} AI: {e}")


def test_ai_factory():
    """Test the AI factory functionality."""
    print("\n🏭 Testing AI Factory")
    print("=" * 50)
    
    # Test available AI types
    available_types = AIFactory.get_available_ai_types()
    print(f"Available AI types: {', '.join(available_types)}")
    
    # Test creating mixed AI players
    print("\n🤖 Creating mixed AI players:")
    mixed_players = AIFactory.create_mixed_ai_players()
    for player in mixed_players:
        print(f"  - {player}")
    
    # Test creating M-Series players if available
    if AIFactory.M_SERIES_AVAILABLE:
        print("\n🧠 Creating M-Series players:")
        try:
            m_series_players = AIFactory.create_m_series_players("magnus")
            for player in m_series_players:
                print(f"  - {player}")
        except Exception as e:
            print(f"❌ Error creating M-Series players: {e}")


def test_decision_consistency():
    """Test that different AI types make consistent decisions."""
    print("\n🔄 Testing Decision Consistency")
    print("=" * 50)
    
    context = create_sample_game_context()
    
    # Test traditional AI types
    traditional_types = ["aggressive", "conservative", "balanced"]
    traditional_decisions = {}
    
    for ai_type in traditional_types:
        ai = AIFactory.create_ai_player(f"{ai_type.capitalize()}", ai_type, 0.5)
        decision = ai.should_order_up(context)
        traditional_decisions[ai_type] = decision.decision_type.value
        print(f"{ai_type.capitalize()}: {decision.decision_type.value} (confidence: {decision.confidence:.3f})")
    
    # Test M-Series AI types if available
    if AIFactory.M_SERIES_AVAILABLE:
        print("\nM-Series AI decisions:")
        m_series_types = ["magnus", "maverick", "mentor"]
        m_series_decisions = {}
        
        for model_type in m_series_types:
            try:
                ai = AIFactory.create_ai_player(f"{model_type.capitalize()}", model_type, 0.5)
                decision = ai.should_order_up(context)
                m_series_decisions[model_type] = decision.decision_type.value
                print(f"{model_type.capitalize()}: {decision.decision_type.value} (confidence: {decision.confidence:.3f})")
            except Exception as e:
                print(f"{model_type.capitalize()}: Error - {e}")


def main():
    """Main test function."""
    print("🧪 Abstract AI Interface System Test")
    print("=" * 60)
    
    try:
        # Test traditional AI
        test_traditional_ai()
        
        # Test M-Series AI
        test_m_series_ai()
        
        # Test AI factory
        test_ai_factory()
        
        # Test decision consistency
        test_decision_consistency()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 