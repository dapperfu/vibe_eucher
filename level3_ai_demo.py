#!/usr/bin/env python3
"""
Level 3 AI Demo Script

This script demonstrates how to use the Level 3 AI implementation
in various scenarios, showing its capabilities and integration.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-13
Context: Demonstrating Level 3 AI capabilities
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

def demo_level3_ai_creation():
    """Demonstrate creating different types of Level 3 AI."""
    print("🎯 Demo: Creating Level 3 AI Players")
    print("=" * 50)
    
    try:
        from euchre.ai.ai_factory import AIFactory
        
        # Create different Level 3 AI types
        ai_types = [
            "level3_strategic",
            "level3_aggressive", 
            "level3_balanced",
            "level3_conservative",
            "level3_opportunistic"
        ]
        
        for ai_type in ai_types:
            ai = AIFactory.create_ai_player(f"Demo_{ai_type}", ai_type, 0.5)
            print(f"✅ Created {ai_type}: {ai}")
            print(f"   Risk Profile: {ai.risk_profile}")
            print(f"   Model Type: {ai.model_type}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create Level 3 AI: {e}")
        return False

def demo_level3_ai_decisions():
    """Demonstrate Level 3 AI decision making."""
    print("🎯 Demo: Level 3 AI Decision Making")
    print("=" * 50)
    
    try:
        from euchre.ai.ai_factory import AIFactory
        from euchre.ai.base_ai_interface import GameContext
        from euchre.models import Card, Suit, Rank
        
        # Create a Level 3 AI
        ai = AIFactory.create_ai_player("DecisionDemo", "level3_strategic", 0.5)
        
        # Create a mock game context
        mock_hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.DIAMONDS, Rank.QUEEN),
            Card(Suit.CLUBS, Rank.JACK),
            Card(Suit.SPADES, Rank.TEN)
        ]
        
        mock_context = GameContext(
            hand=mock_hand,
            position=0,
            is_dealer=False,
            partner_position=2,
            flipped_card=Card(Suit.HEARTS, Rank.NINE),
            trump_suit=None,
            current_trick=[],
            trick_suit=None,
            team1_score=0,
            team2_score=0,
            tricks_won_team1=0,
            tricks_won_team2=0,
            current_trick_number=1,
            partner_is_dealer=False,
            partner_hand_size=5,
            opponent1_hand_size=5,
            opponent2_hand_size=5
        )
        
        print(f"🤖 AI: {ai.name}")
        print(f"🎴 Hand: {[str(card) for card in mock_hand]}")
        print(f"🃏 Flipped Card: {mock_context.flipped_card}")
        print()
        
        # Test different decisions
        print("📊 Decision Analysis:")
        
        # Trump decision
        trump_decision = ai.should_order_up(mock_context)
        print(f"🎯 Trump Decision: {trump_decision.decision_type.value}")
        print(f"   Confidence: {trump_decision.confidence:.3f}")
        print(f"   Reasoning: {trump_decision.reasoning}")
        print()
        
        # Card playing
        card_decision = ai.play_card(mock_context)
        print(f"🃏 Card Decision: {card_decision.decision_type.value}")
        print(f"   Confidence: {card_decision.confidence:.3f}")
        print(f"   Reasoning: {card_decision.reasoning}")
        print()
        
        # Suit selection
        suit_decision = ai.select_trump_suit(mock_context)
        print(f"♠️ Suit Decision: {suit_decision.decision_type.value}")
        print(f"   Confidence: {suit_decision.confidence:.3f}")
        print(f"   Reasoning: {suit_decision.reasoning}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to demonstrate Level 3 AI decisions: {e}")
        return False

def main():
    """Run all Level 3 AI demos."""
    print("🚀 Level 3 AI Demo Suite")
    print("=" * 60)
    print()
    
    demos = [
        demo_level3_ai_creation,
        demo_level3_ai_decisions
    ]
    
    passed = 0
    total = len(demos)
    
    for demo in demos:
        try:
            if demo():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Demo {demo.__name__} crashed: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Demo Results: {passed}/{total} demos completed successfully")
    
    if passed == total:
        print("🎉 All Level 3 AI demos completed successfully!")
        print("\n✨ Level 3 AI is now fully featured and ready for use!")
        return 0
    else:
        print("💥 Some Level 3 AI demos failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 