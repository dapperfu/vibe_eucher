#!/usr/bin/env python3
"""
Test script for Level 3 AI implementation

This script tests the Level 3 AI to ensure it:
1. Can be created through the AI factory
2. Implements the BaseAIInterface correctly
3. Can make decisions (even with untrained models)
4. Integrates properly with the game system

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-13
Context: Testing Level 3 AI implementation
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

def test_level3_ai_creation():
    """Test that Level 3 AI can be created through the factory."""
    print("🧪 Testing Level 3 AI creation...")
    
    try:
        from euchre.ai.ai_factory import AIFactory
        
        # Test creating Level 3 AI
        ai = AIFactory.create_ai_player("TestAI", "level3_strategic", 0.5)
        print(f"✅ Successfully created Level 3 AI: {ai}")
        print(f"   Type: {type(ai).__name__}")
        print(f"   Name: {ai.name}")
        print(f"   Risk Profile: {ai.risk_profile}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create Level 3 AI: {e}")
        return False

def test_level3_ai_interface():
    """Test that Level 3 AI implements the BaseAIInterface correctly."""
    print("\n🧪 Testing Level 3 AI interface implementation...")
    
    try:
        from euchre.ai.ai_factory import AIFactory
        from euchre.ai.base_ai_interface import BaseAIInterface, GameContext, DecisionType
        
        # Create Level 3 AI
        ai = AIFactory.create_ai_player("TestAI", "level3_strategic", 0.5)
        
        # Check if it implements the interface
        if not isinstance(ai, BaseAIInterface):
            print("❌ Level 3 AI does not implement BaseAIInterface")
            return False
        
        print("✅ Level 3 AI correctly implements BaseAIInterface")
        
        # Test that it has all required methods
        required_methods = [
            'should_order_up',
            'should_call_trump', 
            'select_trump_suit',
            'play_card',
            'discard_card'
        ]
        
        for method_name in required_methods:
            if not hasattr(ai, method_name):
                print(f"❌ Missing required method: {method_name}")
                return False
        
        print("✅ All required methods are present")
        return True
        
    except Exception as e:
        print(f"❌ Failed to test Level 3 AI interface: {e}")
        return False

def test_level3_ai_decisions():
    """Test that Level 3 AI can make decisions."""
    print("\n🧪 Testing Level 3 AI decision making...")
    
    try:
        from euchre.ai.ai_factory import AIFactory
        from euchre.ai.base_ai_interface import GameContext, DecisionType
        from euchre.models import Card, Suit, Rank
        
        # Create Level 3 AI
        ai = AIFactory.create_ai_player("TestAI", "level3_strategic", 0.5)
        
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
        
        # Test trump decision
        print("   Testing trump decision...")
        trump_decision = ai.should_order_up(mock_context)
        print(f"   Trump decision: {trump_decision.decision_type.value}")
        print(f"   Confidence: {trump_decision.confidence:.3f}")
        print(f"   Reasoning: {trump_decision.reasoning}")
        
        # Test card playing
        print("   Testing card playing...")
        card_decision = ai.play_card(mock_context)
        print(f"   Card decision: {card_decision.decision_type.value}")
        print(f"   Confidence: {card_decision.confidence:.3f}")
        print(f"   Reasoning: {card_decision.reasoning}")
        
        print("✅ Level 3 AI can make decisions")
        return True
        
    except Exception as e:
        print(f"❌ Failed to test Level 3 AI decisions: {e}")
        return False

def test_level3_ai_factory_integration():
    """Test that Level 3 AI integrates properly with the factory."""
    print("\n🧪 Testing Level 3 AI factory integration...")
    
    try:
        from euchre.ai.ai_factory import AIFactory
        
        # Test available AI types
        available_types = AIFactory.get_available_ai_types()
        print(f"Available AI types: {available_types}")
        
        # Check if Level 3 types are included
        level3_types = [t for t in available_types if t.startswith("level3_")]
        print(f"Level 3 types found: {level3_types}")
        
        if not level3_types:
            print("❌ No Level 3 types found in available AI types")
            return False
        
        # Test Level 3 type validation
        for level3_type in level3_types:
            if not AIFactory.validate_ai_type(level3_type):
                print(f"❌ Level 3 type {level3_type} failed validation")
                return False
        
        print("✅ All Level 3 types pass validation")
        
        # Test Level 3 type checking
        for level3_type in level3_types:
            if not AIFactory.is_level3_type(level3_type):
                print(f"❌ Level 3 type {level3_type} not recognized as Level 3")
                return False
        
        print("✅ All Level 3 types are correctly identified")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test Level 3 AI factory integration: {e}")
        return False

def main():
    """Run all Level 3 AI tests."""
    print("🚀 Starting Level 3 AI tests...\n")
    
    tests = [
        test_level3_ai_creation,
        test_level3_ai_interface,
        test_level3_ai_decisions,
        test_level3_ai_factory_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Level 3 AI tests passed!")
        return 0
    else:
        print("💥 Some Level 3 AI tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 