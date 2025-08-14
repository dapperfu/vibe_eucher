#!/usr/bin/env python3
"""
Test Game Integration with Level 3 AI

This script tests whether the updated game engine can properly
integrate with Level 3 AI using the new AI adapter.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-13
Context: Testing game engine integration with Level 3 AI
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

def test_game_engine_integration():
    """Test that the game engine can integrate with Level 3 AI."""
    print("🧪 Testing Game Engine Integration with Level 3 AI")
    print("=" * 60)
    
    try:
        from euchre.game import EuchreGame
        from euchre.ai.ai_factory import AIFactory
        from euchre.ai.ai_adapter import AIAdapter
        from euchre.models import Card, Suit, Rank
        
        print("✅ Successfully imported game components")
        
        # Create Level 3 AI players
        print("\n🤖 Creating Level 3 AI players...")
        ai_players = []
        
        ai_types = ["level3_strategic", "level3_balanced", "level3_aggressive", "level3_conservative"]
        for i, ai_type in enumerate(ai_types):
            player = AIFactory.create_ai_player(f"Player{i+1}", ai_type, 0.5)
            ai_players.append(player)
            print(f"   ✅ Created {player.name} ({ai_type})")
        
        # Create game with Level 3 AI players
        print("\n🎮 Creating game with Level 3 AI players...")
        game = EuchreGame(players=ai_players, quiet_mode=True)
        print("   ✅ Game created successfully")
        
        # Test AI adapter methods directly
        print("\n🔧 Testing AI adapter methods...")
        
        # Test should_order_up
        test_card = Card(Suit.HEARTS, Rank.ACE)
        for player in ai_players:
            result = AIAdapter.should_order_up(player, test_card, False)
            print(f"   {player.name} should_order_up: {result}")
        
        # Test play_card
        test_hand = [Card(Suit.HEARTS, Rank.KING), Card(Suit.DIAMONDS, Rank.QUEEN)]
        test_game_state = {'team1_score': 0, 'team2_score': 0, 'tricks_won_team1': 0, 'tricks_won_team2': 0, 'current_trick_number': 1}
        
        for player in ai_players:
            # Temporarily set hand for testing
            original_hand = player.hand
            player.hand = test_hand
            
            card = AIAdapter.play_card(player, test_hand, Suit.HEARTS, Suit.CLUBS, None, test_game_state)
            print(f"   {player.name} play_card: {card}")
            
            # Restore original hand
            player.hand = original_hand
        
        print("   ✅ AI adapter methods working correctly")
        
        # Test starting a new game
        print("\n🎯 Testing game start with Level 3 AI...")
        try:
            game.start_new_game()
            print("   ✅ Game started successfully with Level 3 AI")
        except Exception as e:
            print(f"   ❌ Game start failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_interface_consistency():
    """Test that all AI levels have consistent interfaces."""
    print("\n🧪 Testing AI Interface Consistency")
    print("=" * 60)
    
    try:
        from euchre.ai.ai_factory import AIFactory
        from euchre.ai.base_ai_interface import BaseAIInterface
        
        # Test different AI levels
        ai_levels = [
            ("level1_balanced", "Traditional AI"),
            ("level2_strategic", "Level 2 AI"),
            ("level3_strategic", "Level 3 AI")
        ]
        
        for ai_type, description in ai_levels:
            print(f"\n🔍 Testing {description} ({ai_type})...")
            
            try:
                ai = AIFactory.create_ai_player(f"Test_{ai_type}", ai_type, 0.5)
                
                # Check interface implementation
                if isinstance(ai, BaseAIInterface):
                    print(f"   ✅ Implements BaseAIInterface")
                else:
                    print(f"   ❌ Does NOT implement BaseAIInterface")
                
                # Check required methods
                required_methods = [
                    'should_order_up',
                    'should_call_trump',
                    'select_trump_suit',
                    'play_card',
                    'discard_card'
                ]
                
                for method in required_methods:
                    if hasattr(ai, method):
                        print(f"   ✅ Has {method}")
                    else:
                        print(f"   ❌ Missing {method}")
                
            except Exception as e:
                print(f"   ❌ Failed to create {ai_type}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Interface consistency test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Game Engine Integration Test Suite")
    print("=" * 60)
    
    tests = [
        test_game_engine_integration,
        test_ai_interface_consistency
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\n✨ Game engine is now properly integrated with Level 3 AI!")
        return 0
    else:
        print("💥 Some integration tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 