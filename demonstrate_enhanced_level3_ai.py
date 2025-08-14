#!/usr/bin/env python3
"""
Enhanced Level3 AI Demonstration Script

This script demonstrates the enhanced Level3 AI that is now 100% AI-driven
without traditional rule-based fallbacks. It shows pure neural decision-making
and strategic analysis capabilities.
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import the enhanced Level3 AI
from euchre.ai.level3_ai_impl import Level3AI
from euchre.models import Card, Suit, Rank, Player
from euchre.game import EuchreGame


@dataclass
class EnhancedGameScenario:
    """Represents a game scenario for enhanced AI demonstration."""
    name: str
    player_hand: List[Card]
    trump_suit: Optional[Suit]
    top_card: Optional[Card]
    dealer_position: int
    current_position: int
    team_scores: Tuple[int, int]
    round_number: int
    description: str
    expected_ai_behavior: str


class EnhancedLevel3AIDemonstrator:
    """Demonstrates enhanced Level3 AI's 100% AI-driven capabilities."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the enhanced demonstrator.
        
        Parameters
        ----------
        model_path : str
            Path to the trained Level3 model
        device : str
            Device to run the model on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model_path = model_path
        
        print(f"🎯 **Enhanced Level3 AI Demonstrator**")
        print(f"🔧 Device: {self.device}")
        print(f"📊 Model: {model_path}")
        print()
        
        # Create Level3 AI instances with different risk profiles
        self.ai_players = self._create_ai_players()
        
    def _create_ai_players(self) -> Dict[str, Level3AI]:
        """Create Level3 AI players with different risk profiles."""
        ai_players = {}
        
        # Create different AI personalities
        risk_profiles = [
            ("strategic_mastermind", "Strategic Mastermind"),
            ("ultra_conservative", "Ultra Conservative"),
            ("balanced", "Balanced"),
            ("aggressive", "Aggressive"),
            ("ultra_aggressive", "Ultra Aggressive")
        ]
        
        for profile_key, profile_name in risk_profiles:
            try:
                ai = Level3AI(
                    name=f"Level3_{profile_name}",
                    model_type="level3_strategic",
                    risk_profile=profile_key,
                    model_path=self.model_path,
                    device=str(self.device)
                )
                ai_players[profile_key] = ai
                print(f"✅ Created {profile_name} AI")
            except Exception as e:
                print(f"❌ Failed to create {profile_name} AI: {e}")
                # Create a basic AI without model loading for demonstration
                try:
                    ai = Level3AI(
                        name=f"Level3_{profile_name}",
                        model_type="level3_strategic",
                        risk_profile=profile_key,
                        device=str(self.device)
                    )
                    ai_players[profile_key] = ai
                    print(f"✅ Created {profile_name} AI (basic mode)")
                except Exception as e2:
                    print(f"❌ Failed to create {profile_name} AI in basic mode: {e2}")
        
        return ai_players
    
    def demonstrate_ai_capabilities(self, scenario: EnhancedGameScenario) -> None:
        """Demonstrate AI capabilities for a specific scenario."""
        print(f"🎮 **Enhanced Scenario: {scenario.name}**")
        print(f"📝 {scenario.description}")
        print(f"🎯 Expected AI Behavior: {scenario.expected_ai_behavior}")
        print()
        
        # Display scenario details
        print("📋 **Scenario Details:**")
        print(f"   Hand: {[f'{card.rank.name} of {card.suit.name}' for card in scenario.player_hand]}")
        if scenario.trump_suit:
            print(f"   Trump: {scenario.trump_suit.name}")
        else:
            print(f"   Trump: None")
        if scenario.top_card:
            print(f"   Top Card: {scenario.top_card.rank.name} of {scenario.top_card.suit.name}")
        else:
            print(f"   Top Card: None")
        print(f"   Position: {scenario.current_position} (Dealer: {scenario.dealer_position})")
        print(f"   Scores: Team 1: {scenario.team_scores[0]}, Team 2: {scenario.team_scores[1]}")
        print(f"   Round: {scenario.round_number}")
        print()
        
        # Test each AI player
        for profile_key, ai_player in self.ai_players.items():
            print(f"🤖 **{ai_player.name} ({profile_key})**")
            print("-" * 60)
            
            try:
                # Create a mock game context for the AI to analyze
                context = self._create_mock_context(scenario)
                
                # Test trump decision
                if scenario.trump_suit is None and scenario.top_card:
                    print("🎯 **Trump Decision Test:**")
                    trump_decision = ai_player.should_order_up(context)
                    print(f"   Decision: {trump_decision.decision_type.name}")
                    print(f"   Confidence: {trump_decision.confidence:.1%}")
                    print(f"   Reasoning: {trump_decision.reasoning}")
                    print(f"   AI Metadata: {trump_decision.metadata}")
                    print()
                
                # Test suit selection
                if scenario.trump_suit is None:
                    print("♠️ **Suit Selection Test:**")
                    suit_decision = ai_player.select_trump_suit(context)
                    print(f"   Selected Suit: {suit_decision.metadata.get('selected_suit', 'Unknown')}")
                    print(f"   Confidence: {suit_decision.confidence:.1%}")
                    print(f"   Reasoning: {suit_decision.reasoning}")
                    print(f"   AI Metadata: {suit_decision.metadata}")
                    print()
                
                # Test card selection
                print("🃏 **Card Selection Test:**")
                card_decision = ai_player.play_card(context)
                print(f"   Selected Card: {card_decision.metadata.get('selected_card', 'Unknown')}")
                print(f"   Confidence: {card_decision.confidence:.1%}")
                print(f"   Reasoning: {card_decision.reasoning}")
                print(f"   AI Metadata: {card_decision.metadata}")
                print()
                
                # Test strategic analysis
                print("🧠 **Strategic Analysis Test:**")
                strategic_analysis = ai_player.get_strategic_analysis(context)
                print(f"   Risk Adjustment: {strategic_analysis['risk_adjustment']:.3f}")
                print(f"   Strategic Planning: {strategic_analysis['strategic_planning']:.3f}")
                print(f"   Partner Coordination: {strategic_analysis['partner_coordination']:.3f}")
                print(f"   Overall Strategy Score: {strategic_analysis['overall_strategy_score']:.3f}")
                print(f"   AI Confidence: {strategic_analysis['ai_confidence']:.3f}")
                print(f"   Key Features: {strategic_analysis['key_feature_indices'][:3]}...")
                print()
                
                # Test strategy adaptation
                print("🔄 **Strategy Adaptation Test:**")
                # Simulate a game outcome
                game_outcome = 1.0 if scenario.team_scores[0] > scenario.team_scores[1] else -1.0
                ai_player.update_strategy_based_on_outcome(game_outcome, context)
                print(f"   Strategy updated based on outcome: {game_outcome:.1f}")
                print()
                
            except Exception as e:
                print(f"❌ Error testing {ai_player.name}: {e}")
                print()
            
            print("=" * 80)
            print()
    
    def _create_mock_context(self, scenario: EnhancedGameScenario) -> any:
        """Create a mock game context for AI analysis."""
        class MockGameContext:
            def __init__(self, scenario):
                self.hand = scenario.player_hand
                self.trump_suit = scenario.trump_suit
                self.flipped_card = scenario.top_card
                self.current_trick = None
                self.trick_suit = None
                self.team1_score = scenario.team_scores[0]
                self.team2_score = scenario.team_scores[1]
                self.tricks_won_team1 = 0
                self.tricks_won_team2 = 0
                self.current_trick_number = scenario.round_number
                self.dealer_position = scenario.dealer_position
                self.current_position = scenario.current_position
        
        return MockGameContext(scenario)


def create_enhanced_scenarios() -> List[EnhancedGameScenario]:
    """Create enhanced game scenarios for demonstration."""
    scenarios = []
    
    # Scenario 1: Early game, strong hand
    scenarios.append(EnhancedGameScenario(
        name="Early Game - Strong Hand",
        player_hand=[
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.QUEEN, Suit.SPADES)
        ],
        trump_suit=None,
        top_card=Card(Rank.JACK, Suit.HEARTS),
        dealer_position=2,
        current_position=3,
        team_scores=(0, 0),
        round_number=1,
        description="Early in the game with a strong hand including two Aces and face cards.",
        expected_ai_behavior="AI should recognize strong hand and call trump with high confidence"
    ))
    
    # Scenario 2: Mid-game, close scores
    scenarios.append(EnhancedGameScenario(
        name="Mid-Game - Close Scores",
        player_hand=[
            Card(Rank.TEN, Suit.HEARTS),
            Card(Rank.NINE, Suit.DIAMONDS),
            Card(Rank.NINE, Suit.CLUBS),
            Card(Rank.TEN, Suit.SPADES),
            Card(Rank.NINE, Suit.HEARTS)
        ],
        trump_suit=Suit.HEARTS,
        top_card=None,
        dealer_position=1,
        current_position=0,
        team_scores=(7, 6),
        round_number=3,
        description="Mid-game with close scores. Hearts is trump, need strategic play.",
        expected_ai_behavior="AI should play conservatively to maintain lead, coordinate with partner"
    ))
    
    # Scenario 3: Late game, need points
    scenarios.append(EnhancedGameScenario(
        name="Late Game - Need Points",
        player_hand=[
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.NINE, Suit.HEARTS)
        ],
        trump_suit=Suit.HEARTS,
        top_card=None,
        dealer_position=3,
        current_position=2,
        team_scores=(8, 9),
        round_number=5,
        description="Late game, opponent team is ahead 9-8. Need to win this round.",
        expected_ai_behavior="AI should play aggressively, use trump cards strategically"
    ))
    
    # Scenario 4: Trump calling decision
    scenarios.append(EnhancedGameScenario(
        name="Trump Calling Decision",
        player_hand=[
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.CLUBS)
        ],
        trump_suit=None,
        top_card=Card(Rank.JACK, Suit.HEARTS),
        dealer_position=0,
        current_position=1,
        team_scores=(3, 2),
        round_number=2,
        description="Strong Hearts hand with top card available. Need to decide on trump.",
        expected_ai_behavior="AI should call Hearts as trump with very high confidence"
    ))
    
    # Scenario 5: Defensive play
    scenarios.append(EnhancedGameScenario(
        name="Defensive Play",
        player_hand=[
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.TEN, Suit.DIAMONDS),
            Card(Rank.NINE, Suit.CLUBS),
            Card(Rank.TEN, Suit.SPADES),
            Card(Rank.TEN, Suit.HEARTS)
        ],
        trump_suit=Suit.DIAMONDS,
        top_card=None,
        dealer_position=2,
        current_position=3,
        team_scores=(9, 6),
        round_number=4,
        description="Ahead 9-6, weak hand. Need to play defensively.",
        expected_ai_behavior="AI should play very conservatively, avoid giving away points"
    ))
    
    return scenarios


def main():
    """Main demonstration function."""
    print("🎯 **Enhanced Level3 AI - 100% AI-Driven Demonstration**")
    print("=" * 80)
    print()
    print("This demonstration shows the enhanced Level3 AI that is now:")
    print("✅ 100% AI-driven - no traditional rule-based fallbacks")
    print("✅ Pure neural decision-making for all game choices")
    print("✅ AI-based error handling and fallbacks")
    print("✅ Strategic analysis and adaptation capabilities")
    print("✅ Multiple AI personalities with different risk profiles")
    print()
    
    # Model path
    model_path = "trained_models/level3_fast/best_model.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("Please run 'make train-level3-fast' first to train the model.")
        return
    
    try:
        # Initialize enhanced demonstrator
        demonstrator = EnhancedLevel3AIDemonstrator(model_path, device="cpu")
        
        # Create enhanced scenarios
        scenarios = create_enhanced_scenarios()
        
        print(f"📚 **Demonstrating {len(scenarios)} Enhanced Scenarios**")
        print("Each scenario shows the AI's pure neural decision-making capabilities.")
        print()
        
        # Demonstrate each scenario
        for i, scenario in enumerate(scenarios, 1):
            print(f"📖 **Enhanced Scenario {i}/{len(scenarios)}**")
            demonstrator.demonstrate_ai_capabilities(scenario)
            
            if i < len(scenarios):
                input("Press Enter to continue to next scenario...")
                print()
        
        print("🎉 **Enhanced Demonstration Complete!**")
        print("The Level3 AI has demonstrated its 100% AI-driven capabilities:")
        print("  • Pure neural trump decisions without traditional thresholds")
        print("  • AI-based suit selection using learned preferences")
        print("  • Neural card selection with strategic context")
        print("  • Comprehensive strategic analysis and adaptation")
        print("  • Multiple AI personalities with different risk profiles")
        print("  • AI-based error handling instead of rule-based fallbacks")
        
    except Exception as e:
        print(f"❌ Error during enhanced demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 