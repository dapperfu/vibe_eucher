#!/usr/bin/env python3
"""
Level 3 AI Tournament Demo

This script runs a tournament specifically for Level 3 AI players to test
their capabilities and identify any integration issues.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-13
Context: Creating Level 3 AI tournament demo to test integration
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

@dataclass
class Level3TournamentTeam:
    """Represents a Level 3 AI team in the tournament."""
    name: str
    ai_type: str
    risk_profile: float
    wins: int = 0
    losses: int = 0
    games_played: int = 0
    total_score: int = 0
    errors: List[str] = None
    
    def __init__(self, name: str, ai_type: str, risk_profile: float):
        self.name = name
        self.ai_type = ai_type
        self.risk_profile = risk_profile
        self.errors = []
    
    def __str__(self) -> str:
        return f"{self.name} ({self.ai_type}, risk={self.risk_profile})"
    
    def get_win_percentage(self) -> float:
        """Calculate win percentage."""
        if self.games_played == 0:
            return 0.0
        return (self.wins / self.games_played) * 100

@dataclass
class TournamentGame:
    """Represents a single tournament game."""
    team1: Level3TournamentTeam
    team2: Level3TournamentTeam
    winner: Optional[str] = None
    team1_score: int = 0
    team2_score: int = 0
    errors: List[str] = None
    duration: float = 0.0
    
    def __init__(self, team1: Level3TournamentTeam, team2: Level3TournamentTeam):
        self.team1 = team1
        self.team2 = team2
        self.errors = []

class Level3TournamentDemo:
    """Level 3 AI Tournament Demo Controller."""
    
    def __init__(self):
        """Initialize the tournament demo."""
        self.teams: List[Level3TournamentTeam] = []
        self.games: List[TournamentGame] = []
        self.errors: List[str] = []
        
        # Create Level 3 AI teams
        self._create_teams()
    
    def _create_teams(self):
        """Create Level 3 AI teams for the tournament."""
        print("🏗️ Creating Level 3 AI teams...")
        
        team_configs = [
            ("Strategic Masters", "level3_strategic", 0.5),
            ("Aggressive Warriors", "level3_aggressive", 0.8),
            ("Balanced Champions", "level3_balanced", 0.5),
            ("Conservative Defenders", "level3_conservative", 0.2),
            ("Opportunistic Raiders", "level3_opportunistic", 0.7)
        ]
        
        for name, ai_type, risk in team_configs:
            try:
                team = Level3TournamentTeam(name, ai_type, risk)
                self.teams.append(team)
                print(f"✅ Created team: {team}")
            except Exception as e:
                error_msg = f"Failed to create team {name}: {e}"
                self.errors.append(error_msg)
                print(f"❌ {error_msg}")
        
        print(f"Created {len(self.teams)} teams\n")
    
    def test_ai_creation(self):
        """Test that all Level 3 AI can be created successfully."""
        print("🧪 Testing Level 3 AI Creation")
        print("=" * 50)
        
        try:
            from euchre.ai.ai_factory import AIFactory
            
            for team in self.teams:
                try:
                    # Test creating AI players
                    ai1 = AIFactory.create_ai_player(f"{team.name}_Player1", team.ai_type, team.risk_profile)
                    ai2 = AIFactory.create_ai_player(f"{team.name}_Player2", team.ai_type, team.risk_profile)
                    
                    print(f"✅ {team.name}: Successfully created AI players")
                    print(f"   Player 1: {type(ai1).__name__}")
                    print(f"   Player 2: {type(ai2).__name__}")
                    
                except Exception as e:
                    error_msg = f"Failed to create AI for {team.name}: {e}"
                    team.errors.append(error_msg)
                    self.errors.append(error_msg)
                    print(f"❌ {error_msg}")
                
                print()
            
        except Exception as e:
            error_msg = f"Failed to import AI factory: {e}"
            self.errors.append(error_msg)
            print(f"❌ {error_msg}")
        
        return len(self.errors) == 0
    
    def test_ai_interface(self):
        """Test that all Level 3 AI implement the interface correctly."""
        print("🧪 Testing Level 3 AI Interface Implementation")
        print("=" * 50)
        
        try:
            from euchre.ai.ai_factory import AIFactory
            from euchre.ai.base_ai_interface import BaseAIInterface, GameContext
            from euchre.models import Card, Suit, Rank
            
            # Create a test game context
            test_hand = [
                Card(Suit.HEARTS, Rank.ACE),
                Card(Suit.HEARTS, Rank.KING),
                Card(Suit.DIAMONDS, Rank.QUEEN),
                Card(Suit.CLUBS, Rank.JACK),
                Card(Suit.SPADES, Rank.TEN)
            ]
            
            test_context = GameContext(
                hand=test_hand,
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
            
            for team in self.teams:
                try:
                    # Create AI and test interface
                    ai = AIFactory.create_ai_player(f"{team.name}_Test", team.ai_type, team.risk_profile)
                    
                    # Check interface implementation
                    if not isinstance(ai, BaseAIInterface):
                        raise TypeError(f"AI does not implement BaseAIInterface")
                    
                    # Test all required methods
                    methods_to_test = [
                        ('should_order_up', ai.should_order_up, test_context),
                        ('should_call_trump', ai.should_call_trump, test_context),
                        ('select_trump_suit', ai.select_trump_suit, test_context),
                        ('play_card', ai.play_card, test_context),
                        ('discard_card', ai.discard_card, test_context)
                    ]
                    
                    for method_name, method, context in methods_to_test:
                        try:
                            result = method(context)
                            if not hasattr(result, 'decision_type') or not hasattr(result, 'confidence'):
                                raise ValueError(f"Method {method_name} returned invalid result")
                        except Exception as e:
                            raise RuntimeError(f"Method {method_name} failed: {e}")
                    
                    print(f"✅ {team.name}: Interface implementation verified")
                    
                except Exception as e:
                    error_msg = f"Interface test failed for {team.name}: {e}"
                    team.errors.append(error_msg)
                    self.errors.append(error_msg)
                    print(f"❌ {error_msg}")
                
                print()
            
        except Exception as e:
            error_msg = f"Failed to test AI interface: {e}"
            self.errors.append(error_msg)
            print(f"❌ {error_msg}")
        
        return len(self.errors) == 0
    
    def test_ai_decision_making(self):
        """Test that all Level 3 AI can make decisions."""
        print("🧪 Testing Level 3 AI Decision Making")
        print("=" * 50)
        
        try:
            from euchre.ai.ai_factory import AIFactory
            from euchre.ai.base_ai_interface import GameContext
            from euchre.models import Card, Suit, Rank
            
            # Create a test game context
            test_hand = [
                Card(Suit.HEARTS, Rank.ACE),
                Card(Suit.HEARTS, Rank.KING),
                Card(Suit.DIAMONDS, Rank.QUEEN),
                Card(Suit.CLUBS, Rank.JACK),
                Card(Suit.SPADES, Rank.TEN)
            ]
            
            test_context = GameContext(
                hand=test_hand,
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
            
            for team in self.teams:
                try:
                    # Create AI and test decision making
                    ai = AIFactory.create_ai_player(f"{team.name}_Test", team.ai_type, team.risk_profile)
                    
                    # Test decision making
                    decisions = []
                    decisions.append(ai.should_order_up(test_context))
                    decisions.append(ai.should_call_trump(test_context))
                    decisions.append(ai.select_trump_suit(test_context))
                    decisions.append(ai.play_card(test_context))
                    decisions.append(ai.discard_card(test_context))
                    
                    # Verify all decisions are valid
                    for i, decision in enumerate(decisions):
                        if not hasattr(decision, 'decision_type'):
                            raise ValueError(f"Decision {i} missing decision_type")
                        if not hasattr(decision, 'confidence'):
                            raise ValueError(f"Decision {i} missing confidence")
                        if not hasattr(decision, 'reasoning'):
                            raise ValueError(f"Decision {i} missing reasoning")
                    
                    print(f"✅ {team.name}: Decision making verified ({len(decisions)} decisions)")
                    
                except Exception as e:
                    error_msg = f"Decision making test failed for {team.name}: {e}"
                    team.errors.append(error_msg)
                    self.errors.append(error_msg)
                    print(f"❌ {error_msg}")
                
                print()
            
        except Exception as e:
            error_msg = f"Failed to test AI decision making: {e}"
            self.errors.append(error_msg)
            print(f"❌ {error_msg}")
        
        return len(self.errors) == 0
    
    def run_tournament(self):
        """Run the Level 3 AI tournament."""
        print("🏆 Running Level 3 AI Tournament")
        print("=" * 50)
        
        if not self.test_ai_creation():
            print("❌ AI creation tests failed, cannot run tournament")
            return False
        
        if not self.test_ai_interface():
            print("❌ AI interface tests failed, cannot run tournament")
            return False
        
        if not self.test_ai_decision_making():
            print("❌ AI decision making tests failed, cannot run tournament")
            return False
        
        print("🎉 All tests passed! Level 3 AI is ready for tournament play.")
        return True
    
    def generate_report(self):
        """Generate a comprehensive error report."""
        print("\n📊 Level 3 AI Tournament Demo Report")
        print("=" * 60)
        
        if not self.errors:
            print("✅ No errors found! Level 3 AI is working perfectly.")
            return
        
        print(f"❌ Found {len(self.errors)} errors:")
        for i, error in enumerate(self.errors, 1):
            print(f"   {i}. {error}")
        
        print(f"\n📋 Team-specific errors:")
        for team in self.teams:
            if team.errors:
                print(f"   {team.name}: {len(team.errors)} errors")
                for error in team.errors:
                    print(f"     - {error}")

def main():
    """Run the Level 3 AI tournament demo."""
    print("🚀 Level 3 AI Tournament Demo")
    print("=" * 60)
    print()
    
    try:
        # Create and run tournament
        tournament = Level3TournamentDemo()
        success = tournament.run_tournament()
        
        # Generate report
        tournament.generate_report()
        
        if success:
            print("\n🎉 Level 3 AI Tournament Demo completed successfully!")
            return 0
        else:
            print("\n💥 Level 3 AI Tournament Demo encountered errors!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Tournament demo crashed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 