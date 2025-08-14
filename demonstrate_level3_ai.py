#!/usr/bin/env python3
"""
Level3 AI Demonstration Script

This script demonstrates the trained Level3 AI making decisions in various
Euchre game scenarios. It shows the AI's strategic thinking process and
decision-making capabilities.
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import the trained model
from euchre.ai_model.level3_models import Level3NeuralModel, Level3RiskProfile
from euchre.models import Card, Suit, Rank, Player
from euchre.game import EuchreGame


@dataclass
class GameScenario:
    """Represents a game scenario for AI demonstration."""
    name: str
    player_hand: List[Card]
    trump_suit: Optional[Suit]
    top_card: Optional[Card]
    dealer_position: int
    current_position: int
    team_scores: Tuple[int, int]
    round_number: int
    description: str


class Level3AIDemonstrator:
    """Demonstrates Level3 AI decision-making capabilities."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the demonstrator.
        
        Parameters
        ----------
        model_path : str
            Path to the trained Level3 model
        device : str
            Device to run the model on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.risk_profile = Level3RiskProfile()
        
        print(f"🎯 Level3 AI Demonstrator initialized on {device}")
        print(f"📊 Model loaded from: {model_path}")
        print(f"🔧 Device: {self.device}")
        print()
    
    def _load_model(self, model_path: str) -> Level3NeuralModel:
        """Load the trained Level3 model."""
        # Model configuration matching training
        model = Level3NeuralModel(
            input_size=2048,
            hidden_size=512,
            num_layers=4,
            risk_embedding_size=128,
            use_attention=True,
            use_transformer=True,
            use_memory_networks=True
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def _encode_game_state(self, scenario: GameScenario) -> torch.Tensor:
        """
        Encode the game scenario into a feature vector.
        
        Parameters
        ----------
        scenario : GameScenario
            The game scenario to encode
            
        Returns
        -------
        torch.Tensor
            Encoded feature vector
        """
        # Create a comprehensive feature vector
        features = []
        
        # Hand encoding (5 cards * 52 possible cards = 260 features)
        hand_features = [0] * 260
        for card in scenario.player_hand:
            # Convert suit string to index (hearts=0, diamonds=1, clubs=2, spades=3)
            suit_idx = {'hearts': 0, 'diamonds': 1, 'clubs': 2, 'spades': 3}[card.suit.value]
            # Rank is already 0-13 (9=0, 10=1, J=2, Q=3, K=4, A=5)
            rank_idx = card.rank.value - 9  # Convert 9-14 to 0-5
            card_idx = suit_idx * 13 + rank_idx
            hand_features[card_idx] = 1
        features.extend(hand_features)
        
        # Trump suit encoding (5 suits including None = 5 features)
        trump_features = [0] * 5
        if scenario.trump_suit:
            # Convert suit string to index (hearts=0, diamonds=1, clubs=2, spades=3)
            suit_idx = {'hearts': 0, 'diamonds': 1, 'clubs': 2, 'spades': 3}[scenario.trump_suit.value]
            trump_features[suit_idx] = 1
        else:
            trump_features[4] = 1  # No trump
        features.extend(trump_features)
        
        # Top card encoding (52 cards + 1 for no card = 53 features)
        top_card_features = [0] * 53
        if scenario.top_card:
            # Convert suit string to index (hearts=0, diamonds=1, clubs=2, spades=3)
            suit_idx = {'hearts': 0, 'diamonds': 1, 'clubs': 2, 'spades': 3}[scenario.top_card.suit.value]
            # Rank is already 0-13 (9=0, 10=1, J=2, Q=3, K=4, A=5)
            rank_idx = scenario.top_card.rank.value - 9  # Convert 9-14 to 0-5
            card_idx = suit_idx * 13 + rank_idx
            top_card_features[card_idx] = 1
        else:
            top_card_features[52] = 1
        features.extend(top_card_features)
        
        # Position encoding (4 positions = 4 features)
        position_features = [0] * 4
        position_features[scenario.current_position] = 1
        features.extend(position_features)
        
        # Dealer position encoding (4 positions = 4 features)
        dealer_features = [0] * 4
        dealer_features[scenario.dealer_position] = 1
        features.extend(dealer_features)
        
        # Team scores encoding (scores 0-10 = 22 features)
        team1_score_features = [0] * 11
        team1_score_features[scenario.team_scores[0]] = 1
        features.extend(team1_score_features)
        
        team2_score_features = [0] * 11
        team2_score_features[scenario.team_scores[1]] = 1
        features.extend(team2_score_features)
        
        # Round number encoding (1-20 rounds = 20 features)
        round_features = [0] * 20
        if 1 <= scenario.round_number <= 20:
            round_features[scenario.round_number - 1] = 1
        features.extend(round_features)
        
        # Pad to required input size
        while len(features) < 2048:
            features.append(0)
        
        # Truncate if too long
        features = features[:2048]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def _decode_decision(self, outputs: Dict[str, torch.Tensor], scenario: GameScenario) -> Dict[str, any]:
        """
        Decode the model outputs into human-readable decisions.
        
        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs
        scenario : GameScenario
            The game scenario
            
        Returns
        -------
        Dict[str, any]
            Decoded decisions
        """
        decisions = {}
        
        # Trump decision
        trump_probs = torch.softmax(outputs['trump_decision'], dim=1)
        trump_decision = torch.argmax(trump_probs, dim=1).item()
        trump_confidence = trump_probs[0, trump_decision].item()
        
        trump_actions = ['Pass', 'Order Up', 'Call Trump']
        # Handle case where model only outputs 2 probabilities
        if trump_probs.shape[1] == 2:
            trump_actions = ['Pass', 'Order Up']
            trump_decision = min(trump_decision, 1)  # Ensure index is valid
        
        decisions['trump_decision'] = {
            'action': trump_actions[trump_decision],
            'confidence': trump_confidence,
            'probabilities': trump_probs[0].tolist()
        }
        
        # Card selection (if we have cards)
        if scenario.player_hand:
            card_probs = torch.softmax(outputs['card_selection'], dim=1)
            card_decision = torch.argmax(card_probs, dim=1).item()
            card_confidence = card_probs[0, card_decision].item()
            
            # Map to actual cards in hand
            if card_decision < len(scenario.player_hand):
                selected_card = scenario.player_hand[card_decision]
                decisions['card_selection'] = {
                    'card': f"{selected_card.rank.name} of {selected_card.suit.name}",
                    'confidence': card_confidence,
                    'probabilities': card_probs[0, :len(scenario.player_hand)].tolist()
                }
        
        # Suit selection
        suit_probs = torch.softmax(outputs['suit_selection'], dim=1)
        suit_decision = torch.argmax(suit_probs, dim=1).item()
        suit_confidence = suit_probs[0, suit_decision].item()
        
        suit_names = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        decisions['suit_selection'] = {
            'suit': suit_names[suit_decision],
            'confidence': suit_confidence,
            'probabilities': suit_probs[0].tolist()
        }
        
        # Risk adjustment
        risk_adjustment = torch.tanh(outputs['risk_adjustment']).mean().item()
        decisions['risk_adjustment'] = {
            'value': risk_adjustment,
            'interpretation': self._interpret_risk_adjustment(risk_adjustment)
        }
        
        # Strategic planning
        strategic_output = torch.tanh(outputs['strategic_planning']).mean().item()
        decisions['strategic_planning'] = {
            'value': strategic_output,
            'interpretation': self._interpret_strategic_planning(strategic_output)
        }
        
        # Partner coordination
        partner_output = torch.tanh(outputs['partner_coordination']).mean().item()
        decisions['partner_coordination'] = {
            'value': partner_output,
            'interpretation': self._interpret_partner_coordination(partner_output)
        }
        
        return decisions
    
    def _interpret_risk_adjustment(self, value: float) -> str:
        """Interpret risk adjustment values."""
        if value < -0.5:
            return "Very Conservative - Playing it safe"
        elif value < -0.1:
            return "Conservative - Cautious approach"
        elif value < 0.1:
            return "Balanced - Moderate risk"
        elif value < 0.5:
            return "Aggressive - Taking calculated risks"
        else:
            return "Very Aggressive - High risk tolerance"
    
    def _interpret_strategic_planning(self, value: float) -> str:
        """Interpret strategic planning values."""
        if value < -0.5:
            return "Short-term focus - Immediate gains"
        elif value < -0.1:
            return "Near-term planning - Next few tricks"
        elif value < 0.1:
            return "Balanced planning - Mix of short and long term"
        elif value < 0.5:
            return "Long-term planning - Setting up future tricks"
        else:
            return "Strategic mastermind - Complex multi-turn planning"
    
    def _interpret_partner_coordination(self, value: float) -> str:
        """Interpret partner coordination values."""
        if value < -0.5:
            return "Independent play - No partner consideration"
        elif value < -0.1:
            return "Minimal coordination - Basic partner awareness"
        elif value < 0.1:
            return "Moderate coordination - Some partner support"
        elif value < 0.5:
            return "Strong coordination - Actively supporting partner"
        else:
            return "Perfect coordination - Full partner synergy"
    
    def demonstrate_scenario(self, scenario: GameScenario) -> None:
        """
        Demonstrate AI decision-making for a specific scenario.
        
        Parameters
        ----------
        scenario : GameScenario
            The game scenario to demonstrate
        """
        print(f"🎮 **Scenario: {scenario.name}**")
        print(f"📝 {scenario.description}")
        print()
        
        # Display scenario details
        print("📋 **Scenario Details:**")
        print(f"   Hand: {[f'{card.rank.name} of {card.suit.name}' for card in scenario.player_hand]}")
        print(f"   Trump: {scenario.trump_suit.name if scenario.trump_suit else 'None'}")
        if scenario.top_card:
            print(f"   Top Card: {scenario.top_card.rank.name} of {scenario.top_card.suit.name}")
        else:
            print(f"   Top Card: None")
        print(f"   Position: {scenario.current_position} (Dealer: {scenario.dealer_position})")
        print(f"   Scores: Team 1: {scenario.team_scores[0]}, Team 2: {scenario.team_scores[1]}")
        print(f"   Round: {scenario.round_number}")
        print()
        
        # Encode scenario and get AI decision
        with torch.no_grad():
            input_features = self._encode_game_state(scenario)
            outputs = self.model(input_features, self.risk_profile)
            decisions = self._decode_decision(outputs, scenario)
        
        # Display AI decisions
        print("🤖 **Level3 AI Decisions:**")
        print()
        
        # Trump decision
        trump = decisions['trump_decision']
        print(f"🎯 **Trump Decision:** {trump['action']}")
        print(f"   Confidence: {trump['confidence']:.1%}")
        if len(trump['probabilities']) == 2:
            print(f"   Probabilities: Pass: {trump['probabilities'][0]:.1%}, Order Up: {trump['probabilities'][1]:.1%}")
        else:
            print(f"   Probabilities: Pass: {trump['probabilities'][0]:.1%}, Order Up: {trump['probabilities'][1]:.1%}, Call: {trump['probabilities'][2]:.1%}")
        print()
        
        # Card selection (if applicable)
        if 'card_selection' in decisions:
            card = decisions['card_selection']
            print(f"🃏 **Card Selection:** {card['card']}")
            print(f"   Confidence: {card['confidence']:.1%}")
            print()
        
        # Suit selection
        suit = decisions['suit_selection']
        print(f"♠️ **Suit Selection:** {suit['suit']}")
        print(f"   Confidence: {suit['confidence']:.1%}")
        print()
        
        # Strategic analysis
        print("🧠 **Strategic Analysis:**")
        risk = decisions['risk_adjustment']
        print(f"   Risk Profile: {risk['interpretation']} ({risk['value']:.3f})")
        
        strategy = decisions['strategic_planning']
        print(f"   Planning Horizon: {strategy['interpretation']} ({strategy['value']:.3f})")
        
        partner = decisions['partner_coordination']
        print(f"   Partner Coordination: {partner['interpretation']} ({partner['value']:.3f})")
        print()
        
        # Confidence summary
        confidences = [
            trump['confidence'],
            card['confidence'] if 'card_selection' in decisions else 0,
            suit['confidence']
        ]
        avg_confidence = np.mean([c for c in confidences if c > 0])
        print(f"📊 **Overall Confidence: {avg_confidence:.1%}**")
        print("=" * 80)
        print()


def create_demonstration_scenarios() -> List[GameScenario]:
    """Create various game scenarios for demonstration."""
    scenarios = []
    
    # Scenario 1: Early game, strong hand
    scenarios.append(GameScenario(
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
        description="Early in the game with a strong hand including two Aces and face cards. Need to decide whether to call trump."
    ))
    
    # Scenario 2: Mid-game, close scores
    scenarios.append(GameScenario(
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
        description="Mid-game with close scores. Hearts is trump, need to play strategically to maintain lead."
    ))
    
    # Scenario 3: Late game, need points
    scenarios.append(GameScenario(
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
        description="Late game, opponent team is ahead 9-8. Need to win this round to stay in the game."
    ))
    
    # Scenario 4: Trump calling decision
    scenarios.append(GameScenario(
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
        description="Strong Hearts hand with top card available. Need to decide whether to call Hearts as trump."
    ))
    
    # Scenario 5: Defensive play
    scenarios.append(GameScenario(
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
        description="Ahead 9-6, weak hand. Need to play defensively and avoid giving away points."
    ))
    
    return scenarios


def main():
    """Main demonstration function."""
    print("🎯 **Level3 AI Decision-Making Demonstration**")
    print("=" * 80)
    print()
    
    # Model path
    model_path = "trained_models/level3_fast/best_model.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("Please run 'make train-level3-fast' first to train the model.")
        return
    
    try:
        # Initialize demonstrator
        demonstrator = Level3AIDemonstrator(model_path, device="cpu")
        
        # Create scenarios
        scenarios = create_demonstration_scenarios()
        
        print(f"📚 **Demonstrating {len(scenarios)} Game Scenarios**")
        print("Each scenario shows the AI's decision-making process and strategic thinking.")
        print()
        
        # Demonstrate each scenario
        for i, scenario in enumerate(scenarios, 1):
            print(f"📖 **Scenario {i}/{len(scenarios)}**")
            demonstrator.demonstrate_scenario(scenario)
            
            if i < len(scenarios):
                input("Press Enter to continue to next scenario...")
                print()
        
        print("🎉 **Demonstration Complete!**")
        print("The Level3 AI has demonstrated its decision-making capabilities across various game scenarios.")
        print("Key insights:")
        print("  • Trump calling strategy based on hand strength and game state")
        print("  • Card selection considering trump, position, and team scores")
        print("  • Risk assessment and strategic planning")
        print("  • Partner coordination and team play")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 