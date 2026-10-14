"""
Adaptive AI System for Euchre

This module implements an adaptive AI that learns from losses and adjusts
its strategy based on game outcomes, team performance, and historical data.
"""

import random
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from .base_ai import BaseAI
from .ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI
from ..models import Card, Suit, Rank, Player
from ..core.deck import Deck


@dataclass
class AdaptiveMemory:
    """Memory system for adaptive AI learning."""
    
    # Game history
    games_played: int = 0
    games_won: int = 0
    games_lost: int = 0
    
    # Performance metrics
    total_tricks_won: int = 0
    total_tricks_lost: int = 0
    team_sets: int = 0
    times_set: int = 0
    
    # Strategy effectiveness
    trump_calls_won: int = 0
    trump_calls_lost: int = 0
    defensive_plays_successful: int = 0
    offensive_plays_successful: int = 0
    
    # Risk adjustment history
    risk_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Team performance tracking
    team_performance: Dict[str, List[float]] = field(default_factory=lambda: {
        "win_rates": [],
        "trick_efficiency": [],
        "risk_effectiveness": []
    })
    
    def update_game_result(self, won: bool, tricks_won: int, tricks_lost: int, 
                          was_set: bool, set_opponent: bool) -> None:
        """Update memory with game results."""
        self.games_played += 1
        
        if won:
            self.games_won += 1
        else:
            self.games_lost += 1
        
        self.total_tricks_won += tricks_won
        self.total_tricks_lost += tricks_lost
        
        if was_set:
            self.times_set += 1
        if set_opponent:
            self.team_sets += 1
    
    def update_strategy_effectiveness(self, trump_call_won: bool, 
                                    defensive_success: bool, 
                                    offensive_success: bool) -> None:
        """Update strategy effectiveness metrics."""
        if trump_call_won:
            self.trump_calls_won += 1
        else:
            self.trump_calls_lost += 1
        
        if defensive_success:
            self.defensive_plays_successful += 1
        if offensive_success:
            self.offensive_plays_successful += 1
    
    def record_risk_adjustment(self, old_risk: float, new_risk: float, 
                              reason: str, outcome: str) -> None:
        """Record a risk adjustment and its outcome."""
        self.risk_adjustments.append({
            "timestamp": datetime.now().isoformat(),
            "old_risk": old_risk,
            "new_risk": new_risk,
            "reason": reason,
            "outcome": outcome
        })
    
    def calculate_win_rate(self) -> float:
        """Calculate current win rate."""
        if self.games_played == 0:
            return 0.0
        return self.games_won / self.games_played
    
    def calculate_trick_efficiency(self) -> float:
        """Calculate trick efficiency (tricks won vs total tricks)."""
        total_tricks = self.total_tricks_won + self.total_tricks_lost
        if total_tricks == 0:
            return 0.0
        return self.total_tricks_won / total_tricks
    
    def get_recent_performance(self, games: int = 10) -> Dict[str, float]:
        """Get recent performance metrics."""
        return {
            "win_rate": self.calculate_win_rate(),
            "trick_efficiency": self.calculate_trick_efficiency(),
            "team_set_rate": self.team_sets / max(1, self.games_played),
            "times_set_rate": self.times_set / max(1, self.games_played)
        }


class AdaptiveAI(BaseAI):
    """
    Adaptive AI that learns from losses and adjusts strategy.
    
    This AI combines the base AI logic with learning capabilities,
    adjusting its risk tolerance and strategy based on performance.
    """
    
    def __init__(self, name: str, base_ai_type: str = "balanced", 
                 initial_risk_ratio: float = 0.5, learning_rate: float = 0.1,
                 memory_file: Optional[str] = None):
        """
        Initialize adaptive AI.
        
        Parameters
        ----------
        name : str
            Player name
        base_ai_type : str
            Base AI type (aggressive, conservative, balanced, opportunistic)
        initial_risk_ratio : float
            Initial risk ratio (0.0 to 1.0)
        learning_rate : float
            How quickly the AI adapts (0.0 to 1.0)
        memory_file : Optional[str]
            File to persist memory between sessions
        """
        super().__init__(name)
        
        # Create base AI with specified type
        self.base_ai = self._create_base_ai(base_ai_type, initial_risk_ratio)
        self.base_ai_type = base_ai_type
        self.initial_risk_ratio = initial_risk_ratio
        
        # Adaptive parameters
        self.learning_rate = learning_rate
        self.current_risk_ratio = initial_risk_ratio
        self.adaptation_threshold = 0.1  # Minimum change threshold
        
        # Memory and learning
        self.memory = AdaptiveMemory()
        self.memory_file = memory_file or f"adaptive_ai_{name}.json"
        self.load_memory()
        
        # Strategy state
        self.current_strategy = "balanced"
        self.strategy_confidence = 0.5
        self.last_adjustment_game = 0
        
        # Performance tracking
        self.current_game_tricks_won = 0
        self.current_game_tricks_lost = 0
        self.current_game_trump_call = False
        self.current_game_trump_call_won = False
    
    def _create_base_ai(self, ai_type: str, risk_ratio: float) -> BaseAI:
        """Create the base AI instance."""
        ai_classes = {
            "aggressive": AggressiveAI,
            "conservative": ConservativeAI,
            "balanced": BalancedAI,
            "opportunistic": OpportunisticAI
        }
        
        ai_class = ai_classes.get(ai_type, BalancedAI)
        base_ai = ai_class(self.name)
        base_ai.risk_ratio = risk_ratio
        return base_ai
    
    def load_memory(self) -> None:
        """Load memory from file if it exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.memory = AdaptiveMemory(**data)
                print(f"📚 Loaded memory for {self.name} from {self.memory_file}")
            except Exception as e:
                print(f"⚠️  Error loading memory for {self.name}: {e}")
    
    def save_memory(self) -> None:
        """Save memory to file."""
        try:
            # Convert memory to dict for JSON serialization
            memory_dict = {
                "games_played": self.memory.games_played,
                "games_won": self.memory.games_won,
                "games_lost": self.memory.games_lost,
                "total_tricks_won": self.memory.total_tricks_won,
                "total_tricks_lost": self.memory.total_tricks_lost,
                "team_sets": self.memory.team_sets,
                "times_set": self.memory.times_set,
                "trump_calls_won": self.memory.trump_calls_won,
                "trump_calls_lost": self.memory.trump_calls_lost,
                "defensive_plays_successful": self.memory.defensive_plays_successful,
                "offensive_plays_successful": self.memory.offensive_plays_successful,
                "risk_adjustments": self.memory.risk_adjustments,
                "team_performance": self.memory.team_performance
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(memory_dict, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Error saving memory for {self.name}: {e}")
    
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Decide whether to order up the top card."""
        # Use base AI logic
        base_decision = self.base_ai.should_order_up(top_card, is_partner_dealing)
        
        # Track trump call decision
        self.current_game_trump_call = True
        
        # Adjust decision based on learning if needed
        if self._should_adapt_strategy():
            adapted_decision = self._adapt_trump_decision(base_decision, top_card, [])
            return adapted_decision
        
        return base_decision
    
    def _sync_hand_with_base_ai(self) -> None:
        """Sync the hand with the base AI instance."""
        if hasattr(self, 'base_ai') and self.base_ai:
            self.base_ai.hand = self.hand.copy()
    
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play."""
        # Sync hand with base AI before making decision
        self._sync_hand_with_base_ai()
        
        # Use base AI logic
        card = self.base_ai.choose_card_to_play(lead_suit, trump_suit)
        
        # Track play for learning (simplified since we don't have full context)
        self._track_card_play(card, [], trump_suit, lead_suit)
        
        return card
    
    def _should_adapt_strategy(self) -> bool:
        """Determine if strategy should be adapted."""
        # Adapt every 5 games or if performance is poor
        games_since_adjustment = self.memory.games_played - self.last_adjustment_game
        poor_performance = self.memory.calculate_win_rate() < 0.4
        
        return games_since_adjustment >= 5 or poor_performance
    
    def _adapt_trump_decision(self, base_decision: bool, top_card: Card, 
                             hand: List[Card]) -> bool:
        """Adapt trump calling decision based on learning."""
        if not self._should_adapt_strategy():
            return base_decision
        
        # Analyze recent performance
        recent_performance = self.memory.get_recent_performance()
        
        # If losing frequently, be more conservative
        if recent_performance["win_rate"] < 0.4:
            if base_decision:  # Was going to call trump
                # Be more selective about trump calls
                if hand:  # Only evaluate if we have hand information
                    hand_strength = self._evaluate_hand_for_trump(hand, top_card.suit)
                    if hand_strength < 0.7:  # Only call if hand is strong
                        return False
        
        # If winning frequently, be more aggressive
        elif recent_performance["win_rate"] > 0.7:
            if not base_decision:  # Wasn't going to call trump
                # Be more willing to call trump
                if hand:  # Only evaluate if we have hand information
                    hand_strength = self._evaluate_hand_for_trump(hand, top_card.suit)
                    if hand_strength > 0.5:  # Call if hand is decent
                        return True
        
        return base_decision
    
    def _track_card_play(self, card: Card, current_trick: List[Tuple[Player, Card]], 
                         trump_suit: Optional[Suit], lead_suit: Optional[Suit]) -> None:
        """Track card play for learning purposes."""
        # This will be used to analyze play effectiveness
        # For now, just track basic metrics
        pass
    
    def _evaluate_hand_for_trump(self, hand: List[Card], trump_suit: Suit) -> float:
        """
        Evaluate hand strength for a given trump suit.
        
        Parameters
        ----------
        hand : List[Card]
            Player's hand
        trump_suit : Suit
            Potential trump suit
        
        Returns
        -------
        float
            Hand strength (0.0 to 1.0)
        """
        if not hand:
            return 0.0
        
        # Count trump cards
        trump_cards = [card for card in hand if card.suit == trump_suit]
        trump_count = len(trump_cards)
        
        # Count high-value cards (A, K, Q, J, 10)
        high_cards = [card for card in hand if card.rank.value >= 10]
        high_card_count = len(high_cards)
        
        # Calculate strength based on trump count and high cards
        trump_strength = trump_count / 5.0  # Normalize by max possible trump
        high_card_strength = high_card_count / 5.0  # Normalize by hand size
        
        # Weighted combination
        total_strength = (trump_strength * 0.7) + (high_card_strength * 0.3)
        
        return min(1.0, total_strength)
    
    def _get_left_bower_suit(self, trump_suit: Suit) -> Suit:
        """
        Get the suit of the left bower for a given trump suit.
        
        Parameters
        ----------
        trump_suit : Suit
            The trump suit
        
        Returns
        -------
        Suit
            The suit of the left bower
        """
        # Left bower is the Jack of the same color as trump
        if trump_suit in [Suit.HEARTS, Suit.DIAMONDS]:
            return Suit.DIAMONDS if trump_suit == Suit.HEARTS else Suit.HEARTS
        else:  # CLUBS or SPADES
            return Suit.SPADES if trump_suit == Suit.CLUBS else Suit.CLUBS
    
    def _card_value(self, card: Card, trump_suit: Suit, lead_suit: Optional[Suit] = None) -> int:
        """
        Calculate the value of a card in the current trick context.
        
        Parameters
        ----------
        card : Card
            The card to evaluate
        trump_suit : Suit
            The current trump suit
        lead_suit : Optional[Suit]
            The suit that was led (if any)
        
        Returns
        -------
        int
            Card value (higher is better)
        """
        # Get left bower suit
        left_bower_suit = self._get_left_bower_suit(trump_suit)
        
        # Right bower (Jack of trump suit) is highest
        if card.rank == Rank.JACK and card.suit == trump_suit:
            return 100
        
        # Left bower (Jack of left bower suit) is second highest
        if card.rank == Rank.JACK and card.suit == left_bower_suit:
            return 90
        
        # Trump cards (excluding bowers)
        if card.suit == trump_suit:
            return 80 + card.rank.value
        
        # Non-trump cards
        if lead_suit and card.suit == lead_suit:
            # Following suit - use rank value
            return card.rank.value
        else:
            # Off-suit - very low value
            return card.rank.value - 100
    
    def on_game_end(self, won: bool, tricks_won: int, tricks_lost: int, 
                    was_set: bool, set_opponent: bool) -> None:
        """Called when a game ends to update learning."""
        # Update memory with game results
        self.memory.update_game_result(won, tricks_won, tricks_lost, was_set, set_opponent)
        
        # Update strategy effectiveness
        if self.current_game_trump_call:
            self.memory.update_strategy_effectiveness(
                self.current_game_trump_call_won,
                False,  # defensive_success - would need more tracking
                False   # offensive_success - would need more tracking
            )
        
        # Adapt strategy if needed
        if self._should_adapt_strategy():
            self._adapt_strategy()
        
        # Save memory
        self.save_memory()
        
        # Reset current game tracking
        self._reset_game_tracking()
    
    def _adapt_strategy(self) -> None:
        """Adapt the AI strategy based on performance."""
        recent_performance = self.memory.get_recent_performance()
        old_risk = self.current_risk_ratio
        
        # Calculate adaptation
        if recent_performance["win_rate"] < 0.4:
            # Losing frequently - become more conservative
            risk_reduction = self.learning_rate * (0.5 - recent_performance["win_rate"])
            new_risk = max(0.1, old_risk - risk_reduction)
            reason = "Poor performance - becoming more conservative"
        elif recent_performance["win_rate"] > 0.7:
            # Winning frequently - become more aggressive
            risk_increase = self.learning_rate * (recent_performance["win_rate"] - 0.5)
            new_risk = min(0.9, old_risk + risk_increase)
            reason = "Good performance - becoming more aggressive"
        else:
            # Moderate performance - slight adjustment toward balance
            if old_risk > 0.5:
                new_risk = old_risk - self.learning_rate * 0.1
                reason = "Moving toward balance"
            else:
                new_risk = old_risk + self.learning_rate * 0.1
                reason = "Moving toward balance"
        
        # Apply adaptation if significant
        if abs(new_risk - old_risk) >= self.adaptation_threshold:
            self.current_risk_ratio = new_risk
            self.base_ai.risk_ratio = new_risk
            
            # Record adaptation
            outcome = "Applied" if abs(new_risk - old_risk) >= self.adaptation_threshold else "Skipped"
            self.memory.record_risk_adjustment(old_risk, new_risk, reason, outcome)
            
            print(f"🧠 {self.name} adapted: {old_risk:.2f} → {new_risk:.2f} ({reason})")
        
        # Update strategy confidence
        self.strategy_confidence = 1.0 - abs(0.5 - recent_performance["win_rate"])
        self.last_adjustment_game = self.memory.games_played
    
    def _reset_game_tracking(self) -> None:
        """Reset tracking for the current game."""
        self.current_game_tricks_won = 0
        self.current_game_tricks_lost = 0
        self.current_game_trump_call = False
        self.current_game_trump_call_won = False
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get a summary of the AI's adaptation."""
        return {
            "name": self.name,
            "base_type": self.base_ai_type,
            "current_risk_ratio": self.current_risk_ratio,
            "initial_risk_ratio": self.initial_risk_ratio,
            "strategy_confidence": self.strategy_confidence,
            "current_strategy": self.current_strategy,
            "learning_rate": self.learning_rate,
            "memory": {
                "games_played": self.memory.games_played,
                "win_rate": self.memory.calculate_win_rate(),
                "trick_efficiency": self.memory.calculate_trick_efficiency(),
                "total_adjustments": len(self.memory.risk_adjustments)
            }
        }
    
    def reset_learning(self) -> None:
        """Reset all learning and return to initial state."""
        self.memory = AdaptiveMemory()
        self.current_risk_ratio = self.initial_risk_ratio
        self.base_ai.risk_ratio = self.initial_risk_ratio
        self.strategy_confidence = 0.5
        self.current_strategy = "balanced"
        self.last_adjustment_game = 0
        self._reset_game_tracking()
        
        # Remove memory file
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
        
        print(f"🔄 {self.name} learning reset to initial state")


class AdaptiveAIFactory:
    """Factory for creating adaptive AI players."""
    
    @staticmethod
    def create_adaptive_ai(name: str, base_ai_type: str = "balanced", 
                          initial_risk_ratio: float = 0.5, 
                          learning_rate: float = 0.1,
                          memory_dir: str = "adaptive_ai_memory") -> AdaptiveAI:
        """Create an adaptive AI player."""
        # Ensure memory directory exists
        os.makedirs(memory_dir, exist_ok=True)
        
        # Create memory file path
        memory_file = os.path.join(memory_dir, f"{name}_memory.json")
        
        return AdaptiveAI(
            name=name,
            base_ai_type=base_ai_type,
            initial_risk_ratio=initial_risk_ratio,
            learning_rate=learning_rate,
            memory_file=memory_file
        )
    
    @staticmethod
    def create_balanced_team() -> List[AdaptiveAI]:
        """Create a balanced team of adaptive AI players."""
        return [
            AdaptiveAIFactory.create_adaptive_ai("Alice", "conservative", 0.3, 0.15),
            AdaptiveAIFactory.create_adaptive_ai("Bob", "balanced", 0.5, 0.1),
            AdaptiveAIFactory.create_adaptive_ai("Charlie", "aggressive", 0.7, 0.15),
            AdaptiveAIFactory.create_adaptive_ai("David", "opportunistic", 0.6, 0.1)
        ]
    
    @staticmethod
    def create_learning_team() -> List[AdaptiveAI]:
        """Create a team designed for learning and adaptation."""
        return [
            AdaptiveAIFactory.create_adaptive_ai("Learner1", "balanced", 0.5, 0.2),
            AdaptiveAIFactory.create_adaptive_ai("Learner2", "balanced", 0.5, 0.2),
            AdaptiveAIFactory.create_adaptive_ai("Learner3", "balanced", 0.5, 0.2),
            AdaptiveAIFactory.create_adaptive_ai("Learner4", "balanced", 0.5, 0.2)
        ] 