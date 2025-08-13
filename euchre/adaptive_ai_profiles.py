"""Adaptive AI Profiles with Dynamic Risk Ratio Adjustment.

This module provides AI profiles that can adapt their risk ratios based on:
- Game state (winning/losing, team sets, etc.)
- Hand strength
- Trump situation
- Partner performance
- Historical performance
"""

import random
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .models import Card, Suit, Rank, Player, PlayerType
from .ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI


class GameState(Enum):
    """Different game states that affect risk assessment."""
    WINNING = "winning"           # Team is ahead
    LOSING = "losing"             # Team is behind
    CRITICAL = "critical"         # Need to win this round
    SAFE = "safe"                 # Can afford to be conservative
    DESPERATE = "desperate"       # Must take risks to catch up


class HandStrength(Enum):
    """Assessment of hand strength."""
    EXCELLENT = "excellent"       # Very strong hand
    GOOD = "good"                 # Strong hand
    AVERAGE = "average"           # Moderate hand
    WEAK = "weak"                 # Weak hand
    POOR = "poor"                 # Very weak hand


@dataclass
class AdaptiveContext:
    """Context information for adaptive decision making."""
    game_state: GameState
    hand_strength: HandStrength
    team_score: int
    opponent_score: int
    round_number: int
    tricks_won: int
    partner_tricks_won: int
    is_partner_dealing: bool
    trump_suit: Optional[Suit]
    top_card: Optional[Card]
    historical_performance: float  # Average score over last N games


class AdaptiveRiskManager:
    """Manages dynamic risk ratio adjustment based on game context."""
    
    def __init__(self, base_risk_ratio: float = 0.5):
        """Initialize the risk manager.
        
        Parameters
        ----------
        base_risk_ratio : float
            Base risk ratio (0.0 = conservative, 1.0 = aggressive)
        """
        self.base_risk_ratio = base_risk_ratio
        self.current_risk_ratio = base_risk_ratio
        self.adaptation_history: List[Tuple[float, float, str]] = []  # (old, new, reason)
        
        # Risk adjustment factors
        self.game_state_multipliers = {
            GameState.WINNING: 0.8,      # Be more conservative when winning
            GameState.LOSING: 1.3,       # Take more risks when losing
            GameState.CRITICAL: 1.5,     # Take significant risks in critical situations
            GameState.SAFE: 0.7,         # Be very conservative when safe
            GameState.DESPERATE: 1.8     # Take extreme risks when desperate
        }
        
        self.hand_strength_multipliers = {
            HandStrength.EXCELLENT: 1.4, # Play aggressively with excellent hands
            HandStrength.GOOD: 1.2,      # Play somewhat aggressively with good hands
            HandStrength.AVERAGE: 1.0,   # Normal play with average hands
            HandStrength.WEAK: 0.8,      # Play conservatively with weak hands
            HandStrength.POOR: 0.6       # Play very conservatively with poor hands
        }
        
        self.score_difference_multipliers = {
            "large_lead": 0.7,           # Large lead (5+ points)
            "moderate_lead": 0.85,       # Moderate lead (2-4 points)
            "close_game": 1.0,           # Close game (±1 point)
            "moderate_deficit": 1.2,     # Moderate deficit (2-4 points)
            "large_deficit": 1.5         # Large deficit (5+ points)
        }
        
    def calculate_adaptive_risk_ratio(self, context: AdaptiveContext) -> float:
        """Calculate the adaptive risk ratio based on current context.
        
        Parameters
        ----------
        context : AdaptiveContext
            Current game context
            
        Returns
        -------
        float
            Adjusted risk ratio (0.0 to 1.0)
        """
        # Start with base risk ratio
        adjusted_risk = self.base_risk_ratio
        
        # Apply game state multiplier
        game_state_mult = self.game_state_multipliers.get(context.game_state, 1.0)
        adjusted_risk *= game_state_mult
        
        # Apply hand strength multiplier
        hand_strength_mult = self.hand_strength_multipliers.get(context.hand_strength, 1.0)
        adjusted_risk *= hand_strength_mult
        
        # Apply score difference multiplier
        score_diff = context.team_score - context.opponent_score
        if score_diff >= 5:
            score_mult = self.score_difference_multipliers["large_lead"]
        elif score_diff >= 2:
            score_mult = self.score_difference_multipliers["moderate_lead"]
        elif score_diff >= -1:
            score_mult = self.score_difference_multipliers["close_game"]
        elif score_diff >= -4:
            score_mult = self.score_difference_multipliers["moderate_deficit"]
        else:
            score_mult = self.score_difference_multipliers["large_deficit"]
        
        adjusted_risk *= score_mult
        
        # Apply round-based adjustments
        if context.round_number >= 8:  # Late game
            if context.team_score < 7:  # Need to catch up
                adjusted_risk *= 1.3
            else:  # Playing to win
                adjusted_risk *= 0.8
        
        # Apply partner performance adjustment
        if context.partner_tricks_won == 0 and context.tricks_won == 0:
            # Both partners struggling - take more risks
            adjusted_risk *= 1.2
        
        # Apply trump situation adjustment
        if context.trump_suit and context.top_card:
            if context.is_partner_dealing:
                # Partner dealing - can be more aggressive
                adjusted_risk *= 1.1
            else:
                # Opponent dealing - be more conservative
                adjusted_risk *= 0.9
        
        # Apply historical performance adjustment
        if context.historical_performance < -5:
            # Poor historical performance - take fewer risks
            adjusted_risk *= 0.8
        elif context.historical_performance > 5:
            # Good historical performance - can afford more risks
            adjusted_risk *= 1.1
        
        # Clamp to valid range [0.0, 1.0]
        adjusted_risk = max(0.0, min(1.0, adjusted_risk))
        
        # Record adaptation
        if abs(adjusted_risk - self.current_risk_ratio) > 0.05:  # Only record significant changes
            reason = f"Game state: {context.game_state.value}, Hand: {context.hand_strength.value}, Score diff: {score_diff}"
            self.adaptation_history.append((self.current_risk_ratio, adjusted_risk, reason))
            self.current_risk_ratio = adjusted_risk
        
        return adjusted_risk
    
    def get_adaptation_summary(self) -> Dict:
        """Get a summary of risk ratio adaptations.
        
        Returns
        -------
        Dict
            Summary of adaptations made
        """
        if not self.adaptation_history:
            return {"adaptations": 0, "changes": []}
        
        return {
            "total_adaptations": len(self.adaptation_history),
            "base_risk_ratio": self.base_risk_ratio,
            "final_risk_ratio": self.current_risk_ratio,
            "adaptation_range": {
                "min": min(change[1] for change in self.adaptation_history),
                "max": max(change[1] for change in self.adaptation_history)
            },
            "recent_changes": self.adaptation_history[-5:]  # Last 5 changes
        }


class AdaptiveAIProfile:
    """Base class for adaptive AI profiles."""
    
    def __init__(self, name: str, base_risk_ratio: float = 0.5):
        """Initialize adaptive AI profile.
        
        Parameters
        ----------
        name : str
            Name of the AI player
        base_risk_ratio : float
            Base risk ratio (0.0 = conservative, 1.0 = aggressive)
        """
        self.name = name
        self.risk_manager = AdaptiveRiskManager(base_risk_ratio)
        self.performance_history: List[float] = []
        self.game_contexts: List[AdaptiveContext] = []
        
    def assess_hand_strength(self, hand: List[Card], trump_suit: Optional[Suit] = None) -> HandStrength:
        """Assess the strength of a hand.
        
        Parameters
        ----------
        hand : List[Card]
            Cards in hand
        trump_suit : Optional[Suit]
            Current trump suit (if any)
            
        Returns
        -------
        HandStrength
            Assessment of hand strength
        """
        if not hand:
            return HandStrength.POOR
        
        # Calculate base hand value
        total_value = sum(card.rank.value for card in hand)
        
        # Trump bonus
        trump_bonus = 0
        if trump_suit:
            trump_cards = [card for card in hand if card.suit == trump_suit or 
                          (card.rank == Rank.JACK and 
                           ((trump_suit == Suit.HEARTS and card.suit == Suit.DIAMONDS) or
                            (trump_suit == Suit.DIAMONDS and card.suit == Suit.HEARTS) or
                            (trump_suit == Suit.CLUBS and card.suit == Suit.SPADES) or
                            (trump_suit == Suit.SPADES and card.suit == Suit.CLUBS)))]
            trump_bonus = len(trump_cards) * 3
        
        # Suit diversity bonus
        suits = set(card.suit for card in hand)
        suit_bonus = len(suits) * 2
        
        # High card bonus
        high_cards = [card for card in hand if card.rank.value >= 11]
        high_card_bonus = len(high_cards) * 2
        
        total_score = total_value + trump_bonus + suit_bonus + high_card_bonus
        
        # Classify hand strength
        if total_score >= 45:
            return HandStrength.EXCELLENT
        elif total_score >= 35:
            return HandStrength.GOOD
        elif total_score >= 25:
            return HandStrength.AVERAGE
        elif total_score >= 15:
            return HandStrength.WEAK
        else:
            return HandStrength.POOR
    
    def assess_game_state(self, team_score: int, opponent_score: int, round_number: int) -> GameState:
        """Assess the current game state.
        
        Parameters
        ----------
        team_score : int
            Current team score
        opponent_score : int
            Current opponent score
        round_number : int
            Current round number
            
        Returns
        -------
        GameState
            Current game state
        """
        score_diff = team_score - opponent_score
        
        if round_number >= 8:  # Late game
            if score_diff >= 3:
                return GameState.SAFE
            elif score_diff <= -3:
                return GameState.DESPERATE
            elif score_diff >= 1:
                return GameState.WINNING
            else:
                return GameState.CRITICAL
        else:  # Early/mid game
            if score_diff >= 2:
                return GameState.WINNING
            elif score_diff <= -2:
                return GameState.LOSING
            else:
                return GameState.SAFE
    
    def create_context(self, hand: List[Card], team_score: int, opponent_score: int, 
                      round_number: int, tricks_won: int, partner_tricks_won: int,
                      is_partner_dealing: bool, trump_suit: Optional[Suit] = None,
                      top_card: Optional[Card] = None) -> AdaptiveContext:
        """Create adaptive context for decision making.
        
        Parameters
        ----------
        hand : List[Card]
            Current hand
        team_score : int
            Current team score
        opponent_score : int
            Current opponent score
        round_number : int
            Current round number
        tricks_won : int
            Tricks won by this player
        partner_tricks_won : int
            Tricks won by partner
        is_partner_dealing : bool
            Whether partner is dealing
        trump_suit : Optional[Suit]
            Current trump suit
        top_card : Optional[Card]
            Top card for trump selection
            
        Returns
        -------
        AdaptiveContext
            Context for adaptive decision making
        """
        hand_strength = self.assess_hand_strength(hand, trump_suit)
        game_state = self.assess_game_state(team_score, opponent_score, round_number)
        
        # Calculate historical performance (average of last 10 games)
        historical_performance = 0.0
        if self.performance_history:
            recent_performance = self.performance_history[-10:]
            historical_performance = sum(recent_performance) / len(recent_performance)
        
        context = AdaptiveContext(
            game_state=game_state,
            hand_strength=hand_strength,
            team_score=team_score,
            opponent_score=opponent_score,
            round_number=round_number,
            tricks_won=tricks_won,
            partner_tricks_won=partner_tricks_won,
            is_partner_dealing=is_partner_dealing,
            trump_suit=trump_suit,
            top_card=top_card,
            historical_performance=historical_performance
        )
        
        self.game_contexts.append(context)
        return context
    
    def get_adaptive_risk_ratio(self, context: AdaptiveContext) -> float:
        """Get the adaptive risk ratio for current context.
        
        Parameters
        ----------
        context : AdaptiveContext
            Current game context
            
        Returns
        -------
        float
            Adaptive risk ratio
        """
        return self.risk_manager.calculate_adaptive_risk_ratio(context)
    
    def record_performance(self, score: float) -> None:
        """Record performance for historical analysis.
        
        Parameters
        ----------
        score : float
            Performance score from the game
        """
        self.performance_history.append(score)
        
        # Keep only last 50 games for memory efficiency
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary.
        
        Returns
        -------
        Dict
            Performance summary
        """
        if not self.performance_history:
            return {"games_played": 0, "average_score": 0.0}
        
        return {
            "games_played": len(self.performance_history),
            "average_score": sum(self.performance_history) / len(self.performance_history),
            "recent_performance": self.performance_history[-10:],
            "adaptation_summary": self.risk_manager.get_adaptation_summary()
        }


class AdaptiveConservativeAI(AdaptiveAIProfile):
    """Adaptive Conservative AI that adjusts risk based on game state."""
    
    def __init__(self, name: str, base_risk_ratio: float = 0.2):
        """Initialize adaptive conservative AI.
        
        Parameters
        ----------
        name : str
            Name of the AI player
        base_risk_ratio : float
            Base risk ratio (default: 0.2 for conservative)
        """
        super().__init__(name, base_risk_ratio)
        self.base_ai = ConservativeAI(name, base_risk_ratio)
        
        # Initialize player attributes
        self.hand: List[Card] = []
        self.tricks_won: int = 0
        self.player_type = PlayerType.AI
        
    def clear_hand(self) -> None:
        """Clear the player's hand."""
        self.hand.clear()
        self.tricks_won = 0
        
    def add_card(self, card: Card) -> None:
        """Add a card to the player's hand.
        
        Parameters
        ----------
        card : Card
            Card to add to hand
        """
        self.hand.append(card)
        
    def remove_card(self, card: Card) -> None:
        """Remove a card from the player's hand.
        
        Parameters
        ----------
        card : Card
            Card to remove from hand
        """
        if card in self.hand:
            self.hand.remove(card)
            
    def get_hand(self) -> List[Card]:
        """Get the player's current hand.
        
        Returns
        -------
        List[Card]
            List of cards in hand
        """
        return self.hand.copy()
        
    def has_suit(self, suit: Suit) -> bool:
        """Check if the player has any cards of the specified suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to check for
            
        Returns
        -------
        bool
            True if the player has cards of the specified suit
        """
        return any(card.suit == suit for card in self.hand)
        
    def get_cards_of_suit(self, suit: Suit) -> List[Card]:
        """Get all cards of the specified suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to get cards for
            
        Returns
        -------
        List[Card]
            List of cards of the specified suit
        """
        return [card for card in self.hand if card.suit == suit]
        
    def get_playable_cards(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> List[Card]:
        """Get cards that can be played following suit rules.
        
        Parameters
        ----------
        lead_suit : Optional[Suit]
            The suit that was led
        trump_suit : Optional[Suit]
            The current trump suit
            
        Returns
        -------
        List[Card]
            List of playable cards
        """
        if not lead_suit:
            return self.hand.copy()
            
        # If leading, can play any card
        if not any(hasattr(p, 'has_suit') and p.has_suit(lead_suit) for p in [self]):
            return self.hand.copy()
            
        # Must follow suit if possible
        if self.has_suit(lead_suit):
            return self.get_cards_of_suit(lead_suit)
        else:
            return self.hand.copy()
            
    def is_trump_caller(self) -> bool:
        """Check if this player called trump.
        
        Returns
        -------
        bool
            True if this player called trump
        """
        # This would need to be set by the game logic
        return hasattr(self, '_is_trump_caller') and self._is_trump_caller
        
    def set_trump_caller(self, is_caller: bool) -> None:
        """Set whether this player called trump.
        
        Parameters
        ----------
        is_caller : bool
            Whether this player called trump
        """
        self._is_trump_caller = is_caller
        
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Determine if the AI should order up the top card.
        
        Parameters
        ----------
        top_card : Card
            The top card that could be ordered up
        is_partner_dealing : bool
            Whether the AI's partner is dealing
            
        Returns
        -------
        bool
            True if the AI should order up the card
        """
        # Get current adaptive risk ratio
        adaptive_risk = self.risk_manager.current_risk_ratio
        
        # Adjust the base AI's risk ratio temporarily
        original_risk = self.base_ai.risk_ratio
        self.base_ai.risk_ratio = adaptive_risk
        
        try:
            result = self.base_ai.should_order_up(top_card, is_partner_dealing)
            return result
        finally:
            # Restore original risk ratio
            self.base_ai.risk_ratio = original_risk
            
    def choose_trump_suit(self, top_card: Card) -> Optional[Suit]:
        """Choose a trump suit when the top card is rejected.
        
        Parameters
        ----------
        top_card : Card
            The top card that was rejected
            
        Returns
        -------
        Optional[Suit]
            The chosen trump suit or None to pass
        """
        # Delegate to base AI
        return self.base_ai.choose_trump_suit(top_card)
        
    def choose_card(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit], 
                   played_cards: List[Card], current_trick: List[Tuple[str, Card]]) -> Card:
        """Choose a card to play.
        
        Parameters
        ----------
        lead_suit : Optional[Suit]
            The suit that was led
        trump_suit : Optional[Suit]
            The current trump suit
        played_cards : List[Card]
            Cards already played in this trick
        current_trick : List[Tuple[str, Card]]
            Current trick with player names and cards
            
        Returns
        -------
        Card
            The card to play
        """
        # Delegate to base AI
        return self.base_ai.choose_card(lead_suit, trump_suit, played_cards, current_trick)


class AdaptiveBalancedAI(AdaptiveAIProfile):
    """Adaptive Balanced AI that adjusts risk based on game state."""
    
    def __init__(self, name: str, base_risk_ratio: float = 0.5):
        """Initialize adaptive balanced AI.
        
        Parameters
        ----------
        name : str
            Name of the AI player
        base_risk_ratio : float
            Base risk ratio (default: 0.5 for balanced)
        """
        super().__init__(name, base_risk_ratio)
        self.base_ai = BalancedAI(name, base_risk_ratio)
        
        # Initialize player attributes
        self.hand: List[Card] = []
        self.tricks_won: int = 0
        self.player_type = PlayerType.AI
        
    def clear_hand(self) -> None:
        """Clear the player's hand."""
        self.hand.clear()
        self.tricks_won = 0
        
    def add_card(self, card: Card) -> None:
        """Add a card to the player's hand.
        
        Parameters
        ----------
        card : Card
            Card to add to hand
        """
        self.hand.append(card)
        
    def remove_card(self, card: Card) -> None:
        """Remove a card from the player's hand.
        
        Parameters
        ----------
        card : Card
            Card to remove from hand
        """
        if card in self.hand:
            self.hand.remove(card)
            
    def get_hand(self) -> List[Card]:
        """Get the player's current hand.
        
        Returns
        -------
        List[Card]
            List of cards in hand
        """
        return self.hand.copy()
        
    def has_suit(self, suit: Suit) -> bool:
        """Check if the player has any cards of the specified suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to check for
            
        Returns
        -------
        bool
            True if the player has cards of the specified suit
        """
        return any(card.suit == suit for card in self.hand)
        
    def get_cards_of_suit(self, suit: Suit) -> List[Card]:
        """Get all cards of the specified suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to get cards for
            
        Returns
        -------
        List[Card]
            List of cards of the specified suit
        """
        return [card for card in self.hand if card.suit == suit]
        
    def get_playable_cards(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> List[Card]:
        """Get cards that can be played following suit rules.
        
        Parameters
        ----------
        lead_suit : Optional[Suit]
            The suit that was led
        trump_suit : Optional[Suit]
            The current trump suit
            
        Returns
        -------
        List[Card]
            List of playable cards
        """
        if not lead_suit:
            return self.hand.copy()
            
        # If leading, can play any card
        if not any(hasattr(p, 'has_suit') and p.has_suit(lead_suit) for p in [self]):
            return self.hand.copy()
            
        # Must follow suit if possible
        if self.has_suit(lead_suit):
            return self.get_cards_of_suit(lead_suit)
        else:
            return self.hand.copy()
            
    def is_trump_caller(self) -> bool:
        """Check if this player called trump.
        
        Returns
        -------
        bool
            True if this player called trump
        """
        # This would need to be set by the game logic
        return hasattr(self, '_is_trump_caller') and self._is_trump_caller
        
    def set_trump_caller(self, is_caller: bool) -> None:
        """Set whether this player called trump.
        
        Parameters
        ----------
        is_caller : bool
            Whether this player called trump
        """
        self._is_trump_caller = is_caller
        
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Determine if the AI should order up the top card.
        
        Parameters
        ----------
        top_card : Card
            The top card that could be ordered up
        is_partner_dealing : bool
            Whether the AI's partner is dealing
            
        Returns
        -------
        bool
            True if the AI should order up the card
        """
        # Similar to other adaptive AIs
        adaptive_risk = self.risk_manager.current_risk_ratio
        original_risk = self.base_ai.risk_ratio
        self.base_ai.risk_ratio = adaptive_risk
        
        try:
            result = self.base_ai.should_order_up(top_card, is_partner_dealing)
            return result
        finally:
            self.base_ai.risk_ratio = original_risk
            
    def choose_trump_suit(self, top_card: Card) -> Optional[Suit]:
        """Choose a trump suit when the top card is rejected.
        
        Parameters
        ----------
        top_card : Card
            The top card that was rejected
            
        Returns
        -------
        Optional[Suit]
            The chosen trump suit or None to pass
        """
        # Delegate to base AI
        return self.base_ai.choose_trump_suit(top_card)
        
    def choose_card(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit], 
                   played_cards: List[Card], current_trick: List[Tuple[str, Card]]) -> Card:
        """Choose a card to play.
        
        Parameters
        ----------
        lead_suit : Optional[Suit]
            The suit that was led
        trump_suit : Optional[Suit]
            The current trump suit
        played_cards : List[Card]
            Cards already played in this trick
        current_trick : List[Tuple[str, Card]]
            Current trick with player names and cards
            
        Returns
        -------
        Card
            The card to play
        """
        # Delegate to base AI
        return self.base_ai.choose_card(lead_suit, trump_suit, played_cards, current_trick)


class AdaptiveAggressiveAI(AdaptiveAIProfile):
    """Adaptive Aggressive AI that adjusts risk based on game state."""
    
    def __init__(self, name: str, base_risk_ratio: float = 0.8):
        """Initialize adaptive aggressive AI.
        
        Parameters
        ----------
        name : str
            Name of the AI player
        base_risk_ratio : float
            Base risk ratio (default: 0.8 for aggressive)
        """
        super().__init__(name, base_risk_ratio)
        self.base_ai = AggressiveAI(name, base_risk_ratio)
        
        # Initialize player attributes
        self.hand: List[Card] = []
        self.tricks_won: int = 0
        self.player_type = PlayerType.AI
        
    def clear_hand(self) -> None:
        """Clear the player's hand."""
        self.hand.clear()
        self.tricks_won = 0
        
    def add_card(self, card: Card) -> None:
        """Add a card to the player's hand.
        
        Parameters
        ----------
        card : Card
            Card to add to hand
        """
        self.hand.append(card)
        
    def remove_card(self, card: Card) -> None:
        """Remove a card from the player's hand.
        
        Parameters
        ----------
        card : Card
            Card to remove from hand
        """
        if card in self.hand:
            self.hand.remove(card)
            
    def get_hand(self) -> List[Card]:
        """Get the player's current hand.
        
        Returns
        -------
        List[Card]
            List of cards in hand
        """
        return self.hand.copy()
        
    def has_suit(self, suit: Suit) -> bool:
        """Check if the player has any cards of the specified suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to check for
            
        Returns
        -------
        bool
            True if the player has cards of the specified suit
        """
        return any(card.suit == suit for card in self.hand)
        
    def get_cards_of_suit(self, suit: Suit) -> List[Card]:
        """Get all cards of the specified suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to get cards for
            
        Returns
        -------
        List[Card]
            List of cards of the specified suit
        """
        return [card for card in self.hand if card.suit == suit]
        
    def get_playable_cards(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> List[Card]:
        """Get cards that can be played following suit rules.
        
        Parameters
        ----------
        lead_suit : Optional[Suit]
            The suit that was led
        trump_suit : Optional[Suit]
            The current trump suit
            
        Returns
        -------
        List[Card]
            List of playable cards
        """
        if not lead_suit:
            return self.hand.copy()
            
        # If leading, can play any card
        if not any(hasattr(p, 'has_suit') and p.has_suit(lead_suit) for p in [self]):
            return self.hand.copy()
            
        # Must follow suit if possible
        if self.has_suit(lead_suit):
            return self.get_cards_of_suit(lead_suit)
        else:
            return self.hand.copy()
            
    def is_trump_caller(self) -> bool:
        """Check if this player called trump.
        
        Returns
        -------
        bool
            True if this player called trump
        """
        # This would need to be set by the game logic
        return hasattr(self, '_is_trump_caller') and self._is_trump_caller
        
    def set_trump_caller(self, is_caller: bool) -> None:
        """Set whether this player called trump.
        
        Parameters
        ----------
        is_caller : bool
            Whether this player called trump
        """
        self._is_trump_caller = is_caller
        
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Determine if the AI should order up the top card.
        
        Parameters
        ----------
        top_card : Card
            The top card that could be ordered up
        is_partner_dealing : bool
            Whether the AI's partner is dealing
            
        Returns
        -------
        bool
            True if the AI should order up the card
        """
        # Similar to other adaptive AIs
        adaptive_risk = self.risk_manager.current_risk_ratio
        original_risk = self.base_ai.risk_ratio
        self.base_ai.risk_ratio = adaptive_risk
        
        try:
            result = self.base_ai.should_order_up(top_card, is_partner_dealing)
            return result
        finally:
            self.base_ai.risk_ratio = original_risk
            
    def choose_trump_suit(self, top_card: Card) -> Optional[Suit]:
        """Choose a trump suit when the top card is rejected.
        
        Parameters
        ----------
        top_card : Card
            The top card that was rejected
            
        Returns
        -------
        Optional[Suit]
            The chosen trump suit or None to pass
        """
        # Delegate to base AI
        return self.base_ai.choose_trump_suit(top_card)
        
    def choose_card(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit], 
                   played_cards: List[Card], current_trick: List[Tuple[str, Card]]) -> Card:
        """Choose a card to play.
        
        Parameters
        ----------
        lead_suit : Optional[Suit]
            The suit that was led
        trump_suit : Optional[Suit]
            The current trump suit
        played_cards : List[Card]
            Cards already played in this trick
        current_trick : List[Tuple[str, Card]]
            Current trick with player names and cards
            
        Returns
        -------
        Card
            The card to play
        """
        # Delegate to base AI
        return self.base_ai.choose_card(lead_suit, trump_suit, played_cards, current_trick)


class AdaptiveOpportunisticAI(AdaptiveAIProfile):
    """Adaptive Opportunistic AI that adjusts risk based on game state."""
    
    def __init__(self, name: str, base_risk_ratio: float = 0.6):
        """Initialize adaptive opportunistic AI.
        
        Parameters
        ----------
        name : str
            Name of the AI player
        base_risk_ratio : float
            Base risk ratio (default: 0.6 for opportunistic)
        """
        super().__init__(name, base_risk_ratio)
        self.base_ai = OpportunisticAI(name, base_risk_ratio)
        
        # Initialize player attributes
        self.hand: List[Card] = []
        self.tricks_won: int = 0
        self.player_type = PlayerType.AI
        
    def clear_hand(self) -> None:
        """Clear the player's hand."""
        self.hand.clear()
        self.tricks_won = 0
        
    def add_card(self, card: Card) -> None:
        """Add a card to the player's hand.
        
        Parameters
        ----------
        card : Card
            Card to add to hand
        """
        self.hand.append(card)
        
    def remove_card(self, card: Card) -> None:
        """Remove a card from the player's hand.
        
        Parameters
        ----------
        card : Card
            Card to remove from hand
        """
        if card in self.hand:
            self.hand.remove(card)
            
    def get_hand(self) -> List[Card]:
        """Get the player's current hand.
        
        Returns
        -------
        List[Card]
            List of cards in hand
        """
        return self.hand.copy()
        
    def has_suit(self, suit: Suit) -> bool:
        """Check if the player has any cards of the specified suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to check for
            
        Returns
        -------
        bool
            True if the player has cards of the specified suit
        """
        return any(card.suit == suit for card in self.hand)
        
    def get_cards_of_suit(self, suit: Suit) -> List[Card]:
        """Get all cards of the specified suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to get cards for
            
        Returns
        -------
        List[Card]
            List of cards of the specified suit
        """
        return [card for card in self.hand if card.suit == suit]
        
    def get_playable_cards(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> List[Card]:
        """Get cards that can be played following suit rules.
        
        Parameters
        ----------
        lead_suit : Optional[Suit]
            The suit that was led
        trump_suit : Optional[Suit]
            The current trump suit
            
        Returns
        -------
        List[Card]
            List of playable cards
        """
        if not lead_suit:
            return self.hand.copy()
            
        # If leading, can play any card
        if not any(hasattr(p, 'has_suit') and p.has_suit(lead_suit) for p in [self]):
            return self.hand.copy()
            
        # Must follow suit if possible
        if self.has_suit(lead_suit):
            return self.get_cards_of_suit(lead_suit)
        else:
            return self.hand.copy()
            
    def is_trump_caller(self) -> bool:
        """Check if this player called trump.
        
        Returns
        -------
        bool
            True if this player called trump
        """
        # This would need to be set by the game logic
        return hasattr(self, '_is_trump_caller') and self._is_trump_caller
        
    def set_trump_caller(self, is_caller: bool) -> None:
        """Set whether this player called trump.
        
        Parameters
        ----------
        is_caller : bool
            Whether this player called trump
        """
        self._is_trump_caller = is_caller
        
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Determine if the AI should order up the top card.
        
        Parameters
        ----------
        top_card : Card
            The top card that could be ordered up
        is_partner_dealing : bool
            Whether the AI's partner is dealing
            
        Returns
        -------
        bool
            True if the AI should order up the card
        """
        # Similar to other adaptive AIs
        adaptive_risk = self.risk_manager.current_risk_ratio
        original_risk = self.base_ai.risk_ratio
        self.base_ai.risk_ratio = adaptive_risk
        
        try:
            result = self.base_ai.should_order_up(top_card, is_partner_dealing)
            return result
        finally:
            self.base_ai.risk_ratio = original_risk
            
    def choose_trump_suit(self, top_card: Card) -> Optional[Suit]:
        """Choose a trump suit when the top card is rejected.
        
        Parameters
        ----------
        top_card : Card
            The top card that was rejected
            
        Returns
        -------
        Optional[Suit]
            The chosen trump suit or None to pass
        """
        # Delegate to base AI
        return self.base_ai.choose_trump_suit(top_card)
        
    def choose_card(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit], 
                   played_cards: List[Card], current_trick: List[Tuple[str, Card]]) -> Card:
        """Choose a card to play.
        
        Parameters
        ----------
        lead_suit : Optional[Suit]
            The suit that was led
        trump_suit : Optional[Suit]
            The current trump suit
        played_cards : List[Card]
            Cards already played in this trick
        current_trick : List[Tuple[str, Card]]
            Current trick with player names and cards
            
        Returns
        -------
        Card
            The card to play
        """
        # Delegate to base AI
        return self.base_ai.choose_card(lead_suit, trump_suit, played_cards, current_trick)


def create_adaptive_ai_profile(name: str, ai_type: str, base_risk_ratio: float) -> AdaptiveAIProfile:
    """Factory function to create adaptive AI profiles.
    
    Parameters
    ----------
    name : str
        Name of the AI player
    ai_type : str
        Type of AI: "conservative", "balanced", "aggressive", "opportunistic"
    base_risk_ratio : float
        Base risk ratio
        
    Returns
    -------
    AdaptiveAIProfile
        Adaptive AI profile instance
    """
    ai_type = ai_type.lower()
    
    if ai_type == "conservative":
        return AdaptiveConservativeAI(name, base_risk_ratio)
    elif ai_type == "balanced":
        return AdaptiveBalancedAI(name, base_risk_ratio)
    elif ai_type == "aggressive":
        return AdaptiveAggressiveAI(name, base_risk_ratio)
    elif ai_type == "opportunistic":
        return AdaptiveOpportunisticAI(name, base_risk_ratio)
    else:
        # Default to balanced
        return AdaptiveBalancedAI(name, base_risk_ratio) 