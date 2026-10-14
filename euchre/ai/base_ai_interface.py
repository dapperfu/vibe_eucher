"""
Abstract Base Interface for AI Decision Making

This module defines the abstract interface that all AI implementations must follow,
whether they are traditional rule-based AI or modern M-Series neural AI models.

The interface provides a clean abstraction for all Euchre game decisions,
making AI implementations completely interchangeable.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-13
Context: Creating abstract AI interface for unified AI factory system
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

from ..models import Card, Suit, Rank, Player, PlayerType
from ..core.game_state import GameState


class DecisionType(Enum):
    """Types of decisions an AI can make."""
    ORDER_UP = "order_up"
    PASS = "pass"
    CALL_TRUMP = "call_trump"
    PLAY_CARD = "play_card"
    DISCARD = "discard"


@dataclass
class GameContext:
    """Complete game context for AI decision making."""
    
    # Player information
    hand: List[Card]
    position: int  # 0-3 (0=Alice, 1=Bob, 2=Charlie, 3=David)
    is_dealer: bool
    partner_position: int
    
    # Game state
    flipped_card: Optional[Card]
    trump_suit: Optional[Suit]
    current_trick: List[Tuple[int, Card]]  # (position, card)
    trick_suit: Optional[Suit]
    
    # Score and game progress
    team1_score: int
    team2_score: int
    tricks_won_team1: int
    tricks_won_team2: int
    current_trick_number: int
    
    # Partner information
    partner_is_dealer: bool
    partner_hand_size: int
    
    # Opponent information
    opponent1_hand_size: int
    opponent2_hand_size: int
    
    # Game history (optional, for advanced AI)
    game_history: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            'hand': [card.to_dict() for card in self.hand],
            'position': self.position,
            'is_dealer': self.is_dealer,
            'partner_position': self.partner_position,
            'flipped_card': self.flipped_card.to_dict() if self.flipped_card else None,
            'trump_suit': self.trump_suit.name if self.trump_suit else None,
            'current_trick': [(pos, card.to_dict()) for pos, card in self.current_trick],
            'trick_suit': self.trick_suit.name if self.trick_suit else None,
            'team1_score': self.team1_score,
            'team2_score': self.team2_score,
            'tricks_won_team1': self.tricks_won_team1,
            'tricks_won_team2': self.tricks_won_team2,
            'current_trick_number': self.current_trick_number,
            'partner_is_dealer': self.partner_is_dealer,
            'partner_hand_size': self.partner_hand_size,
            'opponent1_hand_size': self.opponent1_hand_size,
            'opponent2_hand_size': self.opponent2_hand_size
        }


@dataclass
class DecisionResult:
    """Result of an AI decision."""
    
    decision_type: DecisionType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'decision_type': self.decision_type.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'metadata': self.metadata or {}
        }


class BaseAIInterface(ABC):
    """
    Abstract base class for all AI implementations.
    
    This interface defines the contract that all AI implementations must follow,
    whether they are traditional rule-based AI or modern neural AI models.
    
    The interface provides methods for all major Euchre decisions:
    - Trump selection (order up, pass, call trump)
    - Card playing (lead, follow, discard)
    - Strategic evaluation
    """
    
    def __init__(self, name: str, risk_profile: float = 0.5):
        """
        Initialize the AI interface.
        
        Parameters
        ----------
        name : str
            The AI's name
        risk_profile : float
            Risk tolerance from 0.0 (conservative) to 1.0 (aggressive)
        """
        self.name = name
        self.risk_profile = max(0.0, min(1.0, risk_profile))
        self.decision_history: List[DecisionResult] = []
    
    @abstractmethod
    def should_order_up(self, context: GameContext) -> DecisionResult:
        """
        Decide whether to order up the flipped card as trump.
        
        Parameters
        ----------
        context : GameContext
            Complete game context including hand, position, etc.
            
        Returns
        -------
        DecisionResult
            Decision with confidence and reasoning
        """
        pass
    
    @abstractmethod
    def should_call_trump(self, context: GameContext) -> DecisionResult:
        """
        Decide whether to call trump if everyone passes on the flipped card.
        
        Parameters
        ----------
        context : GameContext
            Complete game context
            
        Returns
        -------
        DecisionResult
            Decision with confidence and reasoning
        """
        pass
    
    @abstractmethod
    def select_trump_suit(self, context: GameContext) -> DecisionResult:
        """
        Select which suit to call as trump.
        
        Parameters
        ----------
        context : GameContext
            Complete game context
            
        Returns
        -------
        DecisionResult
            Decision with confidence and reasoning
        """
        pass
    
    @abstractmethod
    def play_card(self, context: GameContext) -> DecisionResult:
        """
        Decide which card to play in the current trick.
        
        Parameters
        ----------
        context : GameContext
            Complete game context
            
        Returns
        -------
        DecisionResult
            Decision with confidence and reasoning
        """
        pass
    
    @abstractmethod
    def discard_card(self, context: GameContext) -> DecisionResult:
        """
        Decide which card to discard when partner calls trump.
        
        Parameters
        ----------
        context : GameContext
            Complete game context
            
        Returns
        -------
        DecisionResult
            Decision with confidence and reasoning
        """
        pass
    
    def evaluate_hand_strength(self, context: GameContext) -> float:
        """
        Evaluate the strength of the current hand.
        
        Parameters
        ----------
        context : GameContext
            Complete game context
            
        Returns
        -------
        float
            Hand strength from 0.0 (weak) to 1.0 (strong)
        """
        # Default implementation - can be overridden by subclasses
        hand = context.hand
        trump_suit = context.trump_suit
        
        if not hand:
            return 0.0
        
        # Basic hand strength calculation
        strength = 0.0
        
        for card in hand:
            # High cards are worth more
            if card.rank in [Rank.ACE, Rank.KING, Rank.QUEEN]:
                strength += 0.2
            elif card.rank == Rank.JACK:
                strength += 0.15
            elif card.rank == Rank.TEN:
                strength += 0.1
            elif card.rank == Rank.NINE:
                strength += 0.05
            
            # Trump cards are worth more
            if trump_suit and (card.suit == trump_suit or 
                              (card.rank == Rank.JACK and card.suit == self._get_left_bower_suit(trump_suit))):
                strength += 0.3
        
        # Normalize to 0.0-1.0 range
        return min(1.0, strength / len(hand))
    
    def _get_left_bower_suit(self, trump_suit: Suit) -> Suit:
        """Get the suit of the left bower for a given trump suit."""
        left_bower_map = {
            Suit.HEARTS: Suit.DIAMONDS,
            Suit.DIAMONDS: Suit.HEARTS,
            Suit.CLUBS: Suit.SPADES,
            Suit.SPADES: Suit.CLUBS
        }
        return left_bower_map.get(trump_suit, trump_suit)
    
    def record_decision(self, decision: DecisionResult):
        """Record a decision for analysis and learning."""
        self.decision_history.append(decision)
    
    def get_decision_history(self) -> List[DecisionResult]:
        """Get the history of decisions made by this AI."""
        return self.decision_history.copy()
    
    def clear_history(self):
        """Clear the decision history."""
        self.decision_history.clear()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics based on decision history."""
        if not self.decision_history:
            return {}
        
        total_decisions = len(self.decision_history)
        avg_confidence = sum(d.confidence for d in self.decision_history) / total_decisions
        
        # Count decision types
        decision_counts = {}
        for decision in self.decision_history:
            decision_type = decision.decision_type.value
            decision_counts[decision_type] = decision_counts.get(decision_type, 0) + 1
        
        return {
            'total_decisions': total_decisions,
            'average_confidence': avg_confidence,
            'decision_distribution': decision_counts
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, risk={self.risk_profile:.2f})"
    
    def __repr__(self) -> str:
        return self.__str__() 