"""
Traditional Rule-Based AI Implementation

This module implements the traditional rule-based AI logic using the BaseAIInterface.
The AI makes decisions based on hard-coded rules and heuristics rather than
neural networks or machine learning.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-13
Context: Creating traditional AI implementation using the abstract interface
"""

from typing import List, Optional, Tuple
import random

from .base_ai_interface import (
    BaseAIInterface, GameContext, DecisionResult, DecisionType
)
from ..models import Card, Suit, Rank, Player, PlayerType


class TraditionalAI(Player, BaseAIInterface):
    """
    Traditional rule-based AI implementation.
    
    This AI makes decisions based on hard-coded rules and heuristics:
    - Trump calling based on hand strength and position
    - Card playing based on trick context and partner coordination
    - Risk assessment based on score and game state
    """
    
    def __init__(self, name: str, risk_profile: float = 0.5, ai_style: str = "balanced"):
        """
        Initialize traditional AI.
        
        Parameters
        ----------
        name : str
            The AI's name
        risk_profile : float
            Risk tolerance from 0.0 (conservative) to 1.0 (aggressive)
        ai_style : str
            AI style: "aggressive", "conservative", "balanced", "opportunistic"
        """
        # Initialize as a Player first
        super().__init__(name, PlayerType.AI)
        
        # Initialize AI interface
        BaseAIInterface.__init__(self, name, risk_profile)
        
        self.ai_style = ai_style.lower()
        
        # Style-specific modifiers
        self._setup_style_modifiers()
    
    def _setup_style_modifiers(self):
        """Setup style-specific behavior modifiers."""
        if self.ai_style == "aggressive":
            self.trump_threshold = 0.4  # Lower threshold for calling trump
            self.leading_threshold = 0.3  # More likely to lead
            self.risk_multiplier = 1.5
        elif self.ai_style == "conservative":
            self.trump_threshold = 0.7  # Higher threshold for calling trump
            self.leading_threshold = 0.6  # Less likely to lead
            self.risk_multiplier = 0.5
        elif self.ai_style == "opportunistic":
            self.trump_threshold = 0.5  # Balanced threshold
            self.leading_threshold = 0.4  # Balanced leading
            self.risk_multiplier = 1.2
        else:  # balanced
            self.trump_threshold = 0.55  # Slightly above average
            self.leading_threshold = 0.45  # Slightly below average
            self.risk_multiplier = 1.0
    
    def should_order_up(self, context: GameContext) -> DecisionResult:
        """Decide whether to order up the flipped card as trump."""
        hand = context.hand
        flipped_card = context.flipped_card
        position = context.position
        is_dealer = context.is_dealer
        
        if not flipped_card:
            return DecisionResult(
                decision_type=DecisionType.PASS,
                confidence=1.0,
                reasoning="No flipped card to order up"
            )
        
        # Calculate hand strength with the flipped card as trump
        hand_strength = self._calculate_trump_hand_strength(hand, flipped_card.suit)
        
        # Position-based adjustments
        position_bonus = self._get_position_bonus(position, is_dealer)
        
        # Risk-based threshold adjustment
        adjusted_threshold = self.trump_threshold * self.risk_multiplier
        
        # Decision logic
        should_order = hand_strength + position_bonus > adjusted_threshold
        
        if should_order:
            confidence = min(1.0, (hand_strength + position_bonus) / adjusted_threshold)
            reasoning = f"Strong hand ({hand_strength:.2f}) + position bonus ({position_bonus:.2f}) > threshold ({adjusted_threshold:.2f})"
        else:
            confidence = 1.0 - (hand_strength + position_bonus) / adjusted_threshold
            reasoning = f"Weak hand ({hand_strength:.2f}) + position bonus ({position_bonus:.2f}) < threshold ({adjusted_threshold:.2f})"
        
        decision_type = DecisionType.ORDER_UP if should_order else DecisionType.PASS
        
        result = DecisionResult(
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'hand_strength': hand_strength,
                'position_bonus': position_bonus,
                'threshold': adjusted_threshold,
                'ai_style': self.ai_style
            }
        )
        
        self.record_decision(result)
        return result
    
    def should_call_trump(self, context: GameContext) -> DecisionResult:
        """Decide whether to call trump if everyone passes on the flipped card."""
        hand = context.hand
        position = context.position
        is_dealer = context.is_dealer
        
        # Calculate best possible trump suit strength
        best_trump_strength = 0.0
        best_suit = None
        
        for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
            strength = self._calculate_trump_hand_strength(hand, suit)
            if strength > best_trump_strength:
                best_trump_strength = strength
                best_suit = suit
        
        # Position-based adjustments
        position_bonus = self._get_position_bonus(position, is_dealer)
        
        # Risk-based threshold adjustment
        adjusted_threshold = self.trump_threshold * self.risk_multiplier
        
        # Decision logic
        should_call = best_trump_strength + position_bonus > adjusted_threshold
        
        if should_call:
            confidence = min(1.0, (best_trump_strength + position_bonus) / adjusted_threshold)
            reasoning = f"Best trump strength ({best_trump_strength:.2f}) + position bonus ({position_bonus:.2f}) > threshold ({adjusted_threshold:.2f})"
        else:
            confidence = 1.0 - (best_trump_strength + position_bonus) / adjusted_threshold
            reasoning = f"Best trump strength ({best_trump_strength:.2f}) + position bonus ({position_bonus:.2f}) < threshold ({adjusted_threshold:.2f})"
        
        decision_type = DecisionType.CALL_TRUMP if should_call else DecisionType.PASS
        
        result = DecisionResult(
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'best_trump_strength': best_trump_strength,
                'best_suit': best_suit.name if best_suit else None,
                'position_bonus': position_bonus,
                'threshold': adjusted_threshold,
                'ai_style': self.ai_style
            }
        )
        
        self.record_decision(result)
        return result
    
    def select_trump_suit(self, context: GameContext) -> DecisionResult:
        """Select which suit to call as trump."""
        hand = context.hand
        
        # Calculate strength for each suit
        suit_strengths = {}
        for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
            strength = self._calculate_trump_hand_strength(hand, suit)
            suit_strengths[suit] = strength
        
        # Select the strongest suit
        best_suit = max(suit_strengths, key=suit_strengths.get)
        best_strength = suit_strengths[best_suit]
        
        # Calculate confidence based on how much stronger the best suit is
        other_strengths = [s for s in suit_strengths.values() if s != best_strength]
        if other_strengths:
            max_other = max(other_strengths)
            confidence = min(1.0, (best_strength - max_other) / best_strength + 0.5)
        else:
            confidence = 1.0
        
        reasoning = f"Selected {best_suit.name} with strength {best_strength:.2f} (strongest of all suits)"
        
        result = DecisionResult(
            decision_type=DecisionType.CALL_TRUMP,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'selected_suit': best_suit.name,
                'suit_strengths': {s.name: v for s, v in suit_strengths.items()},
                'ai_style': self.ai_style
            }
        )
        
        self.record_decision(result)
        return result
    
    def play_card(self, context: GameContext) -> DecisionResult:
        """Decide which card to play in the current trick."""
        hand = context.hand
        current_trick = context.current_trick
        trick_suit = context.trick_suit
        trump_suit = context.trump_suit
        position = context.position
        
        if not hand:
            return DecisionResult(
                decision_type=DecisionType.PLAY_CARD,
                confidence=0.0,
                reasoning="No cards in hand"
            )
        
        # If leading the trick
        if not current_trick:
            return self._choose_lead_card(context)
        
        # If following to a trick
        return self._choose_follow_card(context)
    
    def _choose_lead_card(self, context: GameContext) -> DecisionResult:
        """Choose which card to lead."""
        hand = context.hand
        trump_suit = context.trump_suit
        position = context.position
        
        # Calculate hand strength
        hand_strength = self.evaluate_hand_strength(context)
        
        # Decide whether to lead trump or off-suit
        if trump_suit and hand_strength > self.leading_threshold:
            # Lead trump if hand is strong
            trump_cards = [c for c in hand if c.suit == trump_suit or 
                          (c.rank == Rank.JACK and c.suit == self._get_left_bower_suit(trump_suit))]
            if trump_cards:
                # Lead the highest trump
                best_trump = max(trump_cards, key=lambda c: self._get_card_value(c, trump_suit))
                reasoning = f"Leading high trump {best_trump} (strong hand: {hand_strength:.2f})"
                confidence = hand_strength
            else:
                # Lead highest off-suit
                best_off_suit = max(hand, key=lambda c: self._get_card_value(c, trump_suit))
                reasoning = f"Leading high off-suit {best_off_suit} (strong hand: {hand_strength:.2f})"
                confidence = hand_strength
        else:
            # Lead low card if hand is weak
            best_card = min(hand, key=lambda c: self._get_card_value(c, trump_suit))
            reasoning = f"Leading low card {best_card} (weak hand: {hand_strength:.2f})"
            confidence = 1.0 - hand_strength
        
        result = DecisionResult(
            decision_type=DecisionType.PLAY_CARD,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'hand_strength': hand_strength,
                'leading_threshold': self.leading_threshold,
                'ai_style': self.ai_style
            }
        )
        
        self.record_decision(result)
        return result
    
    def _choose_follow_card(self, context: GameContext) -> DecisionResult:
        """Choose which card to follow to a trick."""
        hand = context.hand
        current_trick = context.current_trick
        trick_suit = context.trick_suit
        trump_suit = context.trump_suit
        
        # Must follow suit if possible
        follow_suit_cards = [c for c in hand if c.suit == trick_suit]
        
        if follow_suit_cards:
            # Must follow suit
            if trump_suit and any(c.suit == trump_suit or 
                                 (c.rank == Rank.JACK and c.suit == self._get_left_bower_suit(trump_suit)) 
                                 for c in current_trick):
                # Trump has been played, try to win
                best_follow = max(follow_suit_cards, key=lambda c: self._get_card_value(c, trump_suit))
                reasoning = f"Following suit with high card {best_follow} to try to win"
                confidence = 0.8
            else:
                # No trump played, play low to avoid winning
                best_follow = min(follow_suit_cards, key=lambda c: self._get_card_value(c, trump_suit))
                reasoning = f"Following suit with low card {best_follow} to avoid winning"
                confidence = 0.9
        else:
            # Can't follow suit, play trump if available and beneficial
            if trump_suit:
                trump_cards = [c for c in hand if c.suit == trump_suit or 
                              (c.rank == Rank.JACK and c.suit == self._get_left_bower_suit(trump_suit))]
                if trump_cards:
                    # Play lowest trump
                    best_card = min(trump_cards, key=lambda c: self._get_card_value(c, trump_suit))
                    reasoning = f"Playing low trump {best_card} (can't follow suit)"
                    confidence = 0.7
                else:
                    # Play lowest off-suit
                    best_card = min(hand, key=lambda c: self._get_card_value(c, trump_suit))
                    reasoning = f"Playing low off-suit {best_card} (can't follow suit)"
                    confidence = 0.6
            else:
                # No trump, play lowest card
                best_card = min(hand, key=lambda c: self._get_card_value(c, None))
                reasoning = f"Playing low card {best_card} (can't follow suit)"
                confidence = 0.6
        
        result = DecisionResult(
            decision_type=DecisionType.PLAY_CARD,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'trick_suit': trick_suit.name if trick_suit else None,
                'trump_suit': trump_suit.name if trump_suit else None,
                'ai_style': self.ai_style
            }
        )
        
        self.record_decision(result)
        return result
    
    def discard_card(self, context: GameContext) -> DecisionResult:
        """Decide which card to discard when partner calls trump."""
        hand = context.hand
        trump_suit = context.trump_suit
        
        if not hand or len(hand) != 6:
            return DecisionResult(
                decision_type=DecisionType.DISCARD,
                confidence=0.0,
                reasoning="Invalid hand for discarding"
            )
        
        # Discard the lowest value card
        worst_card = min(hand, key=lambda c: self._get_card_value(c, trump_suit))
        
        reasoning = f"Discarding lowest value card {worst_card}"
        confidence = 0.9
        
        result = DecisionResult(
            decision_type=DecisionType.DISCARD,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'discarded_card': str(worst_card),
                'ai_style': self.ai_style
            }
        )
        
        self.record_decision(result)
        return result
    
    def _calculate_trump_hand_strength(self, hand: List[Card], trump_suit: Suit) -> float:
        """Calculate hand strength with a specific trump suit."""
        if not hand:
            return 0.0
        
        total_strength = 0.0
        
        for card in hand:
            # Base card value
            base_value = self._get_card_value(card, trump_suit)
            
            # Trump bonus
            if card.suit == trump_suit or (card.rank == Rank.JACK and 
                                          card.suit == self._get_left_bower_suit(trump_suit)):
                base_value *= 1.5
            
            total_strength += base_value
        
        # Normalize to 0.0-1.0 range
        return min(1.0, total_strength / (len(hand) * 10))
    
    def _get_card_value(self, card: Card, trump_suit: Optional[Suit]) -> int:
        """Get the value of a card (higher is better)."""
        # Base rank values
        rank_values = {
            Rank.NINE: 1,
            Rank.TEN: 2,
            Rank.JACK: 3,
            Rank.QUEEN: 4,
            Rank.KING: 5,
            Rank.ACE: 6
        }
        
        base_value = rank_values.get(card.rank, 0)
        
        # Trump bonus
        if trump_suit:
            if card.suit == trump_suit:
                base_value += 10
            elif card.rank == Rank.JACK and card.suit == self._get_left_bower_suit(trump_suit):
                base_value += 15  # Left bower is very valuable
        
        return base_value
    
    def _get_position_bonus(self, position: int, is_dealer: bool) -> float:
        """Get position-based bonus for trump calling."""
        if is_dealer:
            return 0.2  # Dealer gets bonus
        elif position == 0:  # First to act
            return 0.1  # Slight bonus for first action
        elif position == 3:  # Last to act
            return 0.15  # Bonus for last action
        else:
            return 0.0  # Middle positions get no bonus 