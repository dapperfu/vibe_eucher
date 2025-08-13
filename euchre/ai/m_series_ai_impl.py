"""
M-Series Neural AI Implementation

This module implements the M-Series neural AI using the BaseAIInterface.
The AI makes decisions based on trained neural network models rather than
hard-coded rules, allowing it to learn optimal strategies from training data.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-13
Context: Creating M-Series neural AI implementation using the abstract interface
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from .base_ai_interface import (
    BaseAIInterface, GameContext, DecisionResult, DecisionType
)
from ..models import Card, Suit, Rank
from ..ai_model.m_series_models import (
    MagnusModel, MaverickModel, MentorModel, MysticModel,
    MSeriesRiskProfile
)


class MSeriesAI(BaseAIInterface):
    """
    M-Series neural AI implementation.
    
    This AI makes decisions using trained neural network models:
    - Trump calling based on neural network evaluation
    - Card playing based on learned strategies
    - Risk assessment based on model confidence
    """
    
    def __init__(self, name: str, model_type: str = "magnus", 
                 risk_profile: float = 0.5, model_path: Optional[str] = None):
        """
        Initialize M-Series AI.
        
        Parameters
        ----------
        name : str
            The AI's name
        model_type : str
            M-Series model type: "magnus", "maverick", "mentor", "mystic"
        risk_profile : float
            Risk tolerance from 0.0 (conservative) to 1.0 (aggressive)
        model_path : str, optional
            Path to trained model file (.pth)
        """
        super().__init__(name, risk_profile)
        self.model_type = model_type.lower()
        self.model_path = model_path
        
        # Create risk profile
        self.risk_profile_obj = MSeriesRiskProfile(
            trump_calling_aggression=risk_profile,
            leading_aggression=risk_profile,
            trump_usage_strategy=risk_profile,
            partner_coordination=0.7,  # Keep coordination high
            set_avoidance=0.8 if risk_profile < 0.5 else 0.6
        )
        
        # Initialize the neural model
        self.model = self._create_model()
        self.model.eval()
        
        # Load trained weights if provided
        if model_path:
            self._load_trained_model(model_path)
    
    def _create_model(self):
        """Create the appropriate M-Series model."""
        if self.model_type == "magnus":
            return MagnusModel(risk_profile=self.risk_profile_obj)
        elif self.model_type == "maverick":
            return MaverickModel(risk_profile=self.risk_profile_obj)
        elif self.model_type == "mentor":
            return MentorModel(risk_profile=self.risk_profile_obj)
        elif self.model_type == "mystic":
            return MysticModel(risk_profile=self.risk_profile_obj)
        else:
            raise ValueError(f"Unknown M-Series model type: {self.model_type}")
    
    def _load_trained_model(self, model_path: str):
        """Load trained model weights."""
        try:
            if not Path(model_path).exists():
                print(f"Warning: Model file {model_path} not found. Using untrained model.")
                return
            
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"✅ Loaded trained {self.model_type} model from {model_path}")
            
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Using untrained model.")
    
    def should_order_up(self, context: GameContext) -> DecisionResult:
        """Decide whether to order up the flipped card as trump using neural network."""
        try:
            # Prepare input for neural network
            input_tensor = self._prepare_input_tensor(context, decision_type="order_up")
            
            # Get model prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                confidence = torch.sigmoid(output).item()
            
            # Apply risk profile adjustment
            adjusted_confidence = self._apply_risk_adjustment(confidence, "trump_calling")
            
            # Decision logic
            should_order = adjusted_confidence > 0.5
            
            if should_order:
                decision_type = DecisionType.ORDER_UP
                reasoning = f"Neural network confidence: {confidence:.3f}, adjusted: {adjusted_confidence:.3f}"
            else:
                decision_type = DecisionType.PASS
                reasoning = f"Neural network confidence: {confidence:.3f}, adjusted: {adjusted_confidence:.3f}"
            
            result = DecisionResult(
                decision_type=decision_type,
                confidence=adjusted_confidence,
                reasoning=reasoning,
                metadata={
                    'raw_confidence': confidence,
                    'adjusted_confidence': adjusted_confidence,
                    'model_type': self.model_type,
                    'risk_profile': self.risk_profile,
                    'ai_style': 'neural'
                }
            )
            
            self.record_decision(result)
            return result
            
        except Exception as e:
            # Fallback to traditional logic if neural network fails
            print(f"Neural network error in should_order_up: {e}")
            return self._fallback_order_up_decision(context)
    
    def should_call_trump(self, context: GameContext) -> DecisionResult:
        """Decide whether to call trump using neural network."""
        try:
            # Prepare input for neural network
            input_tensor = self._prepare_input_tensor(context, decision_type="call_trump")
            
            # Get model prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                confidence = torch.sigmoid(output).item()
            
            # Apply risk profile adjustment
            adjusted_confidence = self._apply_risk_adjustment(confidence, "trump_calling")
            
            # Decision logic
            should_call = adjusted_confidence > 0.5
            
            if should_call:
                decision_type = DecisionType.CALL_TRUMP
                reasoning = f"Neural network confidence: {confidence:.3f}, adjusted: {adjusted_confidence:.3f}"
            else:
                decision_type = DecisionType.PASS
                reasoning = f"Neural network confidence: {confidence:.3f}, adjusted: {adjusted_confidence:.3f}"
            
            result = DecisionResult(
                decision_type=decision_type,
                confidence=adjusted_confidence,
                reasoning=reasoning,
                metadata={
                    'raw_confidence': confidence,
                    'adjusted_confidence': adjusted_confidence,
                    'model_type': self.model_type,
                    'risk_profile': self.risk_profile,
                    'ai_style': 'neural'
                }
            )
            
            self.record_decision(result)
            return result
            
        except Exception as e:
            # Fallback to traditional logic if neural network fails
            print(f"Neural network error in should_call_trump: {e}")
            return self._fallback_call_trump_decision(context)
    
    def select_trump_suit(self, context: GameContext) -> DecisionResult:
        """Select which suit to call as trump using neural network."""
        try:
            # Evaluate each suit
            suit_scores = {}
            for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
                # Create context with this suit as trump
                suit_context = self._create_suit_context(context, suit)
                input_tensor = self._prepare_input_tensor(suit_context, decision_type="select_trump")
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    score = torch.sigmoid(output).item()
                    suit_scores[suit] = score
            
            # Select the best suit
            best_suit = max(suit_scores, key=suit_scores.get)
            best_score = suit_scores[best_suit]
            
            # Apply risk profile adjustment
            adjusted_confidence = self._apply_risk_adjustment(best_score, "trump_calling")
            
            reasoning = f"Neural network selected {best_suit.name} with score {best_score:.3f}, adjusted: {adjusted_confidence:.3f}"
            
            result = DecisionResult(
                decision_type=DecisionType.CALL_TRUMP,
                confidence=adjusted_confidence,
                reasoning=reasoning,
                metadata={
                    'suit_scores': {s.name: v for s, v in suit_scores.items()},
                    'selected_suit': best_suit.name,
                    'model_type': self.model_type,
                    'risk_profile': self.risk_profile,
                    'ai_style': 'neural'
                }
            )
            
            self.record_decision(result)
            return result
            
        except Exception as e:
            # Fallback to traditional logic if neural network fails
            print(f"Neural network error in select_trump_suit: {e}")
            return self._fallback_select_trump_decision(context)
    
    def play_card(self, context: GameContext) -> DecisionResult:
        """Decide which card to play using neural network."""
        try:
            # Evaluate each card in hand
            card_scores = {}
            for card in context.hand:
                # Create context with this card played
                card_context = self._create_card_context(context, card)
                input_tensor = self._prepare_input_tensor(card_context, decision_type="play_card")
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    score = torch.sigmoid(output).item()
                    card_scores[card] = score
            
            # Select the best card
            best_card = max(card_scores, key=card_scores.get)
            best_score = card_scores[best_card]
            
            # Apply risk profile adjustment
            adjusted_confidence = self._apply_risk_adjustment(best_score, "card_playing")
            
            reasoning = f"Neural network selected {best_card} with score {best_score:.3f}, adjusted: {adjusted_confidence:.3f}"
            
            result = DecisionResult(
                decision_type=DecisionType.PLAY_CARD,
                confidence=adjusted_confidence,
                reasoning=reasoning,
                metadata={
                    'card_scores': {str(c): v for c, v in card_scores.items()},
                    'selected_card': str(best_card),
                    'model_type': self.model_type,
                    'risk_profile': self.risk_profile,
                    'ai_style': 'neural'
                }
            )
            
            self.record_decision(result)
            return result
            
        except Exception as e:
            # Fallback to traditional logic if neural network fails
            print(f"Neural network error in play_card: {e}")
            return self._fallback_play_card_decision(context)
    
    def discard_card(self, context: GameContext) -> DecisionResult:
        """Decide which card to discard using neural network."""
        try:
            # Evaluate each card for discarding (lower score = better to discard)
            card_scores = {}
            for card in context.hand:
                # Create context with this card discarded
                card_context = self._create_discard_context(context, card)
                input_tensor = self._prepare_input_tensor(card_context, decision_type="discard")
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    score = torch.sigmoid(output).item()
                    card_scores[card] = score
            
            # Select the card with lowest score (best to discard)
            worst_card = min(card_scores, key=card_scores.get)
            worst_score = card_scores[worst_card]
            
            # Apply risk profile adjustment
            adjusted_confidence = self._apply_risk_adjustment(1.0 - worst_score, "card_playing")
            
            reasoning = f"Neural network selected {worst_card} to discard with score {worst_score:.3f}, adjusted: {adjusted_confidence:.3f}"
            
            result = DecisionResult(
                decision_type=DecisionType.DISCARD,
                confidence=adjusted_confidence,
                reasoning=reasoning,
                metadata={
                    'card_scores': {str(c): v for c, v in card_scores.items()},
                    'discarded_card': str(worst_card),
                    'model_type': self.model_type,
                    'risk_profile': self.risk_profile,
                    'ai_style': 'neural'
                }
            )
            
            self.record_decision(result)
            return result
            
        except Exception as e:
            # Fallback to traditional logic if neural network fails
            print(f"Neural network error in discard_card: {e}")
            return self._fallback_discard_card_decision(context)
    
    def _prepare_input_tensor(self, context: GameContext, decision_type: str) -> torch.Tensor:
        """Prepare input tensor for neural network."""
        # This would need to be implemented based on the specific M-Series model input format
        # For now, create a basic feature vector
        
        features = []
        
        # Hand features
        for card in context.hand:
            features.extend([
                float(card.suit.value),
                float(card.rank.value),
                float(card.suit == context.trump_suit if context.trump_suit else 0.0)
            ])
        
        # Pad to fixed size if needed
        while len(features) < 256:  # Assuming 256 input features
            features.append(0.0)
        
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _apply_risk_adjustment(self, confidence: float, decision_category: str) -> float:
        """Apply risk profile adjustments to neural network confidence."""
        if decision_category == "trump_calling":
            # Aggressive players call trump more often
            if self.risk_profile > 0.7:
                confidence = min(1.0, confidence * 1.2)
            elif self.risk_profile < 0.3:
                confidence = max(0.0, confidence * 0.8)
        
        elif decision_category == "card_playing":
            # Aggressive players play high cards more often
            if self.risk_profile > 0.7:
                confidence = min(1.0, confidence * 1.1)
            elif self.risk_profile < 0.3:
                confidence = max(0.0, confidence * 0.9)
        
        return confidence
    
    def _create_suit_context(self, context: GameContext, suit: Suit) -> GameContext:
        """Create a new context with a specific suit as trump."""
        # This would create a copy of the context with the suit as trump
        # For now, return the original context
        return context
    
    def _create_card_context(self, context: GameContext, card: Card) -> GameContext:
        """Create a new context with a specific card played."""
        # This would create a copy of the context with the card played
        # For now, return the original context
        return context
    
    def _create_discard_context(self, context: GameContext, card: Card) -> GameContext:
        """Create a new context with a specific card discarded."""
        # This would create a copy of the context with the card discarded
        # For now, return the original context
        return context
    
    # Fallback methods using traditional logic
    def _fallback_order_up_decision(self, context: GameContext) -> DecisionResult:
        """Fallback decision using traditional logic."""
        from .traditional_ai_impl import TraditionalAI
        fallback_ai = TraditionalAI(self.name, self.risk_profile, "balanced")
        return fallback_ai.should_order_up(context)
    
    def _fallback_call_trump_decision(self, context: GameContext) -> DecisionResult:
        """Fallback decision using traditional logic."""
        from .traditional_ai_impl import TraditionalAI
        fallback_ai = TraditionalAI(self.name, self.risk_profile, "balanced")
        return fallback_ai.should_call_trump(context)
    
    def _fallback_select_trump_decision(self, context: GameContext) -> DecisionResult:
        """Fallback decision using traditional logic."""
        from .traditional_ai_impl import TraditionalAI
        fallback_ai = TraditionalAI(self.name, self.risk_profile, "balanced")
        return fallback_ai.select_trump_suit(context)
    
    def _fallback_play_card_decision(self, context: GameContext) -> DecisionResult:
        """Fallback decision using traditional logic."""
        from .traditional_ai_impl import TraditionalAI
        fallback_ai = TraditionalAI(self.name, self.risk_profile, "balanced")
        return fallback_ai.play_card(context)
    
    def _fallback_discard_card_decision(self, context: GameContext) -> DecisionResult:
        """Fallback decision using traditional logic."""
        from .traditional_ai_impl import TraditionalAI
        fallback_ai = TraditionalAI(self.name, self.risk_profile, "balanced")
        return fallback_ai.discard_card(context) 