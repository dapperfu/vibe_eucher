"""
Level 2 Neural AI Implementation

This module implements the Level 2 neural AI using the BaseAIInterface.
The AI makes decisions based on trained neural network models rather than
hard-coded rules, allowing it to learn optimal strategies from training data.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-13
Context: Creating Level 2 neural AI implementation using the abstract interface
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from .base_ai_interface import (
    BaseAIInterface, GameContext, DecisionResult, DecisionType
)
from ..models import Card, Suit, Rank, Player, PlayerType
from ..ai_model.level2_models import (
    StrategicModel, AggressiveModel, BalancedModel, IntuitiveModel,
    Level2RiskProfile
)


class Level2AI(Player, BaseAIInterface):
    """
    Level 2 neural AI implementation.
    
    This AI makes decisions using trained neural network models:
    - Trump calling based on neural network evaluation
    - Card playing based on learned strategies
    - Risk assessment based on model confidence
    """
    
    def __init__(self, name: str, model_type: str = "level2_strategic", 
                 risk_profile: float = 0.5, model_path: Optional[str] = None):
        """
        Initialize Level 2 AI.
        
        Parameters
        ----------
        name : str
            The AI's name
        model_type : str
            Level 2 model type: "level2_strategic", "level2_aggressive", "level2_balanced", "level2_intuitive"
        risk_profile : float
            Risk tolerance from 0.0 (conservative) to 1.0 (aggressive)
        model_path : str, optional
            Path to trained model file (.pth)
        """
        # Initialize as a Player first
        super().__init__(name, PlayerType.AI)
        
        # Initialize AI interface
        BaseAIInterface.__init__(self, name, risk_profile)
        
        self.model_type = model_type.lower()
        self.model_path = model_path
        
        # Create risk profile
        self.risk_profile_obj = Level2RiskProfile(
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
        """Create the appropriate Level 2 model."""
        if self.model_type == "level2_strategic":
            return StrategicModel(risk_profile=self.risk_profile_obj)
        elif self.model_type == "level2_aggressive":
            return AggressiveModel(risk_profile=self.risk_profile_obj)
        elif self.model_type == "level2_balanced":
            return BalancedModel(risk_profile=self.risk_profile_obj)
        elif self.model_type == "level2_intuitive":
            return IntuitiveModel(risk_profile=self.risk_profile_obj)
        else:
            raise ValueError(f"Unknown Level 2 model type: {self.model_type}")
    
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
            print(f"✅ Loaded trained Level 2 model from {model_path}")
            
        except Exception as e:
            print(f"Warning: Could not load trained model from {model_path}: {e}")
            print("Using untrained model instead.")
    
    def should_order_up(self, context: GameContext) -> DecisionResult:
        """
        Decide whether to order up the top card.
        
        Parameters
        ----------
        context : GameContext
            Current game context including hand, top card, dealer, etc.
            
        Returns
        -------
        DecisionResult
            Decision with type, confidence, and reasoning
        """
        try:
            # Prepare input tensor for the model
            input_tensor = self._prepare_input_tensor(context)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(input_tensor, self.risk_profile_obj)
                trump_confidence = torch.softmax(outputs['trump_decision'], dim=-1)[1].item()
            
            # Apply risk profile adjustment
            risk_adjustment = self.risk_profile_obj.trump_calling_aggression - 0.5
            adjusted_confidence = trump_confidence + (risk_adjustment * 0.2)
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            # Make decision
            should_order = adjusted_confidence > 0.5
            
            # Generate reasoning
            if should_order:
                reasoning = f"Level 2 AI {self.name} orders up with {adjusted_confidence:.2%} confidence"
                reasoning += f" (base: {trump_confidence:.2%}, risk adjustment: {risk_adjustment:+.2f})"
            else:
                reasoning = f"Level 2 AI {self.name} passes with {adjusted_confidence:.2%} confidence"
                reasoning += f" (base: {trump_confidence:.2%}, risk adjustment: {risk_adjustment:+.2f})"
            
            return DecisionResult(
                decision_type=DecisionType.ORDER_UP if should_order else DecisionType.PASS,
                confidence=adjusted_confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            print(f"Error in Level 2 AI decision: {e}")
            # Fallback to conservative decision
            return DecisionResult(
                decision_type=DecisionType.PASS,
                confidence=0.3,
                reasoning=f"Level 2 AI {self.name} encountered error, defaulting to pass"
            )
    
    def should_call_trump(self, context: GameContext) -> DecisionResult:
        """
        Decide whether to call trump during second round.
        
        Parameters
        ----------
        context : GameContext
            Current game context including hand, dealer, etc.
            
        Returns
        -------
        DecisionResult
            Decision with type, confidence, and reasoning
        """
        try:
            # Prepare input tensor for the model
            input_tensor = self._prepare_input_tensor(context)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(input_tensor, self.risk_profile_obj)
                trump_confidence = torch.softmax(outputs['trump_decision'], dim=-1)[1].item()
            
            # Apply risk profile adjustment
            risk_adjustment = self.risk_profile_obj.trump_calling_aggression - 0.5
            adjusted_confidence = trump_confidence + (risk_adjustment * 0.2)
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            # Make decision
            should_call = adjusted_confidence > 0.5
            
            # Generate reasoning
            if should_call:
                reasoning = f"Level 2 AI {self.name} calls trump with {adjusted_confidence:.2%} confidence"
                reasoning += f" (base: {trump_confidence:.2%}, risk adjustment: {risk_adjustment:+.2f})"
            else:
                reasoning = f"Level 2 AI {self.name} passes with {adjusted_confidence:.2%} confidence"
                reasoning += f" (base: {trump_confidence:.2%}, risk adjustment: {risk_adjustment:+.2f})"
            
            return DecisionResult(
                decision_type=DecisionType.CALL_TRUMP if should_call else DecisionType.PASS,
                confidence=adjusted_confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            print(f"Error in Level 2 AI decision: {e}")
            # Fallback to conservative decision
            return DecisionResult(
                decision_type=DecisionType.PASS,
                confidence=0.3,
                reasoning=f"Level 2 AI {self.name} encountered error, defaulting to pass"
            )
    
    def choose_card_to_play(self, context: GameContext) -> DecisionResult:
        """
        Choose which card to play from hand.
        
        Parameters
        ----------
        context : GameContext
            Current game context including hand, lead suit, trump, etc.
            
        Returns
        -------
        DecisionResult
            Decision with type, confidence, and reasoning
        """
        try:
            # Prepare input tensor for the model
            input_tensor = self._prepare_input_tensor(context)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(input_tensor, self.risk_profile_obj)
                card_probs = torch.softmax(outputs['card_selection'], dim=-1)
                chosen_card_idx = torch.argmax(card_probs).item()
                confidence = card_probs[chosen_card_idx].item()
            
            # Get the chosen card
            if 0 <= chosen_card_idx < len(context.hand):
                chosen_card = context.hand[chosen_card_idx]
            else:
                # Fallback to first card if index is invalid
                chosen_card = context.hand[0]
                confidence = 0.5
            
            # Generate reasoning
            reasoning = f"Level 2 AI {self.name} plays {chosen_card} with {confidence:.2%} confidence"
            
            return DecisionResult(
                decision_type=DecisionType.PLAY_CARD,
                confidence=confidence,
                reasoning=reasoning,
                card=chosen_card
            )
            
        except Exception as e:
            print(f"Error in Level 2 AI card selection: {e}")
            # Fallback to first card
            fallback_card = context.hand[0] if context.hand else None
            return DecisionResult(
                decision_type=DecisionType.PLAY_CARD,
                confidence=0.3,
                reasoning=f"Level 2 AI {self.name} encountered error, defaulting to first card",
                card=fallback_card
            )
    
    def _prepare_input_tensor(self, context: GameContext) -> torch.Tensor:
        """
        Prepare input tensor for the neural network.
        
        Parameters
        ----------
        context : GameContext
            Game context to encode
            
        Returns
        -------
        torch.Tensor
            Input tensor for the model
        """
        try:
            # This is a simplified encoding - in practice, use the full Level2GameStateEncoder
            # For now, create a basic feature vector
            features = []
            
            # Encode hand cards (simplified)
            for card in context.hand:
                # Basic card encoding: suit (4) + rank (6) + trump flag (1)
                suit_idx = card.suit.value
                rank_idx = card.rank.value - 9  # 9=0, 10=1, J=2, Q=3, K=4, A=5
                trump_flag = 1.0 if card.is_trump else 0.0
                
                features.extend([suit_idx, rank_idx, trump_flag])
            
            # Pad to expected size if needed
            expected_features = 5 * 3  # 5 cards * 3 features each
            while len(features) < expected_features:
                features.append(0.0)
            
            # Add game context features (simplified)
            features.append(1.0 if context.dealer == "self" else 0.0)
            features.append(1.0 if context.dealer == "partner" else 0.0)
            features.append(1.0 if context.dealer == "opponent" else 0.0)
            features.append(context.team_score / 10.0)  # Normalize score
            features.append(context.opponent_score / 10.0)
            
            # Pad to 256 features (model input size)
            while len(features) < 256:
                features.append(0.0)
            
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
        except Exception as e:
            print(f"Error preparing input tensor: {e}")
            # Return zero tensor as fallback
            return torch.zeros(1, 256, dtype=torch.float32) 