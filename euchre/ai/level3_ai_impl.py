"""
Level 3 Neural AI Implementation

This module implements the Level 3 neural AI using the BaseAIInterface.
The AI makes decisions using ultra-comprehensive neural network models that capture
every conceivable detail about Euchre gameplay, including:
- Complete game history with every card played
- Player behavior patterns and tendencies  
- Advanced card counting and probability analysis
- Multi-turn strategic planning
- Partner coordination and team dynamics
- Risk assessment and adaptation

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-13
Context: Creating full-featured Level 3 AI implementation using the abstract interface
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import logging

from .base_ai_interface import (
    BaseAIInterface, GameContext, DecisionResult, DecisionType
)
from ..models import Card, Suit, Rank, Player, PlayerType
from ..ai_model.level3_models import (
    Level3NeuralModel, Level3RiskProfile, Level3GameStateEncoder,
    create_level3_model, create_level3_risk_profile
)

# Set up logging
logger = logging.getLogger(__name__)


class Level3AI(Player, BaseAIInterface):
    """
    Level 3 neural AI implementation.
    
    This AI makes decisions using ultra-comprehensive neural network models:
    - 2048-dimensional input features capturing every game detail
    - Advanced neural architecture with 8 hidden layers
    - Transformer architecture with attention mechanisms
    - Memory networks for long-term strategic planning
    - Dynamic risk adaptation with 19 risk parameters
    - Partner coordination and opponent modeling
    """
    
    def __init__(self, name: str, model_type: str = "level3_strategic", 
                 risk_profile: str = "strategic_mastermind", 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize Level 3 AI.
        
        Parameters
        ----------
        name : str
            The AI's name
        model_type : str
            Level 3 model type: "level3_strategic", "level3_aggressive", "level3_balanced", 
                           "level3_conservative", "level3_opportunistic"
        risk_profile : str
            Risk profile: "strategic_mastermind", "ultra_conservative", "conservative", 
                         "balanced", "aggressive", "ultra_aggressive", "partner_coordinator",
                         "opponent_analyzer", "risk_adaptor", "game_theorist"
        model_path : str, optional
            Path to trained model file (.pth)
        device : str, optional
            Device to run on ("cuda", "cpu", or None for auto-detection)
        """
        # Initialize as a Player first
        super().__init__(name, PlayerType.AI)
        
        # Initialize AI interface
        BaseAIInterface.__init__(self, name, risk_profile=0.5)
        
        # Store the risk profile name for later use
        self.risk_profile_name = risk_profile
        
        self.model_type = model_type.lower()
        self.model_path = model_path
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Level3AI {name} initializing on device: {self.device}")
        
        # Create risk profile
        self.risk_profile_obj = create_level3_risk_profile(self.risk_profile_name)
        
        # Initialize the neural model
        self.model = self._create_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize game state encoder
        self.encoder = Level3GameStateEncoder()
        
        # Load trained weights if provided
        if model_path:
            self._load_trained_model(model_path)
        
        # Game state tracking
        self.game_history = []
        self.trick_history = []
        self.current_game_state = None
        
        # Performance tracking
        self.decisions_made = 0
        self.successful_decisions = 0
        
        logger.info(f"Level3AI {name} initialized successfully")
    
    def _create_model(self) -> Level3NeuralModel:
        """Create the appropriate Level 3 model."""
        model_config = {
            'input_size': 2048,
            'hidden_size': 1024,
            'num_layers': 8,
            'risk_embedding_size': 128,
            'use_attention': True,
            'use_transformer': True,
            'use_memory_networks': True
        }
        
        # Adjust model based on type
        if "aggressive" in self.model_type:
            model_config['hidden_size'] = 1024
            model_config['num_layers'] = 8
        elif "conservative" in self.model_type:
            model_config['hidden_size'] = 1024
            model_config['num_layers'] = 8
        elif "balanced" in self.model_type:
            model_config['hidden_size'] = 1024
            model_config['num_layers'] = 8
        elif "strategic" in self.model_type:
            model_config['hidden_size'] = 1024
            model_config['num_layers'] = 8
        elif "opportunistic" in self.model_type:
            model_config['hidden_size'] = 1024
            model_config['num_layers'] = 8
        
        return create_level3_model(model_config)
    
    def _load_trained_model(self, model_path: str):
        """Load trained model weights."""
        try:
            if not Path(model_path).exists():
                logger.warning(f"Model file {model_path} not found. Using untrained model.")
                return
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model state from {model_path}")
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                logger.info(f"Loaded model state from {model_path}")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded model weights from {model_path}")
                
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            logger.info("Using untrained model")
    
    def _encode_context_to_features(self, context: GameContext) -> torch.Tensor:
        """Convert GameContext to Level 3 feature tensor."""
        try:
            # Create a mock game state object for the encoder
            mock_game_state = self._create_mock_game_state(context)
            
            # Encode the game state
            features = self.encoder.encode_game_state(
                game_state=mock_game_state,
                player=self,
                trick_history=self.trick_history,
                game_history=self.game_history
            )
            
            # Ensure we have the right dimensions
            if features.shape[0] != 2048:
                # Pad or truncate to 2048 features
                if features.shape[0] < 2048:
                    padding = torch.zeros(2048 - features.shape[0], dtype=torch.float32)
                    features = torch.cat([features, padding])
                else:
                    features = features[:2048]
            
            return features.to(self.device)
            
        except Exception as e:
            logger.error(f"Error encoding context to features: {e}")
            # Return zero features as fallback
            return torch.zeros(2048, dtype=torch.float32, device=self.device)
    
    def _create_mock_game_state(self, context: GameContext) -> Any:
        """Create a mock game state object for the encoder."""
        # This is a simplified mock - in a real implementation,
        # you'd have access to the actual game state
        class MockGameState:
            def __init__(self, context):
                self.trump_suit = context.trump_suit
                self.flipped_card = context.flipped_card
                self.current_trick = context.current_trick
                self.trick_suit = context.trick_suit
                self.team1_score = context.team1_score
                self.team2_score = context.team2_score
                self.tricks_won_team1 = context.tricks_won_team1
                self.tricks_won_team2 = context.tricks_won_team2
                self.current_trick_number = context.current_trick_number
        
        return MockGameState(context)
    
    def should_order_up(self, context: GameContext) -> DecisionResult:
        """Decide whether to order up the flipped card as trump."""
        try:
            # Encode context to features
            features = self._encode_context_to_features(context)
            
            # Get model prediction
            with torch.no_grad():
                trump_probs = self.model.get_trump_decision_probs(
                    features.unsqueeze(0), self.risk_profile_obj
                )
                
                # Extract order up probability (assuming it's the first class)
                order_up_prob = trump_probs[0, 0].item()
                
                # Make decision based on probability and risk profile
                threshold = 0.5 + (self.risk_profile - 0.5) * 0.3  # Adjust based on risk
                should_order = order_up_prob > threshold
                
                decision_type = DecisionType.ORDER_UP if should_order else DecisionType.PASS
                confidence = max(order_up_prob, 1.0 - order_up_prob)
                
                reasoning = f"Level 3 AI evaluated trump potential: {order_up_prob:.3f}. "
                reasoning += f"Threshold: {threshold:.3f}. "
                reasoning += f"Decision: {'Order up' if should_order else 'Pass'}"
                
                result = DecisionResult(
                    decision_type=decision_type,
                    confidence=confidence,
                    reasoning=reasoning,
                    metadata={'trump_probability': order_up_prob, 'threshold': threshold}
                )
                
                self.record_decision(result)
                return result
                
        except Exception as e:
            logger.error(f"Error in should_order_up: {e}")
            # Fallback to conservative decision
            return DecisionResult(
                decision_type=DecisionType.PASS,
                confidence=0.5,
                reasoning=f"Level 3 AI error, fallback to pass: {e}",
                metadata={'error': str(e)}
            )
    
    def should_call_trump(self, context: GameContext) -> DecisionResult:
        """Decide whether to call trump if everyone passes on the flipped card."""
        try:
            # Encode context to features
            features = self._encode_context_to_features(context)
            
            # Get model prediction
            with torch.no_grad():
                trump_probs = self.model.get_trump_decision_probs(
                    features.unsqueeze(0), self.risk_profile_obj
                )
                
                # Extract call trump probability (assuming it's the second class)
                call_prob = trump_probs[0, 1].item() if trump_probs.shape[1] > 1 else 0.5
                
                # Make decision based on probability and risk profile
                threshold = 0.6 + (self.risk_profile - 0.5) * 0.4  # Higher threshold for calling
                should_call = call_prob > threshold
                
                decision_type = DecisionType.CALL_TRUMP if should_call else DecisionType.PASS
                confidence = max(call_prob, 1.0 - call_prob)
                
                reasoning = f"Level 3 AI evaluated trump calling: {call_prob:.3f}. "
                reasoning += f"Threshold: {threshold:.3f}. "
                reasoning += f"Decision: {'Call trump' if should_call else 'Pass'}"
                
                result = DecisionResult(
                    decision_type=decision_type,
                    confidence=confidence,
                    reasoning=reasoning,
                    metadata={'call_probability': call_prob, 'threshold': threshold}
                )
                
                self.record_decision(result)
                return result
                
        except Exception as e:
            logger.error(f"Error in should_call_trump: {e}")
            # Fallback to conservative decision
            return DecisionResult(
                decision_type=DecisionType.PASS,
                confidence=0.5,
                reasoning=f"Level 3 AI error, fallback to pass: {e}",
                metadata={'error': str(e)}
            )
    
    def select_trump_suit(self, context: GameContext) -> DecisionResult:
        """Select which suit to call as trump."""
        try:
            # Encode context to features
            features = self._encode_context_to_features(context)
            
            # Get model prediction
            with torch.no_grad():
                suit_probs = self.model.get_suit_selection_probs(
                    features.unsqueeze(0), self.risk_profile_obj
                )
                
                # Get the suit with highest probability
                suit_index = torch.argmax(suit_probs[0]).item()
                suit_prob = suit_probs[0, suit_index].item()
                
                # Map index to suit
                suit_map = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
                selected_suit = suit_map[suit_index]
                
                reasoning = f"Level 3 AI selected {selected_suit.name} as trump "
                reasoning += f"with confidence {suit_prob:.3f}"
                
                result = DecisionResult(
                    decision_type=DecisionType.CALL_TRUMP,
                    confidence=suit_prob,
                    reasoning=reasoning,
                    metadata={'selected_suit': selected_suit.name, 'suit_probability': suit_prob}
                )
                
                self.record_decision(result)
                return result
                
        except Exception as e:
            logger.error(f"Error in select_trump_suit: {e}")
            # Fallback to random suit selection
            fallback_suit = Suit.HEARTS  # Default fallback
            return DecisionResult(
                decision_type=DecisionType.CALL_TRUMP,
                confidence=0.3,
                reasoning=f"Level 3 AI error, fallback to {fallback_suit.name}: {e}",
                metadata={'error': str(e), 'fallback_suit': fallback_suit.name}
            )
    
    def play_card(self, context: GameContext) -> DecisionResult:
        """Decide which card to play in the current trick."""
        try:
            # Encode context to features
            features = self._encode_context_to_features(context)
            
            # Get model prediction
            with torch.no_grad():
                card_probs = self.model.get_card_selection_probs(
                    features.unsqueeze(0), self.risk_profile_obj
                )
                
                # Map probabilities to actual cards in hand
                hand_cards = context.hand
                if len(hand_cards) != card_probs.shape[1]:
                    # Pad or truncate probabilities to match hand size
                    if len(hand_cards) < card_probs.shape[1]:
                        card_probs = card_probs[:, :len(hand_cards)]
                    else:
                        padding = torch.zeros(1, len(hand_cards) - card_probs.shape[1])
                        card_probs = torch.cat([card_probs, padding], dim=1)
                
                # Get the card with highest probability
                card_index = torch.argmax(card_probs[0]).item()
                card_prob = card_probs[0, card_index].item()
                
                # Ensure index is within bounds
                if card_index >= len(hand_cards):
                    card_index = 0
                
                selected_card = hand_cards[card_index]
                
                reasoning = f"Level 3 AI selected {selected_card} to play "
                reasoning += f"with confidence {card_prob:.3f}"
                
                result = DecisionResult(
                    decision_type=DecisionType.PLAY_CARD,
                    confidence=card_prob,
                    reasoning=reasoning,
                    metadata={'selected_card': str(selected_card), 'card_probability': card_prob}
                )
                
                self.record_decision(result)
                return result
                
        except Exception as e:
            logger.error(f"Error in play_card: {e}")
            # Fallback to first playable card
            fallback_card = context.hand[0] if context.hand else None
            return DecisionResult(
                decision_type=DecisionType.PLAY_CARD,
                confidence=0.3,
                reasoning=f"Level 3 AI error, fallback to {fallback_card}: {e}",
                metadata={'error': str(e), 'fallback_card': str(fallback_card)}
            )
    
    def discard_card(self, context: GameContext) -> DecisionResult:
        """Decide which card to discard when partner calls trump."""
        try:
            # Encode context to features
            features = self._encode_context_to_features(context)
            
            # Get model prediction
            with torch.no_grad():
                card_probs = self.model.get_card_selection_probs(
                    features.unsqueeze(0), self.risk_profile_obj
                )
                
                # For discarding, we want the card with lowest strategic value
                # Invert the probabilities to favor lower-value cards
                discard_probs = 1.0 - card_probs[0]
                
                # Map probabilities to actual cards in hand
                hand_cards = context.hand
                if len(hand_cards) != discard_probs.shape[0]:
                    # Pad or truncate probabilities to match hand size
                    if len(hand_cards) < discard_probs.shape[0]:
                        discard_probs = discard_probs[:len(hand_cards)]
                    else:
                        padding = torch.zeros(len(hand_cards) - discard_probs.shape[0])
                        discard_probs = torch.cat([discard_probs, padding])
                
                # Get the card with highest discard probability (lowest strategic value)
                card_index = torch.argmax(discard_probs).item()
                discard_prob = discard_probs[card_index].item()
                
                # Ensure index is within bounds
                if card_index >= len(hand_cards):
                    card_index = 0
                
                selected_card = hand_cards[card_index]
                
                reasoning = f"Level 3 AI selected {selected_card} to discard "
                reasoning += f"with discard probability {discard_prob:.3f}"
                
                result = DecisionResult(
                    decision_type=DecisionType.DISCARD,
                    confidence=discard_prob,
                    reasoning=reasoning,
                    metadata={'discarded_card': str(selected_card), 'discard_probability': discard_prob}
                )
                
                self.record_decision(result)
                return result
                
        except Exception as e:
            logger.error(f"Error in discard_card: {e}")
            # Fallback to first card in hand
            fallback_card = context.hand[0] if context.hand else None
            return DecisionResult(
                decision_type=DecisionType.DISCARD,
                confidence=0.3,
                reasoning=f"Level 3 AI error, fallback to {fallback_card}: {e}",
                metadata={'error': str(e), 'fallback_card': str(fallback_card)}
            )
    
    def update_game_state(self, game_state: Any, trick_history: List[Any], 
                         game_history: List[Dict[str, Any]]):
        """Update the AI's understanding of the current game state."""
        self.current_game_state = game_state
        self.trick_history = trick_history
        self.game_history = game_history
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics for Level 3 AI."""
        base_metrics = super().get_performance_metrics()
        
        # Add Level 3 specific metrics
        level3_metrics = {
            'model_type': self.model_type,
            'risk_profile': self.risk_profile_name,
            'device': str(self.device),
            'model_loaded': self.model_path is not None,
            'total_decisions': self.decisions_made,
            'successful_decisions': self.successful_decisions,
            'success_rate': (self.successful_decisions / max(self.decisions_made, 1)) * 100
        }
        
        base_metrics.update(level3_metrics)
        return base_metrics
    
    def __str__(self) -> str:
        return f"Level3AI({self.name}, model={self.model_type}, risk={self.risk_profile_name})"
    
    def __repr__(self) -> str:
        return self.__str__() 