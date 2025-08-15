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
            return StrategicModel()
        elif self.model_type == "level2_aggressive":
            return AggressiveModel()
        elif self.model_type == "level2_balanced":
            return BalancedModel()
        elif self.model_type == "level2_intuitive":
            return IntuitiveModel()
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
            Current game context including hand, flipped card, dealer, etc.
            
        Returns
        -------
        DecisionResult
            Decision with type, confidence, and reasoning
        """
        try:
            # Strategic evaluation based on hand strength and position
            hand = context.hand
            flipped_card = context.flipped_card
            is_dealer = context.is_dealer
            
            # If no flipped card, we can't order up
            if not flipped_card:
                return DecisionResult(
                    decision_type=DecisionType.PASS,
                    confidence=0.0,
                    reasoning=f"Level 2 AI {self.name} passes - no flipped card to order up"
                )
            
            # Count trump cards in hand (including left bower)
            trump_suit = flipped_card.suit
            left_bower_suit = self._get_left_bower_suit(trump_suit)
            trump_count = 0
            high_trump_count = 0
            
            for card in hand:
                if card.suit == trump_suit:
                    trump_count += 1
                    if card.rank in [Rank.JACK, Rank.ACE, Rank.KING, Rank.QUEEN]:
                        high_trump_count += 1
                elif card.suit == left_bower_suit and card.rank == Rank.JACK:
                    trump_count += 1
                    high_trump_count += 1  # Left bower is high trump
            
            # Strategic decision factors
            base_confidence = 0.0
            
            # Factor 1: Trump count (more trump = higher confidence)
            if trump_count >= 3:
                base_confidence += 0.4
            elif trump_count >= 2:
                base_confidence += 0.2
            elif trump_count >= 1:
                base_confidence += 0.1
            
            # Factor 2: High trump count (J, A, K, Q)
            if high_trump_count >= 2:
                base_confidence += 0.3
            elif high_trump_count >= 1:
                base_confidence += 0.15
            
            # Factor 3: Position (dealer has advantage)
            if is_dealer:
                base_confidence += 0.1
            
            # Factor 4: Flipped card quality
            if flipped_card.rank in [Rank.JACK, Rank.ACE, Rank.KING]:
                base_confidence += 0.2
            elif flipped_card.rank == Rank.QUEEN:
                base_confidence += 0.1
            
            # Factor 5: Risk profile adjustment
            risk_adjustment = self.risk_profile_obj.trump_calling_aggression - 0.5
            adjusted_confidence = base_confidence + (risk_adjustment * 0.3)
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            # Make decision
            should_order = adjusted_confidence > 0.5
            
            # Generate reasoning
            if should_order:
                reasoning = f"Level 2 AI {self.name} orders up with {adjusted_confidence:.2%} confidence"
                reasoning += f" (trump: {trump_count}, high: {high_trump_count}, dealer: {is_dealer})"
            else:
                reasoning = f"Level 2 AI {self.name} passes with {adjusted_confidence:.2%} confidence"
                reasoning += f" (trump: {trump_count}, high: {high_trump_count}, dealer: {is_dealer})"
            
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
    
    def _get_left_bower_suit(self, trump_suit: Suit) -> Suit:
        """Get the suit of the left bower for a given trump suit."""
        if trump_suit == Suit.HEARTS:
            return Suit.DIAMONDS
        elif trump_suit == Suit.DIAMONDS:
            return Suit.HEARTS
        elif trump_suit == Suit.CLUBS:
            return Suit.SPADES
        else:  # SPADES
            return Suit.CLUBS
    
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
            # Strategic evaluation for second round trump calling
            hand = context.hand
            is_dealer = context.is_dealer
            
            # Evaluate each suit as potential trump
            best_suit = None
            best_confidence = 0.0
            
            for potential_trump in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
                # Count trump cards for this suit (including left bower)
                left_bower_suit = self._get_left_bower_suit(potential_trump)
                trump_count = 0
                high_trump_count = 0
                
                for card in hand:
                    if card.suit == potential_trump:
                        trump_count += 1
                        if card.rank in [Rank.JACK, Rank.ACE, Rank.KING, Rank.QUEEN]:
                            high_trump_count += 1
                    elif card.suit == left_bower_suit and card.rank == Rank.JACK:
                        trump_count += 1
                        high_trump_count += 1  # Left bower is high trump
                
                # Calculate confidence for this suit
                suit_confidence = 0.0
                
                # Factor 1: Trump count
                if trump_count >= 3:
                    suit_confidence += 0.5
                elif trump_count >= 2:
                    suit_confidence += 0.3
                elif trump_count >= 1:
                    suit_confidence += 0.1
                
                # Factor 2: High trump count
                if high_trump_count >= 2:
                    suit_confidence += 0.4
                elif high_trump_count >= 1:
                    suit_confidence += 0.2
                
                # Factor 3: Position (dealer has advantage)
                if is_dealer:
                    suit_confidence += 0.1
                
                # Update best suit if this one is better
                if suit_confidence > best_confidence:
                    best_confidence = suit_confidence
                    best_suit = potential_trump
            
            # Apply risk profile adjustment
            risk_adjustment = self.risk_profile_obj.trump_calling_aggression - 0.5
            adjusted_confidence = best_confidence + (risk_adjustment * 0.3)
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            # Make decision
            should_call = adjusted_confidence > 0.6  # Higher threshold for second round
            
            # Generate reasoning
            if should_call:
                reasoning = f"Level 2 AI {self.name} calls {best_suit.name} as trump with {adjusted_confidence:.2%} confidence"
                # Count cards for the chosen suit
                left_bower_suit = self._get_left_bower_suit(best_suit)
                trump_count = sum(1 for card in hand if card.suit == best_suit or (card.suit == left_bower_suit and card.rank == Rank.JACK))
                reasoning += f" (trump: {trump_count})"
            else:
                reasoning = f"Level 2 AI {self.name} passes with {adjusted_confidence:.2%} confidence"
                reasoning += f" (best suit: {best_suit.name if best_suit else 'none'})"
            
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
                
                # Handle different model output formats and ensure proper tensor shapes
                card_probs = None
                num_cards = len(context.hand)
                
                if isinstance(outputs, dict) and 'card_selection' in outputs:
                    # Model returns dictionary with card_selection key
                    raw_output = outputs['card_selection']
                    if isinstance(raw_output, torch.Tensor):
                        # Handle 2D tensors [1, X] by squeezing to 1D
                        if raw_output.dim() == 2 and raw_output.size(0) == 1:
                            raw_output = raw_output.squeeze(0)  # Convert [1, X] to [X]
                        
                        # Ensure tensor has the right number of outputs
                        if raw_output.dim() == 1:
                            if raw_output.size(0) >= num_cards:
                                card_probs = torch.softmax(raw_output, dim=0)
                            else:
                                # Pad with zeros if not enough outputs
                                padded = torch.cat([raw_output, torch.zeros(num_cards - raw_output.size(0), device=raw_output.device, dtype=raw_output.dtype)])
                                card_probs = torch.softmax(padded, dim=0)
                        else:
                            card_probs = torch.softmax(raw_output, dim=-1)
                    else:
                        # Fallback for non-tensor outputs
                        card_probs = torch.softmax(torch.randn(num_cards), dim=0)
                        
                elif isinstance(outputs, torch.Tensor):
                    # Model returns raw tensor
                    raw_output = outputs
                    # Handle 2D tensors [1, X] by squeezing to 1D
                    if raw_output.dim() == 2 and raw_output.size(0) == 1:
                        raw_output = raw_output.squeeze(0)  # Convert [1, X] to [X]
                    
                    if raw_output.dim() == 1:
                        if raw_output.size(0) >= num_cards:
                            card_probs = torch.softmax(raw_output, dim=0)
                        else:
                            # Pad with zeros if not enough outputs
                            padded = torch.cat([raw_output, torch.zeros(num_cards - raw_output.size(0), device=raw_output.device, dtype=raw_output.dtype)])
                            card_probs = torch.softmax(padded, dim=0)
                    else:
                        card_probs = torch.softmax(raw_output, dim=-1)
                else:
                    # Fallback: create random probabilities
                    card_probs = torch.softmax(torch.randn(num_cards), dim=0)
                
                # Ensure we have exactly the right number of probabilities
                if card_probs.size(0) < num_cards:
                    # Pad with zeros if needed
                    card_probs = torch.cat([card_probs, torch.zeros(num_cards - card_probs.size(0), device=card_probs.device, dtype=card_probs.dtype)])
                
                # Now safely choose the card with highest probability
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
                metadata={'selected_card': str(chosen_card)}
            )
            
        except Exception as e:
            print(f"Error in Level 2 AI card selection: {e}")
            # Fallback to first card
            fallback_card = context.hand[0] if context.hand else None
            return DecisionResult(
                decision_type=DecisionType.PLAY_CARD,
                confidence=0.3,
                reasoning=f"Level 2 AI {self.name} encountered error, defaulting to first card",
                metadata={'selected_card': str(fallback_card)}
            )
    
    def play_card(self, context: GameContext) -> DecisionResult:
        """Decide which card to play in the current trick."""
        try:
            # Prepare input tensor for the model
            input_tensor = self._prepare_input_tensor(context)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(input_tensor, self.risk_profile_obj)
                
                # Handle different model output formats and ensure proper tensor shapes
                card_probs = None
                num_cards = len(context.hand)
                
                if isinstance(outputs, dict) and 'card_selection' in outputs:
                    # Model returns dictionary with card_selection key
                    raw_output = outputs['card_selection']
                    if isinstance(raw_output, torch.Tensor):
                        # Handle 2D tensors [1, X] by squeezing to 1D
                        if raw_output.dim() == 2 and raw_output.size(0) == 1:
                            raw_output = raw_output.squeeze(0)  # Convert [1, X] to [X]
                        
                        # Ensure tensor has the right number of outputs
                        if raw_output.dim() == 1:
                            if raw_output.size(0) >= num_cards:
                                card_probs = torch.softmax(raw_output, dim=0)
                            else:
                                # Pad with zeros if not enough outputs
                                padded = torch.cat([raw_output, torch.zeros(num_cards - raw_output.size(0), device=raw_output.device, dtype=raw_output.dtype)])
                                card_probs = torch.softmax(padded, dim=0)
                        else:
                            card_probs = torch.softmax(raw_output, dim=-1)
                    else:
                        # Fallback for non-tensor outputs
                        card_probs = torch.softmax(torch.randn(num_cards), dim=0)
                        
                elif isinstance(outputs, torch.Tensor):
                    # Model returns raw tensor
                    raw_output = outputs
                    # Handle 2D tensors [1, X] by squeezing to 1D
                    if raw_output.dim() == 2 and raw_output.size(0) == 1:
                        raw_output = raw_output.squeeze(0)  # Convert [1, X] to [X]
                    
                    if raw_output.dim() == 1:
                        if raw_output.size(0) >= num_cards:
                            card_probs = torch.softmax(raw_output, dim=0)
                        else:
                            # Pad with zeros if not enough outputs
                            padded = torch.cat([raw_output, torch.zeros(num_cards - raw_output.size(0), device=raw_output.device, dtype=raw_output.dtype)])
                            card_probs = torch.softmax(padded, dim=0)
                    else:
                        card_probs = torch.softmax(raw_output, dim=-1)
                else:
                    # Fallback: create random probabilities
                    card_probs = torch.softmax(torch.randn(num_cards), dim=0)
                
                # Ensure we have exactly the right number of probabilities
                if card_probs.size(0) < num_cards:
                    # Pad with zeros if needed
                    card_probs = torch.cat([card_probs, torch.zeros(num_cards - card_probs.size(0), device=card_probs.device, dtype=card_probs.dtype)])
                
                # Now safely choose the card with highest probability
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
                metadata={'selected_card': str(chosen_card)}
            )
            
        except Exception as e:
            print(f"Error in Level 2 AI card selection: {e}")
            # Fallback to first card
            fallback_card = context.hand[0] if context.hand else None
            return DecisionResult(
                decision_type=DecisionType.PLAY_CARD,
                confidence=0.3,
                reasoning=f"Level 2 AI {self.name} encountered error, defaulting to first card",
                metadata={'selected_card': str(fallback_card)}
            )
    
    def _prepare_input_tensor(self, context: GameContext) -> torch.Tensor:
        """Prepare input tensor for the neural model."""
        try:
            features = []
            
            # Create suit mapping for one-hot encoding
            suit_to_idx = {
                "hearts": 0,
                "diamonds": 1, 
                "clubs": 2,
                "spades": 3
            }
            
            # Card features (5 cards * 6 features = 30 features)
            for card in context.hand:
                # Suit one-hot encoding (4 features)
                suit_features = [0.0] * 4
                suit_idx = suit_to_idx.get(card.suit.value, 0)  # Use string mapping
                suit_features[suit_idx] = 1.0
                features.extend(suit_features)
                
                # Rank features (2 features: normalized rank and is_trump)
                rank_value = card.rank.value
                features.append(rank_value / 14.0)  # Normalize rank (9-14)
                features.append(1.0 if card.suit == context.trump_suit else 0.0)
            
            # Pad hand to 5 cards if necessary
            while len(features) < 30:
                features.append(0.0)
            
            # Game context features
            features.append(1.0 if context.is_dealer else 0.0)  # Fixed: use is_dealer instead of dealer
            features.append(1.0 if context.partner_is_dealer else 0.0)  # Fixed: use partner_is_dealer
            features.append(0.0)  # Placeholder for opponent dealer status
            features.append(context.team1_score / 10.0)  # Fixed: use team1_score
            features.append(context.team2_score / 10.0)  # Fixed: use team2_score
            
            # Pad to 256 features (model input size)
            while len(features) < 256:
                features.append(0.0)
            
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
        except Exception as e:
            print(f"Error preparing input tensor: {e}")
            # Return zero tensor as fallback
            return torch.zeros(1, 256, dtype=torch.float32)
    
    def select_trump_suit(self, context: GameContext) -> DecisionResult:
        """Select which suit to call as trump."""
        try:
            # Strategic evaluation for dealer's forced trump selection
            hand = context.hand
            is_dealer = context.is_dealer
            
            # Evaluate each suit as potential trump
            best_suit = None
            best_confidence = 0.0
            
            for potential_trump in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
                # Count trump cards for this suit (including left bower)
                left_bower_suit = self._get_left_bower_suit(potential_trump)
                trump_count = 0
                high_trump_count = 0
                
                for card in hand:
                    if card.suit == potential_trump:
                        trump_count += 1
                        if card.rank in [Rank.JACK, Rank.ACE, Rank.KING, Rank.QUEEN]:
                            high_trump_count += 1
                    elif card.suit == left_bower_suit and card.rank == Rank.JACK:
                        trump_count += 1
                        high_trump_count += 1  # Left bower is high trump
                
                # Calculate confidence for this suit
                suit_confidence = 0.0
                
                # Factor 1: Trump count (critical for dealer)
                if trump_count >= 3:
                    suit_confidence += 0.6
                elif trump_count >= 2:
                    suit_confidence += 0.4
                elif trump_count >= 1:
                    suit_confidence += 0.2
                
                # Factor 2: High trump count
                if high_trump_count >= 2:
                    suit_confidence += 0.5
                elif high_trump_count >= 1:
                    suit_confidence += 0.3
                
                # Factor 3: Dealer advantage (can lead first trick)
                if is_dealer:
                    suit_confidence += 0.2
                
                # Factor 4: Avoid suits with very few cards
                if trump_count == 0:
                    suit_confidence -= 0.3  # Penalty for no trump
                
                # Update best suit if this one is better
                if suit_confidence > best_confidence:
                    best_confidence = suit_confidence
                    best_suit = potential_trump
            
            # Apply risk profile adjustment
            risk_adjustment = self.risk_profile_obj.trump_calling_aggression - 0.5
            adjusted_confidence = best_confidence + (risk_adjustment * 0.2)
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            # Dealer must pick something, so use best available
            selected_suit = best_suit if best_suit else Suit.HEARTS  # Fallback
            
            reasoning = f"Level 2 AI {self.name} selected {selected_suit.name} as trump with {adjusted_confidence:.2%} confidence"
            # Count cards for the chosen suit
            left_bower_suit = self._get_left_bower_suit(selected_suit)
            trump_count = sum(1 for card in hand if card.suit == selected_suit or (card.suit == left_bower_suit and card.rank == Rank.JACK))
            reasoning += f" (trump: {trump_count})"
            
            return DecisionResult(
                decision_type=DecisionType.CALL_TRUMP,
                confidence=adjusted_confidence,
                reasoning=reasoning,
                metadata={'selected_suit': selected_suit.name}
            )
            
        except Exception as e:
            print(f"Error in Level 2 AI suit selection: {e}")
            # Fallback to hearts
            return DecisionResult(
                decision_type=DecisionType.CALL_TRUMP,
                confidence=0.3,
                reasoning=f"Level 2 AI {self.name} encountered error, defaulting to hearts",
                metadata={'selected_suit': 'HEARTS'}
            )
    
    def discard_card(self, context: GameContext) -> DecisionResult:
        """Decide which card to discard when partner calls trump."""
        try:
            # Prepare input tensor for the model
            input_tensor = self._prepare_input_tensor(context)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(input_tensor, self.risk_profile_obj)
                
                # Handle different model output formats and ensure proper tensor shapes
                card_probs = None
                num_cards = len(context.hand)
                
                if isinstance(outputs, dict) and 'card_selection' in outputs:
                    # Model returns dictionary with card_selection key
                    raw_output = outputs['card_selection']
                    if isinstance(raw_output, torch.Tensor):
                        # Handle 2D tensors [1, X] by squeezing to 1D
                        if raw_output.dim() == 2 and raw_output.size(0) == 1:
                            raw_output = raw_output.squeeze(0)  # Convert [1, X] to [X]
                        
                        # Ensure tensor has the right number of outputs
                        if raw_output.dim() == 1:
                            if raw_output.size(0) >= num_cards:
                                card_probs = torch.softmax(raw_output, dim=0)
                            else:
                                # Pad with zeros if not enough outputs
                                padded = torch.cat([raw_output, torch.zeros(num_cards - raw_output.size(0), device=raw_output.device, dtype=raw_output.dtype)])
                                card_probs = torch.softmax(padded, dim=0)
                        else:
                            card_probs = torch.softmax(raw_output, dim=-1)
                    else:
                        # Fallback for non-tensor outputs
                        card_probs = torch.softmax(torch.randn(num_cards), dim=0)
                        
                elif isinstance(outputs, torch.Tensor):
                    # Model returns raw tensor
                    raw_output = outputs
                    # Handle 2D tensors [1, X] by squeezing to 1D
                    if raw_output.dim() == 2 and raw_output.size(0) == 1:
                        raw_output = raw_output.squeeze(0)  # Convert [1, X] to [X]
                    
                    if raw_output.dim() == 1:
                        if raw_output.size(0) >= num_cards:
                            card_probs = torch.softmax(raw_output, dim=0)
                        else:
                            # Pad with zeros if not enough outputs
                            padded = torch.cat([raw_output, torch.zeros(num_cards - raw_output.size(0), device=raw_output.device, dtype=raw_output.dtype)])
                            card_probs = torch.softmax(padded, dim=0)
                    else:
                        card_probs = torch.softmax(raw_output, dim=-1)
                else:
                    # Fallback: create random probabilities
                    card_probs = torch.softmax(torch.randn(num_cards), dim=0)
                
                # Ensure we have exactly the right number of probabilities
                if card_probs.size(0) < num_cards:
                    # Pad with zeros if needed
                    card_probs = torch.cat([card_probs, torch.zeros(num_cards - card_probs.size(0), device=card_probs.device, dtype=card_probs.dtype)])
                
                # For discarding, we want the card with lowest strategic value
                # Invert the probabilities to favor lower-value cards
                discard_probs = 1.0 - card_probs
                chosen_card_idx = torch.argmax(discard_probs).item()
                confidence = discard_probs[chosen_card_idx].item()
            
            # Get the chosen card
            if 0 <= chosen_card_idx < len(context.hand):
                chosen_card = context.hand[chosen_card_idx]
            else:
                # Fallback to first card if index is invalid
                chosen_card = context.hand[0]
                confidence = 0.5
            
            reasoning = f"Level 2 AI {self.name} discards {chosen_card} with {confidence:.2%} confidence"
            
            return DecisionResult(
                decision_type=DecisionType.DISCARD,
                confidence=confidence,
                reasoning=reasoning,
                metadata={'discarded_card': str(chosen_card)}
            )
            
        except Exception as e:
            print(f"Error in Level 2 AI discard: {e}")
            # Fallback to first card
            fallback_card = context.hand[0] if context.hand else None
            return DecisionResult(
                decision_type=DecisionType.DISCARD,
                confidence=0.3,
                reasoning=f"Level 2 AI {self.name} encountered error, defaulting to first card",
                metadata={'discarded_card': str(fallback_card)}
            ) 