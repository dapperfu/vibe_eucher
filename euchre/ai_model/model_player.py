"""Model player that uses trained AI models with risk parameters for decision making."""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

from ..models import Player, PlayerType, Card, Suit, Rank
from .euchre_nn import RiskParameters, create_risk_profile


class ModelPlayer(Player):
    """Player that uses a trained AI model with risk parameters for decision making."""
    
    def __init__(self, name: str, model: nn.Module, device: torch.device, 
                 risk_profile: str = "balanced", custom_risk: Optional[RiskParameters] = None):
        """Initialize the model player.
        
        Parameters
        ----------
        name : str
            The player's name
        model : nn.Module
            The trained PyTorch model
        device : torch.device
            Device the model is running on
        risk_profile : str
            Predefined risk profile name
        custom_risk : Optional[RiskParameters]
            Custom risk parameters (overrides risk_profile if provided)
        """
        super().__init__(name, PlayerType.AI)
        self.model = model
        self.device = device
        self.model.eval()  # Set to evaluation mode
        
        # Set up risk parameters
        if custom_risk is not None:
            self.risk_params = custom_risk.to(device)
        else:
            self.risk_params = create_risk_profile(risk_profile).to(device)
        
        # Game context tracking
        self.game_context = {
            'team_score': 0,
            'opponent_score': 0,
            'position': name,
            'current_round': 1,
            'trump_calls_made': 0,
            'aces_ordered': 0,
            'times_set': 0
        }
        
        # Risk adjustment history
        self.risk_adjustments = []
        
    def _get_left_bower_suit(self, trump_suit: Suit) -> Suit:
        """Get the left bower suit for a given trump suit."""
        if trump_suit == Suit.HEARTS:
            return Suit.DIAMONDS
        elif trump_suit == Suit.DIAMONDS:
            return Suit.HEARTS
        elif trump_suit == Suit.CLUBS:
            return Suit.SPADES
        else:  # SPADES
            return Suit.CLUBS
    
    def _encode_game_state(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> torch.Tensor:
        """Encode the current game state for the model.
        
        Parameters
        ----------
        lead_suit : Optional[Suit]
            The lead suit (None if leading)
        trump_suit : Optional[Suit]
            The trump suit (None if not set)
            
        Returns
        -------
        torch.Tensor
            Encoded game state tensor
        """
        # Suit to index mapping
        suit_to_index = {
            "hearts": 0,
            "diamonds": 1,
            "clubs": 2,
            "spades": 3
        }
        
        # Initialize feature vector
        features = []
        
        # Encode hand cards (one-hot encoding for each card)
        hand_encoding = self._encode_hand()
        features.extend(hand_encoding)
        
        # Encode lead suit
        lead_suit_encoding = [0] * 4
        if lead_suit is not None:
            lead_suit_encoding[suit_to_index[lead_suit.value]] = 1
        features.extend(lead_suit_encoding)
        
        # Encode trump suit
        trump_suit_encoding = [0] * 4
        if trump_suit is not None:
            trump_suit_encoding[suit_to_index[trump_suit.value]] = 1
        features.extend(trump_suit_encoding)
        
        # Encode player position (North=0, East=1, South=2, West=3)
        position_encoding = [0] * 4
        if self.name == "North":
            position_encoding[0] = 1
        elif self.name == "East":
            position_encoding[1] = 1
        elif self.name == "South":
            position_encoding[2] = 1
        elif self.name == "West":
            position_encoding[3] = 1
        features.extend(position_encoding)
        
        # Encode game phase (0=trump selection, 1=playing)
        game_phase = [1] if trump_suit is not None else [0]
        features.extend(game_phase)
        
        # Encode game context - ensure all values are numeric
        context_features = [
            float(self.game_context['team_score']) / 10.0,  # Normalize score
            float(self.game_context['opponent_score']) / 10.0,
            float(self.game_context['current_round']) / 20.0,  # Normalize round
            float(self.game_context['trump_calls_made']) / 5.0,  # Normalize calls
            float(self.game_context['aces_ordered']) / 3.0,  # Normalize aces
            float(self.game_context['times_set']) / 3.0  # Normalize sets
        ]
        features.extend(context_features)
        
        # Convert to tensor - ensure device is properly handled
        try:
            # Convert device string to torch.device if needed
            if isinstance(self.device, str):
                device = torch.device(self.device)
            else:
                device = self.device
            
            # Pad feature vector to 128 dimensions to match model input size
            if len(features) < 128:
                features.extend([0.0] * (128 - len(features)))
            elif len(features) > 128:
                features = features[:128]  # Truncate if too long
            
            return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        except Exception as e:
            print(f"Error creating tensor: {e}")
            print(f"Device: {self.device}, Type: {type(self.device)}")
            # Fallback to CPU if device fails
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _encode_hand(self) -> List[float]:
        """Encode the player's hand as a feature vector."""
        # Create a 24-dimensional vector (6 ranks × 4 suits)
        hand_encoding = [0.0] * 24
        
        # Suit to index mapping
        suit_to_index = {
            "hearts": 0,
            "diamonds": 1,
            "clubs": 2,
            "spades": 3
        }
        
        for card in self.hand:
            # Calculate index: rank_index * 4 + suit_index
            # Convert rank string to index (9=0, 10=1, J=2, Q=3, K=4, A=5)
            rank_to_index = {'9': 0, '10': 1, 'J': 2, 'Q': 3, 'K': 4, 'A': 5}
            rank_index = rank_to_index.get(card.rank.value, 0)
            suit_index = suit_to_index[card.suit.value]  # Convert string suit to integer index
            card_index = rank_index * 4 + suit_index
            
            if 0 <= card_index < 24:
                hand_encoding[card_index] = 1.0
        
        return hand_encoding
    
    def _get_adaptive_risk(self) -> RiskParameters:
        """Get adaptively adjusted risk parameters based on game context."""
        # Check if model supports adaptive risk
        if hasattr(self.model, 'get_adaptive_risk'):
            return self.model.get_adaptive_risk(self.risk_params, self.game_context)
        
        # Simple adaptive logic if model doesn't support it
        base_risk = self.risk_params
        
        # Create new risk parameters with slight adjustments
        return RiskParameters(
            trump_calling_aggression=base_risk.trump_calling_aggression.item(),
            ace_ordering_risk=base_risk.ace_ordering_risk.item(),
            set_risk_tolerance=base_risk.set_risk_tolerance.item(),
            leading_aggression=base_risk.leading_aggression.item()
        )
    
    def _decode_model_output(self, model_output: torch.Tensor, lead_suit: Optional[Suit], 
                           trump_suit: Optional[Suit]) -> Card:
        """Decode model output to select a card with risk awareness."""
        # Get valid cards to play
        valid_cards = self._get_valid_cards(lead_suit, trump_suit)
        
        if not valid_cards:
            raise ValueError("No valid cards to play")
        
        if len(valid_cards) == 1:
            return valid_cards[0]
        
        # Get adaptive risk parameters
        adaptive_risk = self._get_adaptive_risk()
        
        # Convert model output to probabilities
        with torch.no_grad():
            # Apply softmax to get probabilities
            probabilities = torch.softmax(model_output, dim=1)
            
            # Map probabilities to valid cards
            card_scores = self._map_output_to_cards(probabilities, valid_cards, adaptive_risk)
            
            # Select card with highest score
            best_card_idx = np.argmax(card_scores)
            return valid_cards[best_card_idx]
    
    def _get_valid_cards(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> List[Card]:
        """Get list of valid cards to play."""
        if not lead_suit:
            # Leading - can play any card
            return self.hand.copy()
        
        # Must follow suit if possible
        cards_of_suit = self.get_cards_of_suit(lead_suit)
        if cards_of_suit:
            return cards_of_suit
        
        # Can't follow suit - can play any card
        return self.hand.copy()
    
    def _map_output_to_cards(self, probabilities: torch.Tensor, valid_cards: List[Card], 
                            risk_params: RiskParameters) -> np.ndarray:
        """Map model output probabilities to valid cards with risk awareness."""
        card_scores = []
        
        # Suit to index mapping
        suit_to_index = {
            "hearts": 0,
            "diamonds": 1,
            "clubs": 2,
            "spades": 3
        }
        
        # Get the output size from the probabilities tensor
        output_size = probabilities.size(1)
        
        for card in valid_cards:
            # Base score from model probabilities
            # Convert rank string to index (9=0, 10=1, J=2, Q=3, K=4, A=5)
            rank_to_index = {'9': 0, '10': 1, 'J': 2, 'Q': 3, 'K': 4, 'A': 5}
            rank_index = rank_to_index.get(card.rank.value, 0)
            suit_index = suit_to_index[card.suit.value]  # Convert string suit to integer index
            card_index = rank_index * 4 + suit_index
            
            # Handle different output sizes
            if output_size == 24:  # Integer model: 24 possible cards
                if 0 <= card_index < 24:
                    base_score = probabilities[0, card_index].item()
                else:
                    base_score = 0.0
            elif output_size == 5:  # Float model: 5 cards in hand
                # For float model, we need to map the card to a hand position
                # Use a simple heuristic: map by rank and suit priority
                hand_index = min(card_index % 5, output_size - 1)
                base_score = probabilities[0, hand_index].item()
            else:
                # Fallback for other output sizes
                base_score = 0.0
            
            # Risk-based adjustments
            risk_adjustment = 0.0
            
            # Leading aggression adjustment
            if not hasattr(self, '_current_lead_suit') or self._current_lead_suit is None:
                # We're leading
                if risk_params.leading_aggression.item() > 0.7:
                    # Very aggressive - boost high cards
                    # Convert rank string to index for comparison (9=0, 10=1, J=2, Q=3, K=4, A=5)
                    rank_to_index = {'9': 0, '10': 1, 'J': 2, 'Q': 3, 'K': 4, 'A': 5}
                    card_rank_index = rank_to_index.get(card.rank.value, 0)
                    if card_rank_index >= 4:  # K or A
                        risk_adjustment += 0.3
                elif risk_params.leading_aggression.item() < 0.3:
                    # Very conservative - boost low cards
                    if card_rank_index <= 1:  # 9 or 10
                        risk_adjustment += 0.3
            
            # Trump awareness
            if hasattr(self, '_current_trump_suit') and self._current_trump_suit:
                if card.suit == self._current_trump_suit:
                    # Trump card - boost based on set risk tolerance
                    if risk_params.set_risk_tolerance.item() > 0.7:
                        risk_adjustment += 0.4  # High risk tolerance
                    elif risk_params.set_risk_tolerance.item() < 0.3:
                        risk_adjustment += 0.1  # Low risk tolerance
                    else:
                        risk_adjustment += 0.2  # Moderate
                
                # Bower awareness
                if card.rank == Rank.JACK:
                    if card.suit == self._current_trump_suit:
                        risk_adjustment += 0.5  # Right bower
                    else:
                        # Check if left bower
                        left_bower_suit = self._get_left_bower_suit(self._current_trump_suit)
                        if card.suit == left_bower_suit:
                            risk_adjustment += 0.4  # Left bower
            
            # Final score
            final_score = base_score + risk_adjustment
            card_scores.append(final_score)
        
        # Normalize scores
        if card_scores:
            max_score = max(card_scores)
            if max_score > 0:
                card_scores = [score / max_score for score in card_scores]
        
        return np.array(card_scores)
    
    def should_order_up(self, top_card: Card) -> bool:
        """Decide whether to order up the top card using risk-aware model."""
        try:
            # Update game context
            self._current_trump_suit = top_card.suit
            
            # Get adaptive risk parameters
            adaptive_risk = self._get_adaptive_risk()
            
            # Encode the decision state
            state_tensor = self._encode_decision_state(top_card)
            
            # Get model prediction
            with torch.no_grad():
                model_output = self.model(state_tensor, adaptive_risk)
                
                # Extract trump decision
                if 'trump_decision' in model_output:
                    trump_probs = torch.softmax(model_output['trump_decision'], dim=1)
                    order_up_prob = trump_probs[0, 1].item()  # Probability of ordering up
                else:
                    # Fallback for models without specific output structure
                    order_up_prob = 0.5
                
                # Apply risk-based threshold adjustment
                base_threshold = 0.5
                risk_adjustment = (adaptive_risk.trump_calling_aggression.item() - 0.5) * 0.4
                adjusted_threshold = base_threshold - risk_adjustment
                
                # Special handling for aces
                if top_card.rank == Rank.ACE:
                    ace_risk_factor = adaptive_risk.ace_ordering_risk.item()
                    if ace_risk_factor < 0.3:
                        # Very conservative with aces
                        adjusted_threshold += 0.3
                    elif ace_risk_factor > 0.7:
                        # Very aggressive with aces
                        adjusted_threshold -= 0.2
                
                # Make decision
                should_order = order_up_prob > adjusted_threshold
                
                # Update context
                if should_order:
                    self.game_context['trump_calls_made'] += 1
                    if top_card.rank == Rank.ACE:
                        self.game_context['aces_ordered'] += 1
                
                return should_order
                
        except Exception as e:
            # Fallback to heuristic if model fails
            print(f"Model prediction failed: {e}, using fallback heuristic")
            return self._fallback_order_up_decision(top_card)
    
    def _encode_decision_state(self, top_card: Card) -> torch.Tensor:
        """Encode the state for trump ordering decision."""
        # Initialize feature vector
        features = []
        
        # Encode hand cards
        hand_encoding = self._encode_hand()
        features.extend(hand_encoding)
        
        # Encode top card - convert string values to integers
        top_card_encoding = [0] * 24
        
        # Convert rank string to index (9=0, 10=1, J=2, Q=3, K=4, A=5)
        rank_to_index = {'9': 0, '10': 1, 'J': 2, 'Q': 3, 'K': 4, 'A': 5}
        rank_index = rank_to_index.get(top_card.rank.value, 0)
        
        # Convert suit string to index (hearts=0, diamonds=1, clubs=2, spades=3)
        suit_to_index = {'hearts': 0, 'diamonds': 1, 'clubs': 2, 'spades': 3}
        suit_index = suit_to_index.get(top_card.suit.value, 0)
        
        card_index = rank_index * 4 + suit_index
        if 0 <= card_index < 24:
            top_card_encoding[card_index] = 1
        features.extend(top_card_encoding)
        
        # Encode potential trump suit - convert suit string to index
        trump_suit_encoding = [0] * 4
        trump_suit_encoding[suit_index] = 1
        features.extend(trump_suit_encoding)
        
        # Encode player position
        position_encoding = [0] * 4
        if self.name == "North":
            position_encoding[0] = 1
        elif self.name == "East":
            position_encoding[1] = 1
        elif self.name == "South":
            position_encoding[2] = 1
        elif self.name == "West":
            position_encoding[3] = 1
        features.extend(position_encoding)
        
        # Encode game context - ensure all values are numeric
        context_features = [
            float(self.game_context['team_score']) / 10.0,
            float(self.game_context['opponent_score']) / 10.0,
            float(self.game_context['current_round']) / 20.0,
            float(self.game_context['trump_calls_made']) / 5.0
        ]
        features.extend(context_features)
        
        # Debug: check feature types
        for i, feature in enumerate(features):
            if not isinstance(feature, (int, float)):
                print(f"Warning: Decision feature {i} is {type(feature)}: {feature}")
        
        # Convert to tensor - ensure device is properly handled
        try:
            # Convert device string to torch.device if needed
            if isinstance(self.device, str):
                device = torch.device(self.device)
            else:
                device = self.device
            
            # Pad feature vector to 128 dimensions to match model input size
            if len(features) < 128:
                features.extend([0.0] * (128 - len(features)))
            elif len(features) > 128:
                features = features[:128]  # Truncate if too long
            
            return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        except Exception as e:
            print(f"Error creating tensor: {e}")
            print(f"Device: {self.device}, Type: {type(self.device)}")
            # Fallback to CPU if device fails
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _fallback_order_up_decision(self, top_card: Card) -> bool:
        """Fallback heuristic for trump ordering decision."""
        # Count cards of the potential trump suit
        cards_of_suit = self.get_cards_of_suit(top_card.suit)
        
        # Count left bower
        left_bower_suit = self._get_left_bower_suit(top_card.suit)
        left_bower_cards = self.get_cards_of_suit(left_bower_suit)
        
        total_trump_potential = len(cards_of_suit) + len(left_bower_cards)
        
        # Adjust threshold based on risk parameters
        base_threshold = 3
        risk_adjustment = int((self.risk_params.trump_calling_aggression.item() - 0.5) * 2)
        adjusted_threshold = max(1, base_threshold - risk_adjustment)
        
        # Special handling for aces
        if top_card.rank == Rank.ACE:
            ace_threshold_adjustment = int((0.5 - self.risk_params.ace_ordering_risk.item()) * 2)
            adjusted_threshold += ace_threshold_adjustment
        
        return total_trump_potential >= adjusted_threshold
    
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play using risk-aware model."""
        try:
            # Update game context
            self._current_lead_suit = lead_suit
            self._current_trump_suit = trump_suit
            
            # Encode game state
            state_tensor = self._encode_game_state(lead_suit, trump_suit)
            
            # Get model prediction
            model_output = self.model(state_tensor, self.risk_params)
            
            # Extract card probabilities
            if isinstance(model_output, dict):
                card_probs = model_output['card_selection']
            else:
                # Fallback for models that return tuple
                card_probs = model_output[1] if len(model_output) > 1 else torch.randn(1, 24)
            
            # Decode model output to select card
            selected_card = self._decode_model_output(card_probs, lead_suit, trump_suit)
            
            # Remove selected card from hand
            self.hand.remove(selected_card)
            
            return selected_card
            
        except Exception as e:
            # Fallback to heuristic-based card selection
            return self._fallback_card_selection(lead_suit, trump_suit)
    
    def _fallback_card_selection(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Fallback heuristic for card selection."""
        valid_cards = self._get_valid_cards(lead_suit, trump_suit)
        
        if not valid_cards:
            raise ValueError("No valid cards to play")
        
        if len(valid_cards) == 1:
            card = valid_cards[0]
            self.hand.remove(card)
            return card
        
        # Risk-aware card selection
        adaptive_risk = self._get_adaptive_risk()
        
        if not lead_suit:
            # Leading - use leading aggression
            if adaptive_risk.leading_aggression.item() > 0.7:
                # Aggressive - play highest card
                # Convert rank string to index for comparison (9=0, 10=1, J=2, Q=3, K=4, A=5)
                rank_to_index = {'9': 0, '10': 1, 'J': 2, 'Q': 3, 'K': 4, 'A': 5}
                best_card = max(valid_cards, key=lambda c: rank_to_index.get(c.rank.value, 0))
            elif adaptive_risk.leading_aggression.item() < 0.3:
                # Conservative - play lowest card
                best_card = min(valid_cards, key=lambda c: rank_to_index.get(c.rank.value, 0))
            else:
                # Balanced - play middle card
                sorted_cards = sorted(valid_cards, key=lambda c: rank_to_index.get(c.rank.value, 0))
                best_card = sorted_cards[len(sorted_cards) // 2]
        else:
            # Following suit - use set risk tolerance
            if adaptive_risk.set_risk_tolerance.item() > 0.7:
                # High risk tolerance - play highest card
                best_card = max(valid_cards, key=lambda c: rank_to_index.get(c.rank.value, 0))
            elif adaptive_risk.set_risk_tolerance.item() < 0.3:
                # Low risk tolerance - play lowest card
                best_card = min(valid_cards, key=lambda c: rank_to_index.get(c.rank.value, 0))
            else:
                # Moderate - play middle card
                sorted_cards = sorted(valid_cards, key=lambda c: rank_to_index.get(c.rank.value, 0))
                best_card = sorted_cards[len(sorted_cards) // 2]
        
        self.hand.remove(best_card)
        return best_card
    
    def update_game_context(self, team_score: int, opponent_score: int, 
                           current_round: int, was_set: bool = False):
        """Update game context information."""
        self.game_context['team_score'] = team_score
        self.game_context['opponent_score'] = opponent_score
        self.game_context['current_round'] = current_round
        
        if was_set:
            self.game_context['times_set'] += 1
        
        # Update risk parameters if model supports it
        if hasattr(self.model, 'update_risk_dynamics'):
            try:
                state_tensor = self._encode_game_state(None, None)
                self.risk_params = self.model.update_risk_dynamics(
                    state_tensor, self.risk_params
                )
                self.risk_adjustments.append(self.risk_params)
            except Exception as e:
                print(f"Risk dynamics update failed: {e}")
    
    def get_risk_summary(self) -> Dict[str, float]:
        """Get a summary of current risk parameters."""
        return {
            'trump_calling_aggression': self.risk_params.trump_calling_aggression.item(),
            'ace_ordering_risk': self.risk_params.ace_ordering_risk.item(),
            'set_risk_tolerance': self.risk_params.set_risk_tolerance.item(),
            'leading_aggression': self.risk_params.leading_aggression.item()
        }
    
    def set_risk_profile(self, profile_name: str):
        """Change the risk profile during gameplay."""
        new_risk = create_risk_profile(profile_name).to(self.device)
        self.risk_params = new_risk
        print(f"Risk profile changed to: {profile_name}")
    
    def adjust_risk_parameters(self, **kwargs):
        """Manually adjust risk parameters."""
        current_risk = self.risk_params
        
        new_risk = RiskParameters(
            trump_calling_aggression=kwargs.get('trump_calling_aggression', 
                                             current_risk.trump_calling_aggression.item()),
            ace_ordering_risk=kwargs.get('ace_ordering_risk', 
                                       current_risk.ace_ordering_risk.item()),
            set_risk_tolerance=kwargs.get('set_risk_tolerance', 
                                        current_risk.set_risk_tolerance.item()),
            leading_aggression=kwargs.get('leading_aggression', 
                                        current_risk.leading_aggression.item())
        ).to(self.device)
        
        self.risk_params = new_risk
        print("Risk parameters adjusted manually") 