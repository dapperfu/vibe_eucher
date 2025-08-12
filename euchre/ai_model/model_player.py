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
        # Initialize feature vector
        features = []
        
        # Encode hand cards (one-hot encoding for each card)
        hand_encoding = self._encode_hand()
        features.extend(hand_encoding)
        
        # Encode lead suit
        lead_suit_encoding = [0] * 4
        if lead_suit is not None:
            lead_suit_encoding[lead_suit.value] = 1
        features.extend(lead_suit_encoding)
        
        # Encode trump suit
        trump_suit_encoding = [0] * 4
        if trump_suit is not None:
            trump_suit_encoding[trump_suit.value] = 1
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
        
        # Encode game context
        context_features = [
            self.game_context['team_score'] / 10.0,  # Normalize score
            self.game_context['opponent_score'] / 10.0,
            self.game_context['current_round'] / 20.0,  # Normalize round
            self.game_context['trump_calls_made'] / 5.0,  # Normalize calls
            self.game_context['aces_ordered'] / 3.0,  # Normalize aces
            self.game_context['times_set'] / 3.0  # Normalize sets
        ]
        features.extend(context_features)
        
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def _encode_hand(self) -> List[float]:
        """Encode the player's hand as a feature vector."""
        # Create a 24-dimensional vector (6 ranks × 4 suits)
        hand_encoding = [0.0] * 24
        
        for card in self.hand:
            # Calculate index: rank_index * 4 + suit_index
            card_index = (card.rank.value - 9) * 4 + card.suit.value
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
        
        # Adjust based on score difference
        score_diff = self.game_context['team_score'] - self.game_context['opponent_score']
        
        if score_diff < -3:  # Behind significantly
            # Increase trump calling aggression
            adjusted_risk = RiskParameters(
                trump_calling_aggression=min(1.0, base_risk.trump_calling_aggression + 0.2),
                ace_ordering_risk=min(1.0, base_risk.ace_ordering_risk + 0.1),
                set_risk_tolerance=min(1.0, base_risk.set_risk_tolerance + 0.2),
                leading_aggression=base_risk.leading_aggression
            )
        elif score_diff > 3:  # Ahead significantly
            # Decrease risk
            adjusted_risk = RiskParameters(
                trump_calling_aggression=max(0.0, base_risk.trump_calling_aggression - 0.2),
                ace_ordering_risk=max(0.0, base_risk.ace_ordering_risk - 0.2),
                set_risk_tolerance=max(0.0, base_risk.set_risk_tolerance - 0.2),
                leading_aggression=max(0.0, base_risk.leading_aggression - 0.1)
            )
        else:
            # Close game - use base risk
            adjusted_risk = base_risk
        
        return adjusted_risk
    
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
        
        for card in valid_cards:
            # Base score from model probabilities
            card_index = (card.rank.value - 9) * 4 + card.suit.value
            if 0 <= card_index < 24:
                base_score = probabilities[0, card_index].item()
            else:
                base_score = 0.0
            
            # Risk-based adjustments
            risk_adjustment = 0.0
            
            # Leading aggression adjustment
            if not hasattr(self, '_current_lead_suit') or self._current_lead_suit is None:
                # We're leading
                if risk_params.leading_aggression > 0.7:
                    # Very aggressive - boost high cards
                    if card.rank.value >= Rank.KING.value:
                        risk_adjustment += 0.3
                elif risk_params.leading_aggression < 0.3:
                    # Very conservative - boost low cards
                    if card.rank.value <= Rank.TEN.value:
                        risk_adjustment += 0.3
            
            # Trump awareness
            if hasattr(self, '_current_trump_suit') and self._current_trump_suit:
                if card.suit == self._current_trump_suit:
                    # Trump card - boost based on set risk tolerance
                    if risk_params.set_risk_tolerance > 0.7:
                        risk_adjustment += 0.4  # High risk tolerance
                    elif risk_params.set_risk_tolerance < 0.3:
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
                risk_adjustment = (adaptive_risk.trump_calling_aggression - 0.5) * 0.4
                adjusted_threshold = base_threshold - risk_adjustment
                
                # Special handling for aces
                if top_card.rank == Rank.ACE:
                    ace_risk_factor = adaptive_risk.ace_ordering_risk
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
        
        # Encode top card
        top_card_encoding = [0] * 24
        card_index = (top_card.rank.value - 9) * 4 + top_card.suit.value
        if 0 <= card_index < 24:
            top_card_encoding[card_index] = 1
        features.extend(top_card_encoding)
        
        # Encode potential trump suit
        trump_suit_encoding = [0] * 4
        trump_suit_encoding[top_card.suit.value] = 1
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
        
        # Encode game context
        context_features = [
            self.game_context['team_score'] / 10.0,
            self.game_context['opponent_score'] / 10.0,
            self.game_context['current_round'] / 20.0,
            self.game_context['trump_calls_made'] / 5.0
        ]
        features.extend(context_features)
        
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
    
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
        risk_adjustment = int((self.risk_params.trump_calling_aggression - 0.5) * 2)
        adjusted_threshold = max(1, base_threshold - risk_adjustment)
        
        # Special handling for aces
        if top_card.rank == Rank.ACE:
            ace_threshold_adjustment = int((0.5 - self.risk_params.ace_ordering_risk) * 2)
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
            with torch.no_grad():
                model_output = self.model(state_tensor, self.risk_params)
                
                # Extract card selection
                if 'card_selection' in model_output:
                    card_probs = model_output['card_selection']
                else:
                    # Fallback for models without specific output structure
                    card_probs = torch.ones(1, 24) / 24
                
                # Decode to card selection
                selected_card = self._decode_model_output(card_probs, lead_suit, trump_suit)
                
                # Remove card from hand
                if selected_card in self.hand:
                    self.hand.remove(selected_card)
                
                return selected_card
                
        except Exception as e:
            # Fallback to heuristic if model fails
            print(f"Model prediction failed: {e}, using fallback heuristic")
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
            if adaptive_risk.leading_aggression > 0.7:
                # Aggressive - play highest card
                best_card = max(valid_cards, key=lambda c: c.rank.value)
            elif adaptive_risk.leading_aggression < 0.3:
                # Conservative - play lowest card
                best_card = min(valid_cards, key=lambda c: c.rank.value)
            else:
                # Balanced - play middle card
                sorted_cards = sorted(valid_cards, key=lambda c: c.rank.value)
                best_card = sorted_cards[len(sorted_cards) // 2]
        else:
            # Following suit - use set risk tolerance
            if adaptive_risk.set_risk_tolerance > 0.7:
                # High risk tolerance - play highest card
                best_card = max(valid_cards, key=lambda c: c.rank.value)
            elif adaptive_risk.set_risk_tolerance < 0.3:
                # Low risk tolerance - play lowest card
                best_card = min(valid_cards, key=lambda c: c.rank.value)
            else:
                # Moderate - play middle card
                sorted_cards = sorted(valid_cards, key=lambda c: c.rank.value)
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