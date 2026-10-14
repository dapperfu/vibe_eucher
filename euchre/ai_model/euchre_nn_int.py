"""
Integer-based Euchre Neural Network for performance benchmarking.

This module provides an integer-based implementation of the EuchreNN to compare
performance against the float-based version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from ..models import Card, Suit, Rank


@dataclass
class RiskParametersInt:
    """Integer-based risk parameters for the Euchre AI model.
    
    All values are integers in range [0, 100] for faster computation.
    """
    trump_calling_aggression: int  # 0-100: Higher = more aggressive trump calling
    ace_ordering_risk: int         # 0-100: Higher = more willing to order up aces
    set_risk_tolerance: int        # 0-100: Higher = more willing to risk being set
    leading_aggression: int        # 0-100: Higher = more aggressive leading
    
    def __post_init__(self):
        """Clamp values to valid range."""
        self.trump_calling_aggression = max(0, min(100, self.trump_calling_aggression))
        self.ace_ordering_risk = max(0, min(100, self.ace_ordering_risk))
        self.set_risk_tolerance = max(0, min(100, self.set_risk_tolerance))
        self.leading_aggression = max(0, min(100, self.leading_aggression))
    
    def get_risk_vector(self) -> torch.Tensor:
        """Convert to tensor for neural network input."""
        return torch.tensor([
            self.trump_calling_aggression / 100.0,  # Convert back to 0-1 for compatibility
            self.ace_ordering_risk / 100.0,
            self.set_risk_tolerance / 100.0,
            self.leading_aggression / 100.0
        ], dtype=torch.float32)
    
    @classmethod
    def from_float(cls, risk_params) -> 'RiskParametersInt':
        """Convert from float-based RiskParameters."""
        return cls(
            trump_calling_aggression=int(risk_params.trump_calling_aggression * 100),
            ace_ordering_risk=int(risk_params.ace_ordering_risk * 100),
            set_risk_tolerance=int(risk_params.set_risk_tolerance * 100),
            leading_aggression=int(risk_params.leading_aggression * 100)
        )


class EuchreNNInt(nn.Module):
    """Integer-based Euchre Neural Network for performance benchmarking."""
    
    def __init__(self, input_size: int = 128, hidden_size: int = 256, 
                 risk_embedding_size: int = 32, device: str = "cpu"):
        super().__init__()
        
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.risk_embedding_size = risk_embedding_size
        
        # Main network layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size // 2)
        ])
        
        # Risk parameter embedding
        self.risk_embedding = nn.Linear(4, risk_embedding_size)
        
        # Output layers
        self.trump_decision = nn.Linear(hidden_size // 2 + risk_embedding_size, 1)
        self.card_selection = nn.Linear(hidden_size // 2 + risk_embedding_size, 52)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Move to device
        self.to(device)
    
    def forward(self, x: torch.Tensor, risk_params: RiskParametersInt) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with integer-based risk parameters.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            risk_params: Integer-based risk parameters
            
        Returns:
            Tuple of (trump_decision_logits, card_selection_logits)
        """
        batch_size = x.size(0)
        
        # Ensure input is integer-based (quantize to integers)
        x_int = torch.round(x * 1000).to(torch.int32)  # Scale up and convert to int
        x_float = x_int.float() / 1000.0  # Convert back to float for neural network
        
        # Main network forward pass
        x_float = F.relu(self.input_layer(x_float))
        x_float = self.dropout(x_float)
        
        for layer in self.hidden_layers:
            x_float = F.relu(layer(x_float))
            x_float = self.dropout(x_float)
        
        # Embed risk parameters (convert from int to float for embedding)
        risk_vector = risk_params.get_risk_vector().unsqueeze(0).expand(batch_size, -1)
        risk_embedded = self.risk_embedding(risk_vector)
        
        # Concatenate features
        combined_input = torch.cat([x_float, risk_embedded], dim=1)
        
        # Output layers
        trump_logits = self.trump_decision(combined_input)
        card_logits = self.card_selection(combined_input)
        
        return trump_logits, card_logits
    
    def get_trump_decision_probs(self, x: torch.Tensor, risk_params: RiskParametersInt) -> torch.Tensor:
        """Get trump decision probabilities."""
        trump_logits, _ = self.forward(x, risk_params)
        return torch.sigmoid(trump_logits)
    
    def get_card_selection_probs(self, x: torch.Tensor, risk_params: RiskParametersInt) -> torch.Tensor:
        """Get card selection probabilities."""
        _, card_logits = self.forward(x, risk_params)
        return F.softmax(card_logits, dim=1)
    
    def get_risk_adjusted_probs(self, x: torch.Tensor, risk_params: RiskParametersInt, 
                               adjustment_factor: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get risk-adjusted probabilities."""
        trump_probs = self.get_trump_decision_probs(x, risk_params)
        card_probs = self.get_card_selection_probs(x, risk_params)
        
        # Apply risk adjustments
        if adjustment_factor != 1.0:
            trump_probs = torch.clamp(trump_probs * adjustment_factor, 0.0, 1.0)
            card_probs = F.softmax(card_probs * adjustment_factor, dim=1)
        
        return trump_probs, card_probs


class RiskAwareEuchreNNInt(EuchreNNInt):
    """Integer-based risk-aware Euchre neural network with dynamic risk adjustment."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_history = []
    
    def update_risk_dynamics(self, game_context: dict) -> None:
        """Update risk parameters based on game context."""
        # Store risk history for analysis
        self.risk_history.append({
            'context': game_context.copy(),
            'timestamp': torch.cuda.Event() if torch.cuda.is_available() else None
        })
    
    def get_adaptive_risk(self, base_risk: RiskParametersInt, game_context: dict) -> RiskParametersInt:
        """Get adaptively adjusted risk parameters."""
        # Simple adaptive logic based on score difference
        score_diff = game_context.get('team_score', 0) - game_context.get('opponent_score', 0)
        
        if score_diff < -3:  # Behind significantly
            # Increase risk
            adjusted_risk = RiskParametersInt(
                trump_calling_aggression=min(100, base_risk.trump_calling_aggression + 20),
                ace_ordering_risk=min(100, base_risk.ace_ordering_risk + 10),
                set_risk_tolerance=min(100, base_risk.set_risk_tolerance + 20),
                leading_aggression=base_risk.leading_aggression
            )
        elif score_diff > 3:  # Ahead significantly
            # Decrease risk
            adjusted_risk = RiskParametersInt(
                trump_calling_aggression=max(0, base_risk.trump_calling_aggression - 20),
                ace_ordering_risk=max(0, base_risk.ace_ordering_risk - 20),
                set_risk_tolerance=max(0, base_risk.set_risk_tolerance - 20),
                leading_aggression=max(0, base_risk.leading_aggression - 10)
            )
        else:
            # Balanced - use base risk
            adjusted_risk = base_risk
        
        return adjusted_risk


def create_euchre_model_int(input_size: int = 128, hidden_size: int = 256, 
                           risk_embedding_size: int = 32, device: str = "cpu") -> EuchreNNInt:
    """Create an integer-based Euchre neural network model."""
    return EuchreNNInt(
        input_size=input_size,
        hidden_size=hidden_size,
        risk_embedding_size=risk_embedding_size,
        device=device
    )


def create_risk_profile_int(profile_name: str) -> RiskParametersInt:
    """Create predefined integer-based risk profiles."""
    profiles = {
        "aggressive": RiskParametersInt(80, 70, 75, 85),
        "conservative": RiskParametersInt(20, 15, 25, 20),
        "balanced": RiskParametersInt(50, 50, 50, 50),
        "opportunistic": RiskParametersInt(60, 80, 65, 40)
    }
    
    if profile_name not in profiles:
        raise ValueError(f"Unknown risk profile: {profile_name}")
    
    return profiles[profile_name] 