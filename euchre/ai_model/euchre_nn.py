"""Enhanced neural network for euchre with risk parameters."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class RiskParameters:
    """Risk parameters that control the AI's playing style."""
    
    def __init__(self, 
                 trump_calling_aggression: float = 0.5,
                 ace_ordering_risk: float = 0.5,
                 set_risk_tolerance: float = 0.5,
                 leading_aggression: float = 0.5):
        """Initialize risk parameters.
        
        Parameters
        ----------
        trump_calling_aggression : float
            How aggressively to call trump (0.0 = very conservative, 1.0 = very aggressive)
        ace_ordering_risk : float
            Willingness to order up aces to opposing teams (0.0 = never, 1.0 = always)
        set_risk_tolerance : float
            Tolerance for being set (0.0 = avoid at all costs, 1.0 = high risk for high reward)
        leading_aggression : float
            How aggressively to lead (0.0 = lead low, 1.0 = lead high)
        """
        self.trump_calling_aggression = torch.clamp(torch.tensor(trump_calling_aggression), 0.0, 1.0)
        self.ace_ordering_risk = torch.clamp(torch.tensor(ace_ordering_risk), 0.0, 1.0)
        self.set_risk_tolerance = torch.clamp(torch.tensor(set_risk_tolerance), 0.0, 1.0)
        self.leading_aggression = torch.clamp(torch.tensor(leading_aggression), 0.0, 1.0)
    
    def to(self, device: torch.device) -> 'RiskParameters':
        """Move risk parameters to device."""
        return RiskParameters(
            trump_calling_aggression=self.trump_calling_aggression.to(device),
            ace_ordering_risk=self.ace_ordering_risk.to(device),
            set_risk_tolerance=self.set_risk_tolerance.to(device),
            leading_aggression=self.leading_aggression.to(device)
        )
    
    def get_risk_vector(self) -> torch.Tensor:
        """Get risk parameters as a tensor."""
        return torch.stack([
            self.trump_calling_aggression,
            self.ace_ordering_risk,
            self.set_risk_tolerance,
            self.leading_aggression
        ])


class EuchreNN(nn.Module):
    """Enhanced neural network for euchre with risk-aware decision making."""
    
    def __init__(self, 
                 input_size: int = 128,
                 hidden_size: int = 256,
                 output_size: int = 64,
                 risk_embedding_size: int = 32,
                 use_risk_attention: bool = True):
        """Initialize the enhanced euchre neural network.
        
        Parameters
        ----------
        input_size : int
            Size of input feature vector
        hidden_size : int
            Size of hidden layers
        output_size : int
            Size of output layer
        risk_embedding_size : int
            Size of risk parameter embeddings
        use_risk_attention : bool
            Whether to use attention mechanism for risk parameters
        """
        super(EuchreNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.risk_embedding_size = risk_embedding_size
        self.use_risk_attention = use_risk_attention
        
        # Risk parameter embedding
        self.risk_embedding = nn.Linear(4, risk_embedding_size)
        
        # Main network layers
        self.input_layer = nn.Linear(input_size + risk_embedding_size, hidden_size)
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Risk-aware attention mechanism
        if use_risk_attention:
            self.risk_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                batch_first=True
            )
            self.risk_query = nn.Linear(risk_embedding_size, hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Risk-specific output heads
        self.trump_decision_head = nn.Linear(hidden_size, 2)  # Order up or not
        self.card_selection_head = nn.Linear(hidden_size, 5)  # 5 cards in hand
        self.risk_adjustment_head = nn.Linear(hidden_size, 4)  # Risk parameter adjustments
        
    def forward(self, x: torch.Tensor, risk_params: RiskParameters) -> Dict[str, torch.Tensor]:
        """Forward pass with risk parameters.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features [batch_size, input_size]
        risk_params : RiskParameters
            Risk parameters for this forward pass
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing various outputs
        """
        batch_size = x.size(0)
        
        # Embed risk parameters
        risk_vector = risk_params.get_risk_vector().unsqueeze(0).expand(batch_size, -1)
        risk_embedded = self.risk_embedding(risk_vector)
        
        # Debug: check tensor types and shapes (commented out for training)
        # print(f"Input tensor x: shape={x.shape}, dtype={x.dtype}")
        # print(f"Risk embedded: shape={risk_embedded.shape}, dtype={risk_embedded.dtype}")
        
        # Concatenate input features with risk embeddings
        try:
            combined_input = torch.cat([x, risk_embedded], dim=1)
        except Exception as e:
            print(f"Error in torch.cat: {e}")
            print(f"x content sample: {x[0, :5] if x.numel() > 0 else 'empty'}")
            print(f"risk_embedded content sample: {risk_embedded[0, :5] if risk_embedded.numel() > 0 else 'empty'}")
            raise
        
        # Main network forward pass
        h1 = F.relu(self.input_layer(combined_input))
        h1 = self.dropout(h1)
        h1 = self.layer_norm1(h1)
        
        h2 = F.relu(self.hidden_layer1(h1))
        h2 = self.dropout(h2)
        h2 = self.layer_norm2(h2)
        
        h3 = F.relu(self.hidden_layer2(h2))
        h3 = self.dropout(h3)
        
        # Apply risk-aware attention if enabled
        if self.use_risk_attention:
            # Create query from risk parameters
            risk_query = self.risk_query(risk_embedded).unsqueeze(1)  # [batch, 1, hidden]
            h3_expanded = h3.unsqueeze(1)  # [batch, 1, hidden]
            
            # Apply attention
            attended, _ = self.risk_attention(risk_query, h3_expanded, h3_expanded)
            h3 = h3 + attended.squeeze(1)  # Residual connection
        
        # Generate outputs
        outputs = {
            'trump_decision': self.trump_decision_head(h3),
            'card_selection': self.card_selection_head(h3),
            'risk_adjustment': self.risk_adjustment_head(h3),
            'hidden_features': h3
        }
        
        return outputs
    
    def get_trump_decision(self, x: torch.Tensor, risk_params: RiskParameters) -> torch.Tensor:
        """Get trump ordering decision with risk awareness."""
        outputs = self.forward(x, risk_params)
        return F.softmax(outputs['trump_decision'], dim=1)
    
    def get_card_selection(self, x: torch.Tensor, risk_params: RiskParameters) -> torch.Tensor:
        """Get card selection probabilities with risk awareness."""
        outputs = self.forward(x, risk_params)
        return F.softmax(outputs['card_selection'], dim=1)
    
    def get_risk_adjustment(self, x: torch.Tensor, risk_params: RiskParameters) -> torch.Tensor:
        """Get risk parameter adjustments based on game state."""
        outputs = self.forward(x, risk_params)
        return torch.tanh(outputs['risk_adjustment'])  # Output in [-1, 1] range

    def save_model(self, save_path: str) -> None:
        """Save the model to a file.
        
        Parameters
        ----------
        save_path : str
            Path where to save the model
        """
        torch.save(self.state_dict(), save_path)
    
    def load_model(self, load_path: str) -> None:
        """Load the model from a file.
        
        Parameters
        ----------
        load_path : str
            Path from where to load the model
        """
        self.load_state_dict(torch.load(load_path))
        self.eval()


class RiskAwareEuchreNN(EuchreNN):
    """Enhanced version with dynamic risk adjustment during gameplay."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional layers for dynamic risk adjustment
        self.game_state_encoder = nn.Linear(self.hidden_size, 64)
        self.risk_dynamics = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.risk_predictor = nn.Linear(32, 4)
        
        # Risk history tracking
        self.risk_history = []
        
    def update_risk_dynamics(self, game_state: torch.Tensor, 
                           current_risk: RiskParameters,
                           game_outcome: Optional[float] = None) -> RiskParameters:
        """Update risk parameters based on game dynamics.
        
        Parameters
        ----------
        game_state : torch.Tensor
            Current game state features
        current_risk : RiskParameters
            Current risk parameters
        game_outcome : Optional[float]
            Game outcome (-1.0 = loss, 0.0 = tie, 1.0 = win)
            
        Returns
        -------
        RiskParameters
            Updated risk parameters
        """
        # Encode game state
        with torch.no_grad():
            # Get hidden features
            outputs = self.forward(game_state, current_risk)
            hidden = outputs['hidden_features']
            
            # Encode game state
            encoded_state = self.game_state_encoder(hidden)
            
            # Update risk history
            self.risk_history.append(encoded_state.detach())
            
            # Keep only recent history
            if len(self.risk_history) > 10:
                self.risk_history.pop(0)
            
            # Process risk dynamics
            if len(self.risk_history) > 1:
                history_tensor = torch.stack(self.risk_history).unsqueeze(0)
                lstm_out, _ = self.risk_dynamics(history_tensor)
                
                # Get risk adjustment prediction
                risk_adjustment = self.risk_predictor(lstm_out[:, -1, :])
                risk_adjustment = torch.tanh(risk_adjustment) * 0.1  # Small adjustments
                
                # Apply adjustments to current risk parameters
                new_risk = RiskParameters(
                    trump_calling_aggression=torch.clamp(
                        current_risk.trump_calling_aggression + risk_adjustment[0], 0.0, 1.0
                    ),
                    ace_ordering_risk=torch.clamp(
                        current_risk.ace_ordering_risk + risk_adjustment[1], 0.0, 1.0
                    ),
                    set_risk_tolerance=torch.clamp(
                        current_risk.set_risk_tolerance + risk_adjustment[2], 0.0, 1.0
                    ),
                    leading_aggression=torch.clamp(
                        current_risk.leading_aggression + risk_adjustment[3], 0.0, 1.0
                    )
                )
                
                return new_risk
        
        return current_risk
    
    def get_adaptive_risk(self, base_risk: RiskParameters, 
                         game_context: Dict[str, Any]) -> RiskParameters:
        """Get adaptively adjusted risk parameters based on game context.
        
        Parameters
        ----------
        base_risk : RiskParameters
            Base risk parameters
        game_context : Dict[str, Any]
            Current game context (score, position, etc.)
            
        Returns
        -------
        RiskParameters
            Contextually adjusted risk parameters
        """
        # Create context tensor
        context_features = []
        
        # Score-based adjustments
        team_score = game_context.get('team_score', 0)
        opponent_score = game_context.get('opponent_score', 0)
        score_diff = team_score - opponent_score
        
        # Adjust risk based on score difference
        if score_diff < -3:  # Behind significantly
            context_features.extend([1.0, 0.0, 0.0, 0.0])  # High trump calling, low set risk
        elif score_diff > 3:  # Ahead significantly
            context_features.extend([0.0, 0.0, 0.0, 1.0])  # Conservative, safe leading
        else:  # Close game
            context_features.extend([0.5, 0.5, 0.5, 0.5])  # Balanced approach
        
        # Position-based adjustments
        position = game_context.get('position', 'unknown')
        if position in ['North', 'South']:  # Team 1
            context_features.extend([0.0, 0.0, 0.0, 0.0])
        else:  # Team 2
            context_features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Round-based adjustments
        current_round = game_context.get('current_round', 1)
        if current_round <= 3:  # Early game
            context_features.extend([0.0, 0.0, 0.0, 0.0])
        elif current_round >= 8:  # Late game
            context_features.extend([0.0, 0.0, 0.0, 0.0])
        else:  # Mid game
            context_features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Convert to tensor
        context_tensor = torch.tensor(context_features, dtype=torch.float32, device=base_risk.trump_calling_aggression.device)
        
        # Apply context adjustments
        adjusted_risk = RiskParameters(
            trump_calling_aggression=torch.clamp(
                base_risk.trump_calling_aggression + context_tensor[0] * 0.2, 0.0, 1.0
            ),
            ace_ordering_risk=torch.clamp(
                base_risk.ace_ordering_risk + context_tensor[1] * 0.2, 0.0, 1.0
            ),
            set_risk_tolerance=torch.clamp(
                base_risk.set_risk_tolerance + context_tensor[2] * 0.2, 0.0, 1.0
            ),
            leading_aggression=torch.clamp(
                base_risk.leading_aggression + context_tensor[3] * 0.2, 0.0, 1.0
            )
        )
        
        return adjusted_risk


def create_euchre_model(model_config: Dict[str, Any]) -> EuchreNN:
    """Create an euchre model based on configuration.
    
    Parameters
    ----------
    model_config : Dict[str, Any]
        Model configuration dictionary
        
    Returns
    -------
    EuchreNN
        Configured euchre neural network
    """
    model_type = model_config.get('type', 'standard')
    
    if model_type == 'risk_aware':
        return RiskAwareEuchreNN(
            input_size=model_config.get('input_size', 128),
            hidden_size=model_config.get('hidden_size', 256),
            output_size=model_config.get('output_size', 64),
            risk_embedding_size=model_config.get('risk_embedding_size', 32),
            use_risk_attention=model_config.get('use_risk_attention', True)
        )
    else:
        return EuchreNN(
            input_size=model_config.get('input_size', 128),
            hidden_size=model_config.get('hidden_size', 256),
            output_size=model_config.get('output_size', 64),
            risk_embedding_size=model_config.get('risk_embedding_size', 32),
            use_risk_attention=model_config.get('use_risk_attention', True)
        )


def create_risk_profile(profile_name: str) -> RiskParameters:
    """Create predefined risk profiles.
    
    Parameters
    ----------
    profile_name : str
        Name of the risk profile
        
    Returns
    -------
    RiskParameters
        Configured risk parameters
    """
    profiles = {
        'ultra_conservative': RiskParameters(0.1, 0.0, 0.1, 0.1),
        'conservative': RiskParameters(0.3, 0.2, 0.3, 0.3),
        'balanced': RiskParameters(0.5, 0.5, 0.5, 0.5),
        'aggressive': RiskParameters(0.7, 0.6, 0.7, 0.7),
        'ultra_aggressive': RiskParameters(0.9, 0.8, 0.9, 0.9),
        'ace_hunter': RiskParameters(0.6, 0.9, 0.7, 0.4),  # Loves ordering up aces
        'trump_caller': RiskParameters(0.9, 0.3, 0.6, 0.5),  # Always calls trump
        'safe_player': RiskParameters(0.2, 0.1, 0.2, 0.8),  # Conservative but leads high
        'risk_taker': RiskParameters(0.8, 0.7, 0.9, 0.6),  # High risk tolerance
        'adaptive': RiskParameters(0.5, 0.5, 0.5, 0.5)  # Balanced, will be adjusted dynamically
    }
    
    if profile_name not in profiles:
        print(f"Warning: Unknown profile '{profile_name}', using 'balanced'")
        return profiles['balanced']
    
    return profiles[profile_name] 