"""Policy network for ReinforcementEucher."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from eucher.players.computer.reinforcement_eucher.action_space import ACTION_SPACE_SIZE
from eucher.players.computer.reinforcement_eucher.config import ReinforcementEucherConfig


class PolicyNetwork(nn.Module):
    """Policy network outputting action probabilities.

    Parameters
    ----------
    config : ReinforcementEucherConfig
        Configuration with network parameters.
    """

    def __init__(self, config: ReinforcementEucherConfig) -> None:
        """Initialize policy network.

        Parameters
        ----------
        config : ReinforcementEucherConfig
            Configuration with network parameters.
        """
        super().__init__()
        self.config = config
        state_dim = config.state_dim

        if config.use_transformer:
            self._build_transformer(state_dim)
        else:
            self._build_mlp(state_dim)

    def _build_mlp(self, state_dim: int) -> None:
        """Build MLP architecture.

        Parameters
        ----------
        state_dim : int
            Input state dimension.
        """
        layers = []
        input_dim = state_dim

        # Build hidden layers with residual connections
        for i in range(self.config.num_layers):
            layers.append(nn.Linear(input_dim, self.config.hidden_size))
            layers.append(nn.LayerNorm(self.config.hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            input_dim = self.config.hidden_size

        self.layers = nn.Sequential(*layers)

        # Output layer with small initialization to avoid extreme values
        self.output = nn.Linear(self.config.hidden_size, ACTION_SPACE_SIZE)
        # Initialize output layer weights to small values
        nn.init.xavier_uniform_(self.output.weight, gain=0.1)
        nn.init.zeros_(self.output.bias)

    def _build_transformer(self, state_dim: int) -> None:
        """Build Transformer architecture.

        Parameters
        ----------
        state_dim : int
            Input state dimension.
        """
        # Project input to hidden size
        self.input_proj = nn.Linear(state_dim, self.config.hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size,
            nhead=self.config.num_attention_heads,
            dim_feedforward=self.config.hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)

        # Output layer with small initialization
        self.output = nn.Linear(self.config.hidden_size, ACTION_SPACE_SIZE)
        # Initialize output layer weights to small values
        nn.init.xavier_uniform_(self.output.weight, gain=0.1)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        state: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through policy network.

        Parameters
        ----------
        state : torch.Tensor
            State tensor [batch_size, state_dim].
        legal_mask : Optional[torch.Tensor]
            Legal action mask [batch_size, ACTION_SPACE_SIZE] or [ACTION_SPACE_SIZE].

        Returns
        -------
        torch.Tensor
            Action logits [batch_size, ACTION_SPACE_SIZE].
        """
        if self.config.use_transformer:
            # Add sequence dimension for transformer
            if state.dim() == 2:
                state = state.unsqueeze(1)  # [batch, 1, state_dim]
            state = self.input_proj(state)
            x = self.transformer(state)
            x = x.squeeze(1) if x.shape[1] == 1 else x.mean(dim=1)
        else:
            x = self.layers(state)

        logits = self.output(x)

        # Apply legal mask if provided
        if legal_mask is not None:
            if legal_mask.dim() == 1:
                legal_mask = legal_mask.unsqueeze(0).expand_as(logits)
            # Set illegal actions to very negative value (not -inf to avoid NaN)
            # Use a large negative value that won't cause NaN in softmax
            logits = logits.masked_fill(legal_mask == 0, -1e9)

        return logits

    def get_action_probs(
        self,
        state: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Get action probabilities from state.

        Parameters
        ----------
        state : torch.Tensor
            State tensor [batch_size, state_dim].
        legal_mask : Optional[torch.Tensor]
            Legal action mask.
        temperature : float
            Temperature for sampling (1.0 = normal, >1.0 = more random).

        Returns
        -------
        torch.Tensor
            Action probabilities [batch_size, ACTION_SPACE_SIZE].
        """
        logits = self.forward(state, legal_mask)
        logits = logits / temperature
        
        # Apply legal mask again if provided
        if legal_mask is not None:
            if legal_mask.dim() == 1:
                legal_mask = legal_mask.unsqueeze(0).expand_as(logits)
            logits = logits.masked_fill(legal_mask == 0, -1e9)
        
        # Check for NaN
        if torch.isnan(logits).any():
            # Fallback: uniform over legal actions
            if legal_mask is not None:
                probs = legal_mask.float()
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)
            else:
                probs = torch.ones_like(logits) / logits.shape[-1]
        else:
            probs = F.softmax(logits, dim=-1)
        
        # Ensure no NaN
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)
        
        return probs

    def sample_action(
        self,
        state: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Parameters
        ----------
        state : torch.Tensor
            State tensor [batch_size, state_dim].
        legal_mask : Optional[torch.Tensor]
            Legal action mask.
        temperature : float
            Temperature for sampling.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (action_id, log_prob).
        """
        probs = self.get_action_probs(state, legal_mask, temperature)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

