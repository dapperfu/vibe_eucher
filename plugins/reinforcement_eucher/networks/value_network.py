"""Value network for ReinforcementEucher."""

from typing import Optional

import torch
import torch.nn as nn

from ..config import ReinforcementEucherConfig


class ValueNetwork(nn.Module):
    """Value network predicting expected return.

    Parameters
    ----------
    config : ReinforcementEucherConfig
        Configuration with network parameters.
    """

    def __init__(self, config: ReinforcementEucherConfig) -> None:
        """Initialize value network.

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

        # Output layer (single value)
        self.output = nn.Linear(self.config.hidden_size, 1)

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

        # Output layer
        self.output = nn.Linear(self.config.hidden_size, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network.

        Parameters
        ----------
        state : torch.Tensor
            State tensor [batch_size, state_dim].

        Returns
        -------
        torch.Tensor
            Value estimate [batch_size, 1].
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

        value = self.output(x)
        return value



