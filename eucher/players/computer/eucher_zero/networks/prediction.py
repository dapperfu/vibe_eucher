"""Prediction network for EucherZero."""

import torch
import torch.nn as nn

from eucher.players.computer.eucher_zero.config import EucherZeroConfig


class PredictionNetwork(nn.Module):
    """Output policy and value estimates for MCTS."""

    def __init__(self, config: EucherZeroConfig) -> None:
        """
        Initialize prediction network.

        Parameters
        ----------
        config : EucherZeroConfig
            Configuration object.
        """
        super().__init__()
        self.config = config

        # Shared layers
        shared_layers = []
        input_size = config.latent_size

        for output_size in config.prediction_layers:
            shared_layers.append(nn.Linear(input_size, output_size))
            shared_layers.append(nn.ReLU())
            input_size = output_size

        self.shared = nn.Sequential(*shared_layers)

        # Policy head
        self.policy_head = nn.Linear(input_size, config.action_space_size)

        # Value heads
        self.value_head = nn.Linear(input_size, 1)
        self.risk_value_head = nn.Linear(input_size, 1)

    def forward(
        self, latent_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        latent_state : torch.Tensor
            Latent state [batch_size, latent_size] or [latent_size].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (policy_logits, value, risk_value) where:
            - policy_logits: [batch_size, action_space_size] or [action_space_size]
            - value: [batch_size, 1] or [1]
            - risk_value: [batch_size, 1] or [1]
        """
        # Handle single sample
        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        # Shared processing
        hidden = self.shared(latent_state)

        # Heads
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        risk_value = self.risk_value_head(hidden)

        if single_sample:
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)
            risk_value = risk_value.squeeze(0)

        return policy_logits, value, risk_value

