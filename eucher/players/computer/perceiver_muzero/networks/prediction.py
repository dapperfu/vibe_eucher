"""Prediction network for EucherPerceiverMuZero."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PerceiverMuZeroConfig


class PredictionNetwork(nn.Module):
    """Output policy and value estimates for MCTS with risk modulation."""

    def __init__(self, config: PerceiverMuZeroConfig) -> None:
        """
        Initialize prediction network.

        Parameters
        ----------
        config : PerceiverMuZeroConfig
            Configuration object.
        """
        super().__init__()
        self.config = config

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(config.latent_size, config.latent_size),
            nn.LayerNorm(config.latent_size),
            nn.GELU(),
            nn.Linear(config.latent_size, config.latent_size),
        )

        # Policy head
        self.policy_head = nn.Linear(config.latent_size, config.action_space_size)

        # Value head
        self.value_head = nn.Linear(config.latent_size, 1)

        # Risk score head for temperature modulation
        self.risk_score_head = nn.Sequential(
            nn.Linear(config.latent_size, config.latent_size // 2),
            nn.GELU(),
            nn.Linear(config.latent_size // 2, 1),
            nn.Sigmoid(),
        )

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
            (policy_logits, value, risk_score) where:
            - policy_logits: [batch_size, action_space_size] or [action_space_size]
            - value: [batch_size, 1] or [1]
            - risk_score: [batch_size, 1] or [1] (for temperature scaling)
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
        risk_score = self.risk_score_head(hidden)

        if single_sample:
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)
            risk_score = risk_score.squeeze(0)

        return policy_logits, value, risk_score

    def compute_temperature(self, risk_score: torch.Tensor) -> torch.Tensor:
        """
        Compute temperature from risk score.

        Parameters
        ----------
        risk_score : torch.Tensor
            Risk score [batch_size, 1] or [1].

        Returns
        -------
        torch.Tensor
            Temperature [batch_size, 1] or [1].
        """
        # temp = base_temp * (1 + risk_score * RISK_FACTOR)
        temperature = self.config.base_temperature * (
            1.0 + risk_score * self.config.risk_factor
        )
        return temperature

