"""Dynamics network for EucherPerceiverMuZero."""

import torch
import torch.nn as nn

from ..config import PerceiverMuZeroConfig


class DynamicsNetwork(nn.Module):
    """Predict next latent state and immediate reward after action."""

    def __init__(self, config: PerceiverMuZeroConfig) -> None:
        """
        Initialize dynamics network.

        Parameters
        ----------
        config : PerceiverMuZeroConfig
            Configuration object.
        """
        super().__init__()
        self.config = config

        # Input: latent (128) + action (20) = 148
        input_size = config.latent_size + config.action_space_size

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, config.latent_size),
            nn.LayerNorm(config.latent_size),
            nn.GELU(),
            nn.Linear(config.latent_size, config.latent_size),
        )

        # Split heads
        self.state_head = nn.Linear(config.latent_size, config.latent_size)
        self.reward_head = nn.Linear(config.latent_size, 1)

    def forward(
        self, latent_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        latent_state : torch.Tensor
            Current latent state [batch_size, latent_size] or [latent_size].
        action : torch.Tensor
            Action encoding [batch_size, action_space_size] or [action_space_size].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (next_latent_state, reward) where:
            - next_latent_state: [batch_size, latent_size] or [latent_size]
            - reward: [batch_size, 1] or [1]
        """
        # Handle single sample
        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
            action = action.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        # Concatenate latent and action
        combined = torch.cat([latent_state, action], dim=-1)

        # Shared processing
        hidden = self.shared(combined)

        # Split heads
        next_latent = self.state_head(hidden)
        reward = self.reward_head(hidden)

        if single_sample:
            next_latent = next_latent.squeeze(0)
            reward = reward.squeeze(0)

        return next_latent, reward

