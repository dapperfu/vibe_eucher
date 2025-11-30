"""Representation network for EucherZero."""

from typing import Dict

import torch
import torch.nn as nn

from eucher.players.computer.eucher_zero.config import EucherZeroConfig


class RepresentationNetwork(nn.Module):
    """Encode game state to latent representation."""

    def __init__(self, config: EucherZeroConfig) -> None:
        """
        Initialize representation network.

        Parameters
        ----------
        config : EucherZeroConfig
            Configuration object.
        """
        super().__init__()
        self.config = config

        # Input processing layers
        layers = []
        input_size = config.state_dim

        for output_size in config.representation_layers:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.BatchNorm1d(output_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_size = output_size

        # Final layer to latent
        layers.append(nn.Linear(input_size, config.latent_size))

        self.network = nn.Sequential(*layers)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state_tensor : torch.Tensor
            State tensor [batch_size, state_dim] or [state_dim].

        Returns
        -------
        torch.Tensor
            Latent state vector [batch_size, latent_size] or [latent_size].
        """
        # Handle single sample
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        latent = self.network(state_tensor)

        if single_sample:
            latent = latent.squeeze(0)

        return latent

