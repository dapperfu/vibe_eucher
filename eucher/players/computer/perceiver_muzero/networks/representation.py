"""Representation network for EuchrePerceiverMuZero."""

import torch
import torch.nn as nn

from ..config import PerceiverMuZeroConfig


class RepresentationNetwork(nn.Module):
    """MLP that processes Perceiver-IO output to latent representation."""

    def __init__(self, config: PerceiverMuZeroConfig) -> None:
        """
        Initialize representation network.

        Parameters
        ----------
        config : PerceiverMuZeroConfig
            Configuration object.
        """
        super().__init__()
        self.config = config

        # MLP: PerceiverOutput (128-dim) -> latent (128-dim)
        self.network = nn.Sequential(
            nn.Linear(config.latent_size, config.latent_size),
            nn.LayerNorm(config.latent_size),
            nn.GELU(),
            nn.Linear(config.latent_size, config.latent_size),
        )

    def forward(self, perceiver_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        perceiver_output : torch.Tensor
            Perceiver-IO output tensor [batch_size, latent_size] or [latent_size].

        Returns
        -------
        torch.Tensor
            Latent state vector [batch_size, latent_size] or [latent_size].
        """
        # Handle single sample
        if perceiver_output.dim() == 1:
            perceiver_output = perceiver_output.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        latent = self.network(perceiver_output)

        if single_sample:
            latent = latent.squeeze(0)

        return latent

