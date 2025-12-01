"""Shared trunk network for EucherGo."""

from typing import Optional

import torch
import torch.nn as nn


class EucherGoTrunk(nn.Module):
    """Shared trunk network with residual layers.

    Supports both Conv1D and MLP architectures based on configuration.
    Uses layer normalization and nonlinear activations.

    Parameters
    ----------
    input_size : int
        Size of input state tensor.
    hidden_size : int
        Hidden layer size.
    num_layers : int
        Number of residual layers.
    use_conv1d : bool
        If True, use Conv1D layers; otherwise use MLP.
    """

    def __init__(
        self,
        input_size: int = 712,
        hidden_size: int = 256,
        num_layers: int = 4,
        use_conv1d: bool = False,
    ) -> None:
        """Initialize trunk network."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_conv1d = use_conv1d

        if use_conv1d:
            # Conv1D architecture
            self.input_proj = nn.Conv1d(1, hidden_size, kernel_size=1)
            self.layers = nn.ModuleList(
                [ResidualConv1DBlock(hidden_size) for _ in range(num_layers)]
            )
        else:
            # MLP architecture
            self.input_proj = nn.Linear(input_size, hidden_size)
            self.layers = nn.ModuleList([ResidualMLPBlock(hidden_size) for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through trunk.

        Parameters
        ----------
        x : torch.Tensor
            Input state tensor [batch_size, input_size].

        Returns
        -------
        torch.Tensor
            Output tensor [batch_size, hidden_size].
        """
        if self.use_conv1d:
            # Reshape for Conv1D: [batch, 1, input_size]
            x = x.unsqueeze(1)
            x = self.input_proj(x)  # [batch, hidden_size, input_size]
            for layer in self.layers:
                x = layer(x)
            # Global average pooling
            x = x.mean(dim=2)  # [batch, hidden_size]
        else:
            x = self.input_proj(x)
            for layer in self.layers:
                x = layer(x)

        x = self.layer_norm(x)
        return x


class ResidualConv1DBlock(nn.Module):
    """Residual block for Conv1D architecture."""

    def __init__(self, hidden_size: int) -> None:
        """Initialize residual Conv1D block."""
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch, hidden_size, seq_len].

        Returns
        -------
        torch.Tensor
            Output tensor [batch, hidden_size, seq_len].
        """
        residual = x
        # Transpose for layer norm: [batch, seq_len, hidden_size]
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = torch.relu(x)
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = torch.relu(x)
        x = self.conv2(x)
        return x + residual


class ResidualMLPBlock(nn.Module):
    """Residual block for MLP architecture."""

    def __init__(self, hidden_size: int) -> None:
        """Initialize residual MLP block."""
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch, hidden_size].

        Returns
        -------
        torch.Tensor
            Output tensor [batch, hidden_size].
        """
        residual = x
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.linear1(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x + residual



