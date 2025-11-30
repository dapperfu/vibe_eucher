"""Value head network for EucherGo."""

import torch
import torch.nn as nn


class EucherGoValueHead(nn.Module):
    """Value head that outputs scalar win probability.

    Parameters
    ----------
    hidden_size : int
        Size of hidden layer from trunk.
    """

    def __init__(self, hidden_size: int = 256) -> None:
        """Initialize value head."""
        super().__init__()
        self.hidden_size = hidden_size

        self.value_linear1 = nn.Linear(hidden_size, hidden_size // 2)
        self.value_linear2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, trunk_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value head.

        Parameters
        ----------
        trunk_output : torch.Tensor
            Output from trunk [batch_size, hidden_size].

        Returns
        -------
        torch.Tensor
            Value prediction [batch_size, 1] (win probability in [-1, 1]).
        """
        x = torch.relu(self.value_linear1(trunk_output))
        value = torch.tanh(self.value_linear2(x))  # Output in [-1, 1]
        return value


