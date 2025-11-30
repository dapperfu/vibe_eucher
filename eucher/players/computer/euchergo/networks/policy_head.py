"""Policy head network for EucherGo with action masking."""

from typing import Optional

import torch
import torch.nn as nn

from eucher.players.computer.euchergo.action_space import ActionEncoder


class EucherGoPolicyHead(nn.Module):
    """Policy head that outputs action logits with action masking support.

    Parameters
    ----------
    hidden_size : int
        Size of hidden layer from trunk.
    action_space_size : int
        Size of action space (default: 18, supports 6-card discard).
    """

    def __init__(self, hidden_size: int = 256, action_space_size: int = 18) -> None:
        """Initialize policy head."""
        super().__init__()
        self.hidden_size = hidden_size
        self.action_space_size = action_space_size

        self.policy_linear = nn.Linear(hidden_size, action_space_size)

    def forward(
        self, trunk_output: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through policy head.

        Parameters
        ----------
        trunk_output : torch.Tensor
            Output from trunk [batch_size, hidden_size].
        action_mask : Optional[torch.Tensor]
            Action mask [batch_size, action_space_size] where 1 = legal, 0 = illegal.
            If None, all actions are considered legal.

        Returns
        -------
        torch.Tensor
            Policy logits [batch_size, action_space_size].
        """
        logits = self.policy_linear(trunk_output)

        # Apply action mask: set illegal actions to large negative value
        if action_mask is not None:
            logits = logits + (action_mask - 1) * 1e9

        return logits

    def get_policy_distribution(
        self, trunk_output: torch.Tensor, action_mask: Optional[torch.Tensor] = None, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Get policy distribution (softmax over logits).

        Parameters
        ----------
        trunk_output : torch.Tensor
            Output from trunk [batch_size, hidden_size].
        action_mask : Optional[torch.Tensor]
            Action mask [batch_size, action_space_size].
        temperature : float
            Temperature for softmax (1.0 = normal, >1.0 = more exploration).

        Returns
        -------
        torch.Tensor
            Policy distribution [batch_size, action_space_size].
        """
        logits = self.forward(trunk_output, action_mask)
        logits = logits / temperature
        return torch.softmax(logits, dim=-1)

