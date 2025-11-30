"""Complete EucherGo model combining trunk, policy, and value heads."""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from eucher.players.computer.euchergo.networks.policy_head import EucherGoPolicyHead
from eucher.players.computer.euchergo.networks.trunk import EucherGoTrunk
from eucher.players.computer.euchergo.networks.value_head import EucherGoValueHead


class EucherGoModel(nn.Module):
    """Complete EucherGo model with shared trunk and policy/value heads.

    Parameters
    ----------
    input_size : int
        Size of input state tensor.
    hidden_size : int
        Hidden layer size.
    num_layers : int
        Number of residual layers in trunk.
    use_conv1d : bool
        If True, use Conv1D architecture; otherwise use MLP.
    action_space_size : int
        Size of action space (default: 18, supports 6-card discard).
    """

    def __init__(
        self,
        input_size: int = 712,
        hidden_size: int = 256,
        num_layers: int = 4,
        use_conv1d: bool = False,
        action_space_size: int = 18,
    ) -> None:
        """Initialize EucherGo model."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.action_space_size = action_space_size

        self.trunk = EucherGoTrunk(input_size, hidden_size, num_layers, use_conv1d)
        self.policy_head = EucherGoPolicyHead(hidden_size, action_space_size)
        self.value_head = EucherGoValueHead(hidden_size)

    def forward(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through model.

        Parameters
        ----------
        state : torch.Tensor
            Input state tensor [batch_size, input_size].
        action_mask : Optional[torch.Tensor]
            Action mask [batch_size, action_space_size].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (policy_logits, value) where:
            - policy_logits: [batch_size, action_space_size]
            - value: [batch_size, 1]
        """
        trunk_output = self.trunk(state)
        policy_logits = self.policy_head(trunk_output, action_mask)
        value = self.value_head(trunk_output)
        return policy_logits, value

    def get_policy_and_value(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get policy distribution and value.

        Parameters
        ----------
        state : torch.Tensor
            Input state tensor [batch_size, input_size].
        action_mask : Optional[torch.Tensor]
            Action mask [batch_size, action_space_size].
        temperature : float
            Temperature for policy softmax.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (policy_distribution, value) where:
            - policy_distribution: [batch_size, action_space_size]
            - value: [batch_size, 1]
        """
        trunk_output = self.trunk(state)
        policy_dist = self.policy_head.get_policy_distribution(trunk_output, action_mask, temperature)
        value = self.value_head(trunk_output)
        return policy_dist, value

    def save_checkpoint(self, path: Path) -> None:
        """
        Save model checkpoint.

        Parameters
        ----------
        path : Path
            Path to save checkpoint.
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "action_space_size": self.action_space_size,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        """
        Load model checkpoint.

        Parameters
        ----------
        path : Path
            Path to load checkpoint from.
            
        Note
        ----
        If the checkpoint was saved with a different action_space_size, loading may fail.
        Checkpoints saved with action_space_size=17 (old) are incompatible with action_space_size=18 (new).
        """
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint_action_size = checkpoint.get("action_space_size", None)
        if checkpoint_action_size is not None and checkpoint_action_size != self.action_space_size:
            raise ValueError(
                f"Checkpoint action_space_size ({checkpoint_action_size}) does not match "
                f"model action_space_size ({self.action_space_size}). "
                f"Models trained with the old action space (17) are incompatible with the new action space (18)."
            )
        self.load_state_dict(checkpoint["model_state_dict"])

    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.eval()

    def train_mode(self) -> None:
        """Set model to training mode."""
        self.train()

