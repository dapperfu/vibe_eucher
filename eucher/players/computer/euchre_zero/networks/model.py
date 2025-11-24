"""Unified EuchreZero model combining all networks."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from eucher.players.computer.euchre_zero.config import EuchreZeroConfig
from eucher.players.computer.euchre_zero.networks.dynamics import DynamicsNetwork
from eucher.players.computer.euchre_zero.networks.prediction import PredictionNetwork
from eucher.players.computer.euchre_zero.networks.representation import (
    RepresentationNetwork,
)


class EuchreZeroModel(nn.Module):
    """Unified model combining representation, dynamics, and prediction networks."""

    def __init__(self, config: EuchreZeroConfig) -> None:
        """
        Initialize EuchreZero model.

        Parameters
        ----------
        config : EuchreZeroConfig
            Configuration object.
        """
        super().__init__()
        self.config = config
        self.device = config.device

        self.representation_net = RepresentationNetwork(config)
        self.dynamics_net = DynamicsNetwork(config)
        self.prediction_net = PredictionNetwork(config)

        self.to(self.device)

    def forward(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through model.

        Parameters
        ----------
        state : torch.Tensor
            Game state tensor.
        action : Optional[torch.Tensor]
            Action tensor (for dynamics).

        Returns
        -------
        Tuple[torch.Tensor, ...]
            If action provided: (next_latent, reward, policy_logits, value, risk_value)
            Otherwise: (policy_logits, value, risk_value)
        """
        latent = self.representation_net(state)

        if action is not None:
            next_latent, reward = self.dynamics_net(latent, action)
            policy_logits, value, risk_value = self.prediction_net(latent)
            return next_latent, reward, policy_logits, value, risk_value
        else:
            policy_logits, value, risk_value = self.prediction_net(latent)
            return policy_logits, value, risk_value

    def train_mode(self) -> None:
        """Set all networks to training mode."""
        self.train()
        self.representation_net.train()
        self.dynamics_net.train()
        self.prediction_net.train()

    def eval_mode(self) -> None:
        """Set all networks to evaluation mode."""
        self.eval()
        self.representation_net.eval()
        self.dynamics_net.eval()
        self.prediction_net.eval()

    def save_checkpoint(self, checkpoint_path: Path, metadata: Optional[Dict] = None) -> None:
        """
        Save model checkpoint.

        Parameters
        ----------
        checkpoint_path : Path
            Path to save checkpoint.
        metadata : Optional[Dict]
            Additional metadata to save.
        """
        checkpoint = {
            "representation_net": self.representation_net.state_dict(),
            "dynamics_net": self.dynamics_net.state_dict(),
            "prediction_net": self.prediction_net.state_dict(),
            "config": {
                "latent_size": self.config.latent_size,
                "state_dim": self.config.state_dim,
                "action_space_size": self.config.action_space_size,
            },
        }
        if metadata:
            checkpoint["metadata"] = metadata

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path) -> Dict:
        """
        Load model checkpoint.

        Parameters
        ----------
        checkpoint_path : Path
            Path to load checkpoint from.

        Returns
        -------
        Dict
            Metadata from checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.representation_net.load_state_dict(checkpoint["representation_net"])
        self.dynamics_net.load_state_dict(checkpoint["dynamics_net"])
        self.prediction_net.load_state_dict(checkpoint["prediction_net"])

        return checkpoint.get("metadata", {})

