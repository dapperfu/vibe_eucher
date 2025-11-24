"""Unified EuchrePerceiverMuZero model combining all networks."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..config import PerceiverMuZeroConfig
from .dynamics import DynamicsNetwork
from .perceiver_encoder import PerceiverEncoder
from .prediction import PredictionNetwork
from .representation import RepresentationNetwork


class PerceiverMuZeroModel(nn.Module):
    """Unified model combining Perceiver-IO encoder, representation, dynamics, and prediction networks."""

    def __init__(self, config: PerceiverMuZeroConfig) -> None:
        """
        Initialize EuchrePerceiverMuZero model.

        Parameters
        ----------
        config : PerceiverMuZeroConfig
            Configuration object.
        """
        super().__init__()
        self.config = config
        self.device = config.device

        self.perceiver_encoder = PerceiverEncoder(config)
        self.representation_net = RepresentationNetwork(config)
        self.dynamics_net = DynamicsNetwork(config)
        self.prediction_net = PredictionNetwork(config)

        self.to(self.device)

    def forward(
        self,
        input_tokens: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through model.

        Parameters
        ----------
        input_tokens : torch.Tensor
            Input token sequence [batch_size, num_tokens, token_dim] or [num_tokens, token_dim].
        action : Optional[torch.Tensor]
            Action tensor (for dynamics) [batch_size, action_space_size] or [action_space_size].

        Returns
        -------
        Tuple[torch.Tensor, ...]
            If action provided: (next_latent, reward, policy_logits, value, risk_score)
            Otherwise: (policy_logits, value, risk_score)
        """
        # Encode tokens through Perceiver-IO
        perceiver_output = self.perceiver_encoder(input_tokens)

        # Process through representation network
        latent = self.representation_net(perceiver_output)

        if action is not None:
            next_latent, reward = self.dynamics_net(latent, action)
            policy_logits, value, risk_score = self.prediction_net(latent)
            return next_latent, reward, policy_logits, value, risk_score
        else:
            policy_logits, value, risk_score = self.prediction_net(latent)
            return policy_logits, value, risk_score

    def train_mode(self) -> None:
        """Set all networks to training mode."""
        self.train()
        self.perceiver_encoder.train()
        self.representation_net.train()
        self.dynamics_net.train()
        self.prediction_net.train()

    def eval_mode(self) -> None:
        """Set all networks to evaluation mode."""
        self.eval()
        self.perceiver_encoder.eval()
        self.representation_net.eval()
        self.dynamics_net.eval()
        self.prediction_net.eval()

    def save_checkpoint(
        self, checkpoint_path: Path, metadata: Optional[Dict] = None
    ) -> None:
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
            "perceiver_encoder": self.perceiver_encoder.state_dict(),
            "representation_net": self.representation_net.state_dict(),
            "dynamics_net": self.dynamics_net.state_dict(),
            "prediction_net": self.prediction_net.state_dict(),
            "config": {
                "latent_size": self.config.latent_size,
                "latent_slots": self.config.latent_slots,
                "token_dim": self.config.token_dim,
                "action_space_size": self.config.action_space_size,
                "enable_screw_the_dealer": self.config.enable_screw_the_dealer,
                "enable_nine_ten_tradein": self.config.enable_nine_ten_tradein,
                "enable_go_alone": self.config.enable_go_alone,
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

        self.perceiver_encoder.load_state_dict(checkpoint["perceiver_encoder"])
        self.representation_net.load_state_dict(checkpoint["representation_net"])
        self.dynamics_net.load_state_dict(checkpoint["dynamics_net"])
        self.prediction_net.load_state_dict(checkpoint["prediction_net"])

        return checkpoint.get("metadata", {})

