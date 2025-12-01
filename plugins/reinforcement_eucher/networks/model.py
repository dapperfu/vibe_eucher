"""Unified model combining policy and value networks."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..action_space import ACTION_SPACE_SIZE
from ..config import ReinforcementEucherConfig
from .policy_network import PolicyNetwork
from .value_network import ValueNetwork


class ReinforcementEucherModel(nn.Module):
    """Unified model combining policy and value networks.

    Parameters
    ----------
    config : ReinforcementEucherConfig
        Configuration with network parameters.
    """

    def __init__(self, config: ReinforcementEucherConfig) -> None:
        """Initialize unified model.

        Parameters
        ----------
        config : ReinforcementEucherConfig
            Configuration with network parameters.
        """
        super().__init__()
        self.config = config

        # Shared trunk (optional - can be used for parameter sharing)
        self.use_shared_trunk = False
        if self.use_shared_trunk:
            self._build_shared_trunk(config.state_dim)
            policy_input_dim = config.hidden_size
            value_input_dim = config.hidden_size
        else:
            policy_input_dim = config.state_dim
            value_input_dim = config.state_dim

        # Policy and value networks
        self.policy_net = PolicyNetwork(config)
        self.value_net = ValueNetwork(config)

    def _build_shared_trunk(self, state_dim: int) -> None:
        """Build shared trunk network.

        Parameters
        ----------
        state_dim : int
            Input state dimension.
        """
        layers = []
        input_dim = state_dim

        for i in range(self.config.num_layers - 1):  # One less layer for shared trunk
            layers.append(nn.Linear(input_dim, self.config.hidden_size))
            layers.append(nn.LayerNorm(self.config.hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            input_dim = self.config.hidden_size

        self.shared_trunk = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through model.

        Parameters
        ----------
        state : torch.Tensor
            State tensor [batch_size, state_dim].
        legal_mask : Optional[torch.Tensor]
            Legal action mask.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (action_logits, value_estimate).
        """
        if self.use_shared_trunk:
            shared_features = self.shared_trunk(state)
            action_logits = self.policy_net(shared_features, legal_mask)
            value = self.value_net(shared_features)
        else:
            action_logits = self.policy_net(state, legal_mask)
            value = self.value_net(state)

        return action_logits, value

    def get_policy_and_value(
        self,
        state: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get policy distribution and value estimate.

        Parameters
        ----------
        state : torch.Tensor
            State tensor [batch_size, state_dim].
        legal_mask : Optional[torch.Tensor]
            Legal action mask.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (action_probs, value_estimate).
        """
        logits, value = self.forward(state, legal_mask)
        probs = torch.softmax(logits, dim=-1)
        return probs, value

    def sample_action(
        self,
        state: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and get value estimate.

        Parameters
        ----------
        state : torch.Tensor
            State tensor [batch_size, state_dim].
        legal_mask : Optional[torch.Tensor]
            Legal action mask.
        temperature : float
            Temperature for sampling.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (action_id, log_prob, value_estimate).
        """
        logits, value = self.forward(state, legal_mask)
        logits = logits / temperature
        
        # Apply legal mask again to ensure illegal actions are masked
        if legal_mask is not None:
            if legal_mask.dim() == 1:
                legal_mask = legal_mask.unsqueeze(0).expand_as(logits)
            logits = logits.masked_fill(legal_mask == 0, -1e9)
        
        # Check for NaN or all -inf
        if torch.isnan(logits).any() or (logits == -1e9).all(dim=-1).any():
            # Fallback: uniform distribution over legal actions
            if legal_mask is not None:
                probs = legal_mask.float()
                probs = probs / probs.sum(dim=-1, keepdim=True)
            else:
                probs = torch.ones_like(logits) / logits.shape[-1]
        else:
            probs = torch.softmax(logits, dim=-1)
        
        # Ensure no NaN in probs
        probs = torch.nan_to_num(probs, nan=0.0)
        # Renormalize if needed
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def get_parameter_count(self) -> int:
        """Get total number of parameters.

        Returns
        -------
        int
            Parameter count.
        """
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Parameters
        ----------
        path : str
            Path to save checkpoint.
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        config: Optional[ReinforcementEucherConfig] = None,
        device: Optional[str] = None,
    ) -> "ReinforcementEucherModel":
        """Load model from checkpoint.

        Parameters
        ----------
        path : str
            Path to checkpoint.
        config : Optional[ReinforcementEucherConfig]
            Configuration. If None, loaded from checkpoint.
        device : Optional[str]
            Device to load on.

        Returns
        -------
        ReinforcementEucherModel
            Loaded model.
        """
        checkpoint = torch.load(path, map_location=device)
        if config is None:
            config = ReinforcementEucherConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

