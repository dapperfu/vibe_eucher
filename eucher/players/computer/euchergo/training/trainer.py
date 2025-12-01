"""Trainer for EucherGo with policy and value losses."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from eucher.players.computer.euchergo.config import EucherGoConfig
from eucher.players.computer.euchergo.networks.model import EucherGoModel
from eucher.players.computer.euchergo.training.replay_buffer import ReplayBuffer


class EucherGoTrainer:
    """Trainer for EucherGo model."""

    def __init__(
        self,
        model: EucherGoModel,
        config: EucherGoConfig,
        replay_buffer: ReplayBuffer,
    ) -> None:
        """
        Initialize trainer.

        Parameters
        ----------
        model : EucherGoModel
            Model to train.
        config : EucherGoConfig
            Configuration.
        replay_buffer : ReplayBuffer
            Replay buffer.
        """
        self.model = model
        self.config = config
        self.replay_buffer = replay_buffer

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=getattr(config, "learning_rate", 1e-3),
        )

        # Loss functions
        self.policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.value_loss_fn = nn.MSELoss()

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns
        -------
        Dict[str, float]
            Dictionary of loss values.
        """
        batch_size = getattr(self.config, "batch_size", 32)
        if len(self.replay_buffer) < batch_size:
            return {
                "policy": 0.0,
                "value": 0.0,
                "total": 0.0,
            }

        # Sample batch
        states, policies, values = self.replay_buffer.sample_batch(batch_size)

        # Move to device
        states = states.to(self.config.device)
        policies = policies.to(self.config.device)
        values = values.to(self.config.device)

        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()

        # Get policy and value from model
        policy_logits, value = self.model(states, action_mask=None)

        # Compute losses
        # Policy loss: KL divergence
        policy_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = self.policy_loss_fn(policy_probs, policies)

        # Value loss: MSE
        value_loss = self.value_loss_fn(value, values)

        # Total loss
        total_loss = policy_loss + value_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy": policy_loss.item(),
            "value": value_loss.item(),
            "total": total_loss.item(),
        }

