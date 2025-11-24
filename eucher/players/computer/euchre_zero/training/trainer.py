"""Trainer for EuchreZero with multi-component loss."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from eucher.players.computer.euchre_zero.config import EuchreZeroConfig
from eucher.players.computer.euchre_zero.networks.model import EuchreZeroModel
from eucher.players.computer.euchre_zero.training.replay_buffer import ReplayBuffer


class EuchreZeroTrainer:
    """Trainer for EuchreZero model."""

    def __init__(
        self,
        model: EuchreZeroModel,
        config: EuchreZeroConfig,
        replay_buffer: ReplayBuffer,
    ) -> None:
        """
        Initialize trainer.

        Parameters
        ----------
        model : EuchreZeroModel
            Model to train.
        config : EuchreZeroConfig
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
            lr=config.learning_rate,
        )

        # Loss functions
        self.policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.value_loss_fn = nn.MSELoss()
        self.dynamics_loss_fn = nn.MSELoss()
        self.reward_loss_fn = nn.MSELoss()

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns
        -------
        Dict[str, float]
            Dictionary of loss values.
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {
                "policy": 0.0,
                "value": 0.0,
                "risk_value": 0.0,
                "dynamics": 0.0,
                "reward": 0.0,
                "total": 0.0,
            }

        # Sample batch
        states, policies, values, rewards, next_states = self.replay_buffer.sample_batch(
            self.config.batch_size
        )

        # Move to device
        states = states.to(self.model.device)
        policies = policies.to(self.model.device)
        values = values.to(self.model.device)
        rewards = rewards.to(self.model.device)
        next_states = next_states.to(self.model.device)

        # Forward pass
        self.model.train_mode()
        self.optimizer.zero_grad()

        # Representation
        latent = self.model.representation_net(states)

        # Prediction (policy and value)
        policy_logits, value, risk_value = self.model.prediction_net(latent)

        # Dynamics (for next states)
        # Create dummy action tensor for dynamics (simplified)
        action_tensor = torch.zeros(
            states.shape[0],
            self.config.action_space_size,
            device=self.model.device,
        )
        next_latent_pred, reward_pred = self.model.dynamics_net(latent, action_tensor)

        # Compute losses
        # Policy loss: KL divergence
        policy_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = self.policy_loss_fn(policy_probs, policies)

        # Value loss: MSE
        value_loss = self.value_loss_fn(value, values)

        # Risk value loss: MSE (simplified - use same values)
        risk_value_loss = self.value_loss_fn(risk_value, values)

        # Dynamics loss: MSE on next latent (simplified - compare to next state encoding)
        next_latent_target = self.model.representation_net(next_states)
        dynamics_loss = self.dynamics_loss_fn(next_latent_pred, next_latent_target.detach())

        # Reward loss: MSE
        reward_loss = self.reward_loss_fn(reward_pred, rewards)

        # Total loss
        total_loss = (
            self.config.loss_weights["policy"] * policy_loss
            + self.config.loss_weights["value"] * value_loss
            + self.config.loss_weights["risk_value"] * risk_value_loss
            + self.config.loss_weights["dynamics"] * dynamics_loss
            + self.config.loss_weights["reward"] * reward_loss
        )

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy": policy_loss.item(),
            "value": value_loss.item(),
            "risk_value": risk_value_loss.item(),
            "dynamics": dynamics_loss.item(),
            "reward": reward_loss.item(),
            "total": total_loss.item(),
        }

