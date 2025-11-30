"""Trainer for EucherPerceiverMuZero with multi-component loss."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PerceiverMuZeroConfig
from ..networks.model import PerceiverMuZeroModel
from .replay_buffer import ReplayBuffer


class PerceiverMuZeroTrainer:
    """Trainer for EucherPerceiverMuZero model."""

    def __init__(
        self,
        model: PerceiverMuZeroModel,
        config: PerceiverMuZeroConfig,
        replay_buffer: ReplayBuffer,
    ) -> None:
        """
        Initialize trainer.

        Parameters
        ----------
        model : PerceiverMuZeroModel
            Model to train.
        config : PerceiverMuZeroConfig
            Configuration.
        replay_buffer : ReplayBuffer
            Replay buffer.
        """
        self.model = model
        self.config = config
        self.replay_buffer = replay_buffer

        # Optimizer: AdamW with lr=1e-4, weight_decay=1e-3
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
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
                "reward": 0.0,
                "entropy": 0.0,
                "total": 0.0,
            }

        # Sample batch
        (
            input_tokens,
            actions,
            policies,
            values,
            rewards,
            next_input_tokens,
            weights,
        ) = self.replay_buffer.sample_batch(self.config.batch_size)

        # Move to device
        input_tokens = input_tokens.to(self.model.device)
        actions = actions.to(self.model.device)
        policies = policies.to(self.model.device)
        values = values.to(self.model.device)
        rewards = rewards.to(self.model.device)
        next_input_tokens = next_input_tokens.to(self.model.device)
        if weights is not None:
            weights = weights.to(self.model.device)

        # Forward pass
        self.model.train_mode()
        self.optimizer.zero_grad()

        # Encode tokens through Perceiver-IO and representation
        perceiver_output = self.model.perceiver_encoder(input_tokens)
        latent = self.model.representation_net(perceiver_output)

        # Prediction (policy and value)
        policy_logits, value, risk_score = self.model.prediction_net(latent)

        # Dynamics: predict next state and reward
        # Encode actions as one-hot
        action_onehot = F.one_hot(actions, num_classes=self.config.action_space_size).float()
        next_latent_pred, reward_pred = self.model.dynamics_net(latent, action_onehot)

        # Compute losses
        # Policy loss: cross-entropy between predicted and target policy
        policy_loss = self.policy_loss_fn(policy_logits, policies.argmax(dim=1))

        # Value loss: MSE between predicted and target value
        value_loss = self.value_loss_fn(value, values)

        # Reward loss: MSE between predicted and target reward
        reward_loss = self.reward_loss_fn(reward_pred, rewards)

        # Entropy bonus: encourage exploration
        policy_probs = F.softmax(policy_logits, dim=-1)
        entropy = -(policy_probs * F.log_softmax(policy_logits, dim=-1)).sum(dim=1).mean()
        entropy_bonus = -entropy  # Negative because we want to maximize entropy

        # Total loss: L = value_loss + policy_loss + reward_loss + entropy_bonus
        total_loss = (
            self.config.loss_weights["value"] * value_loss
            + self.config.loss_weights["policy"] * policy_loss
            + self.config.loss_weights["reward"] * reward_loss
            + self.config.loss_weights["entropy"] * entropy_bonus
        )

        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            total_loss = (total_loss * weights.squeeze()).mean()

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy": policy_loss.item(),
            "value": value_loss.item(),
            "reward": reward_loss.item(),
            "entropy": entropy_bonus.item(),
            "total": total_loss.item(),
        }

