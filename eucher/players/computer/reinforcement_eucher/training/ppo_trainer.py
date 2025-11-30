"""PPO trainer for ReinforcementEucher."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from eucher.players.computer.reinforcement_eucher.config import ReinforcementEucherConfig
from eucher.players.computer.reinforcement_eucher.networks.model import ReinforcementEucherModel
from eucher.players.computer.reinforcement_eucher.training.experience_buffer import ExperienceBuffer


class PPOTrainer:
    """PPO trainer for policy gradient learning.

    Parameters
    ----------
    model : ReinforcementEucherModel
        Model to train.
    config : ReinforcementEucherConfig
        Configuration.
    """

    def __init__(
        self,
        model: ReinforcementEucherModel,
        config: ReinforcementEucherConfig,
    ) -> None:
        """Initialize PPO trainer.

        Parameters
        ----------
        model : ReinforcementEucherModel
            Model to train.
        config : ReinforcementEucherConfig
            Configuration.
        """
        self.model = model
        self.config = config
        self.device = config.torch_device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4,
        )

    def train_step(
        self,
        batch: List[dict],
    ) -> Dict[str, float]:
        """Perform one PPO training step.

        Parameters
        ----------
        batch : List[dict]
            Batch of experiences.

        Returns
        -------
        Dict[str, float]
            Training metrics.
        """
        from eucher.players.computer.reinforcement_eucher.training.experience_buffer import ExperienceBuffer

        buffer = ExperienceBuffer()
        states, actions, rewards, advantages, legal_masks = buffer.get_batch_tensors(batch, self.device)

        # Get old log probs if available
        old_log_probs = None
        if batch[0].get("log_prob") is not None:
            old_log_probs = torch.tensor([exp["log_prob"] for exp in batch], dtype=torch.float32).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        # Multiple epochs per update
        for epoch in range(self.config.num_epochs_per_update):
            # Forward pass
            logits, values = self.model(states, legal_masks)

            # Policy loss
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            if old_log_probs is not None:
                # PPO clipped objective
                ratio = torch.exp(action_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.config.ppo_clip_epsilon, 1 + self.config.ppo_clip_epsilon)
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            else:
                # Simple policy gradient
                policy_loss = -(action_log_probs * advantages).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(-1), rewards)

            # Entropy bonus
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # Total loss
            total_loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                - self.config.entropy_coef * entropy
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        return {
            "policy_loss": total_policy_loss / self.config.num_epochs_per_update,
            "value_loss": total_value_loss / self.config.num_epochs_per_update,
            "entropy": total_entropy / self.config.num_epochs_per_update,
        }

    def compute_advantages(
        self,
        rewards: List[float],
        values: List[float],
        dones: Optional[List[bool]] = None,
    ) -> Tuple[List[float], List[float]]:
        """Compute advantages and returns using GAE.

        Parameters
        ----------
        rewards : List[float]
            Rewards.
        values : List[float]
            Value estimates.
        dones : Optional[List[bool]]
            Done flags.

        Returns
        -------
        Tuple[List[float], List[float]]
            (advantages, returns).
        """
        if dones is None:
            dones = [False] * len(rewards)

        advantages = []
        returns = []

        gae = 0.0
        next_value = 0.0

        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            if dones[t]:
                gae = 0.0
                next_value = 0.0

            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * 0.95 * gae  # GAE lambda = 0.95
            advantages.insert(0, gae)

            return_val = rewards[t] + self.config.gamma * next_value
            returns.insert(0, return_val)

            next_value = values[t]

        return advantages, returns


