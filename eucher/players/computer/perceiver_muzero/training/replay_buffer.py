"""Replay buffer for EuchrePerceiverMuZero training."""

from typing import List, Optional, Tuple

import numpy as np
import torch

from ..config import PerceiverMuZeroConfig


class ReplayBuffer:
    """Replay buffer with optional prioritized experience replay."""

    def __init__(self, config: PerceiverMuZeroConfig) -> None:
        """
        Initialize replay buffer.

        Parameters
        ----------
        config : PerceiverMuZeroConfig
            Configuration object.
        """
        self.config = config
        self.max_size = config.replay_buffer_size
        self.use_prioritized = config.use_prioritized_replay

        self.buffer: List[Tuple] = []
        self.priorities: List[float] = []
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4  # Importance sampling exponent
        self.beta_increment = 0.001
        self.max_beta = 1.0

    def add(
        self,
        input_tokens: torch.Tensor,
        action: int,
        policy: np.ndarray,
        value: float,
        reward: float,
        next_input_tokens: Optional[torch.Tensor] = None,
        priority: Optional[float] = None,
    ) -> None:
        """
        Add experience to buffer.

        Parameters
        ----------
        input_tokens : torch.Tensor
            Input token sequence.
        action : int
            Action taken.
        policy : np.ndarray
            Policy distribution.
        value : float
            Value estimate.
        reward : float
            Immediate reward.
        next_input_tokens : Optional[torch.Tensor]
            Next state tokens (optional).
        priority : Optional[float]
            Priority for prioritized replay (optional).
        """
        self.buffer.append(
            (input_tokens, action, policy, value, reward, next_input_tokens)
        )

        if self.use_prioritized:
            if priority is None:
                # Use TD error as priority
                priority = abs(reward + value)
            self.priorities.append(priority)

        # Remove oldest if buffer is full
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            if self.use_prioritized:
                self.priorities.pop(0)

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
    ]:
        """
        Sample a batch from buffer.

        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            (input_tokens, actions, policies, values, rewards, next_input_tokens, weights).
            weights is None if not using prioritized replay.
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        if self.use_prioritized and len(self.priorities) > 0:
            return self._sample_prioritized(batch_size)
        else:
            return self._sample_uniform(batch_size)

    def _sample_uniform(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
    ]:
        """Sample uniformly from buffer."""
        # Sample random indices
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            replace=False,
        )

        # Collect samples
        input_tokens_list = []
        actions = []
        policies = []
        values = []
        rewards = []
        next_input_tokens_list = []

        for idx in indices:
            (
                input_tokens,
                action,
                policy,
                value,
                reward,
                next_input_tokens,
            ) = self.buffer[idx]
            input_tokens_list.append(input_tokens)
            actions.append(action)
            policies.append(policy)
            values.append(value)
            rewards.append(reward)
            if next_input_tokens is not None:
                next_input_tokens_list.append(next_input_tokens)
            else:
                next_input_tokens_list.append(input_tokens)

        # Convert to tensors
        # Note: input_tokens may have variable length, so we pad or handle separately
        # For now, assume we can stack them (they should be same size from state encoder)
        input_tokens_tensor = torch.stack(input_tokens_list)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        policies_tensor = torch.tensor(np.array(policies), dtype=torch.float32)
        values_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_input_tokens_tensor = torch.stack(next_input_tokens_list)

        return (
            input_tokens_tensor,
            actions_tensor,
            policies_tensor,
            values_tensor,
            rewards_tensor,
            next_input_tokens_tensor,
            None,  # No weights for uniform sampling
        )

    def _sample_prioritized(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Sample using prioritized experience replay."""
        # Compute sampling probabilities
        priorities_array = np.array(self.priorities)
        probabilities = priorities_array ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            replace=False,
            p=probabilities,
        )

        # Compute importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        self.beta = min(self.max_beta, self.beta + self.beta_increment)

        # Collect samples
        input_tokens_list = []
        actions = []
        policies = []
        values = []
        rewards = []
        next_input_tokens_list = []

        for idx in indices:
            (
                input_tokens,
                action,
                policy,
                value,
                reward,
                next_input_tokens,
            ) = self.buffer[idx]
            input_tokens_list.append(input_tokens)
            actions.append(action)
            policies.append(policy)
            values.append(value)
            rewards.append(reward)
            if next_input_tokens is not None:
                next_input_tokens_list.append(next_input_tokens)
            else:
                next_input_tokens_list.append(input_tokens)

        # Convert to tensors
        input_tokens_tensor = torch.stack(input_tokens_list)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        policies_tensor = torch.tensor(np.array(policies), dtype=torch.float32)
        values_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_input_tokens_tensor = torch.stack(next_input_tokens_list)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        return (
            input_tokens_tensor,
            actions_tensor,
            policies_tensor,
            values_tensor,
            rewards_tensor,
            next_input_tokens_tensor,
            weights_tensor,
        )

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities for prioritized replay.

        Parameters
        ----------
        indices : List[int]
            Indices to update.
        priorities : List[float]
            New priorities.
        """
        if not self.use_prioritized:
            return

        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority

    def __len__(self) -> int:
        """
        Get buffer size.

        Returns
        -------
        int
            Buffer size.
        """
        return len(self.buffer)

    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()
        if self.use_prioritized:
            self.priorities.clear()

