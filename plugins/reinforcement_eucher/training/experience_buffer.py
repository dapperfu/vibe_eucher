"""Experience buffer for storing training trajectories."""

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch


class ExperienceBuffer:
    """Buffer for storing (state, action, reward, advantage, legal_mask) tuples.

    Parameters
    ----------
    max_size : int
        Maximum buffer size.
    """

    def __init__(self, max_size: int = 100000) -> None:
        """Initialize experience buffer.

        Parameters
        ----------
        max_size : int
            Maximum buffer size.
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        advantage: float,
        legal_mask: torch.Tensor,
        value: Optional[float] = None,
        log_prob: Optional[float] = None,
    ) -> None:
        """Add experience to buffer.

        Parameters
        ----------
        state : torch.Tensor
            State tensor.
        action : int
            Action taken.
        reward : float
            Reward received.
        advantage : float
            Advantage estimate.
        legal_mask : torch.Tensor
            Legal action mask.
        value : Optional[float]
            Value estimate (optional).
        log_prob : Optional[float]
            Log probability of action (optional).
        """
        experience = {
            "state": state.cpu() if isinstance(state, torch.Tensor) else state,
            "action": action,
            "reward": reward,
            "advantage": advantage,
            "legal_mask": legal_mask.cpu() if isinstance(legal_mask, torch.Tensor) else legal_mask,
            "value": value,
            "log_prob": log_prob,
        }
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[dict]:
        """Sample batch of experiences.

        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        List[dict]
            Batch of experiences.
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def sample_all(self) -> List[dict]:
        """Sample all experiences.

        Returns
        -------
        List[dict]
            All experiences.
        """
        return list(self.buffer)

    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """Get buffer size.

        Returns
        -------
        int
            Buffer size.
        """
        return len(self.buffer)

    def get_batch_tensors(
        self,
        batch: List[dict],
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert batch to tensors.

        Parameters
        ----------
        batch : List[dict]
            Batch of experiences.
        device : str
            Device for tensors.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (states, actions, rewards, advantages, legal_masks).
        """
        states = torch.stack([exp["state"] for exp in batch]).to(device)
        actions = torch.tensor([exp["action"] for exp in batch], dtype=torch.long).to(device)
        rewards = torch.tensor([exp["reward"] for exp in batch], dtype=torch.float32).to(device)
        advantages = torch.tensor([exp["advantage"] for exp in batch], dtype=torch.float32).to(device)
        legal_masks = torch.stack([exp["legal_mask"] for exp in batch]).to(device)

        return states, actions, rewards, advantages, legal_masks



