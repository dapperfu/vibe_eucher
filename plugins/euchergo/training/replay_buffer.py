"""Replay buffer for EucherGo training."""

from typing import List, Optional, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Simple replay buffer for storing training examples."""

    def __init__(self, max_size: int = 10000) -> None:
        """
        Initialize replay buffer.

        Parameters
        ----------
        max_size : int
            Maximum buffer size.
        """
        self.max_size = max_size
        self.buffer: List[Tuple] = []

    def add(
        self,
        state: torch.Tensor,
        policy: np.ndarray,
        value: float,
    ) -> None:
        """
        Add experience to buffer.

        Parameters
        ----------
        state : torch.Tensor
            State tensor.
        policy : np.ndarray
            Policy distribution.
        value : float
            Value estimate.
        """
        self.buffer.append((state, policy, value))

        # Remove oldest if buffer is full
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch from buffer.

        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            (states, policies, values).
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        # Sample random indices
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)

        # Collect samples using list comprehensions (faster than explicit loops)
        samples = [self.buffer[idx] for idx in indices]
        states = [s[0] for s in samples]
        policies = [s[1] for s in samples]
        values = [s[2] for s in samples]

        # Convert to tensors using vectorized operations
        states_tensor = torch.stack(states)
        policies_tensor = torch.tensor(np.array(policies, dtype=np.float32), dtype=torch.float32)
        values_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

        return states_tensor, policies_tensor, values_tensor

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


