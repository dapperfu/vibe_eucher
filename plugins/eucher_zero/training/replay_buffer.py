"""Replay buffer for EucherZero training."""

import threading
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
        self.lock = threading.Lock()  # Thread safety for parallel game generation

    def add(
        self,
        state: torch.Tensor,
        policy: np.ndarray,
        value: float,
        reward: float,
        next_state: Optional[torch.Tensor] = None,
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
        reward : float
            Immediate reward.
        next_state : Optional[torch.Tensor]
            Next state tensor (optional).
        """
        with self.lock:
            self.buffer.append((state, policy, value, reward, next_state))

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
            (states, policies, values, rewards, next_states).
        """
        with self.lock:
            if len(self.buffer) == 0:
                raise ValueError("Buffer is empty")
            
            buffer_size = len(self.buffer)
        
        # Sample random indices (outside lock to minimize lock time)
        indices = np.random.choice(buffer_size, size=min(batch_size, buffer_size), replace=False)
        
        with self.lock:
            # Collect samples using list comprehensions (faster than explicit loops)
            samples = [self.buffer[idx] for idx in indices]

            # Extract all samples at once
            states = [s[0] for s in samples]
            policies = [s[1] for s in samples]
            values = [s[2] for s in samples]
            rewards = [s[3] for s in samples]
            next_states = [s[4] if s[4] is not None else s[0] for s in samples]

        # Convert to tensors using vectorized operations
        # Stack states and next_states in parallel-friendly way
        states_tensor = torch.stack(states)
        policies_tensor = torch.tensor(np.array(policies, dtype=np.float32), dtype=torch.float32)
        values_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.stack(next_states)

        return states_tensor, policies_tensor, values_tensor, rewards_tensor, next_states_tensor

    def __len__(self) -> int:
        """
        Get buffer size.

        Returns
        -------
        int
            Buffer size.
        """
        with self.lock:
            return len(self.buffer)

    def clear(self) -> None:
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()

