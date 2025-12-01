"""Reinforcement learning models for Euchre decision making."""

import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from eucher.cards import Card, Suit
from ..ml_config import MLConfig


class ExperienceReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int = 10000) -> None:
        """
        Initialize the replay buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of experiences to store.
        """
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add an experience to the buffer.

        Parameters
        ----------
        state : np.ndarray
            Current state.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Next state.
        done : bool
            Whether episode is done.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """
        Sample a batch of experiences.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        List[Tuple[np.ndarray, int, float, np.ndarray, bool]]
            List of sampled experiences.
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        """
        Get the current size of the buffer.

        Returns
        -------
        int
            Number of experiences in buffer.
        """
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Deep Q-Network for reinforcement learning."""

    def __init__(self, input_size: int = 260, hidden_size: int = 128, output_size: int = 24) -> None:
        """
        Initialize the DQN.

        Parameters
        ----------
        input_size : int
            Size of input feature vector.
        hidden_size : int
            Size of hidden layers.
        output_size : int
            Size of output (number of possible actions/cards).
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Q-values for each action.
        """
        return self.network(x)


class RLAgent:
    """Reinforcement learning agent for Euchre (stub - requires training implementation)."""

    def __init__(self, config: Optional[MLConfig] = None) -> None:
        """
        Initialize the RL agent.

        Parameters
        ----------
        config : Optional[MLConfig]
            ML configuration. If None, creates a new one.
        """
        if config is None:
            config = MLConfig()
        self.config = config
        self.device = config.device

        self.q_network = DQNNetwork().to(self.device)
        self.target_network = DQNNetwork().to(self.device)
        # Initialize target network with same weights as Q-network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.step_count = 0

    def select_action(
        self,
        features: np.ndarray,
        valid_card_indices: List[int],
        training: bool = False,
        temperature: Optional[float] = None,
    ) -> int:
        """
        Select an action using epsilon-greedy policy with optional temperature threshold.

        Parameters
        ----------
        features : np.ndarray
            Feature vector.
        valid_card_indices : List[int]
            List of valid card indices.
        training : bool
            Whether in training mode (affects exploration).
        temperature : Optional[float]
            Temperature threshold (0.0-1.0) for filtering by decision weights.
            If None, uses standard Q-value selection.

        Returns
        -------
        int
            Selected card index.
        """
        if not valid_card_indices:
            return 0

        # Epsilon-greedy: random with probability epsilon, else greedy
        if training and np.random.random() < self.epsilon:
            # Random exploration
            return int(np.random.choice(valid_card_indices))

        # Greedy action selection
        # Convert features to tensor
        if isinstance(features, np.ndarray):
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            features_tensor = features.to(self.device)

        # Get Q-values
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(features_tensor)
            q_vals = q_values.squeeze(0).cpu().numpy()

        # Convert Q-values to decision weights (normalize to [0, 1])
        # Normalize Q-values to probabilities using softmax
        q_vals_tensor = torch.tensor(q_vals, dtype=torch.float32)
        q_vals_normalized = torch.softmax(q_vals_tensor, dim=0).numpy()

        # Apply temperature threshold if provided
        if temperature is not None:
            from eucher.players.computer.ml.ml_decision_weights import apply_temperature_threshold

            filtered_indices, filtered_weights = apply_temperature_threshold(
                q_vals_normalized, temperature, valid_card_indices
            )
            if filtered_indices:
                # Select from filtered actions
                selected_idx = int(max(filtered_indices, key=lambda i: filtered_weights[i]))
                return selected_idx

        # Fallback: standard Q-value selection
        # Mask invalid actions
        masked_q_vals = np.full(len(q_vals), -np.inf)
        for idx in valid_card_indices:
            if 0 <= idx < len(q_vals):
                masked_q_vals[idx] = q_vals[idx]

        # Select action with highest Q-value
        selected_idx = int(np.argmax(masked_q_vals))
        return selected_idx

    def update_epsilon(self) -> None:
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath: Path, training_stats: Optional[dict] = None) -> None:
        """
        Save the model to disk.

        Parameters
        ----------
        filepath : Path
            Path to save the model.
        training_stats : Optional[dict]
            Optional dictionary containing training statistics to save.
            Can include: loss, rewards, games_played, etc.
        """
        save_dict = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "epsilon": self.epsilon,
        }
        if training_stats is not None:
            save_dict["training_stats"] = training_stats
        torch.save(save_dict, filepath)

    def load(self, filepath: Path) -> Optional[dict]:
        """
        Load the model from disk.

        Parameters
        ----------
        filepath : Path
            Path to load the model from.

        Returns
        -------
        Optional[dict]
            Training statistics if available, None otherwise.
        """
        if filepath.exists():
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
            return checkpoint.get("training_stats")
        return None

