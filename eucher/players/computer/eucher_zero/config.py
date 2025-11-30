"""Configuration for EucherZero."""

from pathlib import Path
from typing import Optional

import torch


class EucherZeroConfig:
    """Configuration for EucherZero model and training."""

    def __init__(
        self,
        latent_size: int = 128,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        num_games: int = 5,
        num_simulations: int = 10,
        exploration_constant: float = 1.0,
        device: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize EucherZero configuration.

        Parameters
        ----------
        latent_size : int
            Size of latent vector (default: 128 for fast training).
        batch_size : int
            Training batch size (default: 8).
        learning_rate : float
            Learning rate (default: 1e-3).
        num_games : int
            Number of games per training iteration (default: 5 for 1-min training).
        num_simulations : int
            Number of MCTS simulations per decision (default: 10 for speed).
        exploration_constant : float
            UCB exploration constant (default: 1.0).
        device : Optional[str]
            Device to use ("cuda", "cpu", or None for auto).
        checkpoint_dir : Optional[str]
            Checkpoint directory (default: models/checkpoints/eucher_zero).
        """
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Checkpoint directory
        if checkpoint_dir is None:
            self.checkpoint_dir = Path("models/checkpoints/eucher_zero")
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss weights
        self.loss_weights = {
            "policy": 1.0,
            "value": 1.0,
            "risk_value": 0.5,
            "dynamics": 0.5,
            "reward": 0.1,
        }

        # Network architecture
        self.representation_layers = [256, 128, 64]
        self.dynamics_layers = [128, 128]
        self.prediction_layers = [64]

        # Action space
        self.action_space_size = 18

        # State encoding dimensions
        # Calculated from StateEncoder.encode_state_dict_to_tensor():
        # hand(120) + upcard(24) + is_dealer(1) + player_seat(4) + partner_seat(4) +
        # trump_context(5) + bidding_stage(3) + score(2) + trick_history(40) +
        # perfect_memory(96) + deduction_map(72) + suit_voids(12) + trump_counts(3) +
        # risk_factor(1) = 387
        self.state_dim = 387

