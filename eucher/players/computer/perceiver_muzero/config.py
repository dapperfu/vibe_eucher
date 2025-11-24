"""Configuration for EuchrePerceiverMuZero."""

from enum import Enum
from pathlib import Path
from typing import Optional

import torch


class Phase(Enum):
    """Game phase enumeration."""

    DEAL = "deal"
    PICKUP = "pickup"
    CALL_TRUMP = "call_trump"
    PLAY = "play"
    SCORE = "score"


class Mode(Enum):
    """Game mode enumeration."""

    NORMAL = "normal"
    ALONE = "alone"


class PerceiverMuZeroConfig:
    """Configuration for EuchrePerceiverMuZero model and training."""

    def __init__(
        self,
        # Variant flags
        enable_screw_the_dealer: bool = False,
        enable_nine_ten_tradein: bool = False,
        enable_go_alone: bool = True,
        # Perceiver-IO parameters
        latent_size: int = 128,
        latent_slots: int = 32,
        # MCTS parameters
        num_simulations: int = 128,
        c_puct: float = 1.5,
        max_depth: int = 8,
        # Training parameters
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        replay_buffer_size: int = 100000,
        use_prioritized_replay: bool = False,
        # Other parameters
        device: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        num_games: int = 10,
        action_space_size: int = 20,
    ) -> None:
        """
        Initialize EuchrePerceiverMuZero configuration.

        Parameters
        ----------
        enable_screw_the_dealer : bool
            Enable screw the dealer variant (default: False).
        enable_nine_ten_tradein : bool
            Enable 9-10 trade-in variant (default: False).
        enable_go_alone : bool
            Enable going alone variant (default: True).
        latent_size : int
            Size of latent vector (default: 128).
        latent_slots : int
            Number of latent slots in Perceiver-IO (default: 32).
        num_simulations : int
            Number of MCTS simulations per move (default: 128, range: 128-800).
        c_puct : float
            MCTS exploration constant (default: 1.5).
        max_depth : int
            Maximum MCTS search depth (default: 8, range: 4-12).
        batch_size : int
            Training batch size (default: 32).
        learning_rate : float
            Learning rate for AdamW optimizer (default: 1e-4).
        weight_decay : float
            Weight decay for AdamW optimizer (default: 1e-3).
        replay_buffer_size : int
            Size of replay buffer (default: 100000, range: 50k-200k).
        use_prioritized_replay : bool
            Use prioritized experience replay (default: False).
        device : Optional[str]
            Device to use ("cuda", "cpu", or None for auto).
        checkpoint_dir : Optional[str]
            Checkpoint directory (default: models/checkpoints/perceiver_muzero).
        num_games : int
            Number of games per training iteration (default: 10).
        action_space_size : int
            Size of action space (default: 18).
        """
        # Variant flags
        self.enable_screw_the_dealer = enable_screw_the_dealer
        self.enable_nine_ten_tradein = enable_nine_ten_tradein
        self.enable_go_alone = enable_go_alone

        # Perceiver-IO parameters
        self.latent_size = latent_size
        self.latent_slots = latent_slots

        # MCTS parameters
        # Allow lower values for testing, but clamp to reasonable range
        self.num_simulations = max(1, min(800, num_simulations))  # Clamp to 1-800 (128+ recommended)
        self.c_puct = c_puct
        self.max_depth = max(1, min(12, max_depth))  # Clamp to 1-12 (4+ recommended)

        # Training parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.replay_buffer_size = max(50000, min(200000, replay_buffer_size))  # Clamp to 50k-200k
        self.use_prioritized_replay = use_prioritized_replay
        self.num_games = num_games
        self.action_space_size = action_space_size

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Checkpoint directory
        if checkpoint_dir is None:
            self.checkpoint_dir = Path("models/checkpoints/perceiver_muzero")
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss weights
        self.loss_weights = {
            "policy": 1.0,
            "value": 1.0,
            "reward": 1.0,
            "entropy": 0.01,
        }

        # Token embedding dimension for Perceiver-IO input
        self.token_dim = 64

        # Risk modulation parameters
        self.base_temperature = 1.0
        self.risk_factor = 1.0

