"""Configuration for ReinforcementEucher."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class ReinforcementEucherConfig:
    """Configuration for ReinforcementEucher model and training.

    Parameters
    ----------
    state_dim : int
        Dimension of state representation.
    action_space_size : int
        Size of action space.
    hidden_size : int
        Hidden layer size for networks.
    num_layers : int
        Number of hidden layers.
    learning_rate : float
        Learning rate for optimizer.
    gamma : float
        Discount factor for future rewards.
    ppo_clip_epsilon : float
        PPO clipping parameter.
    entropy_coef : float
        Entropy bonus coefficient.
    value_loss_coef : float
        Value loss coefficient.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    batch_size : int
        Training batch size.
    num_epochs_per_update : int
        Number of epochs per PPO update.
    trick_reward_win : float
        Reward for winning a trick.
    trick_reward_lose : float
        Reward for losing a trick.
    hand_reward_win : float
        Reward for winning a hand.
    hand_reward_lose : float
        Reward for losing a hand.
    hand_reward_lone_success : float
        Extra reward for successful lone hand.
    hand_reward_euchred : float
        Extra penalty for getting euchred.
    training_stage : str
        Current training stage: "trick_only" or "full_hand".
    checkpoint_dir : Optional[str]
        Directory for checkpoints.
    device : Optional[str]
        Device to use ("cuda", "cpu", or None for auto).
    use_transformer : bool
        Whether to use Transformer architecture (else MLP).
    num_attention_heads : int
        Number of attention heads if using Transformer.
    """

    # Network architecture
    state_dim: int = 200  # Will be calculated from state encoder
    action_space_size: int = 18  # Unified action space
    hidden_size: int = 256
    num_layers: int = 3
    use_transformer: bool = False
    num_attention_heads: int = 4

    # RL hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    ppo_clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training parameters
    batch_size: int = 64
    num_epochs_per_update: int = 4
    num_games_per_update: int = 10
    checkpoint_interval: int = 100  # Save checkpoint every N games

    # Reward structure
    trick_reward_win: float = 0.1
    trick_reward_lose: float = -0.1
    hand_reward_win: float = 1.0
    hand_reward_lose: float = -1.0
    hand_reward_lone_success: float = 1.0
    hand_reward_euchred: float = -1.0

    # Training stage
    training_stage: str = "trick_only"  # "trick_only" or "full_hand"

    # Paths and device
    checkpoint_dir: Optional[str] = None
    device: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize derived attributes."""
        # Device setup
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(self.device)

        # Checkpoint directory
        if self.checkpoint_dir is None:
            self.checkpoint_dir = Path("models/checkpoints/reinforcement_eucher")
        else:
            self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert config to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary.
        """
        return {
            "state_dim": self.state_dim,
            "action_space_size": self.action_space_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "use_transformer": self.use_transformer,
            "num_attention_heads": self.num_attention_heads,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "ppo_clip_epsilon": self.ppo_clip_epsilon,
            "entropy_coef": self.entropy_coef,
            "value_loss_coef": self.value_loss_coef,
            "max_grad_norm": self.max_grad_norm,
            "batch_size": self.batch_size,
            "num_epochs_per_update": self.num_epochs_per_update,
            "num_games_per_update": self.num_games_per_update,
            "checkpoint_interval": self.checkpoint_interval,
            "trick_reward_win": self.trick_reward_win,
            "trick_reward_lose": self.trick_reward_lose,
            "hand_reward_win": self.hand_reward_win,
            "hand_reward_lose": self.hand_reward_lose,
            "hand_reward_lone_success": self.hand_reward_lone_success,
            "hand_reward_euchred": self.hand_reward_euchred,
            "training_stage": self.training_stage,
            "checkpoint_dir": str(self.checkpoint_dir),
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ReinforcementEucherConfig":
        """Create config from dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary.

        Returns
        -------
        ReinforcementEucherConfig
            Configuration object.
        """
        # Remove checkpoint_dir from dict if present, handle separately
        checkpoint_dir = config_dict.pop("checkpoint_dir", None)
        device = config_dict.pop("device", None)
        config = cls(**config_dict)
        if checkpoint_dir:
            config.checkpoint_dir = Path(checkpoint_dir)
        if device:
            config.device = device
        return config



