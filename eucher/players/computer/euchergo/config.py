"""Configuration for EucherGo."""

from typing import Optional

import torch


class EucherGoConfig:
    """Configuration class for EucherGo.

    Parameters
    ----------
    input_size : int
        Size of input state tensor.
    hidden_size : int
        Hidden layer size.
    num_layers : int
        Number of residual layers in trunk.
    use_conv1d : bool
        If True, use Conv1D architecture; otherwise use MLP.
    num_simulations : int
        Number of MCTS simulations per move.
    exploration_constant : float
        Exploration constant for PUCT.
    device : Optional[torch.device]
        Device to run on.
    """

    def __init__(
        self,
        input_size: int = 712,
        hidden_size: int = 256,
        num_layers: int = 4,
        use_conv1d: bool = False,
        num_simulations: int = 100,
        exploration_constant: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize configuration."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_conv1d = use_conv1d
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device


