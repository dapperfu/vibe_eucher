"""PyTorch model architectures for Euchre AI decisions."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.ml_config import MLConfig


class TrumpSelectionNet(nn.Module):
    """Neural network for trump selection decisions (order_up and call_trump)."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2) -> None:
        """
        Initialize the trump selection network.

        Parameters
        ----------
        input_size : int
            Size of input feature vector.
        hidden_size : int
            Size of hidden layers.
        num_layers : int
            Number of hidden layers.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.features = nn.Sequential(*layers)

        # Output layers
        # Binary decision: order_up (sigmoid)
        self.order_up_output = nn.Linear(hidden_size, 1)

        # Suit selection: call_trump (softmax over 4 suits + pass)
        self.call_trump_output = nn.Linear(hidden_size, 5)  # 4 suits + pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (batch_size, input_size).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (order_up_logits, call_trump_logits).
            order_up_logits: (batch_size, 1) - logits for binary decision
            call_trump_logits: (batch_size, 5) - logits for suit selection
        """
        features = self.features(x)
        order_up = self.order_up_output(features)
        call_trump = self.call_trump_output(features)
        return order_up, call_trump


class CardPlayNet(nn.Module):
    """Neural network for card selection during tricks."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2) -> None:
        """
        Initialize the card play network.

        Parameters
        ----------
        input_size : int
            Size of input feature vector.
        hidden_size : int
            Size of hidden layers.
        num_layers : int
            Number of hidden layers.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.features = nn.Sequential(*layers)

        # Output: probability distribution over 24 possible cards
        # Note: During inference, we'll mask invalid cards
        self.card_output = nn.Linear(hidden_size, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            Card selection logits of shape (batch_size, 24).
        """
        features = self.features(x)
        return self.card_output(features)


class DiscardNet(nn.Module):
    """Neural network for dealer discard decisions."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2) -> None:
        """
        Initialize the discard network.

        Parameters
        ----------
        input_size : int
            Size of input feature vector.
        hidden_size : int
            Size of hidden layers.
        num_layers : int
            Number of hidden layers.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.features = nn.Sequential(*layers)

        # Output: probability distribution over 6 cards (dealer has 6 after picking up)
        self.discard_output = nn.Linear(hidden_size, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            Discard selection logits of shape (batch_size, 6).
        """
        features = self.features(x)
        return self.discard_output(features)


class EuchreMLModel:
    """Container for all ML models used in Euchre AI."""

    def __init__(self, config: Optional[MLConfig] = None) -> None:
        """
        Initialize the ML model container.

        Parameters
        ----------
        config : Optional[MLConfig]
            ML configuration. If None, creates a new config.
        """
        self.config = config or MLConfig()
        self.device = self.config.device

        # Input size determined by feature encoder
        # Hand (5*24=120) + Turned (24) + Trump (4) + Led (4) + Trick (3*24=72) + Played (24) + Positional (12) = 260
        input_size = 260

        # Initialize models
        self.trump_net = TrumpSelectionNet(input_size, self.config.hidden_size, self.config.num_layers, self.config.dropout)
        self.card_play_net = CardPlayNet(input_size, self.config.hidden_size, self.config.num_layers, self.config.dropout)
        self.discard_net = DiscardNet(input_size, self.config.hidden_size, self.config.num_layers, self.config.dropout)

        # Move models to device
        self.trump_net.to(self.device)
        self.card_play_net.to(self.device)
        self.discard_net.to(self.device)

        # Set to evaluation mode by default
        self.trump_net.eval()
        self.card_play_net.eval()
        self.discard_net.eval()

    def load_weights(self, trump_path: Optional[str] = None, card_play_path: Optional[str] = None, discard_path: Optional[str] = None) -> None:
        """
        Load model weights from files.

        Parameters
        ----------
        trump_path : Optional[str]
            Path to trump selection model weights.
        card_play_path : Optional[str]
            Path to card play model weights.
        discard_path : Optional[str]
            Path to discard model weights.
        """
        if trump_path and Path(trump_path).exists():
            self.trump_net.load_state_dict(torch.load(trump_path, map_location=self.device))
        if card_play_path and Path(card_play_path).exists():
            self.card_play_net.load_state_dict(torch.load(card_play_path, map_location=self.device))
        if discard_path and Path(discard_path).exists():
            self.discard_net.load_state_dict(torch.load(discard_path, map_location=self.device))

    def save_weights(self, trump_path: str, card_play_path: str, discard_path: str) -> None:
        """
        Save model weights to files.

        Parameters
        ----------
        trump_path : str
            Path to save trump selection model weights.
        card_play_path : str
            Path to save card play model weights.
        discard_path : str
            Path to save discard model weights.
        """
        torch.save(self.trump_net.state_dict(), trump_path)
        torch.save(self.card_play_net.state_dict(), card_play_path)
        torch.save(self.discard_net.state_dict(), discard_path)

    def train_mode(self) -> None:
        """Set all models to training mode."""
        self.trump_net.train()
        self.card_play_net.train()
        self.discard_net.train()

    def eval_mode(self) -> None:
        """Set all models to evaluation mode."""
        self.trump_net.eval()
        self.card_play_net.eval()
        self.discard_net.eval()

