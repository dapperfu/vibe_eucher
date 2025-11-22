"""Configuration for ML models and training."""

import os
from pathlib import Path
from typing import Optional

import torch


class MLConfig:
    """Configuration for ML models."""

    def __init__(self) -> None:
        """Initialize ML configuration."""
        # Device selection
        self.device = self._get_device()

        # Model paths
        self.models_dir = Path("models")
        self.training_data_dir = Path("training_data")

        # Model hyperparameters
        self.hidden_size = 128
        self.num_layers = 3
        self.dropout = 0.2

        # Training parameters
        self.batch_size = 64
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.validation_split = 0.2

        # Risk/temperature parameters for decision-making
        # Temperature acts as threshold: actions with weight >= temperature are considered valid
        # Low temperature (0.0-0.3): Conservative (only high-confidence actions)
        # Medium temperature (0.3-0.7): Balanced
        # High temperature (0.7-1.0): Risky (accepts lower-confidence actions)
        self.trump_selection_risk: float = 0.5  # Risk factor for trump selection decisions (order up, call trump)
        self.gameplay_risk: float = 0.5  # Risk factor for gameplay decisions (play card, discard)

    def _get_device(self) -> torch.device:
        """
        Get the best available device (GPU or CPU).

        Returns
        -------
        torch.device
            The device to use for computation.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def get_model_path(self, model_name: str) -> Path:
        """
        Get the path to a model file.

        Parameters
        ----------
        model_name : str
            Name of the model file.

        Returns
        -------
        Path
            Path to the model file.
        """
        self.models_dir.mkdir(exist_ok=True)
        return self.models_dir / model_name

    def get_training_data_path(self, filename: str) -> Path:
        """
        Get the path to a training data file.

        Parameters
        ----------
        filename : str
            Name of the training data file.

        Returns
        -------
        Path
            Path to the training data file.
        """
        self.training_data_dir.mkdir(exist_ok=True)
        return self.training_data_dir / filename

