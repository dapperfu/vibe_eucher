"""GAN-based models for Euchre decision making (stub implementation)."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.cards import Card, Suit
from src.ml_config import MLConfig


class EuchreGANGenerator(nn.Module):
    """Generator network for GAN-based card play decisions."""

    def __init__(self, input_size: int = 260, hidden_size: int = 128, output_size: int = 24) -> None:
        """
        Initialize the generator.

        Parameters
        ----------
        input_size : int
            Size of input feature vector.
        hidden_size : int
            Size of hidden layers.
        output_size : int
            Size of output (number of possible cards).
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1),
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
            Output probabilities.
        """
        return self.network(x)


class EuchreGANDiscriminator(nn.Module):
    """Discriminator network for GAN-based decision evaluation."""

    def __init__(self, input_size: int = 284, hidden_size: int = 128) -> None:
        """
        Initialize the discriminator.

        Parameters
        ----------
        input_size : int
            Size of input (features + decision one-hot = 260 + 24 = 284).
        hidden_size : int
            Size of hidden layers.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
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
            Probability that decision is optimal.
        """
        return self.network(x)


class GANCardPlayModel:
    """GAN model for card play decisions.
    
    IMPORTANT: This model REQUIRES training before use. Without training,
    the model will use random weights and make essentially random decisions.
    
    To train the model, use the training functions in src.training.train_gan.
    The trained model should be saved and loaded using the save() and load()
    methods before making predictions.
    """

    def __init__(self, config: Optional[MLConfig] = None) -> None:
        """
        Initialize the GAN model.

        Parameters
        ----------
        config : Optional[MLConfig]
            ML configuration. If None, creates a new one.
            
        Note
        ----
        The model will be initialized with random weights. You must train
        the model (or load a trained model) before using it for predictions.
        """
        if config is None:
            config = MLConfig()
        self.config = config
        self.device = config.device

        self.generator = EuchreGANGenerator().to(self.device)
        # Discriminator input: features (260) + decision one-hot (24) = 284
        self.discriminator = EuchreGANDiscriminator(input_size=284).to(self.device)

    def predict(self, features: np.ndarray, valid_card_indices: List[int]) -> int:
        """
        Predict card to play using generator.

        Parameters
        ----------
        features : np.ndarray
            Feature vector.
        valid_card_indices : List[int]
            List of valid card indices.

        Returns
        -------
        int
            Predicted card index.
        """
        if not valid_card_indices:
            return 0

        # Convert features to tensor
        if isinstance(features, np.ndarray):
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            features_tensor = features.to(self.device)

        # Run through generator
        self.generator.eval()
        with torch.no_grad():
            probabilities = self.generator(features_tensor)
            probs = probabilities.squeeze(0).cpu().numpy()

        # Mask invalid cards
        masked_probs = np.full(len(probs), -np.inf)
        for idx in valid_card_indices:
            if 0 <= idx < len(probs):
                masked_probs[idx] = probs[idx]

        # Select card with highest probability
        selected_idx = int(np.argmax(masked_probs))
        return selected_idx

    def predict_batch(self, features_batch: np.ndarray, valid_card_indices_list: List[List[int]]) -> List[int]:
        """
        Predict cards for a batch of game states.

        Parameters
        ----------
        features_batch : np.ndarray
            Batch of feature vectors (batch_size, feature_size).
        valid_card_indices_list : List[List[int]]
            List of valid card indices for each sample.

        Returns
        -------
        List[int]
            List of predicted card indices.
        """
        if isinstance(features_batch, np.ndarray):
            features_tensor = torch.tensor(features_batch, dtype=torch.float32).to(self.device)
        else:
            features_tensor = features_batch.to(self.device)

        self.generator.eval()
        with torch.no_grad():
            probabilities = self.generator(features_tensor)
            probs = probabilities.cpu().numpy()

        predictions = []
        for i, valid_indices in enumerate(valid_card_indices_list):
            if not valid_indices:
                predictions.append(0)
                continue

            # Mask invalid cards
            masked_probs = np.full(len(probs[i]), -np.inf)
            for idx in valid_indices:
                if 0 <= idx < len(probs[i]):
                    masked_probs[idx] = probs[i][idx]

            selected_idx = int(np.argmax(masked_probs))
            predictions.append(selected_idx)

        return predictions

    def save(self, filepath: Path) -> None:
        """
        Save the model to disk.

        Parameters
        ----------
        filepath : Path
            Path to save the model.
        """
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: Path) -> None:
        """
        Load the model from disk.

        Parameters
        ----------
        filepath : Path
            Path to load the model from.
        """
        if filepath.exists():
            checkpoint = torch.load(filepath, map_location=self.device)
            self.generator.load_state_dict(checkpoint["generator"])
            self.discriminator.load_state_dict(checkpoint["discriminator"])

