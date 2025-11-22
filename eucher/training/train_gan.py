"""Train GAN models for Euchre decision making."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from eucher.players.computer.ml.ml_config import MLConfig
from eucher.players.computer.ml.ml_models_gan import EuchreGANGenerator, EuchreGANDiscriminator, GANCardPlayModel


def load_training_data_for_gan(
    data_dir: Path, prefix: str = "training", decision_type: str = "play_card"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load training data for GAN training.

    Parameters
    ----------
    data_dir : Path
        Directory containing training data files.
    prefix : str
        Prefix for data filenames.
    decision_type : str
        Type of decision: "play_card", "order_up", "call_trump", "discard"

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (features, decisions) where decisions are card indices.
    """
    data_file = data_dir / f"{prefix}_{decision_type}.json"

    if not data_file.exists():
        raise FileNotFoundError(f"Training data file not found: {data_file}")

    with open(data_file, "r") as f:
        data = json.load(f)

    if not data:
        raise ValueError(f"No data found in {data_file}")

    features = np.array([item["features"] for item in data])
    decisions = np.array([item["decision"] for item in data])

    return features, decisions


def train_gan_model(
    features: np.ndarray,
    decisions: np.ndarray,
    config: Optional[MLConfig] = None,
    num_epochs: int = 100,
    batch_size: int = 64,
    generator_lr: float = 0.0002,
    discriminator_lr: float = 0.0002,
    output_dir: Optional[Path] = None,
    checkpoint_interval: int = 10,
) -> GANCardPlayModel:
    """
    Train a GAN model on decision data.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, feature_size).
    decisions : np.ndarray
        Decision labels (card indices) (n_samples,).
    config : Optional[MLConfig]
        ML configuration. If None, creates a new one.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    generator_lr : float
        Learning rate for generator.
    discriminator_lr : float
        Learning rate for discriminator.
    output_dir : Optional[Path]
        Directory to save trained model and checkpoints.
    checkpoint_interval : int
        Save checkpoint every N epochs.

    Returns
    -------
    GANCardPlayModel
        Trained GAN model.
    """
    if config is None:
        config = MLConfig()

    device = config.device
    print(f"Training GAN on device: {device}")

    # Create model
    gan_model = GANCardPlayModel(config)
    generator = gan_model.generator
    discriminator = gan_model.discriminator

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=generator_lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # Convert data to tensors
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    decisions_tensor = torch.tensor(decisions, dtype=torch.long).to(device)

    # Create dataset and dataloader
    dataset = TensorDataset(features_tensor, decisions_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    print(f"Training GAN for {num_epochs} epochs...")
    print(f"Dataset size: {len(features)} samples")

    for epoch in range(num_epochs):
        g_losses = []
        d_losses = []

        for batch_features, batch_decisions in dataloader:
            batch_size_actual = batch_features.size(0)
            real_labels = torch.ones(batch_size_actual, 1).to(device)
            fake_labels = torch.zeros(batch_size_actual, 1).to(device)

            # Train Discriminator
            d_optimizer.zero_grad()

            # Real data
            # Create one-hot encoded decisions
            real_decisions_onehot = torch.zeros(batch_size_actual, 24).to(device)
            real_decisions_onehot.scatter_(1, batch_decisions.unsqueeze(1), 1.0)

            # Concatenate features with decisions for discriminator
            real_input = torch.cat([batch_features, real_decisions_onehot], dim=1)
            d_real_output = discriminator(real_input)
            d_real_loss = criterion(d_real_output, real_labels)

            # Fake data (from generator)
            fake_decisions_probs = generator(batch_features)
            # Sample from probabilities
            fake_decisions = torch.multinomial(fake_decisions_probs, 1).squeeze(1)
            fake_decisions_onehot = torch.zeros(batch_size_actual, 24).to(device)
            fake_decisions_onehot.scatter_(1, fake_decisions.unsqueeze(1), 1.0)

            fake_input = torch.cat([batch_features, fake_decisions_onehot], dim=1)
            d_fake_output = discriminator(fake_input.detach())
            d_fake_loss = criterion(d_fake_output, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            # Generator wants discriminator to think fake is real
            fake_input_gen = torch.cat([batch_features, fake_decisions_onehot], dim=1)
            d_fake_output_gen = discriminator(fake_input_gen)
            g_loss = criterion(d_fake_output_gen, real_labels)

            g_loss.backward()
            g_optimizer.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        avg_g_loss = np.mean(g_losses)
        avg_d_loss = np.mean(d_losses)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}"
            )

        # Save checkpoint
        if output_dir is not None and (epoch + 1) % checkpoint_interval == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = output_dir / f"gan_checkpoint_epoch_{epoch + 1}.pth"
            gan_model.save(checkpoint_path)

    print("GAN training complete!")

    # Save final model
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "gan_model.pth"
        gan_model.save(final_path)
        print(f"Saved final model to {final_path}")

    return gan_model


def train_gan_for_decision_type(
    data_dir: Path,
    decision_type: str = "play_card",
    config: Optional[MLConfig] = None,
    num_epochs: int = 100,
    output_dir: Optional[Path] = None,
    prefix: str = "training",
) -> GANCardPlayModel:
    """
    Train GAN model for a specific decision type.

    Parameters
    ----------
    data_dir : Path
        Directory containing training data.
    decision_type : str
        Type of decision to train on.
    config : Optional[MLConfig]
        ML configuration.
    num_epochs : int
        Number of training epochs.
    output_dir : Optional[Path]
        Directory to save trained model.
    prefix : str
        Prefix for data filenames.

    Returns
    -------
    GANCardPlayModel
        Trained GAN model.
    """
    if config is None:
        config = MLConfig()

    if output_dir is None:
        output_dir = config.models_dir

    print(f"Loading training data for {decision_type}...")
    features, decisions = load_training_data_for_gan(data_dir, prefix, decision_type)

    print(f"Training GAN for {decision_type}...")
    model = train_gan_model(
        features=features,
        decisions=decisions,
        config=config,
        num_epochs=num_epochs,
        output_dir=output_dir,
    )

    return model


if __name__ == "__main__":
    from pathlib import Path

    config = MLConfig()
    train_gan_for_decision_type(
        data_dir=config.training_data_dir,
        decision_type="play_card",
        num_epochs=50,
        output_dir=config.models_dir,
    )
