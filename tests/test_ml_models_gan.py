"""Tests for GAN models."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from eucher.players.computer.ml.ml_config import MLConfig
from eucher.players.computer.ml.ml_models_gan import EuchreGANGenerator, EuchreGANDiscriminator, GANCardPlayModel
from tests.fixtures.ml_test_data import generate_game_state_features, generate_random_hand


class TestEuchreGANGenerator:
    """Tests for GAN generator."""

    def test_forward_pass_shape(self) -> None:
        """Test that forward pass produces correct shape."""
        generator = EuchreGANGenerator(input_size=260, hidden_size=128, output_size=24)
        input_tensor = torch.randn(1, 260)
        output = generator(input_tensor)
        assert output.shape == (1, 24)

    def test_forward_pass_batch(self) -> None:
        """Test forward pass with batch."""
        generator = EuchreGANGenerator(input_size=260, hidden_size=128, output_size=24)
        input_tensor = torch.randn(5, 260)
        output = generator(input_tensor)
        assert output.shape == (5, 24)

    def test_output_probabilities_sum_to_one(self) -> None:
        """Test that output probabilities sum to 1."""
        generator = EuchreGANGenerator(input_size=260, hidden_size=128, output_size=24)
        input_tensor = torch.randn(1, 260)
        output = generator(input_tensor)
        assert torch.allclose(output.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_device_handling(self) -> None:
        """Test device handling (CPU/GPU)."""
        config = MLConfig()
        generator = EuchreGANGenerator().to(config.device)
        input_tensor = torch.randn(1, 260).to(config.device)
        output = generator(input_tensor)
        assert output.device == config.device


class TestEuchreGANDiscriminator:
    """Tests for GAN discriminator."""

    def test_forward_pass_shape(self) -> None:
        """Test that forward pass produces correct shape."""
        # Discriminator input is features + decision (260 + 24 = 284)
        discriminator = EuchreGANDiscriminator(input_size=284, hidden_size=128)
        input_tensor = torch.randn(1, 284)
        output = discriminator(input_tensor)
        assert output.shape == (1, 1)

    def test_output_is_probability(self) -> None:
        """Test that output is in [0, 1] range."""
        discriminator = EuchreGANDiscriminator(input_size=284, hidden_size=128)
        input_tensor = torch.randn(1, 284)
        output = discriminator(input_tensor)
        assert 0.0 <= output.item() <= 1.0


class TestGANCardPlayModel:
    """Tests for GAN card play model."""

    def test_predict_with_valid_cards(self) -> None:
        """Test prediction with valid cards."""
        config = MLConfig()
        model = GANCardPlayModel(config)
        hand = generate_random_hand(5)
        features = generate_game_state_features(hand=hand)
        valid_indices = [0, 1, 2, 3, 4]  # First 5 cards

        prediction = model.predict(features, valid_indices)
        assert prediction in valid_indices

    def test_predict_invalid_card_masking(self) -> None:
        """Test that invalid cards are masked."""
        config = MLConfig()
        model = GANCardPlayModel(config)
        hand = generate_random_hand(5)
        features = generate_game_state_features(hand=hand)
        valid_indices = [0, 1, 2]  # Only first 3 cards valid

        prediction = model.predict(features, valid_indices)
        assert prediction in valid_indices

    def test_predict_batch(self) -> None:
        """Test batch prediction."""
        config = MLConfig()
        model = GANCardPlayModel(config)
        batch_size = 3
        features_batch = np.random.randn(batch_size, 260).astype(np.float32)
        valid_indices_list = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        predictions = model.predict_batch(features_batch, valid_indices_list)
        assert len(predictions) == batch_size
        for i, pred in enumerate(predictions):
            assert pred in valid_indices_list[i]

    def test_save_and_load(self) -> None:
        """Test model save and load."""
        config = MLConfig()
        model = GANCardPlayModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.pth"
            model.save(filepath)
            assert filepath.exists()

            # Create new model and load
            new_model = GANCardPlayModel(config)
            new_model.load(filepath)

            # Verify weights are loaded
            assert (
                model.generator.state_dict()["network.0.weight"]
                == new_model.generator.state_dict()["network.0.weight"]
            ).all()

    def test_training_step(self) -> None:
        """Test a single training step (forward/backward)."""
        config = MLConfig()
        model = GANCardPlayModel(config)
        generator = model.generator
        discriminator = model.discriminator

        # Create dummy data
        batch_size = 4
        features = torch.randn(batch_size, 260).to(config.device)
        decisions_onehot = torch.zeros(batch_size, 24).to(config.device)
        decisions_onehot.scatter_(1, torch.randint(0, 24, (batch_size, 1)), 1.0)

        # Generator forward
        fake_decisions = generator(features)

        # Discriminator forward (real)
        real_input = torch.cat([features, decisions_onehot], dim=1)
        d_real = discriminator(real_input)

        # Discriminator forward (fake)
        fake_input = torch.cat([features, fake_decisions], dim=1)
        d_fake = discriminator(fake_input)

        assert d_real.shape == (batch_size, 1)
        assert d_fake.shape == (batch_size, 1)

