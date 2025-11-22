"""Integration tests for ML training pipelines."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from eucher.players.computer.ml.ml_config import MLConfig
from eucher.players.computer.ml.ml_models_gan import GANCardPlayModel
from eucher.players.computer.ml.ml_models_rl import RLAgent
from eucher.training.train_gan import train_gan_model
from eucher.training.train_rl import ExperienceReplayBuffer, train_rl_agent
from tests.fixtures.ml_test_data import generate_synthetic_decisions


class TestGANTraining:
    """Tests for GAN training."""

    def test_gan_training_produces_valid_model(self) -> None:
        """Test that GAN training produces a valid model."""
        config = MLConfig()
        features, decisions = generate_synthetic_decisions(num_samples=50, decision_type="play_card")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            model = train_gan_model(
                features=features,
                decisions=decisions,
                config=config,
                num_epochs=5,  # Small number for testing
                batch_size=8,
                output_dir=output_dir,
            )

            # Verify model can make predictions
            test_features = features[0]
            valid_indices = [0, 1, 2, 3, 4]
            prediction = model.predict(test_features, valid_indices)
            assert prediction in valid_indices

    def test_gan_model_loading(self) -> None:
        """Test that trained GAN model can be loaded."""
        config = MLConfig()
        features, decisions = generate_synthetic_decisions(num_samples=50, decision_type="play_card")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            model = train_gan_model(
                features=features,
                decisions=decisions,
                config=config,
                num_epochs=5,
                batch_size=8,
                output_dir=output_dir,
            )

            model_path = output_dir / "gan_model.pth"
            assert model_path.exists()

            # Load model
            new_model = GANCardPlayModel(config)
            new_model.load(model_path)

            # Verify it works
            test_features = features[0]
            valid_indices = [0, 1, 2, 3, 4]
            prediction = new_model.predict(test_features, valid_indices)
            assert prediction in valid_indices


class TestRLTraining:
    """Tests for RL training."""

    def test_rl_agent_training_step(self) -> None:
        """Test that RL agent can be trained on a batch."""
        config = MLConfig()
        agent = RLAgent(config)
        replay_buffer = ExperienceReplayBuffer(capacity=100)

        # Add some experiences
        for i in range(20):
            state = np.random.randn(260).astype(np.float32)
            action = np.random.randint(0, 24)
            reward = np.random.randn()
            next_state = np.random.randn(260).astype(np.float32)
            done = False
            replay_buffer.push(state, action, reward, next_state, done)

        # Train agent
        train_rl_agent(agent, replay_buffer, batch_size=8)

        # Verify agent can select actions
        features = np.random.randn(260).astype(np.float32)
        valid_indices = [0, 1, 2, 3, 4]
        action = agent.select_action(features, valid_indices, training=False)
        assert action in valid_indices

    def test_rl_agent_save_and_load(self) -> None:
        """Test that RL agent can be saved and loaded."""
        config = MLConfig()
        agent = RLAgent(config)
        agent.epsilon = 0.3

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_rl_agent.pth"
            agent.save(filepath)
            assert filepath.exists()

            new_agent = RLAgent(config)
            new_agent.load(filepath)
            assert new_agent.epsilon == 0.3


class TestDataCollection:
    """Tests for data collection."""

    def test_data_collection_format(self) -> None:
        """Test that collected data is in correct format."""
        from eucher.training.data_collector import GameDataCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = GameDataCollector(output_dir=Path(tmpdir))
            collector.start_game("test_game")

            # Record a decision
            from eucher.cards import Card, Rank, Suit

            hand = [Card(Suit.HEARTS, Rank.ACE), Card(Suit.HEARTS, Rank.KING)]
            turned_card = Card(Suit.HEARTS, Rank.QUEEN)
            collector.record_order_up_decision(
                player_id=0,
                hand=hand,
                turned_card=turned_card,
                dealer_id=0,
                decision=True,
                game_state={"trick_number": 0, "tricks_won_team0": 0, "tricks_won_team1": 0},
            )

            # Save and verify format
            collector.save_data(prefix="test")
            assert (Path(tmpdir) / "test_order_up.json").exists()

