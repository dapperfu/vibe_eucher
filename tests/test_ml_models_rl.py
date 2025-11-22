"""Tests for RL models."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from eucher.players.computer.ml.ml_config import MLConfig
from eucher.players.computer.ml.ml_models_rl import DQNNetwork, ExperienceReplayBuffer, RLAgent
from tests.fixtures.ml_test_data import generate_game_state_features, generate_random_hand


class TestDQNNetwork:
    """Tests for DQN network."""

    def test_forward_pass_produces_q_values(self) -> None:
        """Test that forward pass produces Q-values."""
        network = DQNNetwork(input_size=260, hidden_size=128, output_size=24)
        input_tensor = torch.randn(1, 260)
        output = network(input_tensor)
        assert output.shape == (1, 24)

    def test_output_dimensions(self) -> None:
        """Test correct output dimensions."""
        network = DQNNetwork(input_size=260, hidden_size=128, output_size=24)
        batch_size = 5
        input_tensor = torch.randn(batch_size, 260)
        output = network(input_tensor)
        assert output.shape == (batch_size, 24)

    def test_device_handling(self) -> None:
        """Test device handling."""
        config = MLConfig()
        network = DQNNetwork().to(config.device)
        input_tensor = torch.randn(1, 260).to(config.device)
        output = network(input_tensor)
        assert output.device == config.device


class TestExperienceReplayBuffer:
    """Tests for experience replay buffer."""

    def test_push_and_sample(self) -> None:
        """Test pushing experiences and sampling."""
        buffer = ExperienceReplayBuffer(capacity=100)
        state = np.random.randn(260)
        action = 5
        reward = 1.0
        next_state = np.random.randn(260)
        done = False

        buffer.push(state, action, reward, next_state, done)
        assert len(buffer) == 1

        batch = buffer.sample(1)
        assert len(batch) == 1
        s, a, r, ns, d = batch[0]
        assert np.array_equal(s, state)
        assert a == action
        assert r == reward
        assert np.array_equal(ns, next_state)
        assert d == done

    def test_capacity_limit(self) -> None:
        """Test that buffer respects capacity limit."""
        buffer = ExperienceReplayBuffer(capacity=10)
        state = np.random.randn(260)
        next_state = np.random.randn(260)

        # Push more than capacity
        for i in range(15):
            buffer.push(state, i, 1.0, next_state, False)

        assert len(buffer) == 10

    def test_sample_less_than_capacity(self) -> None:
        """Test sampling when buffer has fewer items than requested."""
        buffer = ExperienceReplayBuffer(capacity=100)
        state = np.random.randn(260)
        next_state = np.random.randn(260)

        # Push 5 items
        for i in range(5):
            buffer.push(state, i, 1.0, next_state, False)

        # Try to sample 10
        batch = buffer.sample(10)
        assert len(batch) == 5  # Should return only what's available


class TestRLAgent:
    """Tests for RL agent."""

    def test_epsilon_greedy_action_selection(self) -> None:
        """Test epsilon-greedy action selection."""
        config = MLConfig()
        agent = RLAgent(config)
        features = generate_game_state_features(hand=generate_random_hand(5))
        valid_indices = [0, 1, 2, 3, 4]

        # In training mode with high epsilon, should sometimes explore
        agent.epsilon = 1.0
        action = agent.select_action(features, valid_indices, training=True)
        assert action in valid_indices

        # With epsilon = 0, should always be greedy
        agent.epsilon = 0.0
        action = agent.select_action(features, valid_indices, training=True)
        assert action in valid_indices

    def test_action_selection_masks_invalid(self) -> None:
        """Test that invalid actions are masked."""
        config = MLConfig()
        agent = RLAgent(config)
        features = generate_game_state_features(hand=generate_random_hand(5))
        valid_indices = [0, 1, 2]  # Only first 3 valid

        agent.epsilon = 0.0  # Greedy
        action = agent.select_action(features, valid_indices, training=False)
        assert action in valid_indices

    def test_update_epsilon(self) -> None:
        """Test epsilon decay."""
        config = MLConfig()
        agent = RLAgent(config)
        initial_epsilon = agent.epsilon

        agent.update_epsilon()
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_min

    def test_save_and_load(self) -> None:
        """Test model save and load."""
        config = MLConfig()
        agent = RLAgent(config)
        agent.epsilon = 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_rl_model.pth"
            agent.save(filepath)
            assert filepath.exists()

            # Create new agent and load
            new_agent = RLAgent(config)
            new_agent.load(filepath)

            # Verify epsilon is loaded
            assert new_agent.epsilon == 0.5

            # Verify networks are loaded
            assert (
                agent.q_network.state_dict()["network.0.weight"]
                == new_agent.q_network.state_dict()["network.0.weight"]
            ).all()

