"""Tests for game module."""

import pytest

from eucher.cards import Card, Rank, Suit
from eucher.game import Game


class TestGame:
    """Tests for Game class."""

    def test_game_creation(self) -> None:
        """Test game creation."""
        player_config = [
            ("Player 1", "weighted_heuristic"),
            ("Player 2", "weighted_heuristic"),
            ("Player 3", "weighted_heuristic"),
            ("Player 4", "weighted_heuristic"),
        ]
        game = Game(player_config)
        assert len(game.players) == 4
        assert game.dealer_id == 0

    def test_game_creation_wrong_player_count(self) -> None:
        """Test game creation with wrong number of players."""
        player_config = [("Player 1", "weighted_heuristic"), ("Player 2", "weighted_heuristic")]
        with pytest.raises(ValueError):
            Game(player_config)

    def test_game_scores(self) -> None:
        """Test game scoring."""
        player_config = [
            ("Player 1", "weighted_heuristic"),
            ("Player 2", "weighted_heuristic"),
            ("Player 3", "weighted_heuristic"),
            ("Player 4", "weighted_heuristic"),
        ]
        game = Game(player_config)
        scores = game.get_scores()
        assert scores == (0, 0)

    def test_game_not_over_initially(self) -> None:
        """Test that game is not over initially."""
        player_config = [
            ("Player 1", "weighted_heuristic"),
            ("Player 2", "weighted_heuristic"),
            ("Player 3", "weighted_heuristic"),
            ("Player 4", "weighted_heuristic"),
        ]
        game = Game(player_config)
        assert game.get_winner() is None

    def test_set_tui(self) -> None:
        """Test setting TUI."""
        from eucher.tui import TextTUI

        player_config = [
            ("Player 1", "human"),
            ("Player 2", "ai"),
            ("Player 3", "ai"),
            ("Player 4", "ai"),
        ]
        game = Game(player_config)
        tui = TextTUI()
        game.set_tui(tui)
        # Check that human profile has TUI set
        from eucher.players.profiles import HumanProfile
        assert isinstance(game.players[0].profile, HumanProfile)
        assert game.players[0].profile.tui == tui

    def test_invalid_profile_type(self) -> None:
        """Test invalid profile type."""
        player_config = [
            ("Player 1", "invalid"),
            ("Player 2", "ai"),
            ("Player 3", "ai"),
            ("Player 4", "ai"),
        ]
        with pytest.raises(ValueError):
            Game(player_config)
