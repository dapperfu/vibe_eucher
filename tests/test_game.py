"""Tests for game module."""

import pytest

from src.ai import AIDecisionMaker
from src.cards import Card, Rank, Suit
from src.game import Game
from src.players import AIPlayer


class TestGame:
    """Tests for Game class."""

    def test_game_creation(self) -> None:
        """Test game creation."""
        player_config = [
            ("Player 1", False),
            ("Player 2", False),
            ("Player 3", False),
            ("Player 4", False),
        ]
        game = Game(player_config)
        assert len(game.players) == 4
        assert game.dealer_id == 0

    def test_game_creation_wrong_player_count(self) -> None:
        """Test game creation with wrong number of players."""
        player_config = [("Player 1", False), ("Player 2", False)]
        with pytest.raises(ValueError):
            Game(player_config)

    def test_game_scores(self) -> None:
        """Test game scoring."""
        player_config = [
            ("Player 1", False),
            ("Player 2", False),
            ("Player 3", False),
            ("Player 4", False),
        ]
        game = Game(player_config)
        scores = game.get_scores()
        assert scores == (0, 0)

    def test_game_not_over_initially(self) -> None:
        """Test that game is not over initially."""
        player_config = [
            ("Player 1", False),
            ("Player 2", False),
            ("Player 3", False),
            ("Player 4", False),
        ]
        game = Game(player_config)
        assert game.get_winner() is None

    def test_set_ui(self) -> None:
        """Test setting UI."""
        from src.ui import TextUI

        player_config = [
            ("Player 1", True),
            ("Player 2", False),
            ("Player 3", False),
            ("Player 4", False),
        ]
        game = Game(player_config)
        ui = TextUI()
        game.set_ui(ui)
        # Check that human player has UI set
        assert isinstance(game.players[0], AIPlayer) or hasattr(game.players[0], "_ui")

