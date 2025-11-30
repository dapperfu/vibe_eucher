"""Computer player implementations."""

from eucher.players.computer.ai import AIDecisionMaker, AIPlayer
from eucher.players.computer.base import ComputerPlayer
from eucher.players.computer.heuristic import HeuristicPlayer
from eucher.players.computer.random import RandomPlayer
from eucher.plugins import register_plugin

__all__ = ["ComputerPlayer", "HeuristicPlayer", "RandomPlayer", "AIDecisionMaker", "AIPlayer"]


def _create_random_player(game=None, **kwargs):
    """Factory for RandomPlayer."""
    return RandomPlayer()


def _create_heuristic_player(game=None, **kwargs):
    """Factory for HeuristicPlayer."""
    return HeuristicPlayer()


def _create_ai_player(game=None, **kwargs):
    """Factory for AIPlayer."""
    return AIPlayer(AIDecisionMaker())


# Register plugins
register_plugin(
    name="random",
    factory=_create_random_player,
    requires_game=False,
    description="Random player that makes all decisions randomly",
)

register_plugin(
    name="heuristic",
    factory=_create_heuristic_player,
    requires_game=False,
    description="Heuristic-based player using rule-based logic",
)

register_plugin(
    name="ai",
    factory=_create_ai_player,
    requires_game=False,
    description="AI player using weighted heuristics and strategic decision-making",
)

