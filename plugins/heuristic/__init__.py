"""Heuristic player plugin."""

from .heuristic import HeuristicPlayer
from eucher.plugins import register_plugin


def _create_heuristic_player(game=None, **kwargs):
    """Factory for HeuristicPlayer."""
    return HeuristicPlayer()


register_plugin(
    name="heuristic",
    factory=_create_heuristic_player,
    requires_game=False,
    description="Heuristic-based player using rule-based logic",
)

__all__ = ["HeuristicPlayer"]


