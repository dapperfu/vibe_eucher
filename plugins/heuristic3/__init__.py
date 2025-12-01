"""Heuristic3 advanced player plugin."""

from .heuristic3 import HeuristicPlayer3
from eucher.plugins import register_plugin


def _create_heuristic3_player(game=None, **kwargs):
    """Factory for HeuristicPlayer3."""
    return HeuristicPlayer3()


register_plugin(
    name="heuristic3",
    factory=_create_heuristic3_player,
    requires_game=False,
    description="Advanced heuristic-based player with card counting and opponent analysis",
)

__all__ = ["HeuristicPlayer3"]


