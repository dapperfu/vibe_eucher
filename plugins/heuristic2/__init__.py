"""Heuristic2 player plugin with tunable risk factor."""

from .heuristic2 import HeuristicPlayer2
from eucher.plugins import register_plugin


def _create_heuristic2_player(game=None, **kwargs):
    """Factory for HeuristicPlayer2."""
    risk_factor = kwargs.get("risk_factor", 0.5)
    return HeuristicPlayer2(risk_factor=risk_factor)


register_plugin(
    name="heuristic2",
    factory=_create_heuristic2_player,
    requires_game=False,
    description="Heuristic-based player with tunable risk factor",
)

__all__ = ["HeuristicPlayer2"]


