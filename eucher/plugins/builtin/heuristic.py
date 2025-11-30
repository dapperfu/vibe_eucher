"""Heuristic player plugins."""

from typing import Optional

from eucher.players.computer.heuristic import HeuristicPlayer
from eucher.players.computer.heuristic2 import HeuristicPlayer2
from eucher.plugins.registry import register_plugin


def create_heuristic_player(game: Optional[object] = None, **kwargs: object) -> HeuristicPlayer:
    """
    Create a heuristic player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (not used by heuristic player).
    **kwargs : object
        Additional arguments (not used).

    Returns
    -------
    HeuristicPlayer
        A new heuristic player instance.
    """
    return HeuristicPlayer()


def create_heuristic2_player(game: Optional[object] = None, **kwargs: object) -> HeuristicPlayer2:
    """
    Create a heuristic2 player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (not used by heuristic2 player).
    **kwargs : object
        Additional arguments (not used).

    Returns
    -------
    HeuristicPlayer2
        A new heuristic2 player instance.
    """
    return HeuristicPlayer2()


# Register the plugins
register_plugin(
    name="heuristic",
    factory=create_heuristic_player,
    display_name="Heuristic Player",
    description="Rule-based player using heuristics",
    requires_game=False,
    supports_kwargs=False,
)

register_plugin(
    name="heuristic2",
    factory=create_heuristic2_player,
    display_name="Heuristic Player 2",
    description="Alternative rule-based player using heuristics",
    requires_game=False,
    supports_kwargs=False,
)

