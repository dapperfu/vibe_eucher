"""Random player plugin."""

from typing import Optional

from eucher.players.computer.random import RandomPlayer
from eucher.plugins.registry import register_plugin


def create_random_player(game: Optional[object] = None, **kwargs: object) -> RandomPlayer:
    """
    Create a random player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (not used by random player).
    **kwargs : object
        Additional arguments (not used).

    Returns
    -------
    RandomPlayer
        A new random player instance.
    """
    return RandomPlayer()


# Register the plugin
register_plugin(
    name="random",
    factory=create_random_player,
    display_name="Random Player",
    description="Makes all decisions randomly",
    requires_game=False,
    supports_kwargs=False,
    model_name=None,
)

