"""Random player plugin."""

from .random import RandomPlayer
from eucher.plugins import register_plugin


def _create_random_player(game=None, **kwargs):
    """Factory for RandomPlayer."""
    return RandomPlayer()


register_plugin(
    name="random",
    factory=_create_random_player,
    requires_game=False,
    description="Random player that makes all decisions randomly",
)

__all__ = ["RandomPlayer"]

