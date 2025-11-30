"""EuchreZero player plugin."""

from typing import Optional

from eucher.plugins.registry import register_plugin


def create_euchre_zero_player(
    game: Optional[object] = None,
    **kwargs: object,
) -> object:
    """
    Create an EuchreZero player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (required for EuchreZero).
    **kwargs : object
        Additional arguments (config, etc.).

    Returns
    -------
    EuchreZeroPlayer
        A new EuchreZero player instance.

    Raises
    ------
    ValueError
        If game is None (EuchreZero requires game instance).
    """
    if game is None:
        raise ValueError("EuchreZero player requires game instance")

    from eucher.players.computer.euchre_zero.config import EuchreZeroConfig
    from eucher.players.computer.euchre_zero.player import EuchreZeroPlayer

    config = kwargs.get("config")
    if config is None:
        config = EuchreZeroConfig()

    return EuchreZeroPlayer(config=config, game=game)


# Register the plugin
register_plugin(
    name="euchre_zero",
    factory=create_euchre_zero_player,
    display_name="EuchreZero Player",
    description="EuchreZero MCTS-based player",
    requires_game=True,
    supports_kwargs=True,
)

