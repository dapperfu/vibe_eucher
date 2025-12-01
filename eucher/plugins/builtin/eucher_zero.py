"""EucherZero player plugin."""

from typing import Optional

from eucher.plugins.registry import register_plugin


def create_eucher_zero_player(
    game: Optional[object] = None,
    **kwargs: object,
) -> object:
    """
    Create an EucherZero player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (required for EucherZero).
    **kwargs : object
        Additional arguments (config, etc.).

    Returns
    -------
    EucherZeroPlayer
        A new EucherZero player instance.

    Raises
    ------
    ValueError
        If game is None (EucherZero requires game instance).
    """
    if game is None:
        raise ValueError("EucherZero player requires game instance")

    from eucher.players.computer.eucher_zero.config import EucherZeroConfig
    from eucher.players.computer.eucher_zero.player import EucherZeroPlayer

    config = kwargs.get("config")
    if config is None:
        config = EucherZeroConfig()

    model_path = kwargs.get("model_path", None)
    num_simulations = kwargs.get("num_simulations", None)
    risk_factor = kwargs.get("risk_factor", 0.0)

    return EucherZeroPlayer(
        model_path=model_path,
        config=config,
        num_simulations=num_simulations,
        risk_factor=risk_factor,
        game=game,
    )


# Register the plugin
register_plugin(
    name="eucher_zero",
    factory=create_eucher_zero_player,
    display_name="EucherZero Player",
    description="EucherZero MCTS-based player",
    requires_game=True,
    supports_kwargs=True,
    model_name="EucherZeroModel",
)

