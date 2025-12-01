"""EucherGo player plugin."""

from typing import Optional

from eucher.plugins.registry import register_plugin


def create_euchergo_player(
    game: Optional[object] = None,
    **kwargs: object,
) -> object:
    """
    Create an EucherGo player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (required for EucherGo).
    **kwargs : object
        Additional arguments (model_path, num_simulations, risk_factor, config).

    Returns
    -------
    EucherGoPlayer
        A new EucherGo player instance.
    """
    from eucher.players.computer.euchergo.config import EucherGoConfig
    from eucher.players.computer.euchergo.player import EucherGoPlayer

    config = kwargs.get("config")
    if config is None:
        config = EucherGoConfig()

    model_path = kwargs.get("model_path", None)
    num_simulations = kwargs.get("num_simulations", None)
    risk_factor = kwargs.get("risk_factor", 0.0)

    return EucherGoPlayer(
        model_path=model_path,
        config=config,
        num_simulations=num_simulations,
        risk_factor=risk_factor,
        game=game,
    )


# Register the plugin
register_plugin(
    name="euchergo",
    factory=create_euchergo_player,
    display_name="EucherGo Player",
    description="EucherGo player using AlphaZero/MuZero-style MCTS and neural networks",
    requires_game=True,
    supports_kwargs=True,
    model_name="EucherGoModel",
)

