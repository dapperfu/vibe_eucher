"""EucherGo player plugin registration."""

from typing import Optional

from eucher.plugins.registry import register_plugin

from .config import EucherGoConfig
from .player import EucherGoPlayer


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
        Additional arguments (config, model_path, etc.).

    Returns
    -------
    EucherGoPlayer
        A new EucherGo player instance.
    """
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


def _register_euchergo_plugin() -> None:
    """Register the EucherGo plugin."""
    register_plugin(
        name="euchergo",
        factory=create_euchergo_player,
        display_name="EucherGo Player",
        description="EucherGo MCTS-based player",
        requires_game=True,
        supports_kwargs=True,
        model_name="EucherGoModel",
    )


