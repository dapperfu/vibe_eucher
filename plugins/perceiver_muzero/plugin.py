"""PerceiverMuZero player plugin registration."""

from typing import Optional

from eucher.plugins.registry import register_plugin

from .config import PerceiverMuZeroConfig
from .player import EucherPerceiverMuZeroPlayer


def create_perceiver_muzero_player(
    game: Optional[object] = None,
    **kwargs: object,
) -> object:
    """
    Create a PerceiverMuZero player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (required for PerceiverMuZero).
    **kwargs : object
        Additional arguments (config, model_path, etc.).

    Returns
    -------
    EucherPerceiverMuZeroPlayer
        A new PerceiverMuZero player instance.
    """
    config = kwargs.get("config")
    if config is None:
        config = PerceiverMuZeroConfig()

    model_path = kwargs.get("model_path", None)
    num_simulations = kwargs.get("num_simulations", None)
    risk_factor = kwargs.get("risk_factor", 0.0)
    fast_mode = kwargs.get("fast_mode", False)

    return EucherPerceiverMuZeroPlayer(
        model_path=model_path,
        config=config,
        num_simulations=num_simulations,
        risk_factor=risk_factor,
        game=game,
        fast_mode=fast_mode,
    )


def _register_perceiver_muzero_plugin() -> None:
    """Register the PerceiverMuZero plugin."""
    register_plugin(
        name="perceiver_muzero",
        factory=create_perceiver_muzero_player,
        display_name="PerceiverMuZero Player",
        description="PerceiverMuZero MCTS-based player",
        requires_game=True,
        supports_kwargs=True,
        model_name="PerceiverMuZeroModel",
    )


