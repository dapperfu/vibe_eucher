"""ReinforcementEucher player plugin registration."""

from typing import Optional

from eucher.plugins.registry import register_plugin

from .config import ReinforcementEucherConfig
from .networks.model import ReinforcementEucherModel
from .player import ReinforcementEucherPlayer


def create_reinforcement_eucher_player(
    game: Optional[object] = None,
    **kwargs: object,
) -> object:
    """
    Create a ReinforcementEucher player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (required for ReinforcementEucher).
    **kwargs : object
        Additional arguments (config, model_path, etc.).

    Returns
    -------
    ReinforcementEucherPlayer
        A new ReinforcementEucher player instance.
    """
    config = kwargs.get("config")
    if config is None:
        config = ReinforcementEucherConfig()

    model_path = kwargs.get("model_path", None)
    temperature = kwargs.get("temperature", 1.0)

    # Load or create model
    if model_path is not None:
        model = ReinforcementEucherModel(config)
        model.load_checkpoint(model_path)
    else:
        model = ReinforcementEucherModel(config)

    return ReinforcementEucherPlayer(
        model=model,
        game=game,
        temperature=temperature,
    )


def _register_reinforcement_eucher_plugin() -> None:
    """Register the ReinforcementEucher plugin."""
    register_plugin(
        name="reinforcement_eucher",
        factory=create_reinforcement_eucher_player,
        display_name="ReinforcementEucher Player",
        description="Reinforcement learning-based Euchre player",
        requires_game=True,
        supports_kwargs=True,
        model_name="ReinforcementEucherModel",
    )


