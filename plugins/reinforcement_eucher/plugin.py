"""ReinforcementEucher player plugin registration."""

from typing import Optional

import torch

from eucher.personality_training import assert_checkpoint_adequate, resolve_checkpoint_path
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
    require_trained = not bool(kwargs.get("allow_untrained", False))
    if not isinstance(model_path, str) and model_path is not None:
        model_path = str(model_path)

    model = _load_reinforcement_model(
        config,
        model_path if isinstance(model_path, str) else None,
        require_trained,
    )

    return ReinforcementEucherPlayer(
        model=model,
        game=game,
        temperature=temperature,
    )


def _load_reinforcement_model(
    config: ReinforcementEucherConfig,
    model_path: Optional[str],
    require_trained: bool,
) -> ReinforcementEucherModel:
    """Load a trained ReinforcementEucher network or build a fresh one.

    Parameters
    ----------
    config : ReinforcementEucherConfig
        Model configuration.
    model_path : Optional[str]
        Explicit checkpoint path.
    require_trained : bool
        When True, a checkpoint with completed training is required.

    Returns
    -------
    ReinforcementEucherModel
        Network for play or training.

    Raises
    ------
    UntrainedPersonalityError
        When a trained checkpoint is required and none is adequate.
    """
    checkpoint = resolve_checkpoint_path(
        "reinforcement_eucher",
        model_path,
        config.checkpoint_dir,
        require_trained,
    )
    if checkpoint is None:
        return ReinforcementEucherModel(config)

    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        payload = {}
    if require_trained:
        assert_checkpoint_adequate("reinforcement_eucher", payload, checkpoint)
    model = ReinforcementEucherModel.load_checkpoint(str(checkpoint), config=config)
    return model


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


