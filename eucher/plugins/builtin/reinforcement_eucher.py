"""ReinforcementEucher player plugin."""

from typing import Optional

import torch

from eucher.plugins.registry import register_plugin


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
        Additional arguments (config, model, temperature, etc.).

    Returns
    -------
    ReinforcementEucherPlayer
        A new ReinforcementEucher player instance.

    Raises
    ------
    ValueError
        If game is None (ReinforcementEucher requires game instance).
    """
    if game is None:
        raise ValueError("ReinforcementEucher player requires game instance")

    from eucher.players.computer.reinforcement_eucher.config import ReinforcementEucherConfig
    from eucher.players.computer.reinforcement_eucher.networks.model import (
        ReinforcementEucherModel,
    )
    from eucher.players.computer.reinforcement_eucher.player import (
        ReinforcementEucherPlayer,
    )

    config = kwargs.get("config")
    if config is None:
        config = ReinforcementEucherConfig()

    model = kwargs.get("model")
    if model is None:
        model = ReinforcementEucherModel(config)

        # Try to load checkpoint if available
        checkpoint_dir = config.checkpoint_dir
        latest_checkpoint = checkpoint_dir / "latest_checkpoint.pt"
        if latest_checkpoint.exists() and latest_checkpoint.is_symlink():
            resolved = latest_checkpoint.resolve()
            if resolved.exists():
                checkpoint = torch.load(resolved, map_location=config.torch_device)
                model.load_state_dict(checkpoint["model_state_dict"])

    temperature = kwargs.get("temperature", 1.0)

    return ReinforcementEucherPlayer(model=model, game=game, temperature=temperature)


# Register the plugin
register_plugin(
    name="reinforcement_eucher",
    factory=create_reinforcement_eucher_player,
    display_name="ReinforcementEucher Player",
    description="ReinforcementEucher player using pure reinforcement learning",
    requires_game=True,
    supports_kwargs=True,
    model_name="ReinforcementEucherModel",
)

