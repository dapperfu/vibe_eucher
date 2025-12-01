"""Transformer RL player plugin."""

from pathlib import Path
from typing import Optional

from eucher.plugins.registry import register_plugin


def create_transformer_rl_player(
    game: Optional[object] = None,
    **kwargs: object,
) -> object:
    """
    Create a Transformer RL player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (not required for Transformer RL).
    **kwargs : object
        Additional arguments (model_path, device, risk_factor, use_deduction, etc.).

    Returns
    -------
    TransformerRLPlayer
        A new Transformer RL player instance.
    """
    from src.ai_players.transformer_rl.transformer_rl_player import TransformerRLPlayer

    model_path = kwargs.get("model_path")
    device = kwargs.get("device")
    risk_factor = kwargs.get("risk_factor", 0.5)
    use_deduction = kwargs.get("use_deduction", True)

    return TransformerRLPlayer(
        model_path=model_path,
        device=device,
        risk_factor=risk_factor,
        use_deduction=use_deduction,
    )


# Register the plugin
register_plugin(
    name="transformer_rl",
    factory=create_transformer_rl_player,
    display_name="Transformer RL Player",
    description="Transformer RL player using actor-critic agent with perfect memory and deduction",
    requires_game=False,
    supports_kwargs=True,
    model_name="ActorCriticAgent",
)

