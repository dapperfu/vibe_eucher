"""EucherGo player implementation."""

from eucher.plugins import register_plugin


def _create_euchergo_player(game=None, **kwargs):
    """Factory for EucherGoPlayer."""
    from eucher.players.computer.euchergo.player import EucherGoPlayer
    from eucher.players.computer.euchergo.config import EucherGoConfig

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


register_plugin(
    name="euchergo",
    factory=_create_euchergo_player,
    requires_game=True,
    description="EucherGo player using AlphaZero/MuZero-style MCTS and neural networks",
)

