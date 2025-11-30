"""PyTorch-based AI player implementation."""

from eucher.players.computer.ml.pytorch.pytorch_player import PyTorchStrategicPlayer
from eucher.plugins import register_plugin

__all__ = ["PyTorchStrategicPlayer"]


def _create_pytorch_player(game=None, **kwargs):
    """Factory for PyTorchStrategicPlayer."""
    model_path = kwargs.get("model_path", None)
    device = kwargs.get("device", None)
    exploration_epsilon = kwargs.get("exploration_epsilon", 0.0)
    return PyTorchStrategicPlayer(
        model_path=model_path, device=device, exploration_epsilon=exploration_epsilon
    )


register_plugin(
    name="pytorch_ai",
    factory=_create_pytorch_player,
    requires_game=False,
    description="PyTorch-based strategic AI player with deep neural networks",
)

register_plugin(
    name="pytorch_strategic",
    factory=_create_pytorch_player,
    requires_game=False,
    description="PyTorch-based strategic AI player (alias for pytorch_ai)",
)

register_plugin(
    name="ml_pytorch",
    factory=_create_pytorch_player,
    requires_game=False,
    description="PyTorch-based ML player (alias for pytorch_ai)",
)

