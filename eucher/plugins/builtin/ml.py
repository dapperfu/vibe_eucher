"""ML player plugins."""

from typing import Optional

from eucher.players.computer.ml.ml_config import MLConfig
from eucher.players.computer.ml.ml_features import GameStateEncoder
from eucher.players.computer.ml.ml_model import EuchreMLModel
from eucher.players.computer.ml.player import MLPlayer
from eucher.players.profiles import MLBasedProfile
from eucher.plugins.registry import register_plugin


def create_ml_sklearn_player(
    game: Optional[object] = None,
    player_id: int = 0,
    backend: str = "supervised",
    **kwargs: object,
) -> MLPlayer:
    """
    Create an ML sklearn player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (used for game state provider).
    player_id : int
        Player ID for game state access.
    backend : str
        ML backend: "supervised", "gan", or "rl".
    **kwargs : object
        Additional arguments (trump_selection_risk, gameplay_risk).

    Returns
    -------
    MLPlayer
        A new ML sklearn player instance.
    """
    # Create game state provider function
    def get_game_state() -> tuple[int, int, int]:
        """Get current game state for ML player."""
        if game is not None:
            trick_num = getattr(game, "_current_trick_number", 0)
            tricks_won = getattr(game, "_current_tricks_won", [0, 0])
            return (trick_num, tricks_won[0], tricks_won[1])
        return (0, 0, 0)

    trump_selection_risk = kwargs.get("trump_selection_risk", None)
    gameplay_risk = kwargs.get("gameplay_risk", None)

    return MLPlayer(
        backend=backend,
        model_type="random_forest",
        game_state_provider=get_game_state,
        trump_selection_risk=trump_selection_risk,
        gameplay_risk=gameplay_risk,
    )


def create_ml_pytorch_player(
    game: Optional[object] = None,
    player_id: int = 0,
    **kwargs: object,
) -> MLBasedProfile:
    """
    Create an ML PyTorch player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (used for game state provider).
    player_id : int
        Player ID for game state access.
    **kwargs : object
        Additional arguments (trump_selection_risk, gameplay_risk).

    Returns
    -------
    MLBasedProfile
        A new ML PyTorch player instance.
    """
    # Lazy initialization of ML components
    config = MLConfig()
    ml_model = EuchreMLModel(config)
    ml_encoder = GameStateEncoder()

    # Try to load existing weights
    trump_path = config.get_model_path("order_up.pth")
    card_path = config.get_model_path("play_card.pth")
    discard_path = config.get_model_path("discard.pth")

    if trump_path.exists() or card_path.exists() or discard_path.exists():
        ml_model.load_weights(
            trump_path=str(trump_path) if trump_path.exists() else None,
            card_play_path=str(card_path) if card_path.exists() else None,
            discard_path=str(discard_path) if discard_path.exists() else None,
        )

    # Create game state provider function
    def get_game_state() -> tuple[int, int, int]:
        """Get current game state for ML profile."""
        if game is not None:
            trick_num = getattr(game, "_current_trick_number", 0)
            tricks_won = getattr(game, "_current_tricks_won", [0, 0])
            return (trick_num, tricks_won[0], tricks_won[1])
        return (0, 0, 0)

    trump_selection_risk = kwargs.get("trump_selection_risk", None)
    gameplay_risk = kwargs.get("gameplay_risk", None)

    return MLBasedProfile(
        ml_model,
        ml_encoder,
        get_game_state,
        trump_selection_risk=trump_selection_risk,
        gameplay_risk=gameplay_risk,
    )


def create_pytorch_strategic_player(
    game: Optional[object] = None,
    **kwargs: object,
) -> object:
    """
    Create a PyTorch strategic player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (not used).
    **kwargs : object
        Additional arguments (model_path, device, use_strategic_overrides, etc.).

    Returns
    -------
    PyTorchStrategicPlayer
        A new PyTorch strategic player instance.
    """
    from eucher.ai_players.pytorch_player import PyTorchStrategicPlayer

    return PyTorchStrategicPlayer(
        model_path=kwargs.get("model_path"),
        device=kwargs.get("device"),
        use_strategic_overrides=kwargs.get("use_strategic_overrides", True),
        trump_selection_risk=kwargs.get("trump_selection_risk"),
        gameplay_risk=kwargs.get("gameplay_risk"),
    )


# Register the plugins
register_plugin(
    name="ml_sklearn",
    factory=create_ml_sklearn_player,
    display_name="ML Sklearn Player",
    description="ML-based player using sklearn models",
    requires_game=True,
    supports_kwargs=True,
)

register_plugin(
    name="ml_rl",
    factory=lambda game=None, player_id=0, **kwargs: create_ml_sklearn_player(
        game, player_id, backend="rl", **kwargs
    ),
    display_name="ML RL Player",
    description="ML-based player using reinforcement learning",
    requires_game=True,
    supports_kwargs=True,
)

register_plugin(
    name="ml_gan",
    factory=lambda game=None, player_id=0, **kwargs: create_ml_sklearn_player(
        game, player_id, backend="gan", **kwargs
    ),
    display_name="ML GAN Player",
    description="ML-based player using GAN models",
    requires_game=True,
    supports_kwargs=True,
)

register_plugin(
    name="ml_pytorch",
    factory=create_ml_pytorch_player,
    display_name="ML PyTorch Player",
    description="ML-based player using PyTorch models",
    requires_game=True,
    supports_kwargs=True,
)

register_plugin(
    name="pytorch_ai",
    factory=create_pytorch_strategic_player,
    display_name="PyTorch Strategic AI",
    description="PyTorch-based strategic AI player",
    requires_game=False,
    supports_kwargs=True,
)

register_plugin(
    name="pytorch_strategic",
    factory=create_pytorch_strategic_player,
    display_name="PyTorch Strategic Player",
    description="PyTorch-based strategic player",
    requires_game=False,
    supports_kwargs=True,
)

