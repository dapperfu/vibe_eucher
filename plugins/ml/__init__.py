"""ML-based player implementations."""

from eucher.plugins import register_plugin


def _create_ml_sklearn_player(game=None, **kwargs):
    """Factory for MLPlayer with supervised backend."""
    from .player import MLPlayer

    player_id = kwargs.get("player_id", 0)
    trump_selection_risk = kwargs.get("trump_selection_risk", None)
    gameplay_risk = kwargs.get("gameplay_risk", None)

    # Create game state provider if game is available
    game_state_provider = None
    if game is not None:

        def get_game_state():
            """Get current game state."""
            if hasattr(game, "current_hand") and game.current_hand is not None:
                trick_number = len(game.current_hand.tricks) if hasattr(game.current_hand, "tricks") else 0
                scores = game.get_scores() if hasattr(game, "get_scores") else (0, 0)
                return (trick_number, scores[0], scores[1])
            return (0, 0, 0)

        game_state_provider = get_game_state

    return MLPlayer(
        backend="supervised",
        model_type="random_forest",
        game_state_provider=game_state_provider,
        trump_selection_risk=trump_selection_risk,
        gameplay_risk=gameplay_risk,
    )


def _create_ml_rl_player(game=None, **kwargs):
    """Factory for MLPlayer with RL backend."""
    from .player import MLPlayer

    player_id = kwargs.get("player_id", 0)
    trump_selection_risk = kwargs.get("trump_selection_risk", None)
    gameplay_risk = kwargs.get("gameplay_risk", None)

    game_state_provider = None
    if game is not None:

        def get_game_state():
            """Get current game state."""
            if hasattr(game, "current_hand") and game.current_hand is not None:
                trick_number = len(game.current_hand.tricks) if hasattr(game.current_hand, "tricks") else 0
                scores = game.get_scores() if hasattr(game, "get_scores") else (0, 0)
                return (trick_number, scores[0], scores[1])
            return (0, 0, 0)

        game_state_provider = get_game_state

    return MLPlayer(
        backend="rl",
        game_state_provider=game_state_provider,
        trump_selection_risk=trump_selection_risk,
        gameplay_risk=gameplay_risk,
    )


def _create_ml_gan_player(game=None, **kwargs):
    """Factory for MLPlayer with GAN backend."""
    from eucher.players.computer.ml.player import MLPlayer

    player_id = kwargs.get("player_id", 0)
    trump_selection_risk = kwargs.get("trump_selection_risk", None)
    gameplay_risk = kwargs.get("gameplay_risk", None)

    game_state_provider = None
    if game is not None:

        def get_game_state():
            """Get current game state."""
            if hasattr(game, "current_hand") and game.current_hand is not None:
                trick_number = len(game.current_hand.tricks) if hasattr(game.current_hand, "tricks") else 0
                scores = game.get_scores() if hasattr(game, "get_scores") else (0, 0)
                return (trick_number, scores[0], scores[1])
            return (0, 0, 0)

        game_state_provider = get_game_state

    return MLPlayer(
        backend="gan",
        game_state_provider=game_state_provider,
        trump_selection_risk=trump_selection_risk,
        gameplay_risk=gameplay_risk,
    )


# Note: Plugin registration is handled by external plugins via entry points.
# This module only provides the factory functions for backward compatibility.


