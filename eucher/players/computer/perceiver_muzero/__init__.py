"""EucherPerceiverMuZero player implementation."""

from eucher.plugins import register_plugin


def _create_perceiver_muzero_player(game=None, **kwargs):
    """Factory for EuchrePerceiverMuZeroPlayer."""
    from eucher.players.computer.perceiver_muzero.player import EuchrePerceiverMuZeroPlayer
    from eucher.players.computer.perceiver_muzero.config import PerceiverMuZeroConfig

    config = PerceiverMuZeroConfig()
    model_path = kwargs.get("model_path", None)
    num_simulations = kwargs.get("num_simulations", None)
    risk_factor = kwargs.get("risk_factor", 0.0)
    fast_mode = kwargs.get("fast_mode", False)

    # Parse simulation count from profile type if provided
    profile_type = kwargs.get("profile_type", "")
    if profile_type and "_" in profile_type:
        parts = profile_type.split("_")
        if len(parts) >= 3:
            try:
                parsed_sims = int(parts[-1])
                if parsed_sims == 1:
                    fast_mode = True
                elif num_simulations is None:
                    num_simulations = parsed_sims
            except ValueError:
                pass

    if fast_mode or (num_simulations is not None and num_simulations == 1):
        return EuchrePerceiverMuZeroPlayer(
            model_path=model_path,
            config=config,
            game=game,
            fast_mode=True,
        )
    else:
        return EuchrePerceiverMuZeroPlayer(
            model_path=model_path,
            config=config,
            num_simulations=num_simulations,
            risk_factor=risk_factor,
            game=game,
            fast_mode=False,
        )


register_plugin(
    name="perceiver_muzero",
    factory=_create_perceiver_muzero_player,
    requires_game=True,
    description="EuchrePerceiverMuZero player using Perceiver architecture and MuZero-style MCTS",
)

