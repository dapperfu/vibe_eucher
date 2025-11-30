"""PerceiverMuZero player plugin."""

from typing import Optional

from eucher.plugins.registry import register_plugin


def create_perceiver_muzero_player(
    game: Optional[object] = None,
    num_simulations: Optional[int] = None,
    fast_mode: Optional[bool] = None,
    **kwargs: object,
) -> object:
    """
    Create a PerceiverMuZero player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (required for PerceiverMuZero).
    num_simulations : Optional[int]
        Number of MCTS simulations. If None or 1, uses fast_mode=True.
    fast_mode : Optional[bool]
        Whether to use fast mode. If None, determined from num_simulations.
    **kwargs : object
        Additional arguments (config, etc.).

    Returns
    -------
    EuchrePerceiverMuZeroPlayer
        A new PerceiverMuZero player instance.

    Raises
    ------
    ValueError
        If game is None (PerceiverMuZero requires game instance).
    """
    if game is None:
        raise ValueError("PerceiverMuZero player requires game instance")

    from eucher.players.computer.perceiver_muzero.config import PerceiverMuZeroConfig
    from eucher.players.computer.perceiver_muzero.player import (
        EuchrePerceiverMuZeroPlayer,
    )

    config = kwargs.get("config")
    if config is None:
        config = PerceiverMuZeroConfig()

    # Determine fast_mode and num_simulations
    if fast_mode is None:
        if num_simulations is None or num_simulations == 1:
            fast_mode = True
            num_simulations = None
        else:
            fast_mode = False

    if fast_mode:
        return EuchrePerceiverMuZeroPlayer(config=config, game=game, fast_mode=True)
    else:
        return EuchrePerceiverMuZeroPlayer(
            config=config, game=game, num_simulations=num_simulations, fast_mode=False
        )


def create_perceiver_muzero_factory(num_simulations: Optional[int] = None) -> object:
    """
    Create a factory function for PerceiverMuZero with specific simulation count.

    Parameters
    ----------
    num_simulations : Optional[int]
        Number of simulations to use.

    Returns
    -------
    Callable
        Factory function for this specific configuration.
    """
    def factory(game: Optional[object] = None, **kwargs: object) -> object:
        return create_perceiver_muzero_player(
            game=game, num_simulations=num_simulations, **kwargs
        )

    return factory


# Register base plugin
register_plugin(
    name="perceiver_muzero",
    factory=create_perceiver_muzero_player,
    display_name="PerceiverMuZero Player",
    description="PerceiverMuZero MCTS-based player",
    requires_game=True,
    supports_kwargs=True,
)

# Register variants with specific simulation counts
for sim_count in [1, 16, 64, 128]:
    register_plugin(
        name=f"perceiver_muzero_{sim_count}",
        factory=create_perceiver_muzero_factory(num_simulations=sim_count),
        display_name=f"PerceiverMuZero ({sim_count} sims)",
        description=f"PerceiverMuZero player with {sim_count} simulations",
        requires_game=True,
        supports_kwargs=True,
    )

