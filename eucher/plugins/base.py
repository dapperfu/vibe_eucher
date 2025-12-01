"""Base classes and metadata structures for Euchre player plugins."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from eucher.game import Game
    from eucher.players.base import PlayerProfile


@dataclass
class PluginMetadata:
    """Metadata for a player plugin.

    Parameters
    ----------
    name : str
        Plugin identifier (e.g., "heuristic", "eucher_zero").
    display_name : str
        Human-readable name for the plugin.
    description : str
        Description of the plugin's behavior.
    factory : Callable
        Factory function that creates a PlayerProfile instance.
        Signature: factory(game: Optional[Game] = None, **kwargs) -> PlayerProfile
    requires_game : bool
        Whether the factory function requires a game instance.
    supports_kwargs : bool
        Whether the plugin accepts additional keyword arguments.
    model_name : Optional[str]
        Name of the model class used by this plugin (e.g., "EucherGoModel", "HybridNetwork").
        None for non-ML bots.
    """

    name: str
    display_name: str
    description: str
    factory: Callable[..., "PlayerProfile"]
    requires_game: bool = False
    supports_kwargs: bool = False
    model_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate plugin metadata after initialization."""
        if not self.name:
            raise ValueError("Plugin name cannot be empty")
        if not self.display_name:
            raise ValueError("Plugin display_name cannot be empty")
        if not callable(self.factory):
            raise ValueError("Plugin factory must be callable")


def create_player_factory(
    factory_func: Callable[..., "PlayerProfile"],
    requires_game: bool = False,
    supports_kwargs: bool = False,
) -> Callable[..., "PlayerProfile"]:
    """
    Create a standardized factory function from a plugin factory.

    Parameters
    ----------
    factory_func : Callable
        The factory function to wrap.
    requires_game : bool
        Whether the factory requires a game instance.
    supports_kwargs : bool
        Whether the factory accepts additional kwargs.

    Returns
    -------
    Callable
        Standardized factory function with signature:
        factory(game: Optional[Game] = None, **kwargs) -> PlayerProfile
    """
    def factory(game: Optional["Game"] = None, **kwargs: object) -> "PlayerProfile":
        """Standardized factory function."""
        if requires_game and game is None:
            raise ValueError(f"Plugin requires game instance but none provided")
        
        if supports_kwargs:
            return factory_func(game=game, **kwargs)
        elif requires_game:
            return factory_func(game=game)
        else:
            return factory_func(**kwargs) if kwargs else factory_func()
    
    return factory

