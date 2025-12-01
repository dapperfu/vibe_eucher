"""Plugin registry for managing player plugins."""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from eucher.plugins.base import PluginMetadata

if TYPE_CHECKING:
    from eucher.players.base import PlayerProfile


class PluginRegistry:
    """Registry for player plugins.

    This class manages the registration and lookup of player plugins.
    Plugins can be registered manually or discovered automatically.
    """

    def __init__(self) -> None:
        """Initialize an empty plugin registry."""
        self._plugins: Dict[str, PluginMetadata] = {}

    def register(self, metadata: PluginMetadata) -> None:
        """
        Register a plugin with the registry.

        Parameters
        ----------
        metadata : PluginMetadata
            Plugin metadata to register.

        Raises
        ------
        ValueError
            If a plugin with the same name is already registered.
        """
        if metadata.name in self._plugins:
            raise ValueError(f"Plugin '{metadata.name}' is already registered")
        self._plugins[metadata.name] = metadata

    def get(self, name: str) -> Optional[PluginMetadata]:
        """
        Get plugin metadata by name.

        Parameters
        ----------
        name : str
            Plugin name to look up.

        Returns
        -------
        Optional[PluginMetadata]
            Plugin metadata if found, None otherwise.
        """
        return self._plugins.get(name)

    def has(self, name: str) -> bool:
        """
        Check if a plugin is registered.

        Parameters
        ----------
        name : str
            Plugin name to check.

        Returns
        -------
        bool
            True if plugin is registered, False otherwise.
        """
        return name in self._plugins

    def list_plugins(self) -> List[str]:
        """
        Get list of all registered plugin names.

        Returns
        -------
        List[str]
            List of plugin names.
        """
        return list(self._plugins.keys())

    def get_all_metadata(self) -> Dict[str, PluginMetadata]:
        """
        Get all registered plugin metadata.

        Returns
        -------
        Dict[str, PluginMetadata]
            Dictionary mapping plugin names to their metadata.
        """
        return self._plugins.copy()

    def clear(self) -> None:
        """Clear all registered plugins."""
        self._plugins.clear()


# Global plugin registry instance
_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """
    Get the global plugin registry instance.

    Returns
    -------
    PluginRegistry
        The global plugin registry.
    """
    return _registry


def register_plugin(
    name: str,
    factory: Callable[..., "PlayerProfile"],
    display_name: Optional[str] = None,
    description: str = "",
    requires_game: bool = False,
    supports_kwargs: bool = False,
    model_name: Optional[str] = None,
) -> None:
    """
    Register a plugin with the global registry.

    This is a convenience function for registering plugins.

    Parameters
    ----------
    name : str
        Plugin identifier.
    factory : Callable
        Factory function that creates a PlayerProfile instance.
    display_name : Optional[str]
        Human-readable name. If None, uses name.
    description : str
        Description of the plugin.
    requires_game : bool
        Whether the factory requires a game instance.
    supports_kwargs : bool
        Whether the plugin accepts additional kwargs.
    model_name : Optional[str]
        Name of the model class used by this plugin. None for non-ML bots.

    Raises
    ------
    ValueError
        If a plugin with the same name is already registered.
    """
    from eucher.plugins.base import PluginMetadata

    metadata = PluginMetadata(
        name=name,
        display_name=display_name or name,
        description=description,
        factory=factory,
        requires_game=requires_game,
        supports_kwargs=supports_kwargs,
        model_name=model_name,
    )
    _registry.register(metadata)

