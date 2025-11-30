"""Plugin registry system for automatic player profile discovery."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type

from eucher.players.base import PlayerProfile


@dataclass
class PluginMetadata:
    """Metadata for a player profile plugin.

    Parameters
    ----------
    name : str
        Plugin identifier (e.g., "random", "heuristic", "euchergo").
    factory : Callable
        Factory function that creates PlayerProfile instances.
        Signature: factory(game: Optional[Game] = None, **kwargs) -> PlayerProfile
    requires_game : bool
        Whether the factory needs a game instance.
    description : str
        Human-readable description of the plugin.
    """

    name: str
    factory: Callable
    requires_game: bool
    description: str


class PluginRegistry:
    """Registry for player profile plugins with auto-discovery support."""

    def __init__(self) -> None:
        """Initialize the plugin registry."""
        self._plugins: Dict[str, PluginMetadata] = {}
        self._discovered: bool = False

    def register(
        self,
        name: str,
        factory: Callable,
        requires_game: bool = False,
        description: str = "",
    ) -> None:
        """
        Register a player profile plugin.

        Parameters
        ----------
        name : str
            Plugin identifier.
        factory : Callable
            Factory function that creates PlayerProfile instances.
        requires_game : bool
            Whether factory needs game instance.
        description : str
            Human-readable description.
        """
        if name in self._plugins:
            raise ValueError(f"Plugin '{name}' is already registered")
        self._plugins[name] = PluginMetadata(
            name=name, factory=factory, requires_game=requires_game, description=description
        )

    def get(self, name: str) -> Optional[PluginMetadata]:
        """
        Retrieve plugin metadata by name.

        Parameters
        ----------
        name : str
            Plugin identifier.

        Returns
        -------
        Optional[PluginMetadata]
            Plugin metadata if found, None otherwise.
        """
        if not self._discovered:
            self.discover_plugins()
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        """
        List all registered plugin names.

        Returns
        -------
        List[str]
            List of plugin names.
        """
        if not self._discovered:
            self.discover_plugins()
        return sorted(self._plugins.keys())

    def discover_plugins(self) -> None:
        """
        Auto-discover plugins by scanning player modules.

        This method scans eucher/players/computer/*/ directories and
        attempts to import and register plugins. Plugins can register
        themselves by calling register_plugin() in their __init__.py.
        """
        if self._discovered:
            return

        # Import player modules to trigger their registration
        try:
            from eucher.players.computer import random  # noqa: F401
        except ImportError:
            pass

        try:
            from eucher.players.computer import heuristic  # noqa: F401
        except ImportError:
            pass

        try:
            from eucher.players.computer import ai  # noqa: F401
        except ImportError:
            pass

        try:
            from eucher.players.computer.ml import player as ml_player  # noqa: F401
        except ImportError:
            pass

        try:
            from eucher.players.computer.ml.pytorch import pytorch_player  # noqa: F401
        except ImportError:
            pass

        try:
            from eucher.players.computer.euchre_zero import player as euchre_zero_player  # noqa: F401
        except ImportError:
            pass

        try:
            from eucher.players.computer.perceiver_muzero import player as perceiver_muzero_player  # noqa: F401
        except ImportError:
            pass

        try:
            from eucher.players.computer import euchergo  # noqa: F401
        except ImportError:
            pass

        self._discovered = True

    def get_all_metadata(self) -> Dict[str, PluginMetadata]:
        """
        Get all registered plugin metadata.

        Returns
        -------
        Dict[str, PluginMetadata]
            Dictionary mapping plugin names to metadata.
        """
        if not self._discovered:
            self.discover_plugins()
        return self._plugins.copy()


# Global registry instance
_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """
    Get the global plugin registry instance.

    Returns
    -------
    PluginRegistry
        The global plugin registry.
    """
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def register_plugin(
    name: str,
    factory: Callable,
    requires_game: bool = False,
    description: str = "",
) -> None:
    """
    Register a plugin with the global registry.

    Convenience function for plugin modules to register themselves.

    Parameters
    ----------
    name : str
        Plugin identifier.
    factory : Callable
        Factory function that creates PlayerProfile instances.
    requires_game : bool
        Whether factory needs game instance.
    description : str
        Human-readable description.
    """
    get_registry().register(name, factory, requires_game, description)

