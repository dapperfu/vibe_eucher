"""Plugin system for Euchre player bots."""

from eucher.plugins.discovery import (
    discover_all_plugins,
    discover_builtin_plugins,
    load_builtin_plugins,
)
from eucher.plugins.registry import get_registry, register_plugin

__all__ = [
    "get_registry",
    "register_plugin",
    "discover_all_plugins",
    "discover_builtin_plugins",
    "load_builtin_plugins",
]

# Load builtin plugins on import
load_builtin_plugins()

# Discover external plugins
discover_all_plugins()

