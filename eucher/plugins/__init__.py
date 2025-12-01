"""Plugin system for Euchre player bots.

All plugins are external and discovered via entry points in pyproject.toml.
"""

from eucher.plugins.discovery import discover_entry_points
from eucher.plugins.registry import get_registry, register_plugin

__all__ = [
    "get_registry",
    "register_plugin",
    "discover_entry_points",
]

# Discover plugins via entry points on import
discover_entry_points()

