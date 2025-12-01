"""Plugin discovery mechanisms for finding and loading player plugins.

All plugins are external and discovered via entry points only.
"""

import sys
from typing import List

from eucher.plugins.base import PluginMetadata
from eucher.plugins.registry import get_registry


def discover_entry_points() -> List[PluginMetadata]:
    """
    Discover plugins from Python package entry points.

    Looks for entry points in the 'eucher.plugins' group.
    This is the ONLY way plugins are discovered - all plugins must be external.
    
    Entry points can point to:
    1. Factory functions (callable) - creates metadata automatically
    2. Modules (importable) - module registers itself via register_plugin()

    Returns
    -------
    List[PluginMetadata]
        List of discovered plugin metadata.
    """
    metadata_list: List[PluginMetadata] = []
    registry = get_registry()

    try:
        import importlib.metadata
        entry_points = importlib.metadata.entry_points(group="eucher.plugins")
    except (AttributeError, ImportError):
        # Fallback for Python < 3.8
        try:
            import pkg_resources
            entry_points = pkg_resources.iter_entry_points("eucher.plugins")
        except ImportError:
            # No entry point support available
            return metadata_list

    for entry_point in entry_points:
        try:
            # Track registry state before loading this entry point
            plugins_before = set(registry.list_plugins())
            
            loaded = entry_point.load()
            
            # Check if it's a callable (factory function) or a module
            if callable(loaded):
                # Factory function - create metadata from entry point
                metadata = PluginMetadata(
                    name=entry_point.name,
                    display_name=entry_point.name.replace("_", " ").title(),
                    description=f"Plugin from {entry_point.module}",
                    factory=loaded,
                    requires_game=False,  # Default, can be overridden by factory inspection
                    supports_kwargs=True,  # Default to True for flexibility
                )
                metadata_list.append(metadata)
                # Register it
                if not registry.has(metadata.name):
                    try:
                        registry.register(metadata)
                    except ValueError:
                        pass
            else:
                # Module - importing it triggers registration
                # The module should call register_plugin() during import
                # Check what plugins were registered by this module
                plugins_after = set(registry.list_plugins())
                new_plugins = plugins_after - plugins_before
                
                # Get metadata for newly registered plugins
                for plugin_name in new_plugins:
                    metadata = registry.get(plugin_name)
                    if metadata:
                        metadata_list.append(metadata)
                        
        except Exception as e:
            # Log error but continue discovering other plugins
            print(f"Warning: Failed to load entry point '{entry_point.name}': {e}", file=sys.stderr)
            continue

    return metadata_list
