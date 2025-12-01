"""Plugin discovery mechanisms for finding and loading player plugins."""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional

from eucher.plugins.base import PluginMetadata
from eucher.plugins.registry import get_registry, register_plugin


def discover_entry_points() -> List[PluginMetadata]:
    """
    Discover plugins from Python package entry points.

    Looks for entry points in the 'eucher.plugins' group.
    
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


def discover_directory_plugins(plugins_dir: Optional[Path] = None) -> List[PluginMetadata]:
    """
    Discover plugins from a directory.

    Scans a directory for Python modules and attempts to load them as plugins.
    Plugins should call register_plugin() during module import.

    Parameters
    ----------
    plugins_dir : Optional[Path]
        Directory to scan for plugins. If None, uses 'plugins/' in project root.

    Returns
    -------
    List[PluginMetadata]
        List of discovered plugin metadata.
    """
    if plugins_dir is None:
        # Find project root (where setup.py or pyproject.toml is)
        current = Path(__file__).resolve()
        # Go up from eucher/plugins/discovery.py to project root
        project_root = current.parent.parent.parent
        plugins_dir = project_root / "plugins"

    if not plugins_dir.exists() or not plugins_dir.is_dir():
        return []

    metadata_list: List[PluginMetadata] = []
    registry_before = set(get_registry().list_plugins())

    # Scan for Python files
    for plugin_file in plugins_dir.glob("*.py"):
        if plugin_file.name.startswith("_"):
            continue  # Skip private modules

        module_name = plugin_file.stem
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                f"plugins.{module_name}", plugin_file
            )
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"plugins.{module_name}"] = module
            spec.loader.exec_module(module)

            # Check what plugins were registered by this module
            registry_after = set(get_registry().list_plugins())
            new_plugins = registry_after - registry_before
            registry_before = registry_after

            # Get metadata for newly registered plugins
            for plugin_name in new_plugins:
                metadata = get_registry().get(plugin_name)
                if metadata:
                    metadata_list.append(metadata)

        except Exception as e:
            # Log error but continue discovering other plugins
            print(
                f"Warning: Failed to load plugin from '{plugin_file}': {e}",
                file=sys.stderr,
            )
            continue

    return metadata_list


def discover_all_plugins(plugins_dir: Optional[Path] = None) -> List[PluginMetadata]:
    """
    Discover all available plugins from all sources.

    Discovers plugins from:
    1. Python package entry points (eucher.plugins group)
    2. Directory-based plugins (plugins/ directory)

    Parameters
    ----------
    plugins_dir : Optional[Path]
        Directory to scan for plugins. If None, uses 'plugins/' in project root.

    Returns
    -------
    List[PluginMetadata]
        List of all discovered plugin metadata.
    """
    all_metadata: List[PluginMetadata] = []

    # Discover from entry points
    entry_point_metadata = discover_entry_points()
    
    # Register entry point plugins (if they're factory functions)
    # Modules that self-register are already registered during discovery
    registry = get_registry()
    for metadata in entry_point_metadata:
        if not registry.has(metadata.name):
            try:
                registry.register(metadata)
            except ValueError:
                # Already registered, skip
                pass
    
    all_metadata.extend(entry_point_metadata)

    # Discover from directory
    directory_metadata = discover_directory_plugins(plugins_dir)
    # Directory plugins are already registered during discovery
    all_metadata.extend(directory_metadata)

    return all_metadata


def discover_builtin_plugins() -> List[PluginMetadata]:
    """
    Auto-discover built-in plugins by scanning eucher.plugins.builtin directory.

    Scans the builtin plugins directory and imports modules to trigger
    their self-registration.

    Returns
    -------
    List[PluginMetadata]
        List of discovered builtin plugin metadata.
    """
    metadata_list: List[PluginMetadata] = []
    
    try:
        # Find the builtin plugins directory
        current = Path(__file__).resolve()
        builtin_dir = current.parent / "builtin"
        
        if not builtin_dir.exists() or not builtin_dir.is_dir():
            return metadata_list
        
        registry_before = set(get_registry().list_plugins())
        
        # Scan for Python files (excluding __init__.py)
        for plugin_file in sorted(builtin_dir.glob("*.py")):
            if plugin_file.name.startswith("_") or plugin_file.name == "__init__.py":
                continue  # Skip private modules and __init__.py
            
            module_name = plugin_file.stem
            try:
                # Import the module - this triggers plugin registration
                module_path = f"eucher.plugins.builtin.{module_name}"
                importlib.import_module(module_path)
                
                # Check what plugins were registered by this module
                registry_after = set(get_registry().list_plugins())
                new_plugins = registry_after - registry_before
                registry_before = registry_after
                
                # Get metadata for newly registered plugins
                for plugin_name in new_plugins:
                    metadata = get_registry().get(plugin_name)
                    if metadata:
                        metadata_list.append(metadata)
                        
            except Exception as e:
                # Log error but continue discovering other plugins
                print(
                    f"Warning: Failed to load builtin plugin '{module_name}': {e}",
                    file=sys.stderr,
                )
                continue
                
    except Exception as e:
        # Log error but don't fail completely
        print(
            f"Warning: Failed to discover builtin plugins: {e}",
            file=sys.stderr,
        )
    
    return metadata_list


def load_builtin_plugins() -> None:
    """
    Load built-in plugins from eucher.plugins.builtin.

    This should be called during module initialization to ensure
    built-in plugins are always available.
    
    Plugins are discovered via entry points in pyproject.toml.
    Falls back to directory scanning if entry points are not available.
    """
    # Try entry points first (idiomatic Python way)
    entry_point_metadata = discover_entry_points()
    
    # If no plugins found via entry points, fall back to directory scanning
    if not entry_point_metadata:
        discover_builtin_plugins()

