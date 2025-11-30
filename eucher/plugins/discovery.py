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

    Returns
    -------
    List[PluginMetadata]
        List of discovered plugin metadata.
    """
    metadata_list: List[PluginMetadata] = []

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
            factory = entry_point.load()
            # Create metadata from entry point
            # Entry point name is the plugin name
            metadata = PluginMetadata(
                name=entry_point.name,
                display_name=entry_point.name.replace("_", " ").title(),
                description=f"Plugin from {entry_point.module}",
                factory=factory,
                requires_game=False,  # Default, can be overridden by factory inspection
                supports_kwargs=True,  # Default to True for flexibility
            )
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
    all_metadata.extend(entry_point_metadata)

    # Register entry point plugins
    registry = get_registry()
    for metadata in entry_point_metadata:
        if not registry.has(metadata.name):
            try:
                registry.register(metadata)
            except ValueError:
                # Already registered, skip
                pass

    # Discover from directory
    directory_metadata = discover_directory_plugins(plugins_dir)
    # Directory plugins are already registered during discovery
    all_metadata.extend(directory_metadata)

    return all_metadata


def load_builtin_plugins() -> None:
    """
    Load built-in plugins from eucher.plugins.builtin.

    This should be called during module initialization to ensure
    built-in plugins are always available.
    
    Note: Builtin plugins register themselves when their modules
    are imported, so this function just ensures the import happens.
    """
    try:
        # Import builtin package - this triggers registration of all builtin plugins
        import eucher.plugins.builtin  # noqa: F401
    except ImportError:
        # Builtin plugins not available
        pass

