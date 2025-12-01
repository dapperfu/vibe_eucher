"""EucherGo player plugin - external plugin package."""

# This module is imported via entry point to register the plugin
from .plugin import _register_euchergo_plugin

# Register the plugin when this module is imported
_register_euchergo_plugin()
