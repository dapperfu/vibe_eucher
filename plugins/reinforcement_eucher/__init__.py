"""ReinforcementEucher player plugin - external plugin package."""

# This module is imported via entry point to register the plugin
from .plugin import _register_reinforcement_eucher_plugin

# Register the plugin when this module is imported
_register_reinforcement_eucher_plugin()

# Export main classes (after plugin registration)
from .config import ReinforcementEucherConfig  # noqa: E402
from .networks.model import ReinforcementEucherModel  # noqa: E402
from .player import ReinforcementEucherPlayer  # noqa: E402

__all__ = [
    "ReinforcementEucherConfig",
    "ReinforcementEucherModel",
    "ReinforcementEucherPlayer",
]



