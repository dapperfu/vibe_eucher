"""Built-in player plugins for Euchre game."""

# Import all builtin plugins to trigger registration
# Each plugin module registers itself when imported
from eucher.plugins.builtin import ai, heuristic, ml, random  # noqa: F401

# Also import complex plugins
try:
    from eucher.plugins.builtin import euchre_zero, perceiver_muzero  # noqa: F401
except ImportError:
    # These may not be available if dependencies are missing
    pass

