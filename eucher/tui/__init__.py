"""TUI modules for Euchre."""

from eucher.tui import TextTUI

__all__ = ["TextTUI", "NetworkTUI"]

try:
    from eucher.tui.network_tui import NetworkTUI
except ImportError:
    # NetworkTUI may not be available if network dependencies are missing
    NetworkTUI = None  # type: ignore

