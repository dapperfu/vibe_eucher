"""Computer player base classes.

All player implementations are now external plugins.
See plugins/ directory for all player implementations.
"""

from eucher.players.computer.base import ComputerPlayer

__all__ = ["ComputerPlayer"]

# Note: All player implementations are external plugins discovered via entry points.
# This module only provides the base ComputerPlayer class.

