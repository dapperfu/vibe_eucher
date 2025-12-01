"""Computer player implementations."""

from eucher.players.computer.ai import AIDecisionMaker, AIPlayer
from eucher.players.computer.base import ComputerPlayer
from eucher.players.computer.heuristic import HeuristicPlayer
from eucher.players.computer.random import RandomPlayer

__all__ = ["ComputerPlayer", "HeuristicPlayer", "RandomPlayer", "AIDecisionMaker", "AIPlayer"]

# Note: Plugin registration is handled by eucher.plugins.builtin modules.
# This module only provides the implementation classes.

