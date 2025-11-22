"""Base class for all computer players in Euchre.

This module re-exports ComputerPlayer from player_profiles to maintain backward compatibility.
"""

# Re-export ComputerPlayer from player_profiles to avoid breaking existing imports
from src.player_profiles import ComputerPlayer  # noqa: F401

