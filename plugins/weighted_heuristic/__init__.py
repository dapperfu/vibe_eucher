"""AI player plugin."""

from .ai import AIDecisionMaker, AIPlayer
from eucher.plugins import register_plugin


def _create_weighted_heuristic_player(game=None, **kwargs):
    """Factory for AIPlayer using weighted heuristics."""
    ai_decision_maker = AIDecisionMaker()
    return AIPlayer(ai_decision_maker)


# Primary registration with descriptive name
register_plugin(
    name="weighted_heuristic",
    factory=_create_weighted_heuristic_player,
    display_name="Weighted Heuristic AI",
    requires_game=False,
    description="AI-based player using weighted heuristics",
)

# Backwards compatibility alias
register_plugin(
    name="ai",
    factory=_create_weighted_heuristic_player,
    display_name="Weighted Heuristic AI",
    requires_game=False,
    description="AI-based player using weighted heuristics (alias for weighted_heuristic)",
)

__all__ = ["AIDecisionMaker", "AIPlayer"]


