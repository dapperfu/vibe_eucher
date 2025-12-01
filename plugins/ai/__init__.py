"""AI player plugin."""

from .ai import AIDecisionMaker, AIPlayer
from eucher.plugins import register_plugin


def _create_ai_player(game=None, **kwargs):
    """Factory for AIPlayer."""
    ai_decision_maker = AIDecisionMaker()
    return AIPlayer(ai_decision_maker)


def _create_weighted_heuristic_player(game=None, **kwargs):
    """Factory for AIPlayer (alias for weighted_heuristic)."""
    ai_decision_maker = AIDecisionMaker()
    return AIPlayer(ai_decision_maker)


register_plugin(
    name="ai",
    factory=_create_ai_player,
    display_name="Weighted Heuristic AI",
    requires_game=False,
    description="AI-based player using weighted heuristics",
)

register_plugin(
    name="weighted_heuristic",
    factory=_create_weighted_heuristic_player,
    display_name="Weighted Heuristic AI",
    requires_game=False,
    description="AI-based player using weighted heuristics (alias for ai)",
)

__all__ = ["AIDecisionMaker", "AIPlayer"]


