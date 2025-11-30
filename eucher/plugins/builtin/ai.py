"""AI player plugin."""

from typing import Optional

from eucher.players.computer.ai import AIDecisionMaker, AIPlayer
from eucher.plugins.registry import register_plugin

# Shared AI decision maker instance
_ai_decision_maker: Optional[AIDecisionMaker] = None


def get_ai_decision_maker() -> AIDecisionMaker:
    """
    Get or create the shared AI decision maker instance.

    Returns
    -------
    AIDecisionMaker
        The shared AI decision maker.
    """
    global _ai_decision_maker
    if _ai_decision_maker is None:
        _ai_decision_maker = AIDecisionMaker()
    return _ai_decision_maker


def create_ai_player(game: Optional[object] = None, **kwargs: object) -> AIPlayer:
    """
    Create an AI player instance.

    Parameters
    ----------
    game : Optional[object]
        Game instance (not used by AI player, but AIDecisionMaker is shared).
    **kwargs : object
        Additional arguments (not used).

    Returns
    -------
    AIPlayer
        A new AI player instance.
    """
    return AIPlayer(get_ai_decision_maker())


# Register the plugin
register_plugin(
    name="ai",
    factory=create_ai_player,
    display_name="AI Player",
    description="AI-based player using weighted heuristics",
    requires_game=False,
    supports_kwargs=False,
)

