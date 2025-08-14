"""AI player implementations for euchre."""

from .base_ai import BaseAI
from .ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI
from .ai_factory import AIFactory

# Import Level 2 and Level 3 AI if available
try:
    from .level2_ai_impl import Level2AI
    LEVEL2_AVAILABLE = True
except ImportError:
    LEVEL2_AVAILABLE = False

try:
    from .level3_ai_impl import Level3AI
    LEVEL3_AVAILABLE = True
except ImportError:
    LEVEL3_AVAILABLE = False

__all__ = [
    'BaseAI', 'AggressiveAI', 'ConservativeAI', 'BalancedAI', 'OpportunisticAI', 
    'AIFactory', 'LEVEL2_AVAILABLE', 'LEVEL3_AVAILABLE'
]

# Add Level 2 and Level 3 AI to exports if available
if LEVEL2_AVAILABLE:
    __all__.append('Level2AI')

if LEVEL3_AVAILABLE:
    __all__.append('Level3AI') 