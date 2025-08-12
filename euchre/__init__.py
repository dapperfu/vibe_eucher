"""Euchre - A CLI card game with AI opponents."""

from .game import EuchreGame
from .models import Player, PlayerType, Card, Suit, Rank, GameState, Trick
from .ai import AIFactory, BaseAI, AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI

__version__ = "0.1.0"
__all__ = [
    'EuchreGame',
    'Player', 'PlayerType', 'Card', 'Suit', 'Rank', 'GameState', 'Trick',
    'AIFactory', 'BaseAI', 'AggressiveAI', 'ConservativeAI', 'BalancedAI', 'OpportunisticAI'
] 