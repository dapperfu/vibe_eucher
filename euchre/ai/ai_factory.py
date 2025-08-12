"""Factory for creating AI players with different profiles."""

from typing import List
from .ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI
from ..models import Player, PlayerType


class AIFactory:
    """Factory for creating AI players with different profiles."""
    
    @staticmethod
    def create_ai_player(name: str, ai_type: str = "balanced", risk_ratio: float = 0.5) -> Player:
        """Create an AI player with a specific profile.
        
        Parameters
        ----------
        name : str
            The player's name
        ai_type : str
            Type of AI: "aggressive", "conservative", "balanced", "opportunistic"
        risk_ratio : float
            Risk tolerance (0.0 = conservative, 1.0 = aggressive)
            
        Returns
        -------
        Player
            The created AI player
        """
        ai_type = ai_type.lower()
        
        if ai_type == "aggressive":
            return AggressiveAI(name, risk_ratio)
        elif ai_type == "conservative":
            return ConservativeAI(name, risk_ratio)
        elif ai_type == "opportunistic":
            return OpportunisticAI(name, risk_ratio)
        else:  # balanced or unknown
            return BalancedAI(name, risk_ratio)
    
    @staticmethod
    def create_ai_players(names: List[str], ai_types: List[str] = None, risk_ratios: List[float] = None) -> List[Player]:
        """Create multiple AI players with specified profiles.
        
        Parameters
        ----------
        names : List[str]
            List of player names
        ai_types : List[str], optional
            List of AI types for each player
        risk_ratios : List[float], optional
            List of risk ratios for each player
            
        Returns
        -------
        List[Player]
            List of created AI players
        """
        if ai_types is None:
            ai_types = ["balanced"] * len(names)
        
        if risk_ratios is None:
            risk_ratios = [0.5] * len(names)
        
        # Ensure lists are the same length
        while len(ai_types) < len(names):
            ai_types.append("balanced")
        
        while len(risk_ratios) < len(names):
            risk_ratios.append(0.5)
        
        players = []
        for name, ai_type, risk_ratio in zip(names, ai_types, risk_ratios):
            player = AIFactory.create_ai_player(name, ai_type, risk_ratio)
            players.append(player)
        
        return players
    
    @staticmethod
    def create_default_ai_players() -> List[Player]:
        """Create default AI players for a 4-player game.
        
        Returns
        -------
        List[Player]
            List of 4 default AI players
        """
        default_names = ["Alice", "Bob", "Charlie", "David"]
        default_types = ["balanced", "balanced", "balanced", "balanced"]
        default_risks = [0.5, 0.5, 0.5, 0.5]
        
        return AIFactory.create_ai_players(default_names, default_types, default_risks)
    
    @staticmethod
    def create_mixed_ai_players() -> List[Player]:
        """Create AI players with mixed playing styles.
        
        Returns
        -------
        List[Player]
            List of 4 AI players with different styles
        """
        names = ["Alice", "Bob", "Charlie", "David"]
        types = ["aggressive", "conservative", "balanced", "opportunistic"]
        risks = [0.8, 0.2, 0.5, 0.6]
        
        return AIFactory.create_ai_players(names, types, risks)
    
    @staticmethod
    def get_available_ai_types() -> List[str]:
        """Get list of available AI types.
        
        Returns
        -------
        List[str]
            List of available AI type names
        """
        return ["aggressive", "conservative", "balanced", "opportunistic"]
    
    @staticmethod
    def validate_ai_type(ai_type: str) -> bool:
        """Validate if an AI type is supported.
        
        Parameters
        ----------
        ai_type : str
            The AI type to validate
            
        Returns
        -------
        bool
            True if the AI type is valid
        """
        return ai_type.lower() in AIFactory.get_available_ai_types()
    
    @staticmethod
    def validate_risk_ratio(risk_ratio: float) -> bool:
        """Validate if a risk ratio is within valid range.
        
        Parameters
        ----------
        risk_ratio : float
            The risk ratio to validate
            
        Returns
        -------
        bool
            True if the risk ratio is valid
        """
        return 0.0 <= risk_ratio <= 1.0 