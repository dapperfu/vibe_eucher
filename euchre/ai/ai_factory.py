"""Factory for creating AI players with different profiles."""

from typing import List, Optional, Union
from .base_ai_interface import BaseAIInterface
from .traditional_ai_impl import TraditionalAI
from .ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI
from ..models import Player, PlayerType

# Import M-Series models if available
try:
    from .m_series_ai_impl import MSeriesAI
    M_SERIES_AVAILABLE = True
except ImportError:
    M_SERIES_AVAILABLE = False


class AIFactory:
    """Factory for creating AI players with different profiles."""
    
    @staticmethod
    def create_ai_player(name: str, ai_type: str = "balanced", risk_ratio: float = 0.5, 
                        model_path: Optional[str] = None) -> Union[BaseAIInterface, Player]:
        """
        Create an AI player with a specific profile.
        
        Parameters
        ----------
        name : str
            The player's name
        ai_type : str
            Type of AI: "aggressive", "conservative", "balanced", "opportunistic",
                       "magnus", "maverick", "mentor", "mystic"
        risk_ratio : float
            Risk tolerance (0.0 = conservative, 1.0 = aggressive)
        model_path : str, optional
            Path to trained M-Series model file (.pth)
            
        Returns
        -------
        Union[BaseAIInterface, Player]
            The created AI player (either new interface or legacy Player)
        """
        ai_type = ai_type.lower()
        
        # Check if this is an M-Series model type
        if ai_type in ["magnus", "maverick", "mentor", "mystic"]:
            if not M_SERIES_AVAILABLE:
                raise ValueError(f"M-Series models not available. Install PyTorch and M-Series dependencies.")
            
            return MSeriesAI(name, ai_type, risk_ratio, model_path)
        
        # Check if this is a traditional AI type using the new interface
        if ai_type in ["aggressive", "conservative", "balanced", "opportunistic"]:
            return TraditionalAI(name, risk_ratio, ai_type)
        
        # Fallback to legacy AI types for backward compatibility
        if ai_type == "aggressive":
            return AggressiveAI(name, risk_ratio)
        elif ai_type == "conservative":
            return ConservativeAI(name, risk_ratio)
        elif ai_type == "opportunistic":
            return OpportunisticAI(name, risk_ratio)
        else:  # balanced or unknown
            return BalancedAI(name, risk_ratio)
    
    @staticmethod
    def create_ai_players(names: List[str], ai_types: List[str] = None, risk_ratios: List[float] = None,
                         model_paths: List[str] = None) -> List[Union[BaseAIInterface, Player]]:
        """Create multiple AI players with specified profiles.
        
        Parameters
        ----------
        names : List[str]
            List of player names
        ai_types : List[str], optional
            List of AI types for each player
        risk_ratios : List[float], optional
            List of risk ratios for each player
        model_paths : List[str], optional
            List of model paths for M-Series players
            
        Returns
        -------
        List[Union[BaseAIInterface, Player]]
            List of created AI players
        """
        if ai_types is None:
            ai_types = ["balanced"] * len(names)
        
        if risk_ratios is None:
            risk_ratios = [0.5] * len(names)
        
        if model_paths is None:
            model_paths = [None] * len(names)
        
        # Ensure lists are the same length
        while len(ai_types) < len(names):
            ai_types.append("balanced")
        
        while len(risk_ratios) < len(names):
            risk_ratios.append(0.5)
        
        while len(model_paths) < len(names):
            model_paths.append(None)
        
        players = []
        for name, ai_type, risk_ratio, model_path in zip(names, ai_types, risk_ratios, model_paths):
            player = AIFactory.create_ai_player(name, ai_type, risk_ratio, model_path)
            players.append(player)
        
        return players
    
    @staticmethod
    def create_default_ai_players() -> List[Union[BaseAIInterface, Player]]:
        """Create default AI players for a 4-player game.
        
        Returns
        -------
        List[Union[BaseAIInterface, Player]]
            List of 4 default AI players
        """
        default_names = ["Alice", "Bob", "Charlie", "David"]
        default_types = ["balanced", "balanced", "balanced", "balanced"]
        default_risks = [0.5, 0.5, 0.5, 0.5]
        
        return AIFactory.create_ai_players(default_names, default_types, default_risks)
    
    @staticmethod
    def create_mixed_ai_players() -> List[Union[BaseAIInterface, Player]]:
        """Create AI players with mixed playing styles.
        
        Returns
        -------
        List[Union[BaseAIInterface, Player]]
            List of 4 AI players with different styles
        """
        mixed_names = ["Alice", "Bob", "Charlie", "David"]
        mixed_types = ["aggressive", "conservative", "balanced", "opportunistic"]
        mixed_risks = [0.8, 0.2, 0.5, 0.7]
        
        return AIFactory.create_ai_players(mixed_names, mixed_types, mixed_risks)
    
    @staticmethod
    def create_m_series_players(model_type: str = "magnus", 
                               model_path: Optional[str] = None) -> List[BaseAIInterface]:
        """Create M-Series AI players.
        
        Parameters
        ----------
        model_type : str
            M-Series model type: "magnus", "maverick", "mentor", "mystic"
        model_path : str, optional
            Path to trained model file
            
        Returns
        -------
        List[BaseAIInterface]
            List of 4 M-Series AI players
        """
        if not M_SERIES_AVAILABLE:
            raise ValueError("M-Series models not available. Install PyTorch and M-Series dependencies.")
        
        names = ["Alice", "Bob", "Charlie", "David"]
        ai_types = [model_type] * 4
        risk_ratios = [0.5, 0.3, 0.7, 0.4]  # Different risk profiles
        
        return AIFactory.create_ai_players(names, ai_types, risk_ratios, [model_path] * 4)
    
    @staticmethod
    def get_available_ai_types() -> List[str]:
        """Get list of available AI types.
        
        Returns
        -------
        List[str]
            List of available AI type names
        """
        traditional_types = ["aggressive", "conservative", "balanced", "opportunistic"]
        
        if M_SERIES_AVAILABLE:
            m_series_types = ["magnus", "maverick", "mentor", "mystic"]
            return traditional_types + m_series_types
        
        return traditional_types
    
    @staticmethod
    def is_m_series_type(ai_type: str) -> bool:
        """Check if an AI type is an M-Series model.
        
        Parameters
        ----------
        ai_type : str
            The AI type to check
            
        Returns
        -------
        bool
            True if it's an M-Series type, False otherwise
        """
        return ai_type.lower() in ["magnus", "maverick", "mentor", "mystic"]
    
    @staticmethod
    def is_new_interface_type(ai_type: str) -> bool:
        """Check if an AI type uses the new interface system.
        
        Parameters
        ----------
        ai_type : str
            The AI type to check
            
        Returns
        -------
        bool
            True if it uses the new interface, False if it's legacy
        """
        new_interface_types = ["aggressive", "conservative", "balanced", "opportunistic", 
                              "magnus", "maverick", "mentor", "mystic"]
        return ai_type.lower() in new_interface_types
    
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