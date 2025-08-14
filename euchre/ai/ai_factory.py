"""Factory for creating AI players with different profiles."""

from typing import List, Optional, Union
from .base_ai_interface import BaseAIInterface
from .traditional_ai_impl import TraditionalAI
from .ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI
from ..models import Player, PlayerType

# Import Level 2 models if available
try:
    from .level2_ai_impl import Level2AI
    LEVEL2_AVAILABLE = True
except ImportError:
    LEVEL2_AVAILABLE = False

# Import Level 3 models if available
try:
    from .level3_ai_impl import Level3AI
    LEVEL3_AVAILABLE = True
except ImportError:
    LEVEL3_AVAILABLE = False


class AIFactory:
    """Factory for creating AI players with different profiles."""
    
    @staticmethod
    def create_ai_player(name: str, ai_type: str = "level1_balanced", risk_ratio: float = 0.5, 
                        model_path: Optional[str] = None) -> Union[BaseAIInterface, Player]:
        """
        Create an AI player with a specific profile.
        
        Parameters
        ----------
        name : str
            The player's name
        ai_type : str
            Type of AI: "level1_aggressive", "level1_conservative", "level1_balanced", "level1_opportunistic",
                       "level2_strategic", "level2_aggressive", "level2_balanced", "level2_intuitive",
                       "level3_strategic", "level3_aggressive", "level3_balanced", "level3_conservative", "level3_opportunistic"
        risk_ratio : float
            Risk tolerance (0.0 = conservative, 1.0 = aggressive)
        model_path : str, optional
            Path to trained Level 2 or Level 3 model file (.pth)
            
        Returns
        -------
        Union[BaseAIInterface, Player]
            The created AI player (either new interface or legacy Player)
        """
        ai_type = ai_type.lower()
        
        # Check if this is a Level 3 model type
        if ai_type in ["level3_strategic", "level3_aggressive", "level3_balanced", "level3_conservative", "level3_opportunistic"]:
            if not LEVEL3_AVAILABLE:
                raise ValueError(f"Level 3 models not available. Install PyTorch and Level 3 dependencies.")
            
            return Level3AI(name, ai_type, risk_ratio, model_path)
        
        # Check if this is a Level 2 model type
        if ai_type in ["level2_strategic", "level2_aggressive", "level2_balanced", "level2_intuitive"]:
            if not LEVEL2_AVAILABLE:
                raise ValueError(f"Level 2 models not available. Install PyTorch and Level 2 dependencies.")
            
            return Level2AI(name, ai_type, risk_ratio, model_path)
        
        # Check if this is a Level 1 AI type using the new interface
        if ai_type in ["level1_aggressive", "level1_conservative", "level1_balanced", "level1_opportunistic"]:
            # Extract the base type from level1_ prefix
            base_type = ai_type.replace("level1_", "")
            return TraditionalAI(name, risk_ratio, base_type)
        
        # Legacy support for old naming (backward compatibility)
        if ai_type in ["aggressive", "conservative", "balanced", "opportunistic"]:
            # Use legacy AI classes for simple types to maintain compatibility
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
            List of model paths for Level 2 players
            
        Returns
        -------
        List[Union[BaseAIInterface, Player]]
            List of created AI players
        """
        if ai_types is None:
            ai_types = ["level1_balanced"] * len(names)
        
        if risk_ratios is None:
            risk_ratios = [0.5] * len(names)
        
        if model_paths is None:
            model_paths = [None] * len(names)
        
        # Ensure lists are the same length
        while len(ai_types) < len(names):
            ai_types.append("level1_balanced")
        
        while len(risk_ratios) < len(names):
            risk_ratios.append(0.5)
        
        while len(model_paths) < len(names):
            model_paths.append(None)
        
        players = []
        for i, name in enumerate(names):
            try:
                player = AIFactory.create_ai_player(
                    name, ai_types[i], risk_ratios[i], model_paths[i]
                )
                players.append(player)
            except Exception as e:
                print(f"Warning: Could not create AI player {name} with type {ai_types[i]}: {e}")
                # Fallback to balanced AI
                fallback_player = AIFactory.create_ai_player(name, "level1_balanced", 0.5)
                players.append(fallback_player)
        
        return players
    
    @staticmethod
    def create_default_ai_players() -> List[BaseAIInterface]:
        """Create default AI players with balanced profiles."""
        names = ["Alice", "Bob", "Charlie", "David"]
        ai_types = ["level1_balanced", "level1_balanced", "level1_balanced", "level1_balanced"]
        risk_ratios = [0.5, 0.5, 0.5, 0.5]
        
        return AIFactory.create_ai_players(names, ai_types, risk_ratios)
    
    @staticmethod
    def create_mixed_ai_players() -> List[BaseAIInterface]:
        """Create AI players with mixed strategies for variety."""
        names = ["Alice", "Bob", "Charlie", "David"]
        ai_types = ["level1_aggressive", "level1_conservative", "level1_balanced", "level1_opportunistic"]
        risk_ratios = [0.8, 0.2, 0.5, 0.7]
        
        return AIFactory.create_ai_players(names, ai_types, risk_ratios)
    
    @staticmethod
    def create_level2_players(model_type: str = "level2_strategic", 
                               model_path: Optional[str] = None) -> List[BaseAIInterface]:
        """Create Level 2 AI players.
        
        Parameters
        ----------
        model_type : str
            Level 2 model type: "level2_strategic", "level2_aggressive", "level2_balanced", "level2_intuitive"
        model_path : str, optional
            Path to trained model file
            
        Returns
        -------
        List[BaseAIInterface]
            List of 4 Level 2 AI players
        """
        if not LEVEL2_AVAILABLE:
            raise ValueError("Level 2 models not available. Install PyTorch and Level 2 dependencies.")
        
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
        level1_types = ["level1_aggressive", "level1_conservative", "level1_balanced", "level1_opportunistic"]
        
        available_types = level1_types.copy()
        
        if LEVEL2_AVAILABLE:
            level2_types = ["level2_strategic", "level2_aggressive", "level2_balanced", "level2_intuitive"]
            available_types.extend(level2_types)
        
        if LEVEL3_AVAILABLE:
            level3_types = ["level3_strategic", "level3_aggressive", "level3_balanced", "level3_conservative", "level3_opportunistic"]
            available_types.extend(level3_types)
        
        return available_types
    
    @staticmethod
    def is_level1_type(ai_type: str) -> bool:
        """Check if an AI type is a Level 1 model.
        
        Parameters
        ----------
        ai_type : str
            The AI type to check
            
        Returns
        -------
        bool
            True if it's a Level 1 type, False otherwise
        """
        return ai_type.lower() in ["level1_aggressive", "level1_conservative", "level1_balanced", "level1_opportunistic"]
    
    @staticmethod
    def is_level2_type(ai_type: str) -> bool:
        """Check if an AI type is a Level 2 model.
        
        Parameters
        ----------
        ai_type : str
            The AI type to check
            
        Returns
        -------
        bool
            True if it's a Level 2 type, False otherwise
        """
        return ai_type.lower() in ["level2_strategic", "level2_aggressive", "level2_balanced", "level2_intuitive"]
    
    @staticmethod
    def is_level3_type(ai_type: str) -> bool:
        """Check if an AI type is a Level 3 model.
        
        Parameters
        ----------
        ai_type : str
            The AI type to check
            
        Returns
        -------
        bool
            True if it's a Level 3 type, False otherwise
        """
        return ai_type.lower() in ["level3_strategic", "level3_aggressive", "level3_balanced", "level3_conservative", "level3_opportunistic"]
    
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
        new_interface_types = ["level1_aggressive", "level1_conservative", "level1_balanced", "level1_opportunistic",
                              "level2_strategic", "level2_aggressive", "level2_balanced", "level2_intuitive",
                              "level3_strategic", "level3_aggressive", "level3_balanced", "level3_conservative", "level3_opportunistic"]
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
    
    @staticmethod
    def create_level3_players(model_type: str = "level3_strategic", 
                               model_path: Optional[str] = None) -> List[BaseAIInterface]:
        """Create Level 3 AI players.
        
        Parameters
        ----------
        model_type : str
            Type of Level 3 model to use
        model_path : str, optional
            Path to trained model file
            
        Returns
        -------
        List[BaseAIInterface]
            List of Level 3 AI players
        """
        if not LEVEL3_AVAILABLE:
            raise ValueError("Level 3 models not available. Install PyTorch and Level 3 dependencies.")
        
        names = ["Alice", "Bob", "Charlie", "David"]
        ai_types = [model_type] * 4
        model_paths = [model_path] * 4
        
        return AIFactory.create_ai_players(names, ai_types, model_paths) 