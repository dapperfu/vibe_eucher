"""Player manager for loading and managing trained euchre AI models."""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from .euchre_nn import create_euchre_model, RiskParameters, create_risk_profile
from .model_player import ModelPlayer
from ..models import Player, PlayerType


class PlayerManager:
    """Manages trained AI players and their configurations."""
    
    def __init__(self, models_dir: str = "trained_models"):
        """Initialize the player manager.
        
        Parameters
        ----------
        models_dir : str
            Directory containing trained model files
        """
        self.models_dir = Path(models_dir)
        self.available_players = {}
        self.loaded_models = {}
        
        # Load available players
        self._discover_players()
        
        logging.info(f"Player manager initialized with {len(self.available_players)} available players")
    
    def _discover_players(self):
        """Discover available trained players in the models directory."""
        if not self.models_dir.exists():
            logging.warning(f"Models directory does not exist: {self.models_dir}")
            return
        
        for model_file in self.models_dir.glob("*.json"):
            if model_file.name == "training_summary.json":
                continue
            
            player_name = model_file.stem
            if not player_name.endswith('_checkpoint'):
                self.available_players[player_name] = {
                    'model_file': model_file,
                    'last_modified': model_file.stat().st_mtime
                }
        
        logging.info(f"Discovered players: {list(self.available_players.keys())}")
    
    def get_available_players(self) -> List[str]:
        """Get list of available trained players.
        
        Returns
        -------
        List[str]
            List of player names
        """
        return list(self.available_players.keys())
    
    def get_player_info(self, player_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific player.
        
        Parameters
        ----------
        player_name : str
            Name of the player
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Player information or None if not found
        """
        if player_name not in self.available_players:
            return None
        
        model_file = self.available_players[player_name]['model_file']
        
        try:
            with open(model_file, 'r') as f:
                checkpoint = json.load(f)
            
            return {
                'name': player_name,
                'model_file': str(model_file),
                'training_games': checkpoint.get('training_games', 'Unknown'),
                'timestamp': checkpoint.get('timestamp', 'Unknown'),
                'model_config': checkpoint.get('model_config', {}),
                'last_modified': self.available_players[player_name]['last_modified']
            }
        except Exception as e:
            logging.error(f"Failed to load player info for {player_name}: {e}")
            return None
    
    def load_player_model(self, player_name: str, device: str = "auto") -> Optional[torch.nn.Module]:
        """Load a trained model for a specific player.
        
        Parameters
        ----------
        player_name : str
            Name of the player to load
        device : str
            Device to load the model on
            
        Returns
        -------
        Optional[torch.nn.Module]
            Loaded model or None if failed
        """
        if player_name not in self.available_players:
            logging.error(f"Player not found: {player_name}")
            return None
        
        # Check if already loaded
        if player_name in self.loaded_models:
            return self.loaded_models[player_name]
        
        # Device setup
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)
        
        try:
            model_file = self.available_players[player_name]['model_file']
            
            # Load checkpoint
            with open(model_file, 'r') as f:
                checkpoint = json.load(f)
            
            # Create model
            model_config = checkpoint.get('model_config', {})
            model = create_euchre_model(model_config)
            model.to(device)
            
            # Load weights
            state_dict = {}
            for param_name, param_data in checkpoint['model_state_dict'].items():
                state_dict[param_name] = torch.tensor(param_data, device=device)
            
            model.load_state_dict(state_dict)
            model.eval()
            
            # Cache the loaded model
            self.loaded_models[player_name] = model
            
            logging.info(f"✅ Loaded model for {player_name} on {device}")
            return model
            
        except Exception as e:
            logging.error(f"Failed to load model for {player_name}: {e}")
            return None
    
    def create_model_player(self, 
                           player_name: str, 
                           risk_profile: str = "balanced",
                           custom_risk: Optional[RiskParameters] = None,
                           device: str = "auto") -> Optional[ModelPlayer]:
        """Create a model player instance.
        
        Parameters
        ----------
        player_name : str
            Name of the player
        risk_profile : str
            Risk profile to use
        custom_risk : Optional[RiskParameters]
            Custom risk parameters
        device : str
            Device to use
            
        Returns
        -------
        Optional[ModelPlayer]
            Model player instance or None if failed
        """
        # Load the model
        model = self.load_player_model(player_name, device)
        if model is None:
            return None
        
        # Create player
        try:
            player = ModelPlayer(
                name=player_name,
                model=model,
                device=device,
                risk_profile=risk_profile,
                custom_risk=custom_risk
            )
            
            logging.info(f"✅ Created model player: {player_name}")
            return player
            
        except Exception as e:
            logging.error(f"Failed to create model player {player_name}: {e}")
            return None
    
    def create_game_with_players(self, 
                                player_configs: List[Dict[str, Any]],
                                enable_logging: bool = True) -> Optional['EuchreGame']:
        """Create a game with specified players.
        
        Parameters
        ----------
        player_configs : List[Dict[str, Any]]
            List of player configurations
        enable_logging : bool
            Whether to enable game logging
            
        Returns
        -------
        Optional[EuchreGame]
            Game instance or None if failed
        """
        try:
            from ..game import EuchreGame
            
            # Create game
            game = EuchreGame(enable_logging=enable_logging)
            
            # Add players
            for config in player_configs:
                player_name = config['name']
                player_type = config.get('type', 'model')
                
                if player_type == 'model':
                    risk_profile = config.get('risk_profile', 'balanced')
                    custom_risk = config.get('custom_risk')
                    
                    player = self.create_model_player(
                        player_name, risk_profile, custom_risk
                    )
                    
                    if player is None:
                        logging.error(f"Failed to create model player: {player_name}")
                        return None
                    
                    game.players.append(player)
                    
                elif player_type == 'ai':
                    risk_ratio = config.get('risk_ratio', 0.5)
                    game.add_ai_player(player_name, "balanced", risk_ratio)
                    
                elif player_type == 'human':
                    # For now, create a placeholder human player
                    from ..models import Player, PlayerType
                    human_player = Player(player_name, PlayerType.HUMAN)
                    game.players.append(human_player)
                
                logging.info(f"✅ Added {player_type} player: {player_name}")
            
            return game
            
        except Exception as e:
            logging.error(f"Failed to create game: {e}")
            return None
    
    def get_player_statistics(self) -> Dict[str, Any]:
        """Get statistics about all available players.
        
        Returns
        -------
        Dict[str, Any]
            Player statistics
        """
        stats = {
            'total_players': len(self.available_players),
            'players': {}
        }
        
        for player_name in self.available_players:
            info = self.get_player_info(player_name)
            if info:
                stats['players'][player_name] = info
        
        return stats
    
    def refresh_players(self):
        """Refresh the list of available players."""
        self.available_players.clear()
        self.loaded_models.clear()
        self._discover_players()
        logging.info("Player list refreshed")
    
    def unload_player(self, player_name: str):
        """Unload a specific player's model from memory.
        
        Parameters
        ----------
        player_name : str
            Name of the player to unload
        """
        if player_name in self.loaded_models:
            del self.loaded_models[player_name]
            logging.info(f"Unloaded model for {player_name}")
    
    def unload_all_players(self):
        """Unload all player models from memory."""
        self.loaded_models.clear()
        logging.info("Unloaded all player models")


def create_sample_game_configs() -> List[Dict[str, Any]]:
    """Create sample game configurations for different scenarios.
    
    Returns
    -------
    List[Dict[str, Any]]
        List of sample configurations
    """
    return [
        # Scenario 1: All trained models
        {
            'name': 'All Trained Models',
            'description': 'Play with 4 trained AI models',
            'players': [
                {'name': 'Alice', 'type': 'model', 'risk_profile': 'balanced'},
                {'name': 'Bob', 'type': 'model', 'risk_profile': 'aggressive'},
                {'name': 'Charlie', 'type': 'model', 'risk_profile': 'conservative'},
                {'name': 'David', 'type': 'model', 'risk_profile': 'ace_hunter'}
            ]
        },
        
        # Scenario 2: Mixed AI and models
        {
            'name': 'Mixed AI and Models',
            'description': 'Mix of trained models and AI profiles',
            'players': [
                {'name': 'Alice', 'type': 'model', 'risk_profile': 'balanced'},
                {'name': 'AI_East', 'type': 'ai', 'risk_ratio': 0.8},
                {'name': 'Charlie', 'type': 'model', 'risk_profile': 'conservative'},
                {'name': 'AI_West', 'type': 'ai', 'risk_ratio': 0.2}
            ]
        },
        
        # Scenario 3: Human vs AI
        {
            'name': 'Human vs AI',
            'description': 'Human player against trained AI models',
            'players': [
                {'name': 'Human', 'type': 'human'},
                {'name': 'Alice', 'type': 'model', 'risk_profile': 'balanced'},
                {'name': 'Bob', 'type': 'model', 'risk_profile': 'aggressive'},
                {'name': 'Charlie', 'type': 'model', 'risk_profile': 'conservative'}
            ]
        },
        
        # Scenario 4: Tournament setup
        {
            'name': 'Tournament Setup',
            'description': 'High-stakes tournament with aggressive players',
            'players': [
                {'name': 'Alice', 'type': 'model', 'risk_profile': 'ultra_aggressive'},
                {'name': 'Bob', 'type': 'model', 'risk_profile': 'trump_caller'},
                {'name': 'Charlie', 'type': 'model', 'risk_profile': 'risk_taker'},
                {'name': 'David', 'type': 'model', 'risk_profile': 'ace_hunter'}
            ]
        }
    ]


def main():
    """Main entry point for player manager demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Player manager for trained euchre AI models")
    parser.add_argument("--models-dir", default="trained_models", help="Models directory")
    parser.add_argument("--list-players", action="store_true", help="List available players")
    parser.add_argument("--player-info", help="Get info for specific player")
    parser.add_argument("--sample-configs", action="store_true", help="Show sample game configurations")
    
    args = parser.parse_args()
    
    # Create player manager
    manager = PlayerManager(args.models_dir)
    
    if args.list_players:
        players = manager.get_available_players()
        print(f"Available players ({len(players)}):")
        for player in players:
            print(f"  - {player}")
    
    elif args.player_info:
        info = manager.get_player_info(args.player_info)
        if info:
            print(f"Player info for {args.player_info}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"Player not found: {args.player_info}")
    
    elif args.sample_configs:
        configs = create_sample_game_configs()
        print("Sample game configurations:")
        for i, config in enumerate(configs, 1):
            print(f"\n{i}. {config['name']}")
            print(f"   Description: {config['description']}")
            print("   Players:")
            for player in config['players']:
                print(f"     - {player['name']} ({player['type']})")
    
    else:
        print("Player manager initialized. Use --help for options.")


if __name__ == "__main__":
    main() 