"""Self-play trainer for training euchre AI models against themselves."""

import os
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from .euchre_nn import EuchreNN, RiskParameters, create_risk_profile, create_euchre_model
from .euchre_nn_int import EuchreNNInt, RiskParametersInt, create_risk_profile_int, create_euchre_model_int
from .model_player import ModelPlayer
from ..game import EuchreGame
from ..game_logger import GameLogger


class SelfPlayTrainer:
    """Trainer that uses self-play to improve euchre AI models."""
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 output_dir: str = "trained_models",
                 device: str = "auto"):
        """Initialize the self-play trainer.
        
        Parameters
        ----------
        model_config : Dict[str, Any]
            Configuration for the model architecture
        output_dir : str
            Directory to save trained models
        device : str
            Device to use for training
        """
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Device setup
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print(f"🚀 CUDA detected: {torch.cuda.get_device_name()}")
                print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print(f"🚀 MPS (Apple Silicon) detected")
            else:
                self.device = torch.device("cpu")
                print(f"🚀 Using CPU (CUDA/MPS not available)")
        else:
            self.device = torch.device(device)
            if device == "cuda" and torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print(f"🚀 CUDA enabled: {torch.cuda.get_device_name()}")
            elif device == "cuda" and not torch.cuda.is_available():
                print(f"⚠️  CUDA requested but not available, falling back to CPU")
                self.device = torch.device("cpu")
        
        # Training state
        self.training_history = []
        self.best_models = {}
        
        print(f"🚀 Self-play trainer initialized on {self.device}")
        print(f"📁 Output directory: {self.output_dir}")
        
        # GPU memory optimization
        if self.device.type == "cuda":
            self._optimize_gpu_memory()
    
    def _optimize_gpu_memory(self) -> None:
        """Optimize GPU memory usage for training."""
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Set memory fraction to avoid OOM
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            
            print(f"💾 GPU memory optimized for training")
            
        except Exception as e:
            print(f"⚠️  GPU optimization failed: {e}")
    
    def _check_gpu_memory(self) -> None:
        """Check and report GPU memory usage."""
        if self.device.type == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(self.device) / 1e9
                reserved = torch.cuda.memory_reserved(self.device) / 1e9
                total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
                
                print(f"💾 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
                
                # Warn if memory usage is high
                if allocated / total > 0.8:
                    print(f"⚠️  High GPU memory usage ({allocated/total*100:.1f}%)")
                    
            except Exception as e:
                print(f"⚠️  Could not check GPU memory: {e}")
    
    def create_player_model(self, player_name: str, risk_profile: str = "balanced") -> Tuple[EuchreNN, ModelPlayer]:
        """Create a model and player instance for a specific player.
        
        Parameters
        ----------
        player_name : str
            Name of the player
        risk_profile : str
            Risk profile for the player
            
        Returns
        -------
        Tuple[EuchreNN, ModelPlayer]
            Model and player instance
        """
        # Create model
        model = create_euchre_model(self.model_config)
        model.to(self.device)
        
        # Create player
        player = ModelPlayer(
            name=player_name,
            model=model,
            device=self.device,
            risk_profile=risk_profile
        )
        
        return model, player
    
    def create_integer_player_model(self, player_name: str, risk_profile: str = "balanced") -> Tuple[EuchreNNInt, ModelPlayer]:
        """Create an integer model and player instance for a specific player.
        
        Parameters
        ----------
        player_name : str
            Name of the player
        risk_profile : str
            Risk profile for the player
            
        Returns
        -------
        Tuple[EuchreNNInt, ModelPlayer]
            Integer model and player instance
        """
        # Create integer model with same architecture parameters
        int_model_config = {
            'input_size': self.model_config.get('input_size', 128),
            'hidden_size': self.model_config.get('hidden_size', 256),
            'risk_embedding_size': self.model_config.get('risk_embedding_size', 32)
        }
        
        model = create_euchre_model_int(**int_model_config, device=self.device)
        model.to(self.device)
        
        # Create player
        player = ModelPlayer(
            name=player_name,
            model=model,
            device=self.device,
            risk_profile=risk_profile
        )
        
        return model, player
    
    def train_self_play(self, 
                        num_games: int = 100000,
                        players: List[str] = None,
                        risk_profiles: List[str] = None,
                        save_interval: int = 1000,
                        evaluation_interval: int = 5000) -> Dict[str, Any]:
        """Train models using self-play.
        
        Parameters
        ----------
        num_games : int
            Number of games to play
        players : List[str]
            List of player names (default: Alice 1-4)
        risk_profiles : List[str]
            List of risk profiles for each player
        save_interval : int
            Save models every N games
        evaluation_interval : int
            Evaluate performance every N games
            
        Returns
        -------
        Dict[str, Any]
            Training results and statistics
        """
        if players is None:
            players = ["Alice", "Bob", "Charlie", "David"]
        
        if risk_profiles is None:
            risk_profiles = ["balanced", "balanced", "balanced", "balanced"]
        
        print(f"🎮 Starting self-play training with {num_games} games")
        print(f"👥 Players: {players}")
        print(f"🎯 Risk profiles: {risk_profiles}")
        print("=" * 80)
        
        # Create models and players
        models = {}
        game_players = {}
        
        for i, (player_name, risk_profile) in enumerate(zip(players, risk_profiles)):
            model, player = self.create_player_model(player_name, risk_profile)
            models[player_name] = model
            game_players[player_name] = player
            print(f"✅ Created {player_name} with {risk_profile} profile")
        
        # Training statistics
        training_stats = {
            'total_games': 0,
            'player_wins': {name: 0 for name in players},
            'player_losses': {name: 0 for name in players},
            'player_ties': {name: 0 for name in players},
            'game_lengths': [],
            'trump_calls': {name: 0 for name in players},
            'aces_ordered': {name: 0 for name in players},
            'times_set': {name: 0 for name in players}
        }
        
        # Training loop
        start_time = time.time()
        
        for game_num in tqdm(range(num_games), desc="Training games"):
            # Play a single game
            game_result = self._play_training_game(game_players, players[:2], players[2:])
            
            # Update statistics
            self._update_training_stats(training_stats, game_result)
            
            # Save models periodically
            if (game_num + 1) % save_interval == 0:
                self._save_models(models, players, game_num + 1)
            
            # Evaluate performance periodically
            if (game_num + 1) % evaluation_interval == 0:
                self._evaluate_performance(models, players, training_stats, game_num + 1)
            
            # Update training history
            self.training_history.append({
                'game_num': game_num + 1,
                'result': game_result,
                'timestamp': time.time()
            })
        
        # Final save
        self._save_models(models, players, num_games, final=True)
        
        # Calculate final statistics
        training_time = time.time() - start_time
        final_stats = self._calculate_final_stats(training_stats, training_time)
        
        # Save training summary
        self._save_training_summary(final_stats, players, risk_profiles)
        
        print("\n" + "=" * 80)
        print("🎉 Self-play training completed!")
        print(f"⏱️  Total time: {training_time/3600:.2f} hours")
        print(f"📊 Final statistics saved to {self.output_dir}")
        print("=" * 80)
        
        return final_stats
    
    def _play_training_game(self, game_players: Dict[str, ModelPlayer], 
                           team1_players: List[str], team2_players: List[str]) -> Dict[str, Any]:
        """Play a single training game.
        
        Parameters
        ----------
        game_players : Dict[str, ModelPlayer]
            Dictionary of player instances
        team1_players : List[str]
            List of team 1 player names
        team2_players : List[str]
            List of team 2 player names
            
        Returns
        -------
        Dict[str, Any]
            Game result data
        """
        # Create game
        game = EuchreGame(enable_logging=False, quiet_mode=True)
        
        # Add players in order (team1 first, then team2)
        all_players = team1_players + team2_players
        for player_name in all_players:
            player = game_players[player_name]
            game.players.append(player)
        
        # Start game
        game.start_new_game()
        
        # Play until completion
        round_num = 1
        while not game.is_game_over():
            results = game.play_round()
            round_num += 1
        
        # Determine winner
        team1_score = game.game_state.team1_score
        team2_score = game.game_state.team2_score
        
        if team1_score > team2_score:
            winner = "team1"
            winning_players = team1_players
            losing_players = team2_players
        elif team2_score > team1_score:
            winner = "team2"
            winning_players = team2_players
            losing_players = team1_players
        else:
            winner = "tie"
            winning_players = []
            losing_players = []
        
        # Collect game data
        game_data = {
            'winner': winner,
            'team1_score': team1_score,
            'team2_score': team2_score,
            'team1_players': team1_players,
            'team2_players': team2_players,
            'winning_players': winning_players,
            'losing_players': losing_players,
            'game_length': round_num - 1,
            'player_stats': {}
        }
        
        # Collect individual player statistics
        for player_name in all_players:
            player = game_players[player_name]
            if hasattr(player, 'game_context'):
                game_data['player_stats'][player_name] = {
                    'trump_calls': player.game_context.get('trump_calls_made', 0),
                    'aces_ordered': player.game_context.get('aces_ordered', 0),
                    'times_set': player.game_context.get('times_set', 0)
                }
        
        return game_data
    
    def _update_training_stats(self, stats: Dict[str, Any], game_result: Dict[str, Any]):
        """Update training statistics with game result."""
        stats['total_games'] += 1
        stats['game_lengths'].append(game_result['game_length'])
        
        # Update win/loss counts
        for player_name in game_result['winning_players']:
            stats['player_wins'][player_name] += 1
        
        for player_name in game_result['losing_players']:
            stats['player_losses'][player_name] += 1
        
        # Update player statistics
        for player_name, player_stats in game_result['player_stats'].items():
            stats['trump_calls'][player_name] += player_stats['trump_calls']
            stats['aces_ordered'][player_name] += player_stats['aces_ordered']
            stats['times_set'][player_name] += player_stats['times_set']
    
    def _save_models(self, models: Dict[str, EuchreNN], 
                    player_names: List[str], 
                    game_num: int, 
                    final: bool = False):
        """Save trained models to files."""
        timestamp = int(time.time())
        
        for player_name in player_names:
            model = models[player_name]
            
            # Create checkpoint data
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': self.model_config,
                'player_name': player_name,
                'training_games': game_num,
                'timestamp': timestamp,
                'device': str(self.device)
            }
            
            # Save to file
            filename = f"{player_name}.json" if final else f"{player_name}_checkpoint_{game_num}.json"
            filepath = self.output_dir / filename
            
            # Convert tensors to lists for JSON serialization
            serializable_checkpoint = self._make_checkpoint_serializable(checkpoint)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_checkpoint, f, indent=2)
            
            if final:
                print(f"💾 Saved final model for {player_name}: {filepath}")
    
    def _make_checkpoint_serializable(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Convert checkpoint to JSON-serializable format."""
        serializable = {}
        
        for key, value in checkpoint.items():
            if key == 'model_state_dict':
                # Convert tensor values to lists
                serializable[key] = {}
                for param_name, param_tensor in value.items():
                    serializable[key][param_name] = param_tensor.cpu().numpy().tolist()
            else:
                serializable[key] = value
        
        return serializable
    
    def _evaluate_performance(self, models: Dict[str, EuchreNN], 
                            player_names: List[str], 
                            stats: Dict[str, Any], 
                            game_num: int):
        """Evaluate current model performance."""
        print(f"\n📊 Performance at game {game_num}:")
        
        for player_name in player_names:
            wins = stats['player_wins'][player_name]
            losses = stats['player_losses'][player_name]
            total = wins + losses
            
            if total > 0:
                win_rate = wins / total
                print(f"  {player_name}: {win_rate:.1%} win rate ({wins}/{total})")
    
    def _calculate_final_stats(self, stats: Dict[str, Any], training_time: float) -> Dict[str, Any]:
        """Calculate final training statistics."""
        final_stats = stats.copy()
        
        # Calculate win rates
        for player_name in stats['player_wins']:
            wins = stats['player_wins'][player_name]
            losses = stats['player_losses'][player_name]
            total = wins + losses
            
            if total > 0:
                final_stats[f'{player_name}_win_rate'] = wins / total
            else:
                final_stats[f'{player_name}_win_rate'] = 0.0
        
        # Calculate averages
        if stats['game_lengths']:
            final_stats['avg_game_length'] = np.mean(stats['game_lengths'])
            final_stats['std_game_length'] = np.std(stats['game_lengths'])
        
        final_stats['training_time_hours'] = training_time / 3600
        final_stats['games_per_hour'] = stats['total_games'] / final_stats['training_time_hours']
        
        return final_stats
    
    def _save_training_summary(self, stats: Dict[str, Any], 
                             players: List[str], 
                             risk_profiles: List[str]):
        """Save training summary to file."""
        summary = {
            'training_summary': {
                'total_games': stats['total_games'],
                'training_time_hours': stats['training_time_hours'],
                'games_per_hour': stats['games_per_hour'],
                'players': players,
                'risk_profiles': risk_profiles,
                'model_config': self.model_config
            },
            'player_statistics': {},
            'training_history': self.training_history[-1000:]  # Last 1000 games
        }
        
        # Add player statistics
        for player_name in players:
            summary['player_statistics'][player_name] = {
                'wins': stats['player_wins'][player_name],
                'losses': stats['player_losses'][player_name],
                'win_rate': stats[f'{player_name}_win_rate'],
                'trump_calls': stats['trump_calls'][player_name],
                'aces_ordered': stats['aces_ordered'][player_name],
                'times_set': stats['times_set'][player_name]
            }
        
        # Save summary
        summary_file = self.output_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Training summary saved to: {summary_file}")
    
    def load_trained_model(self, player_name: str) -> EuchreNN:
        """Load a trained model for a specific player.
        
        Parameters
        ----------
        player_name : str
            Name of the player to load
            
        Returns
        -------
        EuchreNN
            Loaded model instance
        """
        model_file = self.output_dir / f"{player_name}.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load checkpoint
        with open(model_file, 'r') as f:
            checkpoint = json.load(f)
        
        # Create model
        model = create_euchre_model(checkpoint['model_config'])
        model.to(self.device)
        
        # Load weights
        state_dict = {}
        for param_name, param_data in checkpoint['model_state_dict'].items():
            state_dict[param_name] = torch.tensor(param_data, device=self.device)
        
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"✅ Loaded trained model for {player_name}")
        return model
    
    def get_available_players(self) -> List[str]:
        """Get list of available trained players.
        
        Returns
        -------
        List[str]
            List of player names with trained models
        """
        available = []
        
        for file_path in self.output_dir.glob("*.json"):
            if file_path.name != "training_summary.json":
                player_name = file_path.stem
                if not player_name.endswith('_checkpoint'):
                    available.append(player_name)
        
        return sorted(available)
    
    def create_game_with_trained_players(self, 
                                       player_names: List[str],
                                       risk_profiles: List[str] = None) -> EuchreGame:
        """Create a game with trained model players.
        
        Parameters
        ----------
        player_names : List[str]
            List of player names to use
        risk_profiles : List[str]
            Risk profiles for each player (optional)
            
        Returns
        -------
        EuchreGame
            Game instance with trained players
        """
        if risk_profiles is None:
            risk_profiles = ["balanced"] * len(player_names)
        
        # Create game
        game = EuchreGame(enable_logging=True)
        
        # Add trained players
        for i, (player_name, risk_profile) in enumerate(zip(player_names, risk_profiles)):
            try:
                model = self.load_trained_model(player_name)
                game.add_model_player(player_name, model, self.device, risk_profile)
                print(f"✅ Added trained player: {player_name}")
            except FileNotFoundError:
                # Fallback to AI profile if model not found
                game.add_ai_player(player_name, risk_profile, 0.5)
                print(f"⚠️  Model not found for {player_name}, using AI profile")
        
        return game

    def train_integer_vs_float(self, 
                               num_games: int = 200000,
                               integer_players: List[str] = None,
                               float_players: List[str] = None,
                               risk_profiles: List[str] = None,
                               save_interval: int = 1000,
                               evaluation_interval: int = 5000) -> Dict[str, Any]:
        """Train integer models against float models in a mixed training scenario.
        
        Parameters
        ----------
        num_games : int
            Number of games to play
        integer_players : List[str]
            Names of integer model players
        float_players : List[str]
            Names of float model players
        risk_profiles : List[str]
            Risk profiles for each player
        save_interval : int
            How often to save model checkpoints
        evaluation_interval : int
            How often to evaluate model performance
            
        Returns
        -------
        Dict[str, Any]
            Training results and statistics
        """
        if integer_players is None:
            integer_players = ["Integer_Alice", "Integer_Bob"]
        if float_players is None:
            float_players = ["Float_Charlie", "Float_David"]
        if risk_profiles is None:
            risk_profiles = ["balanced", "aggressive", "conservative", "opportunistic"]
        
        print(f"🚀 Starting Integer vs Float training for {num_games:,} games")
        print(f"📊 Integer players: {', '.join(integer_players)}")
        print(f"📊 Float players: {', '.join(float_players)}")
        print(f"🎯 Risk profiles: {', '.join(risk_profiles)}")
        
        # Create models and players
        self.models = {}
        self.players = {}
        
        # Create integer models
        for i, player_name in enumerate(integer_players):
            risk_profile = risk_profiles[i % len(risk_profiles)]
            model, player = self.create_integer_player_model(player_name, risk_profile)
            self.models[player_name] = model
            self.players[player_name] = player
            self.players[player_name].risk_profile_name = risk_profile  # Store the risk profile name
            print(f"✅ Created integer model: {player_name} ({risk_profile})")
        
        # Create float models
        for i, player_name in enumerate(float_players):
            risk_profile = risk_profiles[(i + len(integer_players)) % len(risk_profiles)]
            model, player = self.create_player_model(player_name, risk_profile)
            self.models[player_name] = model
            self.players[player_name] = player
            self.players[player_name].risk_profile_name = risk_profile  # Store the risk profile name
            print(f"✅ Created float model: {player_name} ({risk_profile})")
        
        # Training statistics
        training_stats = {
            "integer_wins": 0,
            "float_wins": 0,
            "integer_team_wins": 0,
            "float_team_wins": 0,
            "game_results": [],
            "model_improvements": {}
        }
        
        # Training loop
        print(f"🎮 Starting training loop...")
        start_time = time.time()
        
        for game_num in tqdm(range(num_games), desc="Training games"):
            # Create teams: Integer players vs Float players
            team1 = [integer_players[0], integer_players[1]]  # Integer team
            team2 = [float_players[0], float_players[1]]      # Float team
            
            # Play game
            game = EuchreGame()
            game.add_model_player(team1[0], self.models[team1[0]], self.device, self.players[team1[0]].risk_profile_name)
            game.add_model_player(team2[0], self.models[team2[0]], self.device, self.players[team2[0]].risk_profile_name)
            game.add_model_player(team1[1], self.models[team1[1]], self.device, self.players[team1[1]].risk_profile_name)
            game.add_model_player(team2[1], self.models[team2[1]], self.device, self.players[team2[1]].risk_profile_name)
            
            # Play the complete game
            game.start_new_game()
            while not game.is_game_over():
                game.play_round()
            
            # Record results
            team1_score = game.game_state.team1_score  # Integer team
            team2_score = game.game_state.team2_score  # Float team
            
            if team1_score > team2_score:
                training_stats["integer_team_wins"] += 1
                training_stats["integer_wins"] += 1
            else:
                training_stats["float_team_wins"] += 1
                training_stats["float_wins"] += 1
            
            # Record game result
            game_result = {
                "game_num": game_num,
                "integer_team_score": team1_score,
                "float_team_score": team2_score,
                "winner": "integer" if team1_score > team2_score else "float",
                "timestamp": time.time()
            }
            training_stats["game_results"].append(game_result)
            
            # Save checkpoints periodically
            if (game_num + 1) % save_interval == 0:
                self._save_mixed_training_checkpoint(game_num + 1, training_stats)
            
            # Evaluate performance periodically
            if (game_num + 1) % evaluation_interval == 0:
                self._evaluate_mixed_training_performance(game_num + 1, training_stats)
                
                # Check GPU memory usage
                self._check_gpu_memory()
        
        # Final save
        self._save_mixed_training_checkpoint(num_games, training_stats, is_final=True)
        
        # Calculate final statistics
        total_time = time.time() - start_time
        training_stats["total_time"] = total_time
        training_stats["games_per_second"] = num_games / total_time
        training_stats["integer_win_rate"] = training_stats["integer_team_wins"] / num_games
        training_stats["float_win_rate"] = training_stats["float_team_wins"] / num_games
        
        print(f"\n🎉 Training completed!")
        print(f"⏱️  Total time: {total_time/3600:.2f} hours")
        print(f"🎮 Games per second: {training_stats['games_per_second']:.2f}")
        print(f"🏆 Integer team wins: {training_stats['integer_team_wins']:,} ({training_stats['integer_win_rate']*100:.1f}%)")
        print(f"🏆 Float team wins: {training_stats['float_team_wins']:,} ({training_stats['float_win_rate']*100:.1f}%)")
        
        return training_stats
    
    def _save_mixed_training_checkpoint(self, game_num: int, training_stats: Dict[str, Any], is_final: bool = False) -> None:
        """Save a checkpoint during mixed training."""
        checkpoint_dir = self.output_dir / "mixed_training_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save models
        for player_name, model in self.models.items():
            model_path = checkpoint_dir / f"{player_name}_checkpoint_{game_num}.json"
            self._save_model_weights(model, model_path)
        
        # Save training statistics
        stats_path = checkpoint_dir / f"training_stats_{game_num}.json"
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f, indent=2, default=str)
        
        if is_final:
            print(f"💾 Final models and stats saved to {checkpoint_dir}")
        else:
            print(f"💾 Checkpoint saved at game {game_num:,}")
    
    def _evaluate_mixed_training_performance(self, game_num: int, training_stats: Dict[str, Any]) -> None:
        """Evaluate performance during mixed training."""
        total_games = game_num
        integer_wins = training_stats["integer_team_wins"]
        float_wins = training_stats["float_team_wins"]
        
        integer_win_rate = integer_wins / total_games
        float_win_rate = float_wins / total_games
        
        print(f"\n📊 Performance at game {total_games:,}:")
        print(f"   Integer team: {integer_wins:,} wins ({integer_win_rate*100:.1f}%)")
        print(f"   Float team:   {float_wins:,} wins ({float_win_rate*100:.1f}%)")
        
        if integer_win_rate > 0.6:
            print(f"   🚀 Integer models are dominating!")
        elif float_win_rate > 0.6:
            print(f"   🚀 Float models are dominating!")
        else:
            print(f"   ⚖️  Models are well balanced")
    
    def train_single_profile(self,
                            player_name: str,
                            risk_profile: str,
                            num_games: int,
                            save_interval: int = 1000,
                            evaluation_interval: int = 2000) -> Dict[str, Any]:
        """Train a single player profile with a specific risk profile.
        
        Parameters
        ----------
        player_name : str
            Name of the player to train
        risk_profile : str
            Risk profile to use for training
        num_games : int
            Number of games to play
        save_interval : int
            How often to save checkpoints
        evaluation_interval : int
            How often to evaluate performance
            
        Returns
        -------
        Dict[str, Any]
            Training results
        """
        print(f"🧠 Training {player_name} with {risk_profile} profile")
        print(f"📊 Games: {num_games:,}")
        print(f"💾 Save interval: {save_interval}")
        print(f"📈 Evaluation interval: {evaluation_interval}")
        print(f"🖥️  Device: {self.device}")
        
        # Create the player model
        model, player = self.create_player_model(player_name, risk_profile)
        
        # Create opponent models for training
        opponent_names = ['Opponent1', 'Opponent2', 'Opponent3']
        opponents = {}
        
        for opp_name in opponent_names:
            # Use different risk profiles for variety
            opp_risk = np.random.choice(['balanced', 'conservative', 'aggressive'])
            _, opp_player = self.create_player_model(opp_name, opp_risk)
            opponents[opp_name] = opp_player
        
        # Training loop
        wins = 0
        total_games = 0
        training_history = []
        
        print(f"🎮 Training {player_name} - Each '.' = 1 hand, Each row = 1 game, '#' = game complete")
        print("=" * 60)
        
        for game_num in range(num_games):
            # Randomly select opponents
            opponent_list = list(opponents.values())
            np.random.shuffle(opponent_list)
            
            # Create game with player and 3 opponents
            game_players = {player_name: player}
            for i, opp in enumerate(opponent_list[:3]):
                game_players[f"Opponent{i+1}"] = opp
            
            # Play training game
            game_result = self._play_training_game(game_players, [player_name], list(game_players.keys())[1:])
            
            # Update statistics
            if game_result['winner'] == 'team1' and player_name in game_result.get('team1_players', []):
                wins += 1
            elif game_result['winner'] == 'team2' and player_name in game_result.get('team2_players', []):
                wins += 1
            
            total_games += 1
            current_win_rate = wins / total_games
            
            # Record training progress
            training_history.append({
                'game': game_num + 1,
                'win_rate': current_win_rate,
                'wins': wins,
                'total': total_games
            })
            
            # Show progress with dots for hands and # for game completion
            hands_played = game_result.get('game_length', 0)
            print('.' * hands_played + '#', end='', flush=True)
            
            # New line every 10 games for readability
            if (game_num + 1) % 10 == 0:
                print(f" ({game_num + 1}/{num_games})")
            elif (game_num + 1) % 5 == 0:
                print(" ", end='', flush=True)
            
            # Progress summary every 1000 games
            if (game_num + 1) % 1000 == 0:
                print(f"\n📊 Progress: {game_num + 1}/{num_games} games, Win rate: {current_win_rate:.3f}")
            
            # Save checkpoints
            if (game_num + 1) % save_interval == 0:
                checkpoint_path = self._save_profile_checkpoint(
                    model, player_name, risk_profile, game_num + 1, current_win_rate
                )
                print(f"\n💾 Checkpoint saved: {checkpoint_path}")
                
                # Check GPU memory usage
                self._check_gpu_memory()
            
            # Evaluate performance
            if (game_num + 1) % evaluation_interval == 0:
                print(f"\n📊 Game {game_num + 1}: {player_name} win rate: {current_win_rate:.3f}")
        
        # Final newline for clean output
        print()
        
        # Training summary
        print(f"\n🎯 Training Summary for {player_name}:")
        print(f"   Games played: {total_games:,}")
        print(f"   Wins: {wins:,}")
        print(f"   Losses: {total_games - wins:,}")
        print(f"   Final win rate: {final_win_rate*100:.1f}%")
        print(f"   Average game length: {sum([h['game_length'] for h in training_history[-100:]]) / min(100, len(training_history)):.1f} hands")
        
        # Final evaluation and save
        final_win_rate = wins / total_games
        final_model_path = self._save_final_profile(model, player_name, risk_profile, final_win_rate, num_games)
        
        results = {
            'player_name': player_name,
            'risk_profile': risk_profile,
            'final_win_rate': final_win_rate,
            'total_games': total_games,
            'wins': wins,
            'training_history': training_history,
            'model_path': final_model_path
        }
        
        print(f"✅ {player_name} training completed!")
        print(f"   Final win rate: {final_win_rate*100:.1f}%")
        print(f"   Model saved: {final_model_path}")
        
        return results

    def _save_profile_checkpoint(self, model: EuchreNN, player_name: str, risk_profile: str, 
                                game_num: int, win_rate: float) -> str:
        """Save a checkpoint during profile training."""
        checkpoint_dir = self.output_dir / "profile_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{player_name}_{risk_profile}_checkpoint_{game_num}.json"
        self._save_model_weights(model, checkpoint_path)
        
        return str(checkpoint_path)

    def _save_final_profile(self, model: EuchreNN, player_name: str, risk_profile: str, 
                           win_rate: float, total_games: int) -> str:
        """Save the final trained profile."""
        final_path = self.output_dir / f"{player_name}.json"
        self._save_model_weights(model, final_path)
        
        return str(final_path)
    
    def _save_model_weights(self, model, model_path: Path) -> None:
        """Save model weights to a JSON file."""
        try:
            # Convert model state dict to JSON-serializable format
            state_dict = model.state_dict()
            json_state = {}
            
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    json_state[key] = tensor.cpu().numpy().tolist()
                else:
                    json_state[key] = tensor
            
            with open(model_path, 'w') as f:
                json.dump(json_state, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save model weights: {e}")


def main():
    """Main entry point for self-play training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train euchre AI models using self-play")
    parser.add_argument("--num-games", type=int, default=100000, help="Number of training games")
    parser.add_argument("--players", nargs="+", default=["Alice", "Bob", "Charlie", "David"], 
                       help="Player names")
    parser.add_argument("--risk-profiles", nargs="+", 
                       default=["balanced", "balanced", "balanced", "balanced"],
                       help="Risk profiles for each player")
    parser.add_argument("--output-dir", default="trained_models", help="Output directory")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--eval-interval", type=int, default=5000, help="Evaluation interval")
    parser.add_argument("--device", default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Model configuration
    model_config = {
        'type': 'standard',
        'input_size': 128,
        'hidden_size': 256,
        'output_size': 64,
        'risk_embedding_size': 32,
        'use_risk_attention': True
    }
    
    # Create trainer
    trainer = SelfPlayTrainer(model_config, args.output_dir, args.device)
    
    # Start training
    results = trainer.train_self_play(
        num_games=args.num_games,
        players=args.players,
        risk_profiles=args.risk_profiles,
        save_interval=args.save_interval,
        evaluation_interval=args.eval_interval
    )
    
    print("🎯 Training completed successfully!")


if __name__ == "__main__":
    main() 