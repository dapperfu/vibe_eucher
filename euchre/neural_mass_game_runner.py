"""Neural network mass game runner for euchre games with trained AI models."""

import os
import uuid
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import multiprocessing as mp
import logging

from .game import EuchreGame
from .ai_model.player_manager import PlayerManager
from .ai_model.model_player import ModelPlayer


class NeuralMassGameRunner:
    """Runs thousands of euchre games with neural network AI players and tracks detailed statistics."""
    
    def __init__(self, models_dir: str = "demo_models", output_dir: str = "neural_games", 
                 max_workers: Optional[int] = None) -> None:
        """Initialize the neural mass game runner.
        
        Parameters
        ----------
        models_dir : str
            Directory containing trained model files
        output_dir : str
            Directory to save game results
        max_workers : Optional[int]
            Maximum number of parallel processes (defaults to CPU count)
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if max_workers is None:
            self.max_workers = mp.cpu_count()
        else:
            self.max_workers = max_workers
        
        # Initialize player manager
        self.player_manager = PlayerManager(str(self.models_dir))
        
        # Get available models
        self.available_models = self.player_manager.get_available_players()
        
        logging.info(f"Neural mass game runner initialized with {len(self.available_models)} available models")
        logging.info(f"Available models: {self.available_models}")
        
    def create_team_config(self, team1_models: List[str], team2_models: List[str], 
                          config_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a team configuration with specific neural network models.
        
        Parameters
        ----------
        team1_models : List[str]
            List of model names for team 1 (North/South)
        team2_models : List[str]
            List of model names for team 2 (East/West)
        config_name : Optional[str]
            Name for this configuration
            
        Returns
        -------
        Dict[str, Any]
            Team configuration dictionary
        """
        if config_name is None:
            config_name = f"{'_'.join(team1_models)}_vs_{'_'.join(team2_models)}"
        
        # Validate that all models exist
        all_models = team1_models + team2_models
        missing_models = [model for model in all_models if model not in self.available_models]
        if missing_models:
            raise ValueError(f"Models not found: {missing_models}. Available: {self.available_models}")
        
        config = {
            "config_name": config_name,
            "team1": [
                ("North", team1_models[0]),
                ("South", team1_models[1] if len(team1_models) > 1 else team1_models[0])
            ],
            "team2": [
                ("East", team2_models[0]),
                ("West", team2_models[1] if len(team2_models) > 1 else team2_models[0])
            ]
        }
        
        return config
    
    def _run_single_neural_game(self, game_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single game with neural network AI players.
        
        Parameters
        ----------
        game_config : Dict[str, Any]
            Configuration for the game including team setups
            
        Returns
        -------
        Dict[str, Any]
            Game results and detailed metadata
        """
        try:
            start_time = time.time()
            
            # Create game without logging (we'll handle that ourselves)
            game = EuchreGame(enable_logging=False)
            
            # Add neural network players
            team1_players = game_config["team1"]
            team2_players = game_config["team2"]
            
            # Load models and create players
            for position, model_name in team1_players + team2_players:
                try:
                    # Load the model
                    model = self.player_manager.load_player_model(model_name)
                    if model is None:
                        raise ValueError(f"Failed to load model: {model_name}")
                    
                    # Create model player
                    model_player = ModelPlayer(name=position, model=model, device="cpu")
                    game.add_player(position, model_player)
                    
                except Exception as e:
                    raise ValueError(f"Failed to create player {position} with model {model_name}: {e}")
            
            # Start the game
            game.start_new_game()
            
            # Track detailed game progress
            rounds_data = []
            round_num = 1
            trick_history = []
            
            # Play rounds until game ends
            while not game.is_game_over():
                # Play the round
                results = game.play_round()
                
                # Get detailed trick information for this round
                round_tricks = []
                for trick_num in range(5):  # Each round has 5 tricks
                    if hasattr(game, 'current_trick') and game.current_trick:
                        trick_cards = game.current_trick.get_cards_played()
                        trick_winner = game.current_trick.get_winner()
                        round_tricks.append({
                            "trick_num": trick_num + 1,
                            "cards_played": [str(card) for card in trick_cards],
                            "winner": trick_winner.name if trick_winner else None,
                            "winner_team": "team1" if trick_winner and trick_winner.name in ["North", "South"] else "team2"
                        })
                
                # Record round data
                round_data = {
                    "round": round_num,
                    "trump_suit": game.game_state.trump_suit.name if game.game_state.trump_suit else None,
                    "trump_caller": game.trump_caller.name if game.trump_caller else None,
                    "trump_caller_team": "team1" if game.trump_caller_team == 0 else "team2",
                    "trick_counts": results,
                    "team1_score": game.game_state.team1_score,
                    "team2_score": game.game_state.team2_score,
                    "tricks": round_tricks
                }
                rounds_data.append(round_data)
                round_num += 1
            
            # Get final results
            winner = game.get_winner()
            final_score_team1 = game.game_state.team1_score
            final_score_team2 = game.game_state.team2_score
            
            # Determine which team won
            if final_score_team1 > final_score_team2:
                winning_team = "team1"
            elif final_score_team2 > final_score_team1:
                winning_team = "team2"
            else:
                winning_team = "tie"
            
            # Calculate detailed statistics
            team1_tricks_total = sum(round_data["trick_counts"][0] + round_data["trick_counts"][2] 
                                   for round_data in rounds_data)
            team2_tricks_total = sum(round_data["trick_counts"][1] + round_data["trick_counts"][3] 
                                   for round_data in rounds_data)
            
            # Check if team was set
            team_set = game.is_team_set()
            set_team = None
            if team_set:
                if game.trump_caller_team == 0:
                    set_team = "team1"
                else:
                    set_team = "team2"
            
            # Compile comprehensive game results
            game_results = {
                "game_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "team_config": game_config["config_name"],
                "team1": {
                    "players": team1_players,
                    "final_score": final_score_team1,
                    "total_tricks": team1_tricks_total,
                    "tricks_per_round": [round_data["trick_counts"][0] + round_data["trick_counts"][2] 
                                        for round_data in rounds_data]
                },
                "team2": {
                    "players": team2_players,
                    "final_score": final_score_team2,
                    "total_tricks": team2_tricks_total,
                    "tricks_per_round": [round_data["trick_counts"][1] + round_data["trick_counts"][3] 
                                        for round_data in rounds_data]
                },
                "winner": winning_team,
                "total_rounds": len(rounds_data),
                "rounds": rounds_data,
                "game_duration": time.time() - start_time,
                "team_set": team_set,
                "set_team": set_team,
                "total_tricks_team1": team1_tricks_total,
                "total_tricks_team2": team2_tricks_total
            }
            
            return game_results
            
        except Exception as e:
            # Return error information
            return {
                "game_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "error": str(e),
                "team_config": game_config.get("config_name", "unknown"),
                "traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else None
            }
    
    def run_neural_games(self, num_games: int, team1_models: List[str], 
                         team2_models: List[str], config_name: Optional[str] = None) -> None:
        """Run multiple games with neural network AI players.
        
        Parameters
        ----------
        num_games : int
            Number of games to run
        team1_models : List[str]
            List of model names for team 1 (North/South)
        team2_models : List[str]
            List of model names for team 2 (East/West)
        config_name : Optional[str]
            Name for this configuration
        """
        # Create team configuration
        config = self.create_team_config(team1_models, team2_models, config_name)
        
        print(f"Starting {num_games} neural network games")
        print(f"Configuration: {config['config_name']}")
        print(f"Team 1 (North/South): {[f'{p[0]} ({p[1]})' for p in config['team1']]}")
        print(f"Team 2 (East/West): {[f'{p[0]} ({p[1]})' for p in config['team2']]}")
        print(f"Using {self.max_workers} parallel workers")
        print(f"Models directory: {self.models_dir}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 80)
        
        # Prepare game configurations
        game_configs = []
        for i in range(num_games):
            game_config = config.copy()
            game_configs.append(game_config)
        
        # Run games in parallel
        completed_games = 0
        successful_games = 0
        failed_games = 0
        
        # Statistics tracking
        team1_wins = 0
        team2_wins = 0
        ties = 0
        team1_total_tricks = 0
        team2_total_tricks = 0
        team_sets = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all games
            future_to_config = {
                executor.submit(self._run_single_neural_game, config): config 
                for config in game_configs
            }
            
            # Process completed games
            for future in as_completed(future_to_config):
                completed_games += 1
                result = future.result()
                
                # Save result to file
                filename = f"{result['game_id']}.json"
                filepath = self.output_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Track progress and statistics
                if "error" in result:
                    failed_games += 1
                    print(f"Game {completed_games}/{num_games}: FAILED - {result['error']}")
                else:
                    successful_games += 1
                    winner = result["winner"]
                    score1 = result["team1"]["final_score"]
                    score2 = result["team2"]["final_score"]
                    rounds = result["total_rounds"]
                    tricks1 = result["total_tricks_team1"]
                    tricks2 = result["total_tricks_team2"]
                    
                    # Update statistics
                    if winner == "team1":
                        team1_wins += 1
                    elif winner == "team2":
                        team2_wins += 1
                    else:
                        ties += 1
                    
                    team1_total_tricks += tricks1
                    team2_total_tricks += tricks2
                    
                    if result["team_set"]:
                        team_sets += 1
                    
                    print(f"Game {completed_games}/{num_games}: {winner.upper()} wins "
                          f"({score1}-{score2}) in {rounds} rounds, "
                          f"Tricks: {tricks1}-{tricks2}")
                
                # Progress update every 100 games
                if completed_games % 100 == 0:
                    success_rate = (successful_games / completed_games) * 100
                    print(f"Progress: {completed_games}/{num_games} games completed "
                          f"({successful_games} successful, {failed_games} failed, {success_rate:.1f}% success)")
        
        # Print final statistics
        print("-" * 80)
        print(f"Completed {num_games} neural network games!")
        print(f"Successful: {successful_games}, Failed: {failed_games}")
        print(f"Team 1 wins: {team1_wins}, Team 2 wins: {team2_wins}, Ties: {ties}")
        print(f"Team 1 total tricks: {team1_total_tricks}, Team 2 total tricks: {team2_total_tricks}")
        print(f"Team sets: {team_sets}")
        print(f"Results saved to: {self.output_dir}")
        
        # Save summary statistics
        summary = {
            "config_name": config["config_name"],
            "total_games": num_games,
            "successful_games": successful_games,
            "failed_games": failed_games,
            "team1_wins": team1_wins,
            "team2_wins": team2_wins,
            "ties": ties,
            "team1_total_tricks": team1_total_tricks,
            "team2_total_tricks": team2_total_tricks,
            "team_sets": team_sets,
            "timestamp": time.time()
        }
        
        summary_file = self.output_dir / f"summary_{config['config_name']}_{int(time.time())}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_file}")
    
    def run_model_vs_model(self, model1: str, model2: str, num_games: int = 10000) -> None:
        """Run games between two specific models (each model plays both positions on their team).
        
        Parameters
        ----------
        model1 : str
            First model name (plays North/South)
        model2 : str
            Second model name (plays East/West)
        num_games : int
            Number of games to run
        """
        print(f"Running {num_games} games: {model1} vs {model2}")
        print(f"Team 1: {model1} (North/South)")
        print(f"Team 2: {model2} (East/West)")
        print("=" * 80)
        
        self.run_neural_games(num_games, [model1, model1], [model2, model2], 
                             f"{model1}_vs_{model2}")
    
    def run_all_model_combinations(self, num_games: int = 1000) -> None:
        """Run games for all possible model combinations.
        
        Parameters
        ----------
        num_games : int
            Number of games to run for each combination
        """
        if len(self.available_models) < 2:
            print("Need at least 2 models to run combinations")
            return
        
        print(f"Running {num_games} games for all model combinations")
        print(f"Available models: {self.available_models}")
        print("=" * 80)
        
        # Generate all unique combinations
        combinations = []
        for i, model1 in enumerate(self.available_models):
            for j, model2 in enumerate(self.available_models):
                if i < j:  # Avoid duplicates and self vs self
                    combinations.append((model1, model2))
        
        print(f"Found {len(combinations)} unique combinations")
        
        for model1, model2 in combinations:
            print(f"\nRunning combination: {model1} vs {model2}")
            self.run_model_vs_model(model1, model2, num_games)
            print(f"Completed combination: {model1} vs {model2}")
            print("=" * 80)
    
    def get_game_files(self) -> List[Path]:
        """Get list of all game result files.
        
        Returns
        -------
        List[Path]
            List of game result file paths
        """
        return list(self.output_dir.glob("*.json"))
    
    def get_game_count(self) -> int:
        """Get total number of completed games.
        
        Returns
        -------
        int
            Number of game result files
        """
        return len(self.get_game_files())
    
    def cleanup_old_games(self, older_than_days: int = 7) -> int:
        """Remove game files older than specified days.
        
        Parameters
        ----------
        older_than_days : int
            Remove files older than this many days
            
        Returns
        -------
        int
            Number of files removed
        """
        import datetime
        
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
        removed_count = 0
        
        for game_file in self.get_game_files():
            try:
                with open(game_file, 'r') as f:
                    data = json.load(f)
                    if data.get('timestamp', 0) < cutoff_time:
                        game_file.unlink()
                        removed_count += 1
            except (json.JSONDecodeError, KeyError):
                # Remove corrupted files
                game_file.unlink()
                removed_count += 1
        
        return removed_count 