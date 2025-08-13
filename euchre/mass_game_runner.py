"""Mass parallel game runner for euchre games."""

import os
import uuid
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import multiprocessing as mp

from .game import EuchreGame
from .ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI


class MassGameRunner:
    """Runs thousands of euchre games in parallel and saves results."""
    
    def __init__(self, output_dir: str = "games", max_workers: Optional[int] = None) -> None:
        """Initialize the mass game runner.
        
        Parameters
        ----------
        output_dir : str
            Directory to save game results
        max_workers : Optional[int]
            Maximum number of parallel processes (defaults to CPU count)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if max_workers is None:
            self.max_workers = mp.cpu_count()
        else:
            self.max_workers = max_workers
            
        # Game configurations
        self.team_configs = {
            "aggressive_vs_conservative": {
                "team1": [("Alice", "aggressive", 0.8), ("Bob", "aggressive", 0.8)],
                "team2": [("Charlie", "conservative", 0.2), ("David", "conservative", 0.2)]
            },
            "aggressive_vs_balanced": {
                "team1": [("Alice", "aggressive", 0.8), ("Bob", "aggressive", 0.8)],
                "team2": [("Charlie", "balanced", 0.5), ("David", "balanced", 0.5)]
            },
            "conservative_vs_balanced": {
                "team1": [("Alice", "conservative", 0.2), ("Bob", "conservative", 0.2)],
                "team2": [("Charlie", "balanced", 0.5), ("David", "balanced", 0.5)]
            },
            "opportunistic_vs_aggressive": {
                "team1": [("Alice", "opportunistic", 0.7), ("Bob", "opportunistic", 0.7)],
                "team2": [("Charlie", "aggressive", 0.8), ("David", "aggressive", 0.8)]
            },
            "mixed_vs_mixed": {
                "team1": [("Alice", "aggressive", 0.8), ("Bob", "conservative", 0.2)],
                "team2": [("Charlie", "balanced", 0.5), ("David", "opportunistic", 0.7)]
            }
        }
        
    def _run_single_game(self, game_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single game with the given configuration.
        
        Parameters
        ----------
        game_config : Dict[str, Any]
            Configuration for the game including team setups
            
        Returns
        -------
        Dict[str, Any]
            Game results and metadata
        """
        try:
            # Create game without logging (we'll handle that ourselves)
            game = EuchreGame(quiet_mode=True)
            
            # Add players based on configuration
            team1_players = game_config["team1"]
            team2_players = game_config["team2"]
            
            for name, profile, risk in team1_players + team2_players:
                game.add_ai_player(name, profile, risk)
            
            # Start the game
            game.start_new_game()
            
            # Track game progress
            rounds_data = []
            round_num = 1
            
            # Play rounds until game ends
            while not game.is_game_over():
                results = game.play_round()
                
                # Record round data
                round_data = {
                    "round": round_num,
                    "trump_suit": game.game_state.trump_suit.name if game.game_state.trump_suit else None,
                    "trick_counts": results,
                    "team1_score": game.game_state.team1_score,
                    "team2_score": game.game_state.team2_score
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
            
            # Compile game results
            game_results = {
                "game_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "team_config": game_config["config_name"],
                "team1": {
                    "players": team1_players,
                    "final_score": final_score_team1
                },
                "team2": {
                    "players": team2_players,
                    "final_score": final_score_team2
                },
                "winner": winning_team,
                "total_rounds": len(rounds_data),
                "rounds": rounds_data,
                "game_duration": time.time() - game_config.get("start_time", time.time())
            }
            
            return game_results
            
        except Exception as e:
            # Return error information
            return {
                "game_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "error": str(e),
                "team_config": game_config.get("config_name", "unknown")
            }
    
    def run_games(self, num_games: int, config_name: str = "aggressive_vs_conservative") -> None:
        """Run multiple games with the specified configuration.
        
        Parameters
        ----------
        num_games : int
            Number of games to run
        config_name : str
            Name of the team configuration to use
        """
        if config_name not in self.team_configs:
            raise ValueError(f"Unknown configuration: {config_name}. Available: {list(self.team_configs.keys())}")
        
        config = self.team_configs[config_name]
        config["config_name"] = config_name
        
        print(f"Starting {num_games} games with configuration: {config_name}")
        print(f"Team 1: {[f'{p[0]} ({p[1]}, risk={p[2]})' for p in config['team1']]}")
        print(f"Team 2: {[f'{p[0]} ({p[1]}, risk={p[2]})' for p in config['team2']]}")
        print(f"Using {self.max_workers} parallel workers")
        print("-" * 80)
        
        # Prepare game configurations
        game_configs = []
        for i in range(num_games):
            game_config = config.copy()
            game_config["start_time"] = time.time()
            game_configs.append(game_config)
        
        # Run games in parallel
        completed_games = 0
        successful_games = 0
        failed_games = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all games
            future_to_config = {
                executor.submit(self._run_single_game, config): config 
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
                
                # Track progress
                if "error" in result:
                    failed_games += 1
                    print(f"Game {completed_games}/{num_games}: FAILED - {result['error']}")
                else:
                    successful_games += 1
                    winner = result["winner"]
                    score1 = result["team1"]["final_score"]
                    score2 = result["team2"]["final_score"]
                    rounds = result["total_rounds"]
                    
                    print(f"Game {completed_games}/{num_games}: {winner.upper()} wins "
                          f"({score1}-{score2}) in {rounds} rounds")
                
                # Progress update every 100 games
                if completed_games % 100 == 0:
                    print(f"Progress: {completed_games}/{num_games} games completed "
                          f"({successful_games} successful, {failed_games} failed)")
        
        print("-" * 80)
        print(f"Completed {num_games} games!")
        print(f"Successful: {successful_games}")
        print(f"Failed: {failed_games}")
        print(f"Results saved to: {self.output_dir}")
    
    def run_all_configurations(self, games_per_config: int = 1000) -> None:
        """Run games for all available team configurations.
        
        Parameters
        ----------
        games_per_config : int
            Number of games to run for each configuration
        """
        print(f"Running {games_per_config} games for each of {len(self.team_configs)} configurations")
        print("=" * 80)
        
        for config_name in self.team_configs.keys():
            print(f"\nRunning configuration: {config_name}")
            self.run_games(games_per_config, config_name)
            print(f"Completed configuration: {config_name}")
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