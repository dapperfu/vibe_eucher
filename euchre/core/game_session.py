"""
Game session management for running multiple euchre games.

This module provides classes for managing game sessions, tournaments,
and statistical analysis across multiple games.
"""

from typing import List, Dict, Any, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
from pathlib import Path

from ..models import Player, Card, Suit
from ..utils.logging_config import GameLogger

if TYPE_CHECKING:
    from ..game import EuchreGame


@dataclass
class GameResult:
    """Results from a single game."""
    
    game_id: str
    timestamp: datetime
    players: List[str]
    winner: str
    winning_team: str
    final_scores: Dict[str, int]
    rounds_played: int
    total_tricks: int
    trump_suits_called: List[str]
    game_duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "game_id": self.game_id,
            "timestamp": self.timestamp.isoformat(),
            "players": self.players,
            "winner": self.winner,
            "winning_team": self.winning_team,
            "final_scores": self.final_scores,
            "rounds_played": self.rounds_played,
            "total_tricks": self.total_tricks,
            "trump_suits_called": self.trump_suits_called,
            "game_duration_seconds": self.game_duration_seconds
        }


@dataclass
class SessionConfig:
    """Configuration for a game session."""
    
    num_games: int = 100
    save_results: bool = True
    output_dir: str = "game_sessions"
    log_level: str = "INFO"
    parallel_games: bool = False
    max_workers: Optional[int] = None
    game_callbacks: List[Callable[["GameResult"], None]] = field(default_factory=list)
    
    def validate(self) -> None:
        """Validate the session configuration."""
        if self.num_games <= 0:
            raise ValueError("num_games must be positive")
        if self.max_workers is not None and self.max_workers <= 0:
            raise ValueError("max_workers must be positive")


class GameSession:
    """Manages a session of multiple euchre games."""
    
    def __init__(self, config: SessionConfig) -> None:
        """Initialize the game session.
        
        Parameters
        ----------
        config : SessionConfig
            Configuration for the session
        """
        self.config = config
        self.config.validate()
        
        # Initialize logging
        verbose = config.log_level in ["DEBUG", "VERBOSE"]
        very_verbose = config.log_level == "DEBUG"
        self.logger = GameLogger(verbose, very_verbose)
        
        # Session state
        self.session_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        self.results: List[GameResult] = []
        self.current_game = 0
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Game session {self.session_id} initialized")
        self.logger.info(f"Configuration: {config.num_games} games, parallel={config.parallel_games}")
    
    def run_session(self, player_factory: Callable[[], List[Player]]) -> List[GameResult]:
        """Run the complete game session.
        
        Parameters
        ----------
        player_factory : Callable[[], List[Player]]
            Factory function to create players for each game
            
        Returns
        -------
        List[GameResult]
            Results from all games in the session
        """
        self.logger.info(f"Starting game session with {self.config.num_games} games")
        
        if self.config.parallel_games:
            return self._run_parallel_session(player_factory)
        else:
            return self._run_sequential_session(player_factory)
    
    def _run_sequential_session(self, player_factory: Callable[[], List[Player]]) -> List[GameResult]:
        """Run games sequentially."""
        for game_num in range(self.config.num_games):
            self.current_game = game_num + 1
            self.logger.info(f"Starting game {self.current_game}/{self.config.num_games}")
            
            try:
                result = self._run_single_game(player_factory())
                self.results.append(result)
                
                # Execute callbacks
                for callback in self.config.game_callbacks:
                    callback(result)
                
                self.logger.info(f"Game {self.current_game} completed: {result.winner} won")
                
            except Exception as e:
                self.logger.error(f"Game {self.current_game} failed: {e}")
                continue
        
        return self.results
    
    def _run_parallel_session(self, player_factory: Callable[[], List[Player]]) -> List[GameResult]:
        """Run games in parallel."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        
        max_workers = self.config.max_workers or mp.cpu_count()
        self.logger.info(f"Running {self.config.num_games} games in parallel with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all games
            future_to_game = {}
            for game_num in range(self.config.num_games):
                future = executor.submit(self._run_single_game_worker, game_num, player_factory())
                future_to_game[future] = game_num
            
            # Collect results as they complete
            for future in as_completed(future_to_game):
                game_num = future_to_game[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    self.logger.info(f"Game {game_num + 1} completed: {result.winner} won")
                except Exception as e:
                    self.logger.error(f"Game {game_num + 1} failed: {e}")
        
        return self.results
    
    def _run_single_game(self, players: List[Player]) -> "GameResult":
        """Run a single game and return the result."""
        import time
        
        start_time = time.time()
        
        # Create and run the game
        from ..game import EuchreGame
        game = EuchreGame(players, quiet_mode=True, verbose=False, very_verbose=False)
        game.start_new_game()
        game.run_full_game()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Extract game data
        game_id = str(uuid.uuid4())
        winner = game.game_state_manager.get_winner()
        final_scores = game.game_state_manager.game_scores
        
        # Create result
        result = GameResult(
            game_id=game_id,
            timestamp=datetime.now(),
            players=[p.name for p in players],
            winner=winner,
            winning_team=winner,
            final_scores=final_scores,
            rounds_played=game.round_number,
            total_tricks=game.round_number * 5,
            trump_suits_called=[],  # Would need to track this in the game
            game_duration_seconds=duration
        )
        
        return result
    
    def _run_single_game_worker(self, game_num: int, players: List[Player]) -> "GameResult":
        """Worker function for parallel game execution."""
        return self._run_single_game(players)
    
    def save_results(self) -> Path:
        """Save session results to file.
        
        Returns
        -------
        Path
            Path to the saved results file
        """
        if not self.config.save_results:
            return Path()
        
        # Create session summary
        session_summary = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_games": len(self.results),
            "config": {
                "num_games": self.config.num_games,
                "parallel_games": self.config.parallel_games,
                "log_level": self.config.log_level
            },
            "results": [result.to_dict() for result in self.results]
        }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{self.session_id[:8]}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(session_summary, f, indent=2)
        
        self.logger.info(f"Session results saved to {filepath}")
        return filepath
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from the session results.
        
        Returns
        -------
        Dict[str, Any]
            Session statistics
        """
        if not self.results:
            return {}
        
        # Player statistics
        player_stats = {}
        for result in self.results:
            for player in result.players:
                if player not in player_stats:
                    player_stats[player] = {"games_played": 0, "wins": 0}
                player_stats[player]["games_played"] += 1
                if result.winner == player:
                    player_stats[player]["wins"] += 1
        
        # Calculate win rates
        for player, stats in player_stats.items():
            stats["win_rate"] = stats["wins"] / stats["games_played"]
        
        # Team statistics
        team_stats = {"Team 1": 0, "Team 2": 0}
        for result in self.results:
            if result.winning_team == "Team 1":
                team_stats["Team 1"] += 1
            else:
                team_stats["Team 2"] += 1
        
        # Game duration statistics
        durations = [r.game_duration_seconds for r in self.results]
        
        return {
            "total_games": len(self.results),
            "player_statistics": player_stats,
            "team_statistics": team_stats,
            "average_game_duration": sum(durations) / len(durations),
            "fastest_game": min(durations),
            "slowest_game": max(durations),
            "average_rounds_per_game": sum(r.rounds_played for r in self.results) / len(self.results)
        }
    
    def print_summary(self) -> None:
        """Print a summary of the session results."""
        if not self.results:
            self.logger.info("No games completed yet")
            return
        
        stats = self.get_statistics()
        
        self.logger.info(f"\n=== SESSION SUMMARY ===")
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(f"Total Games: {stats['total_games']}")
        self.logger.info(f"Average Duration: {stats['average_game_duration']:.2f}s")
        self.logger.info(f"Average Rounds: {stats['average_rounds_per_game']:.1f}")
        
        self.logger.info(f"\nPlayer Statistics:")
        for player, player_stats in stats["player_statistics"].items():
            win_rate = player_stats["win_rate"] * 100
            self.logger.info(f"  {player}: {win_rate:.1f}% win rate ({player_stats['wins']}/{player_stats['games_played']})")
        
        self.logger.info(f"\nTeam Statistics:")
        for team, wins in stats["team_statistics"].items():
            win_rate = (wins / stats["total_games"]) * 100
            self.logger.info(f"  {team}: {win_rate:.1f}% win rate ({wins}/{stats['total_games']})")


class TournamentSession(GameSession):
    """Specialized session for tournament-style play."""
    
    def __init__(self, config: SessionConfig, tournament_name: str) -> None:
        """Initialize tournament session.
        
        Parameters
        ----------
        config : SessionConfig
            Configuration for the session
        tournament_name : str
            Name of the tournament
        """
        super().__init__(config)
        self.tournament_name = tournament_name
        self.brackets: List[List[str]] = []
        self.championship_results: List[GameResult] = []
    
    def setup_brackets(self, players: List[str], bracket_size: int = 4) -> None:
        """Setup tournament brackets.
        
        Parameters
        ----------
        players : List[str]
            List of player names
        bracket_size : int
            Number of players per bracket
        """
        self.brackets = []
        for i in range(0, len(players), bracket_size):
            bracket = players[i:i + bracket_size]
            self.brackets.append(bracket)
        
        self.logger.info(f"Tournament brackets created: {len(self.brackets)} brackets of {bracket_size} players")
    
    def run_tournament(self, player_factory: Callable[[], List[Player]]) -> List[GameResult]:
        """Run the tournament with bracket play."""
        self.logger.info(f"Starting tournament: {self.tournament_name}")
        
        # Run bracket games
        for bracket_num, bracket in enumerate(self.brackets):
            self.logger.info(f"Running bracket {bracket_num + 1}: {bracket}")
            
            # Create players for this bracket
            players = player_factory()
            bracket_players = [p for p in players if p.name in bracket]
            
            # Run games for this bracket
            for game_num in range(self.config.num_games // len(self.brackets)):
                result = self._run_single_game(bracket_players)
                self.results.append(result)
        
        return self.results 