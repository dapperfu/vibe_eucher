"""AI Training Framework for Euchre AI Players.

This module provides a comprehensive training and evaluation system for AI players
with different team configurations and weighted scoring systems.
"""

import random
import time
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path

from .game import EuchreGame
from .models import PlayerType, Suit
from .ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI


class TrainingMode(Enum):
    """Different training modes for AI players."""
    SAME_MODEL = "same_model"  # 4 Alice players
    TEAM_VS_TEAM = "team_vs_team"  # 2 Alice vs 2 Bob
    MIXED_TEAMS = "mixed_teams"  # Alice/Bob vs Charlie/David
    ROUND_ROBIN = "round_robin"  # All combinations


@dataclass
class TrainingConfig:
    """Configuration for AI training sessions."""
    mode: TrainingMode
    num_games: int = 100
    ai_types: List[str] = None
    risk_ratios: List[float] = None
    enable_logging: bool = False
    quiet_mode: bool = True
    output_dir: str = "training_results"
    
    def __post_init__(self):
        if self.ai_types is None:
            self.ai_types = ["balanced", "balanced", "balanced", "balanced"]
        if self.risk_ratios is None:
            self.risk_ratios = [0.5, 0.5, 0.5, 0.5]


@dataclass
class GameResult:
    """Results from a single training game."""
    game_id: str
    config: str
    winner: str
    team1_score: int
    team2_score: int
    team_set: bool
    reneges: int
    total_tricks: int
    trump_calls: List[str]
    duration_seconds: float
    timestamp: str


class WeightedScoring:
    """Weighted scoring system with penalties for reneging and being set."""
    
    def __init__(self):
        # Base scoring weights
        self.win_game_weight = 10.0
        self.lose_game_weight = -5.0
        
        # Penalty weights
        self.team_set_penalty = -15.0  # Heavy penalty for being set
        self.renege_penalty = -25.0    # Very heavy penalty for reneging
        self.lose_after_calling_trump = -8.0  # Penalty for losing after calling trump
        
        # Bonus weights
        self.sweep_bonus = 5.0  # Bonus for winning all 5 tricks
        self.euchre_bonus = 8.0  # Bonus for euchring opponent (winning 5 tricks after they call trump)
        
    def calculate_score(self, game_result: GameResult, player_name: str) -> float:
        """Calculate weighted score for a player based on game results."""
        score = 0.0
        
        # Determine if player won the game
        if game_result.winner == "Team 1":
            if player_name in ["Alice", "Charlie"]:  # Team 1 players
                score += self.win_game_weight
            else:
                score += self.lose_game_weight
        else:  # Team 2 wins
            if player_name in ["Bob", "David"]:  # Team 2 players
                score += self.win_game_weight
            else:
                score += self.lose_game_weight
        
        # Apply penalties
        if game_result.team_set:
            # Find which team was set
            if game_result.team1_score < 3:  # Team 1 was set
                if player_name in ["Alice", "Charlie"]:
                    score += self.team_set_penalty
            else:  # Team 2 was set
                if player_name in ["Bob", "David"]:
                    score += self.team_set_penalty
        
        # Apply renege penalties (if any)
        if game_result.reneges > 0:
            score += self.renege_penalty * game_result.reneges
        
        # Apply sweep bonus
        if game_result.team1_score == 5:
            if player_name in ["Alice", "Charlie"]:
                score += self.sweep_bonus
        elif game_result.team2_score == 5:
            if player_name in ["Bob", "David"]:
                score += self.sweep_bonus
        
        # Apply euchre bonus
        if game_result.team_set:
            # The team that got euchred gets penalty, opponents get bonus
            if game_result.team1_score < 3:  # Team 1 was euchred
                if player_name in ["Bob", "David"]:
                    score += self.euchre_bonus
            else:  # Team 2 was euchred
                if player_name in ["Alice", "Charlie"]:
                    score += self.euchre_bonus
        
        return score


class AITrainingFramework:
    """Main framework for training and evaluating AI players."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the training framework.
        
        Parameters
        ----------
        config : TrainingConfig
            Configuration for the training session
        """
        self.config = config
        self.scoring = WeightedScoring()
        self.results: List[GameResult] = []
        self.player_stats: Dict[str, Dict] = {}
        
        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
        
    def create_team_configuration(self, mode: TrainingMode) -> List[Tuple[str, str, PlayerType, str, float]]:
        """Create team configuration based on training mode.
        
        Parameters
        ----------
        mode : TrainingMode
            The training mode to use
            
        Returns
        -------
        List[Tuple[str, str, PlayerType, str, float]]
            List of (name, display_name, player_type, ai_type, risk_ratio) tuples
        """
        if mode == TrainingMode.SAME_MODEL:
            # 4 Alice players with different AI profiles
            return [
                ("Alice_1", "Alice", PlayerType.AI, self.config.ai_types[0], self.config.risk_ratios[0]),
                ("Alice_2", "Alice", PlayerType.AI, self.config.ai_types[1], self.config.risk_ratios[1]),
                ("Alice_3", "Alice", PlayerType.AI, self.config.ai_types[2], self.config.risk_ratios[2]),
                ("Alice_4", "Alice", PlayerType.AI, self.config.ai_types[3], self.config.risk_ratios[3]),
            ]
            
        elif mode == TrainingMode.TEAM_VS_TEAM:
            # 2 Alice vs 2 Bob
            return [
                ("Alice_1", "Alice", PlayerType.AI, self.config.ai_types[0], self.config.risk_ratios[0]),
                ("Alice_2", "Alice", PlayerType.AI, self.config.ai_types[1], self.config.risk_ratios[1]),
                ("Bob_1", "Bob", PlayerType.AI, self.config.ai_types[2], self.config.risk_ratios[2]),
                ("Bob_2", "Bob", PlayerType.AI, self.config.ai_types[3], self.config.risk_ratios[3]),
            ]
            
        elif mode == TrainingMode.MIXED_TEAMS:
            # Alice/Bob vs Charlie/David
            return [
                ("Alice", "Alice", PlayerType.AI, self.config.ai_types[0], self.config.risk_ratios[0]),
                ("Charlie", "Charlie", PlayerType.AI, self.config.ai_types[1], self.config.risk_ratios[1]),
                ("Bob", "Bob", PlayerType.AI, self.config.ai_types[2], self.config.risk_ratios[2]),
                ("David", "David", PlayerType.AI, self.config.ai_types[3], self.config.risk_ratios[3]),
            ]
            
        elif mode == TrainingMode.ROUND_ROBIN:
            # All combinations - this will be handled differently
            return []
            
        else:
            raise ValueError(f"Unknown training mode: {mode}")
    
    def run_training_session(self) -> Dict:
        """Run a complete training session.
        
        Returns
        -------
        Dict
            Summary of training results
        """
        print(f"Starting AI Training Session: {self.config.mode.value}")
        print(f"Number of games: {self.config.num_games}")
        print(f"AI types: {self.config.ai_types}")
        print(f"Risk ratios: {self.config.risk_ratios}")
        print("-" * 60)
        
        start_time = time.time()
        
        for game_num in range(self.config.num_games):
            if game_num % 10 == 0:
                print(f"Running game {game_num + 1}/{self.config.num_games}")
            
            # Create team configuration
            team_config = self.create_team_configuration(self.config.mode)
            
            # Run the game
            game_result = self._run_single_game(game_num, team_config)
            self.results.append(game_result)
            
            # Update player statistics
            self._update_player_stats(game_result)
        
        # Calculate final results
        final_results = self._calculate_final_results()
        
        # Save results
        self._save_results()
        
        total_time = time.time() - start_time
        print(f"\nTraining session completed in {total_time:.2f} seconds")
        print(f"Results saved to {self.config.output_dir}/")
        
        return final_results
    
    def _run_single_game(self, game_num: int, team_config: List[Tuple]) -> GameResult:
        """Run a single training game.
        
        Parameters
        ----------
        game_num : int
            The game number
        team_config : List[Tuple]
            Team configuration for this game
            
        Returns
        -------
        GameResult
            Results from the game
        """
        # Create game
        game = EuchreGame(
            quiet_mode=self.config.quiet_mode
        )
        
        # Add players based on configuration
        for name, display_name, player_type, ai_type, risk_ratio in team_config:
            if player_type == PlayerType.AI:
                game.add_ai_player(display_name, ai_type, risk_ratio)
            else:
                game.add_player(display_name, player_type)
        
        # Start the game
        game.start_new_game()
        
        # Run the full game
        start_time = time.time()
        game.run_full_game()
        duration = time.time() - start_time
        
        # Extract game results
        game_result = GameResult(
            game_id=f"game_{game_num:06d}",
            config=self.config.mode.value,
            winner=game.get_winner(),
            team1_score=game.game_state.team1_score if game.game_state else 0,
            team2_score=game.game_state.team2_score if game.game_state else 0,
            team_set=game.is_team_set(),
            reneges=game.renege_count,
            total_tricks=sum(player.tricks_won for player in game.players),
            trump_calls=[p.name for p in game.players if p == game.trump_caller],
            duration_seconds=duration,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return game_result
    
    def _update_player_stats(self, game_result: GameResult) -> None:
        """Update player statistics based on game results.
        
        Parameters
        ----------
        game_result : GameResult
            Results from the game
        """
        player_names = ["Alice", "Bob", "Charlie", "David"]
        
        for player_name in player_names:
            if player_name not in self.player_stats:
                self.player_stats[player_name] = {
                    "games_played": 0,
                    "games_won": 0,
                    "total_score": 0.0,
                    "team_sets": 0,
                    "sweeps": 0,
                    "euchres": 0
                }
            
            # Calculate score for this player
            score = self.scoring.calculate_score(game_result, player_name)
            
            # Update stats
            self.player_stats[player_name]["games_played"] += 1
            self.player_stats[player_name]["total_score"] += score
            
            # Determine if player won
            if game_result.winner == "Team 1" and player_name in ["Alice", "Charlie"]:
                self.player_stats[player_name]["games_won"] += 1
            elif game_result.winner == "Team 2" and player_name in ["Bob", "David"]:
                self.player_stats[player_name]["games_won"] += 1
            
            # Update other stats
            if game_result.team_set:
                if (game_result.team1_score < 3 and player_name in ["Alice", "Charlie"]) or \
                   (game_result.team2_score < 3 and player_name in ["Bob", "David"]):
                    self.player_stats[player_name]["team_sets"] += 1
            
            if game_result.team1_score == 5 and player_name in ["Alice", "Charlie"]:
                self.player_stats[player_name]["sweeps"] += 1
            elif game_result.team2_score == 5 and player_name in ["Bob", "David"]:
                self.player_stats[player_name]["sweeps"] += 1
    
    def _calculate_final_results(self) -> Dict:
        """Calculate final training results.
        
        Returns
        -------
        Dict
            Final results summary
        """
        results = {
            "training_config": {
                "mode": self.config.mode.value,
                "num_games": self.config.num_games,
                "ai_types": self.config.ai_types,
                "risk_ratios": self.config.risk_ratios
            },
            "overall_stats": {
                "total_games": len(self.results),
                "team1_wins": sum(1 for r in self.results if r.winner == "Team 1"),
                "team2_wins": sum(1 for r in self.results if r.winner == "Team 2"),
                "total_team_sets": sum(1 for r in self.results if r.team_set),
                "total_reneges": sum(r.reneges for r in self.results),
                "average_game_duration": sum(r.duration_seconds for r in self.results) / len(self.results)
            },
            "player_stats": self.player_stats,
            "game_results": [vars(r) for r in self.results]
        }
        
        return results
    
    def _save_results(self) -> None:
        """Save training results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = Path(self.config.output_dir) / f"training_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self._calculate_final_results(), f, indent=2)
        
        # Save summary
        summary_file = Path(self.config.output_dir) / f"training_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            self._write_summary(f)
        
        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
    
    def _write_summary(self, file_handle) -> None:
        """Write a human-readable summary to the file.
        
        Parameters
        ----------
        file_handle
            File handle to write to
        """
        results = self._calculate_final_results()
        
        file_handle.write("AI TRAINING SESSION SUMMARY\n")
        file_handle.write("=" * 50 + "\n\n")
        
        # Training configuration
        file_handle.write("TRAINING CONFIGURATION:\n")
        file_handle.write(f"Mode: {results['training_config']['mode']}\n")
        file_handle.write(f"Games: {results['training_config']['num_games']}\n")
        file_handle.write(f"AI Types: {', '.join(results['training_config']['ai_types'])}\n")
        file_handle.write(f"Risk Ratios: {', '.join(map(str, results['training_config']['risk_ratios']))}\n\n")
        
        # Overall statistics
        file_handle.write("OVERALL STATISTICS:\n")
        overall = results['overall_stats']
        file_handle.write(f"Total Games: {overall['total_games']}\n")
        file_handle.write(f"Team 1 Wins: {overall['team1_wins']} ({overall['team1_wins']/overall['total_games']*100:.1f}%)\n")
        file_handle.write(f"Team 2 Wins: {overall['team2_wins']} ({overall['team2_wins']/overall['total_games']*100:.1f}%)\n")
        file_handle.write(f"Team Sets: {overall['total_team_sets']}\n")
        file_handle.write(f"Total Reneges: {overall['total_reneges']}\n")
        file_handle.write(f"Average Game Duration: {overall['average_game_duration']:.2f} seconds\n\n")
        
        # Player statistics
        file_handle.write("PLAYER STATISTICS:\n")
        for player, stats in results['player_stats'].items():
            file_handle.write(f"\n{player}:\n")
            file_handle.write(f"  Games Played: {stats['games_played']}\n")
            file_handle.write(f"  Games Won: {stats['games_won']} ({stats['games_won']/stats['games_played']*100:.1f}%)\n")
            file_handle.write(f"  Average Score: {stats['total_score']/stats['games_played']:.2f}\n")
            file_handle.write(f"  Team Sets: {stats['team_sets']}\n")
            file_handle.write(f"  Sweeps: {stats['sweeps']}\n")
            file_handle.write(f"  Euchres: {stats['euchres']}\n")


def run_training_examples():
    """Run example training sessions."""
    
    # Example 1: Same model training (4 Alice players)
    print("=== SAME MODEL TRAINING ===")
    config1 = TrainingConfig(
        mode=TrainingMode.SAME_MODEL,
        num_games=50,
        ai_types=["aggressive", "conservative", "balanced", "opportunistic"],
        risk_ratios=[0.8, 0.2, 0.5, 0.7]
    )
    
    framework1 = AITrainingFramework(config1)
    results1 = framework1.run_training_session()
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Team vs Team training (2 Alice vs 2 Bob)
    print("=== TEAM VS TEAM TRAINING ===")
    config2 = TrainingConfig(
        mode=TrainingMode.TEAM_VS_TEAM,
        num_games=50,
        ai_types=["balanced", "balanced", "aggressive", "aggressive"],
        risk_ratios=[0.5, 0.5, 0.8, 0.8]
    )
    
    framework2 = AITrainingFramework(config2)
    results2 = framework2.run_training_session()
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Mixed teams training
    print("=== MIXED TEAMS TRAINING ===")
    config3 = TrainingConfig(
        mode=TrainingMode.MIXED_TEAMS,
        num_games=50,
        ai_types=["conservative", "balanced", "aggressive", "opportunistic"],
        risk_ratios=[0.3, 0.5, 0.8, 0.6]
    )
    
    framework3 = AITrainingFramework(config3)
    results3 = framework3.run_training_session()


if __name__ == "__main__":
    run_training_examples() 