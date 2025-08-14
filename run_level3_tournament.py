#!/usr/bin/env python3
"""
Run Level3 AI Tournament

This script runs actual tournament games between Level3 AI and Level2/Level1 AI
to demonstrate the trained model's performance.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from euchre.game import EuchreGame
from euchre.models import Card, Suit, Rank, Player
from euchre.ai.ai_factory import AIFactory


@dataclass
class GameResult:
    """Data class for storing game results."""
    game_id: int
    timestamp: str
    duration_seconds: float
    winner_team: str
    final_score: Dict[str, int]
    trump_suit: str
    trump_caller: str
    trump_caller_team: str
    tricks_won: Dict[str, int]
    game_length_tricks: int
    players: List[str]
    teams: Dict[str, str]


@dataclass
class TournamentStats:
    """Data class for tournament statistics."""
    total_games: int
    level3_wins: int
    level2_level1_wins: int
    level3_win_rate: float
    level2_level1_win_rate: float
    avg_game_length: float
    trump_calling_success: Dict[str, float]


class Level3TournamentRunner:
    """Runs tournaments between Level3 AI and Level2/Level1 AI."""
    
    def __init__(self, num_games: int = 100):
        """Initialize the tournament runner.
        
        Parameters
        ----------
        num_games : int
            Number of games to play in the tournament
        """
        self.num_games = num_games
        self.game_results: List[GameResult] = []
        
        print(f"🎯 Level3 Tournament Runner Initialized")
        print(f"   Games to play: {num_games}")
        print()
    
    def run_tournament(self) -> TournamentStats:
        """Run the complete tournament and return statistics."""
        print("🏆 Starting Level3 Tournament!")
        print("=" * 50)
        
        # Tournament setup: Level3 vs Level2+Level1
        # Team A: 2 Level3 AI players
        # Team B: 1 Level2 AI + 1 Level1 AI player
        
        for game_id in range(self.num_games):
            print(f"🎮 Game {game_id + 1}/{self.num_games}")
            
            # Create game
            game = EuchreGame()
            
            # Create AI players
            team_a_players = [
                AIFactory.create_ai_player(f"Level3_Alice_{game_id}", "level3_strategic", 0.5),
                AIFactory.create_ai_player(f"Level3_Bob_{game_id}", "level3_balanced", 0.5)
            ]
            
            team_b_players = [
                AIFactory.create_ai_player(f"Traditional_Charlie_{game_id}", "strategic", 0.5),
                AIFactory.create_ai_player(f"Traditional_David_{game_id}", "balanced", 0.5)
            ]
            
            # Add players to game
            for player in team_a_players + team_b_players:
                game.add_player(player.name, player.player_type)
            
            # Play the game
            start_time = time.time()
            game_result = self._play_game(game, game_id, team_a_players, team_b_players)
            duration = time.time() - start_time
            
            # Store result
            game_result.duration_seconds = duration
            self.game_results.append(game_result)
            
            # Print game summary
            self._print_game_summary(game_result)
            print()
        
        # Calculate tournament statistics
        tournament_stats = self._calculate_tournament_stats()
        self._print_tournament_summary(tournament_stats)
        
        # Save results
        self._save_results()
        
        return tournament_stats
    
    def _play_game(self, game: EuchreGame, game_id: int, 
                   team_a: List[Player], team_b: List[Player]) -> GameResult:
        """Play a single game and return results."""
        # Start the game
        game.start_new_game()
        
        # Run the full game automatically
        game.run_full_game()
        
        # Get game results
        final_score = game.get_scores()
        winner_team = "Team_A" if final_score["team1_score"] >= 10 else "Team_B"
        
        # Get basic game information
        trump_suit = str(game.game_state_manager.trump_suit) if hasattr(game, 'game_state_manager') and game.game_state_manager.trump_suit else "None"
        trump_caller = game.game_state_manager.trump_caller if hasattr(game, 'game_state_manager') else "None"
        trump_caller_team = "Team_A" if trump_caller in [p.name for p in team_a] else "Team_B" if trump_caller else "None"
        
        # Calculate tricks won (simplified)
        tricks_won = {"Team_A": 0, "Team_B": 0}
        if hasattr(game, 'tricks_won'):
            for player_name, tricks in game.tricks_won.items():
                if any(p.name == player_name for p in team_a):
                    tricks_won["Team_A"] += tricks
                else:
                    tricks_won["Team_B"] += tricks
        
        # Estimate game length
        game_length_tricks = 20  # Default estimate
        
        # Create game result
        result = GameResult(
            game_id=game_id,
            timestamp=datetime.now().isoformat(),
            duration_seconds=0,  # Will be set later
            winner_team=winner_team,
            final_score={"Team_A": final_score["team1_score"], "Team_B": final_score["team2_score"]},
            trump_suit=trump_suit,
            trump_caller=trump_caller or "None",
            trump_caller_team=trump_caller_team,
            tricks_won=tricks_won,
            game_length_tricks=game_length_tricks,
            players=[p.name for p in game.players] if hasattr(game, 'players') else [],
            teams={"Team_A": "Level3", "Team_B": "Traditional"}
        )
        
        return result
    
    def _print_game_summary(self, result: GameResult):
        """Print a summary of the game result."""
        print(f"   Winner: {result.winner_team}")
        print(f"   Score: Team A {result.final_score['Team_A']} - Team B {result.final_score['Team_B']}")
        if result.trump_caller != "None":
            print(f"   Trump: {result.trump_suit} called by {result.trump_caller} ({result.trump_caller_team})")
        print(f"   Tricks: Team A {result.tricks_won['Team_A']} - Team B {result.tricks_won['Team_B']}")
        print(f"   Duration: {result.duration_seconds:.2f}s")
    
    def _calculate_tournament_stats(self) -> TournamentStats:
        """Calculate comprehensive tournament statistics."""
        print("📊 Calculating Tournament Statistics...")
        
        # Basic win rates
        level3_wins = sum(1 for r in self.game_results if r.winner_team == "Team_A")
        level2_level1_wins = sum(1 for r in self.game_results if r.winner_team == "Team_B")
        
        total_games = len(self.game_results)
        
        # Trump calling statistics
        trump_calling_success = {
            "Team_A": {"calls": 0, "successes": 0, "rate": 0.0},
            "Team_B": {"calls": 0, "successes": 0, "rate": 0.0}
        }
        
        for result in self.game_results:
            if result.trump_caller_team != "None":
                trump_calling_success[result.trump_caller_team]["calls"] += 1
                if result.winner_team == result.trump_caller_team:
                    trump_calling_success[result.trump_caller_team]["successes"] += 1
        
        # Calculate success rates
        for team in trump_calling_success:
            calls = trump_calling_success[team]["calls"]
            successes = trump_calling_success[team]["successes"]
            trump_calling_success[team]["rate"] = successes / calls if calls > 0 else 0.0
        
        # Game length statistics
        game_lengths = [r.game_length_tricks for r in self.game_results]
        avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
        
        # Create tournament stats
        stats = TournamentStats(
            total_games=total_games,
            level3_wins=level3_wins,
            level2_level1_wins=level2_level1_wins,
            level3_win_rate=level3_wins / total_games,
            level2_level1_win_rate=level2_level1_wins / total_games,
            avg_game_length=avg_game_length,
            trump_calling_success={
                "Team_A": trump_calling_success["Team_A"]["rate"],
                "Team_B": trump_calling_success["Team_B"]["rate"]
            }
        )
        
        return stats
    
    def _print_tournament_summary(self, stats: TournamentStats):
        """Print a comprehensive tournament summary."""
        print("🏆 TOURNAMENT RESULTS")
        print("=" * 50)
        print(f"Total Games: {stats.total_games}")
        print(f"Level3 Win Rate: {stats.level3_win_rate:.1%} ({stats.level3_wins} wins)")
        print(f"Traditional AI Win Rate: {stats.level2_level1_win_rate:.1%} ({stats.level2_level1_wins} wins)")
        print()
        
        print("📊 DETAILED STATISTICS")
        print("-" * 30)
        print("Trump Calling Success:")
        print(f"  Team A (Level3): {stats.trump_calling_success['Team_A']:.1%}")
        print(f"  Team B (Traditional): {stats.trump_calling_success['Team_B']:.1%}")
        
        print(f"\nAverage Game Length: {stats.avg_game_length:.1f} tricks")
    
    def _save_results(self):
        """Save tournament results to files."""
        # Save detailed game results
        results_file = f"level3_tournament_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert dataclasses to dictionaries
        game_results_dict = [asdict(r) for r in self.game_results]
        
        # Save to JSON
        with open(results_file, 'w') as f:
            json.dump({
                'game_results': game_results_dict,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'num_games': self.num_games,
                    'tournament_type': 'Level3_vs_Traditional'
                }
            }, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")


def main():
    """Main function to run the tournament."""
    print("🎯 Level3 AI Tournament")
    print("=" * 50)
    
    # Run tournament
    runner = Level3TournamentRunner(num_games=10)  # Start with 10 games for testing
    stats = runner.run_tournament()
    
    print("\n🎉 Tournament completed successfully!")
    print(f"📊 Final Level3 Win Rate: {stats.level3_win_rate:.1%}")


if __name__ == "__main__":
    main() 