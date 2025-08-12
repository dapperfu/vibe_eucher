#!/usr/bin/env python3
"""
AI Tournament System for Euchre

This script runs a comprehensive tournament to determine which AI teams are the best.
The tournament includes:
- 4 teams of the same AI type (aggressive, conservative, balanced, opportunistic)
- 8 teams with mismatched AI types
- Best-of-3 series for each matchup
- Comprehensive ranking and statistics

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-12
Context: Creating AI tournament system to determine best teams
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from euchre.game import EuchreGame
from euchre.ai.ai_factory import AIFactory
from euchre.models import PlayerType


@dataclass
class TournamentTeam:
    """Represents a team in the tournament."""
    name: str
    player1_ai_type: str
    player2_ai_type: str
    player1_risk: float
    player2_risk: float
    wins: int = 0
    losses: int = 0
    games_played: int = 0
    total_score: int = 0
    team_sets: int = 0
    euchres: int = 0
    
    def __str__(self) -> str:
        return f"{self.name} ({self.player1_ai_type}/{self.player2_ai_type})"
    
    def get_win_percentage(self) -> float:
        """Calculate win percentage."""
        if self.games_played == 0:
            return 0.0
        return (self.wins / self.games_played) * 100


@dataclass
class TournamentMatch:
    """Represents a match between two teams."""
    team1: TournamentTeam
    team2: TournamentTeam
    team1_wins: int = 0
    team2_wins: int = 0
    games: List[Dict] = None
    
    def __init__(self, team1: TournamentTeam, team2: TournamentTeam):
        self.team1 = team1
        self.team2 = team2
        self.team1_wins = 0
        self.team2_wins = 0
        self.games = []
    
    def add_game_result(self, winner: str, game_data: Dict) -> None:
        """Add a game result to the match."""
        self.games.append(game_data)
        if winner == "Team 1":
            self.team1_wins += 1
        else:
            self.team2_wins += 1
    
    def get_winner(self) -> Optional[TournamentTeam]:
        """Get the winner of the match (best of 3)."""
        if self.team1_wins >= 2:
            return self.team1
        elif self.team2_wins >= 2:
            return self.team2
        return None
    
    def is_complete(self) -> bool:
        """Check if the match is complete (best of 3)."""
        return self.team1_wins >= 2 or self.team2_wins >= 2


class AITournament:
    """Main tournament controller."""
    
    def __init__(self, output_dir: str = "tournament_results"):
        """Initialize the tournament.
        
        Parameters
        ----------
        output_dir : str
            Directory to save tournament results
        """
        self.output_dir = output_dir
        self.teams: List[TournamentTeam] = []
        self.matches: List[TournamentMatch] = []
        self.results: Dict = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize teams
        self._create_teams()
        
        # Generate matchups
        self._generate_matchups()
    
    def _create_teams(self) -> None:
        """Create tournament teams."""
        print("🏆 Creating Tournament Teams...")
        
        # 4 teams of the same AI type
        same_type_teams = [
            ("Aggressive Squad", "aggressive", "aggressive", 0.7, 0.7),
            ("Conservative Corps", "conservative", "conservative", 0.3, 0.3),
            ("Balanced Battalion", "balanced", "balanced", 0.5, 0.5),
            ("Opportunistic Outfit", "opportunistic", "opportunistic", 0.6, 0.6)
        ]
        
        # 8 teams with mismatched AI types
        mixed_type_teams = [
            ("Alpha Strike", "aggressive", "balanced", 0.8, 0.5),
            ("Beta Defense", "conservative", "opportunistic", 0.2, 0.7),
            ("Gamma Tactics", "balanced", "aggressive", 0.5, 0.6),
            ("Delta Risk", "opportunistic", "conservative", 0.7, 0.3),
            ("Echo Team", "aggressive", "opportunistic", 0.9, 0.6),
            ("Foxtrot Force", "balanced", "conservative", 0.4, 0.2),
            ("Golf Squad", "conservative", "balanced", 0.3, 0.5),
            ("Hotel Strike", "opportunistic", "aggressive", 0.6, 0.8)
        ]
        
        # Create all teams
        for name, ai1, ai2, risk1, risk2 in same_type_teams + mixed_type_teams:
            team = TournamentTeam(name, ai1, ai2, risk1, risk2)
            self.teams.append(team)
            print(f"  ✅ {team}")
        
        print(f"Created {len(self.teams)} teams for the tournament!")
    
    def _generate_matchups(self) -> None:
        """Generate tournament matchups."""
        print("\n🎯 Generating Tournament Matchups...")
        
        # Round-robin tournament: every team plays every other team
        for i, team1 in enumerate(self.teams):
            for j, team2 in enumerate(self.teams):
                if i < j:  # Avoid duplicate matches and self-matches
                    match = TournamentMatch(team1, team2)
                    self.matches.append(match)
                    print(f"  🆚 {team1.name} vs {team2.name}")
        
        print(f"Generated {len(self.matches)} matches!")
    
    def run_tournament(self) -> None:
        """Run the complete tournament."""
        print(f"\n🚀 Starting AI Tournament!")
        print(f"📊 {len(self.teams)} teams, {len(self.matches)} matches")
        print(f"🏆 Best of 3 series")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run all matches
        for match_num, match in enumerate(self.matches, 1):
            print(f"\n🎮 Match {match_num}/{len(self.matches)}: {match.team1.name} vs {match.team2.name}")
            print("-" * 60)
            
            self._run_match(match)
            
            # Update team statistics
            winner = match.get_winner()
            if winner:
                winner.wins += 1
                loser = match.team2 if winner == match.team1 else match.team1
                loser.losses += 1
                print(f"🏆 {winner.name} wins the series!")
            else:
                print("❌ Match incomplete - this shouldn't happen")
            
            # Progress update
            completed = sum(1 for m in self.matches if m.is_complete())
            print(f"📈 Tournament Progress: {completed}/{len(self.matches)} matches completed")
        
        # Calculate final results
        tournament_time = time.time() - start_time
        self._calculate_final_results(tournament_time)
        
        # Save results
        self._save_tournament_results()
        
        print(f"\n🎉 TOURNAMENT COMPLETED!")
        print(f"⏱️  Total time: {tournament_time/60:.1f} minutes")
        print(f"📊 Results saved to: {self.output_dir}/")
    
    def _run_match(self, match: TournamentMatch) -> None:
        """Run a best-of-3 series between two teams."""
        game_num = 1
        
        while not match.is_complete() and game_num <= 3:
            print(f"  Game {game_num}: {match.team1.name} vs {match.team2.name}")
            
            # Create players for this game
            players = []
            
            # Team 1: Alice & Charlie
            players.append(AIFactory.create_ai_player("Alice", match.team1.player1_ai_type, match.team1.player1_risk))
            players.append(AIFactory.create_ai_player("Charlie", match.team1.player2_ai_type, match.team1.player2_risk))
            
            # Team 2: Bob & David
            players.append(AIFactory.create_ai_player("Bob", match.team2.player1_ai_type, match.team2.player1_risk))
            players.append(AIFactory.create_ai_player("David", match.team2.player2_ai_type, match.team2.player2_risk))
            
            # Create and run the game
            game = EuchreGame(players, quiet_mode=True, verbose=False)
            
            try:
                game.start_new_game()
                game.run_full_game()
                
                # Determine winner
                team1_score = sum(players[i].score for i in [0, 2])  # Alice + Charlie
                team2_score = sum(players[i].score for i in [1, 3])  # Bob + David
                
                if team1_score > team2_score:
                    winner = "Team 1"
                    print(f"    🏆 {match.team1.name} wins Game {game_num} ({team1_score}-{team2_score})")
                else:
                    winner = "Team 2"
                    print(f"    🏆 {match.team2.name} wins Game {game_num} ({team2_score}-{team1_score})")
                
                # Record game result
                game_data = {
                    "game_num": game_num,
                    "team1_score": team1_score,
                    "team2_score": team2_score,
                    "winner": winner,
                    "duration": time.time() - time.time(),  # Placeholder
                    "timestamp": datetime.now().isoformat()
                }
                
                match.add_game_result(winner, game_data)
                
                # Update team statistics
                match.team1.total_score += team1_score
                match.team2.total_score += team2_score
                match.team1.games_played += 1
                match.team2.games_played += 1
                
            except Exception as e:
                print(f"    ❌ Error in Game {game_num}: {e}")
                # Record a default result
                game_data = {
                    "game_num": game_num,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                match.games.append(game_data)
            
            game_num += 1
            
            # Check if match is complete
            if match.is_complete():
                break
        
        # Display match summary
        print(f"  📊 Series Result: {match.team1.name} {match.team1_wins} - {match.team2_wins} {match.team2.name}")
    
    def _calculate_final_results(self, tournament_time: float) -> None:
        """Calculate final tournament results."""
        print("\n📊 Calculating Final Tournament Results...")
        
        # Sort teams by win percentage, then by wins, then by score differential
        sorted_teams = sorted(
            self.teams,
            key=lambda t: (t.get_win_percentage(), t.wins, t.total_score),
            reverse=True
        )
        
        # Create results summary
        self.results = {
            "tournament_info": {
                "total_teams": len(self.teams),
                "total_matches": len(self.matches),
                "total_games": sum(t.games_played for t in self.teams),
                "tournament_duration_minutes": tournament_time / 60,
                "timestamp": datetime.now().isoformat()
            },
            "team_rankings": [],
            "match_results": [],
            "statistics": {
                "best_team": sorted_teams[0].name if sorted_teams else "None",
                "most_games": max(self.teams, key=lambda t: t.games_played).name if self.teams else "None",
                "highest_scoring": max(self.teams, key=lambda t: t.total_score).name if self.teams else "None"
            }
        }
        
        # Add team rankings
        for rank, team in enumerate(sorted_teams, 1):
            self.results["team_rankings"].append({
                "rank": rank,
                "name": team.name,
                "ai_types": f"{team.player1_ai_type}/{team.player2_ai_type}",
                "wins": team.wins,
                "losses": team.losses,
                "win_percentage": team.get_win_percentage(),
                "games_played": team.games_played,
                "total_score": team.total_score,
                "team_sets": team.team_sets,
                "euchres": team.euchres
            })
        
        # Add match results
        for match in self.matches:
            winner = match.get_winner()
            self.results["match_results"].append({
                "team1": match.team1.name,
                "team2": match.team2.name,
                "team1_wins": match.team1_wins,
                "team2_wins": match.team2_wins,
                "winner": winner.name if winner else "Incomplete",
                "games": len(match.games)
            })
    
    def _save_tournament_results(self) -> None:
        """Save tournament results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"{self.output_dir}/tournament_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary
        summary_file = f"{self.output_dir}/tournament_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(self._generate_summary_text())
        
        print(f"📁 Tournament results saved to:")
        print(f"  📄 {results_file}")
        print(f"  📄 {summary_file}")
    
    def _generate_summary_text(self) -> str:
        """Generate human-readable tournament summary."""
        summary = []
        summary.append("🏆 AI TOURNAMENT RESULTS")
        summary.append("=" * 80)
        summary.append("")
        
        # Tournament info
        info = self.results["tournament_info"]
        summary.append(f"Tournament Information:")
        summary.append(f"  Total Teams: {info['total_teams']}")
        summary.append(f"  Total Matches: {info['total_matches']}")
        summary.append(f"  Total Games: {info['total_games']}")
        summary.append(f"  Duration: {info['tournament_duration_minutes']:.1f} minutes")
        summary.append(f"  Completed: {info['timestamp']}")
        summary.append("")
        
        # Team rankings
        summary.append("🏅 FINAL TEAM RANKINGS:")
        summary.append("-" * 80)
        summary.append(f"{'Rank':<4} {'Team':<25} {'AI Types':<20} {'W-L':<8} {'Win%':<8} {'Score':<10}")
        summary.append("-" * 80)
        
        for team_data in self.results["team_rankings"]:
            rank = team_data["rank"]
            name = team_data["name"]
            ai_types = team_data["ai_types"]
            record = f"{team_data['wins']}-{team_data['losses']}"
            win_pct = f"{team_data['win_percentage']:.1f}%"
            score = team_data["total_score"]
            
            summary.append(f"{rank:<4} {name:<25} {ai_types:<20} {record:<8} {win_pct:<8} {score:<10}")
        
        summary.append("")
        
        # Statistics
        stats = self.results["statistics"]
        summary.append("📊 TOURNAMENT STATISTICS:")
        summary.append(f"  🏆 Best Team: {stats['best_team']}")
        summary.append(f"  🎮 Most Games: {stats['most_games']}")
        summary.append(f"  💯 Highest Scoring: {stats['highest_scoring']}")
        summary.append("")
        
        # Match results summary
        summary.append("🎮 MATCH RESULTS SUMMARY:")
        summary.append("-" * 80)
        
        completed_matches = [m for m in self.results["match_results"] if m["winner"] != "Incomplete"]
        summary.append(f"Completed Matches: {len(completed_matches)}/{len(self.results['match_results'])}")
        
        # Count wins by team
        team_wins = {}
        for match in completed_matches:
            winner = match["winner"]
            team_wins[winner] = team_wins.get(winner, 0) + 1
        
        summary.append("Series Wins by Team:")
        for team, wins in sorted(team_wins.items(), key=lambda x: x[1], reverse=True):
            summary.append(f"  {team}: {wins} series wins")
        
        return "\n".join(summary)
    
    def display_results(self) -> None:
        """Display tournament results."""
        print("\n" + "=" * 80)
        print("🏆 TOURNAMENT RESULTS")
        print("=" * 80)
        
        # Display top teams
        print("\n🏅 TOP 5 TEAMS:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Team':<25} {'AI Types':<20} {'W-L':<8} {'Win%':<8} {'Score':<10}")
        print("-" * 80)
        
        for team_data in self.results["team_rankings"][:5]:
            rank = team_data["rank"]
            name = team_data["name"]
            ai_types = team_data["ai_types"]
            record = f"{team_data['wins']}-{team_data['losses']}"
            win_pct = f"{team_data['win_percentage']:.1f}%"
            score = team_data["total_score"]
            
            print(f"{rank:<4} {name:<25} {ai_types:<20} {record:<8} {win_pct:<8} {score:<10}")
        
        # Display champion
        if self.results["team_rankings"]:
            champion = self.results["team_rankings"][0]
            print(f"\n👑 TOURNAMENT CHAMPION: {champion['name']}")
            print(f"   AI Types: {champion['ai_types']}")
            print(f"   Record: {champion['wins']}-{champion['losses']}")
            print(f"   Win Percentage: {champion['win_percentage']:.1f}%")
            print(f"   Total Score: {champion['total_score']}")


def main():
    """Main tournament function."""
    print("🏆 AI Tournament System for Euchre")
    print("=" * 60)
    
    # Create and run tournament
    tournament = AITournament()
    tournament.run_tournament()
    
    # Display results
    tournament.display_results()
    
    print(f"\n✅ Tournament completed successfully!")
    print(f"📊 Check {tournament.output_dir}/ for detailed results")


if __name__ == "__main__":
    main() 