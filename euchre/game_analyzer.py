"""Game results analyzer and statistical analysis tool."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns


class GameAnalyzer:
    """Analyzes euchre game results and generates statistics."""
    
    def __init__(self, games_dir: str = "games") -> None:
        """Initialize the game analyzer.
        
        Parameters
        ----------
        games_dir : str
            Directory containing game result files
        """
        self.games_dir = Path(games_dir)
        self.games_data = None
        self.df = None
        
    def load_games(self, max_games: Optional[int] = None) -> pd.DataFrame:
        """Load and parse all game result files.
        
        Parameters
        ----------
        max_games : Optional[int]
            Maximum number of games to load (None for all)
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing all game data
        """
        game_files = list(self.games_dir.glob("*.json"))
        
        if max_games:
            game_files = game_files[:max_games]
        
        print(f"Loading {len(game_files)} game files...")
        
        games_data = []
        for game_file in game_files:
            try:
                with open(game_file, 'r') as f:
                    game_data = json.load(f)
                    
                # Skip games with errors
                if "error" in game_data:
                    continue
                    
                games_data.append(game_data)
                
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading {game_file}: {e}")
                continue
        
        print(f"Successfully loaded {len(games_data)} games")
        
        # Convert to DataFrame
        self.games_data = games_data
        self.df = pd.DataFrame(games_data)
        
        return self.df
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the loaded games.
        
        Returns
        -------
        Dict[str, Any]
            Basic statistics dictionary
        """
        if self.df is None:
            raise ValueError("No games loaded. Call load_games() first.")
        
        stats = {
            "total_games": len(self.df),
            "team_configs": self.df["team_config"].value_counts().to_dict(),
            "winning_teams": self.df["winner"].value_counts().to_dict(),
            "avg_rounds_per_game": self.df["total_rounds"].mean(),
            "avg_game_duration": self.df["game_duration"].mean(),
            "score_distribution": {
                "team1_scores": self.df["team1"].apply(lambda x: x["final_score"]).describe().to_dict(),
                "team2_scores": self.df["team2"].apply(lambda x: x["final_score"]).describe().to_dict()
            }
        }
        
        return stats
    
    def analyze_team_matchups(self) -> pd.DataFrame:
        """Analyze win rates for different team matchups.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with matchup analysis
        """
        if self.df is None:
            raise ValueError("No games loaded. Call load_games() first.")
        
        # Group by team configuration and analyze
        matchup_stats = []
        
        for config in self.df["team_config"].unique():
            config_games = self.df[self.df["team_config"] == config]
            
            total_games = len(config_games)
            team1_wins = len(config_games[config_games["winner"] == "team1"])
            team2_wins = len(config_games[config_games["winner"] == "team2"])
            ties = len(config_games[config_games["winner"] == "tie"])
            
            # Calculate win rates
            team1_win_rate = team1_wins / total_games if total_games > 0 else 0
            team2_win_rate = team2_wins / total_games if total_games > 0 else 0
            tie_rate = ties / total_games if total_games > 0 else 0
            
            # Get team compositions
            team1_players = config_games.iloc[0]["team1"]["players"]
            team2_players = config_games.iloc[0]["team2"]["players"]
            
            team1_profile = f"{team1_players[0][1]}-{team1_players[1][1]}"
            team2_profile = f"{team2_players[0][1]}-{team2_players[1][1]}"
            
            matchup_stats.append({
                "team_config": config,
                "team1_profile": team1_profile,
                "team2_profile": team2_profile,
                "total_games": total_games,
                "team1_wins": team1_wins,
                "team2_wins": team2_wins,
                "ties": ties,
                "team1_win_rate": team1_win_rate,
                "team2_win_rate": team2_win_rate,
                "tie_rate": tie_rate,
                "team1_avg_score": config_games["team1"].apply(lambda x: x["final_score"]).mean(),
                "team2_avg_score": config_games["team2"].apply(lambda x: x["final_score"]).mean()
            })
        
        return pd.DataFrame(matchup_stats)
    
    def analyze_aggressive_vs_conservative(self) -> Dict[str, Any]:
        """Specifically analyze aggressive vs conservative team matchups.
        
        Returns
        -------
        Dict[str, Any]
            Analysis results for aggressive vs conservative matchups
        """
        if self.df is None:
            raise ValueError("No games loaded. Call load_games() first.")
        
        # Find aggressive vs conservative matchups
        aggressive_vs_conservative = self.df[
            self.df["team_config"] == "aggressive_vs_conservative"
        ]
        
        if len(aggressive_vs_conservative) == 0:
            return {"error": "No aggressive_vs_conservative games found"}
        
        # Analyze results
        total_games = len(aggressive_vs_conservative)
        aggressive_wins = len(aggressive_vs_conservative[
            aggressive_vs_conservative["winner"] == "team1"
        ])
        conservative_wins = len(aggressive_vs_conservative[
            aggressive_vs_conservative["winner"] == "team2"
        ])
        ties = total_games - aggressive_wins - conservative_wins
        
        # Calculate win rates
        aggressive_win_rate = aggressive_wins / total_games
        conservative_win_rate = conservative_wins / total_games
        tie_rate = ties / total_games
        
        # Score analysis
        aggressive_scores = aggressive_vs_conservative["team1"].apply(lambda x: x["final_score"])
        conservative_scores = aggressive_vs_conservative["team2"].apply(lambda x: x["final_score"])
        
        # Round analysis
        rounds_per_game = aggressive_vs_conservative["total_rounds"]
        
        analysis = {
            "total_games": total_games,
            "aggressive_team": {
                "wins": aggressive_wins,
                "win_rate": aggressive_win_rate,
                "avg_score": aggressive_scores.mean(),
                "score_std": aggressive_scores.std(),
                "min_score": aggressive_scores.min(),
                "max_score": aggressive_scores.max()
            },
            "conservative_team": {
                "wins": conservative_wins,
                "win_rate": conservative_win_rate,
                "avg_score": conservative_scores.mean(),
                "score_std": conservative_scores.std(),
                "min_score": conservative_scores.min(),
                "max_score": conservative_scores.max()
            },
            "ties": ties,
            "tie_rate": tie_rate,
            "avg_rounds_per_game": rounds_per_game.mean(),
            "rounds_std": rounds_per_game.std()
        }
        
        return analysis
    
    def analyze_round_by_round(self, team_config: str = "aggressive_vs_conservative") -> pd.DataFrame:
        """Analyze how games progress round by round.
        
        Parameters
        ----------
        team_config : str
            Team configuration to analyze
            
        Returns
        -------
        pd.DataFrame
            Round-by-round analysis
        """
        if self.df is None:
            raise ValueError("No games loaded. Call load_games() first.")
        
        config_games = self.df[self.df["team_config"] == team_config]
        
        if len(config_games) == 0:
            raise ValueError(f"No games found for configuration: {team_config}")
        
        # Extract round data
        round_data = []
        for _, game in config_games.iterrows():
            for round_info in game["rounds"]:
                round_data.append({
                    "game_id": game["game_id"],
                    "round": round_info["round"],
                    "trump_suit": round_info["trump_suit"],
                    "team1_score": round_info["team1_score"],
                    "team2_score": round_info["team2_score"],
                    "trick_counts": round_info["trick_counts"],
                    "team1_tricks": round_info["trick_counts"][0] + round_info["trick_counts"][2],
                    "team2_tricks": round_info["trick_counts"][1] + round_info["trick_counts"][3]
                })
        
        round_df = pd.DataFrame(round_data)
        
        # Group by round and calculate statistics
        round_stats = []
        for round_num in round_df["round"].unique():
            round_games = round_df[round_df["round"] == round_num]
            
            stats = {
                "round": round_num,
                "games_count": len(round_games),
                "avg_team1_score": round_games["team1_score"].mean(),
                "avg_team2_score": round_games["team2_score"].mean(),
                "avg_team1_tricks": round_games["team1_tricks"].mean(),
                "avg_team2_tricks": round_games["team2_tricks"].mean(),
                "team1_leading_count": len(round_games[round_games["team1_score"] > round_games["team2_score"]]),
                "team2_leading_count": len(round_games[round_games["team2_score"] > round_games["team1_score"]]),
                "tied_count": len(round_games[round_games["team1_score"] == round_games["team2_score"]])
            }
            
            # Calculate leading percentages
            total_games = stats["games_count"]
            stats["team1_leading_pct"] = stats["team1_leading_count"] / total_games if total_games > 0 else 0
            stats["team2_leading_pct"] = stats["team2_leading_count"] / total_games if total_games > 0 else 0
            stats["tied_pct"] = stats["tied_count"] / total_games if total_games > 0 else 0
            
            round_stats.append(stats)
        
        return pd.DataFrame(round_stats)
    
    def generate_visualizations(self, output_dir: str = "analysis_plots") -> None:
        """Generate visualization plots for the analysis.
        
        Parameters
        ----------
        output_dir : str
            Directory to save plots
        """
        if self.df is None:
            raise ValueError("No games loaded. Call load_games() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Win rates by team configuration
        plt.figure(figsize=(12, 8))
        matchup_df = self.analyze_team_matchups()
        
        x = np.arange(len(matchup_df))
        width = 0.35
        
        plt.bar(x - width/2, matchup_df["team1_win_rate"], width, 
                label="Team 1 Win Rate", alpha=0.8)
        plt.bar(x + width/2, matchup_df["team2_win_rate"], width, 
                label="Team 2 Win Rate", alpha=0.8)
        
        plt.xlabel("Team Configuration")
        plt.ylabel("Win Rate")
        plt.title("Win Rates by Team Configuration")
        plt.xticks(x, matchup_df["team_config"], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "win_rates_by_config.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Score distribution
        plt.figure(figsize=(12, 8))
        
        team1_scores = self.df["team1"].apply(lambda x: x["final_score"])
        team2_scores = self.df["team2"].apply(lambda x: x["final_score"])
        
        plt.hist(team1_scores, bins=20, alpha=0.7, label="Team 1 Scores", density=True)
        plt.hist(team2_scores, bins=20, alpha=0.7, label="Team 2 Scores", density=True)
        
        plt.xlabel("Final Score")
        plt.ylabel("Density")
        plt.title("Distribution of Final Scores")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "score_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Rounds per game distribution
        plt.figure(figsize=(10, 6))
        
        plt.hist(self.df["total_rounds"], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel("Number of Rounds")
        plt.ylabel("Frequency")
        plt.title("Distribution of Rounds per Game")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "rounds_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Round-by-round progression for aggressive vs conservative
        try:
            round_df = self.analyze_round_by_round("aggressive_vs_conservative")
            
            plt.figure(figsize=(12, 8))
            
            rounds = round_df["round"]
            team1_leading = round_df["team1_leading_pct"]
            team2_leading = round_df["team2_leading_pct"]
            tied = round_df["tied_pct"]
            
            plt.plot(rounds, team1_leading, 'o-', label="Aggressive Team Leading", linewidth=2, markersize=6)
            plt.plot(rounds, team2_leading, 's-', label="Conservative Team Leading", linewidth=2, markersize=6)
            plt.plot(rounds, tied, '^-', label="Tied", linewidth=2, markersize=6)
            
            plt.xlabel("Round Number")
            plt.ylabel("Percentage of Games")
            plt.title("Round-by-Round Progression: Aggressive vs Conservative")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / "round_progression.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Could not generate round progression plot: {e}")
        
        print(f"Visualizations saved to: {output_path}")
    
    def export_analysis(self, output_file: str = "euchre_analysis.xlsx") -> None:
        """Export analysis results to Excel file.
        
        Parameters
        ----------
        output_file : str
            Output Excel filename
        """
        if self.df is None:
            raise ValueError("No games loaded. Call load_games() first.")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Basic stats
            basic_stats = self.get_basic_stats()
            stats_df = pd.DataFrame([basic_stats])
            stats_df.to_excel(writer, sheet_name="Basic_Stats", index=False)
            
            # Team matchups
            matchup_df = self.analyze_team_matchups()
            matchup_df.to_excel(writer, sheet_name="Team_Matchups", index=False)
            
            # Aggressive vs Conservative analysis
            agg_vs_cons = self.analyze_aggressive_vs_conservative()
            if "error" not in agg_vs_cons:
                agg_vs_cons_df = pd.DataFrame([agg_vs_cons])
                agg_vs_cons_df.to_excel(writer, sheet_name="Aggressive_vs_Conservative", index=False)
            
            # Round-by-round analysis
            try:
                round_df = self.analyze_round_by_round("aggressive_vs_conservative")
                round_df.to_excel(writer, sheet_name="Round_by_Round", index=False)
            except Exception as e:
                print(f"Could not export round-by-round analysis: {e}")
            
            # Raw game data
            self.df.to_excel(writer, sheet_name="Raw_Game_Data", index=False)
        
        print(f"Analysis exported to: {output_file}")
    
    def print_summary(self) -> None:
        """Print a summary of the analysis."""
        if self.df is None:
            print("No games loaded. Call load_games() first.")
            return
        
        print("=" * 80)
        print("EUCRE GAME ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Basic stats
        basic_stats = self.get_basic_stats()
        print(f"\nTotal Games: {basic_stats['total_games']}")
        print(f"Average Rounds per Game: {basic_stats['avg_rounds_per_game']:.2f}")
        print(f"Average Game Duration: {basic_stats['avg_game_duration']:.2f} seconds")
        
        # Team configurations
        print(f"\nTeam Configurations:")
        for config, count in basic_stats['team_configs'].items():
            print(f"  {config}: {count} games")
        
        # Win rates
        print(f"\nOverall Win Rates:")
        for team, count in basic_stats['winning_teams'].items():
            rate = count / basic_stats['total_games']
            print(f"  {team}: {count} wins ({rate:.1%})")
        
        # Aggressive vs Conservative analysis
        print(f"\nAggressive vs Conservative Analysis:")
        agg_vs_cons = self.analyze_aggressive_vs_conservative()
        if "error" not in agg_vs_cons:
            print(f"  Total Games: {agg_vs_cons['total_games']}")
            print(f"  Aggressive Win Rate: {agg_vs_cons['aggressive_team']['win_rate']:.1%}")
            print(f"  Conservative Win Rate: {agg_vs_cons['conservative_team']['win_rate']:.1%}")
            print(f"  Tie Rate: {agg_vs_cons['tie_rate']:.1%}")
            print(f"  Average Rounds: {agg_vs_cons['avg_rounds_per_game']:.2f}")
        else:
            print(f"  {agg_vs_cons['error']}")
        
        print("=" * 80) 