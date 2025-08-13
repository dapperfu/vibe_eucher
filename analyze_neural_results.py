#!/usr/bin/env python3
"""Script to analyze neural network tournament results."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns


class NeuralTournamentAnalyzer:
    """Analyzes neural network tournament results."""
    
    def __init__(self, results_dir: str = "neural_tournament_results"):
        """Initialize the analyzer.
        
        Parameters
        ----------
        results_dir : str
            Directory containing tournament results
        """
        self.results_dir = Path(results_dir)
        self.game_files = list(self.results_dir.glob("*.json"))
        self.summary_files = list(self.results_dir.glob("summary_*.json"))
        
    def load_games(self) -> List[Dict[str, Any]]:
        """Load all game results.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of game result dictionaries
        """
        games = []
        for game_file in self.game_files:
            if game_file.name.startswith("summary_"):
                continue
            try:
                with open(game_file, 'r') as f:
                    game_data = json.load(f)
                    if "error" not in game_data:
                        games.append(game_data)
            except Exception as e:
                print(f"Error loading {game_file}: {e}")
        
        return games
    
    def load_summaries(self) -> List[Dict[str, Any]]:
        """Load all summary files.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of summary dictionaries
        """
        summaries = []
        for summary_file in self.summary_files:
            try:
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                    summaries.append(summary_data)
            except Exception as e:
                print(f"Error loading {summary_file}: {e}")
        
        return summaries
    
    def analyze_tournament_results(self) -> Dict[str, Any]:
        """Analyze tournament results and generate comprehensive statistics.
        
        Returns
        -------
        Dict[str, Any]
            Comprehensive analysis results
        """
        games = self.load_games()
        summaries = self.load_summaries()
        
        if not games:
            print("No games found to analyze")
            return {}
        
        print(f"Analyzing {len(games)} games...")
        
        # Basic statistics
        total_games = len(games)
        successful_games = len([g for g in games if "error" not in g])
        
        # Team performance analysis
        team_stats = {}
        model_performance = {}
        
        for game in games:
            if "error" in game:
                continue
                
            config = game["team_config"]
            team1_models = [p[1] for p in game["team1"]["players"]]
            team2_models = [p[1] for p in game["team2"]["players"]]
            
            # Track team performance
            if config not in team_stats:
                team_stats[config] = {
                    "team1_wins": 0,
                    "team2_wins": 0,
                    "ties": 0,
                    "team1_total_tricks": 0,
                    "team2_total_tricks": 0,
                    "team_sets": 0,
                    "games": 0
                }
            
            team_stats[config]["games"] += 1
            
            if game["winner"] == "team1":
                team_stats[config]["team1_wins"] += 1
            elif game["winner"] == "team2":
                team_stats[config]["team2_wins"] += 1
            else:
                team_stats[config]["ties"] += 1
            
            team_stats[config]["team1_total_tricks"] += game["total_tricks_team1"]
            team_stats[config]["team2_total_tricks"] += game["total_tricks_team2"]
            
            if game["team_set"]:
                team_stats[config]["team_sets"] += 1
            
            # Track individual model performance
            for model in team1_models + team2_models:
                if model not in model_performance:
                    model_performance[model] = {
                        "games_played": 0,
                        "wins": 0,
                        "total_tricks": 0,
                        "teams": []
                    }
                
                model_performance[model]["games_played"] += 1
                
                # Determine if this model won
                if (model in team1_models and game["winner"] == "team1") or \
                   (model in team2_models and game["winner"] == "team2"):
                    model_performance[model]["wins"] += 1
                
                # Track tricks
                if model in team1_models:
                    model_performance[model]["total_tricks"] += game["total_tricks_team1"]
                    if "team1" not in model_performance[model]["teams"]:
                        model_performance[model]["teams"].append("team1")
                else:
                    model_performance[model]["total_tricks"] += game["total_tricks_team2"]
                    if "team2" not in model_performance[model]["teams"]:
                        model_performance[model]["teams"].append("team2")
        
        # Calculate win rates and averages
        for config, stats in team_stats.items():
            if stats["games"] > 0:
                stats["team1_win_rate"] = stats["team1_wins"] / stats["games"]
                stats["team2_win_rate"] = stats["team2_wins"] / stats["games"]
                stats["team1_avg_tricks"] = stats["team1_total_tricks"] / stats["games"]
                stats["team2_avg_tricks"] = stats["team2_total_tricks"] / stats["games"]
                stats["team_set_rate"] = stats["team_sets"] / stats["games"]
        
        for model, stats in model_performance.items():
            if stats["games_played"] > 0:
                stats["win_rate"] = stats["wins"] / stats["games_played"]
                stats["avg_tricks"] = stats["total_tricks"] / stats["games_played"]
        
        # Compile analysis
        analysis = {
            "total_games": total_games,
            "successful_games": successful_games,
            "success_rate": successful_games / total_games if total_games > 0 else 0,
            "team_stats": team_stats,
            "model_performance": model_performance,
            "summaries": summaries
        }
        
        return analysis
    
    def print_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print comprehensive analysis results.
        
        Parameters
        ----------
        analysis : Dict[str, Any]
            Analysis results to print
        """
        if not analysis:
            print("No analysis to print")
            return
        
        print("\n" + "="*80)
        print("🧠 NEURAL NETWORK TOURNAMENT ANALYSIS")
        print("="*80)
        
        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total games: {analysis['total_games']}")
        print(f"  Successful games: {analysis['successful_games']}")
        print(f"  Success rate: {analysis['success_rate']:.2%}")
        
        # Team configuration analysis
        print(f"\n🏆 TEAM CONFIGURATION ANALYSIS:")
        for config, stats in analysis["team_stats"].items():
            print(f"\n  {config}:")
            print(f"    Games played: {stats['games']}")
            print(f"    Team 1 wins: {stats['team1_wins']} ({stats['team1_win_rate']:.2%})")
            print(f"    Team 2 wins: {stats['team2_wins']} ({stats['team2_win_rate']:.2%})")
            print(f"    Ties: {stats['ties']}")
            print(f"    Team 1 avg tricks: {stats['team1_avg_tricks']:.2f}")
            print(f"    Team 2 avg tricks: {stats['team2_avg_tricks']:.2f}")
            print(f"    Team sets: {stats['team_sets']} ({stats['team_set_rate']:.2%})")
        
        # Individual model performance
        print(f"\n🧠 INDIVIDUAL MODEL PERFORMANCE:")
        for model, stats in analysis["model_performance"].items():
            print(f"\n  {model}:")
            print(f"    Games played: {stats['games_played']}")
            print(f"    Wins: {stats['wins']} ({stats['win_rate']:.2%})")
            print(f"    Average tricks: {stats['avg_tricks']:.2f}")
            print(f"    Teams: {', '.join(stats['teams'])}")
        
        # Summary analysis
        if analysis["summaries"]:
            print(f"\n📋 SUMMARY FILES:")
            for summary in analysis["summaries"]:
                print(f"  {summary.get('config_name', 'Unknown')}: {summary.get('total_games', 0)} games")
    
    def generate_visualizations(self, output_dir: str = "analysis_plots") -> None:
        """Generate visualization plots for the tournament results.
        
        Parameters
        ----------
        output_dir : str
            Directory to save plots
        """
        analysis = self.analyze_tournament_results()
        if not analysis:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Win rates by configuration
        if analysis["team_stats"]:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            configs = list(analysis["team_stats"].keys())
            team1_win_rates = [analysis["team_stats"][c]["team1_win_rate"] for c in configs]
            team2_win_rates = [analysis["team_stats"][c]["team2_win_rate"] for c in configs]
            
            x = range(len(configs))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], team1_win_rates, width, label='Team 1', alpha=0.8)
            ax.bar([i + width/2 for i in x], team2_win_rates, width, label='Team 2', alpha=0.8)
            
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Win Rate')
            ax.set_title('Win Rates by Team Configuration')
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / "win_rates_by_config.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Model performance comparison
        if analysis["model_performance"]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            models = list(analysis["model_performance"].keys())
            win_rates = [analysis["model_performance"][m]["win_rate"] for m in models]
            avg_tricks = [analysis["model_performance"][m]["avg_tricks"] for m in models]
            
            # Win rates
            bars1 = ax1.bar(models, win_rates, alpha=0.8, color='skyblue')
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Win Rate')
            ax1.set_title('Model Win Rates')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2%}', ha='center', va='bottom')
            
            # Average tricks
            bars2 = ax2.bar(models, avg_tricks, alpha=0.8, color='lightcoral')
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Average Tricks per Game')
            ax2.set_title('Model Average Tricks')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / "model_performance.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Tricks distribution
        if analysis["team_stats"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            configs = list(analysis["team_stats"].keys())
            team1_tricks = [analysis["team_stats"][c]["team1_avg_tricks"] for c in configs]
            team2_tricks = [analysis["team_stats"][c]["team2_avg_tricks"] for c in configs]
            
            x = range(len(configs))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], team1_tricks, width, label='Team 1', alpha=0.8)
            ax.bar([i + width/2 for i in x], team2_tricks, width, label='Team 2', alpha=0.8)
            
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Average Tricks per Game')
            ax.set_title('Tricks Distribution by Team Configuration')
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / "tricks_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\n📊 Visualizations saved to: {output_path}")
    
    def export_analysis(self, filename: str = "neural_tournament_analysis.xlsx") -> None:
        """Export analysis results to Excel file.
        
        Parameters
        ----------
        filename : str
            Output Excel filename
        """
        analysis = self.analyze_tournament_results()
        if not analysis:
            return
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Team stats
            if analysis["team_stats"]:
                team_df = pd.DataFrame(analysis["team_stats"]).T
                team_df.to_excel(writer, sheet_name='Team_Statistics')
            
            # Model performance
            if analysis["model_performance"]:
                model_df = pd.DataFrame(analysis["model_performance"]).T
                model_df.to_excel(writer, sheet_name='Model_Performance')
            
            # Summary
            if analysis["summaries"]:
                summary_df = pd.DataFrame(analysis["summaries"])
                summary_df.to_excel(writer, sheet_name='Tournament_Summaries')
        
        print(f"\n📊 Analysis exported to: {filename}")


def main():
    """Main function to run the analysis."""
    print("🧠 Neural Network Tournament Results Analyzer")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = NeuralTournamentAnalyzer("neural_tournament_results")
    
    # Check if results exist
    if not analyzer.game_files:
        print("❌ No tournament results found in 'neural_tournament_results' directory")
        print("Please run a tournament first using:")
        print("  python run_neural_tournament.py")
        print("  or")
        print("  python -m euchre.cli run-neural-games -m1 Alice -m2 Bob -n 1000")
        return
    
    # Run analysis
    analysis = analyzer.analyze_tournament_results()
    
    # Print results
    analyzer.print_analysis(analysis)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    analyzer.generate_visualizations()
    
    # Export to Excel
    print("\n📊 Exporting analysis to Excel...")
    analyzer.export_analysis()
    
    print("\n✅ Analysis completed!")


if __name__ == "__main__":
    main() 