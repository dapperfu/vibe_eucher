#!/usr/bin/env python3
"""
Train Optimal Classic AI Bots with Balanced Playing Profiles

This script uses the optimal parameters identified in the analysis to create
AI profiles that are more evenly matched and competitive.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

from euchre.game import EuchreGame
from euchre.ai.ai_factory import AIFactory
from euchre.models import Player, PlayerType


def create_optimal_ai_players():
    """Create AI players with optimal parameters for balanced gameplay."""
    players = []
    
    # Use the optimal parameters from the analysis
    profiles = [
        ("Alice", "conservative", 0.15),    # Optimal conservative
        ("Bob", "balanced", 0.45),         # Optimal balanced
        ("Charlie", "aggressive", 0.75),   # Moderate aggressive
        ("David", "opportunistic", 0.65),  # Moderate opportunistic
    ]
    
    for name, ai_type, risk_ratio in profiles:
        player = AIFactory.create_ai_player(name, ai_type, risk_ratio)
        players.append(player)
        print(f"Created {name}: {ai_type} AI with risk ratio {risk_ratio}")
    
    return players


def run_training_games(num_games=3000, output_dir="training_results"):
    """Run training games and collect results."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create AI players
    players = create_optimal_ai_players()
    
    # Initialize results tracking
    results = {
        "training_config": {
            "num_games": num_games,
            "ai_types": ["conservative", "balanced", "aggressive", "opportunistic"],
            "risk_ratios": [0.15, 0.45, 0.75, 0.65],
            "strategy": "Optimal parameters from analysis",
            "timestamp": datetime.now().isoformat()
        },
        "game_results": [],
        "player_stats": {
            "Alice": {"games_played": 0, "games_won": 0, "total_score": 0, "team_sets": 0},
            "Bob": {"games_played": 0, "games_won": 0, "total_score": 0, "team_sets": 0},
            "Charlie": {"games_played": 0, "games_won": 0, "total_score": 0, "team_sets": 0},
            "David": {"games_played": 0, "games_won": 0, "total_score": 0, "team_sets": 0}
        },
        "overall_stats": {
            "total_games": 0,
            "team1_wins": 0,  # Alice & Charlie (Conservative + Aggressive)
            "team2_wins": 0,  # Bob & David (Balanced + Opportunistic)
            "total_team_sets": 0,
            "average_game_duration": 0.0
        }
    }
    
    print(f"\n🎯 Starting optimal training session with {num_games} games...")
    print("=" * 60)
    print("Using optimal parameters from analysis:")
    print("  • Alice (Conservative 0.15) + Charlie (Aggressive 0.75)")
    print("  • Bob (Balanced 0.45) + David (Opportunistic 0.65)")
    print("=" * 60)
    
    total_duration = 0
    
    for game_num in range(1, num_games + 1):
        if game_num % 300 == 0:
            print(f"Playing game {game_num}/{num_games}...")
        
        # Create new game
        game = EuchreGame(players, verbose=False, very_verbose=False)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Run the game
            game.start_new_game()
            game.run_full_game()
            
            # Calculate game duration
            game_duration = time.time() - start_time
            total_duration += game_duration
            
            # Get game results
            team1_score = game.players[0].score + game.players[2].score  # Alice + Charlie
            team2_score = game.players[1].score + game.players[3].score  # Bob + David
            
            # Determine winner
            if team1_score > team2_score:
                winner = "Team 1 (Alice & Charlie)"
                results["overall_stats"]["team1_wins"] += 1
            else:
                winner = "Team 2 (Bob & David)"
                results["overall_stats"]["team2_wins"] += 1
            
            # Check for team set
            team_set = False
            if game.scoring_manager:
                # Check if either team was set
                if team1_score < 3 or team2_score < 3:
                    team_set = True
                    results["overall_stats"]["total_team_sets"] += 1
            
            # Record game result
            game_result = {
                "game_id": f"optimal_game_{game_num:06d}",
                "winner": winner,
                "team1_score": team1_score,
                "team2_score": team2_score,
                "team_set": team_set,
                "duration_seconds": game_duration,
                "timestamp": datetime.now().isoformat()
            }
            results["game_results"].append(game_result)
            
            # Update player statistics
            for i, player in enumerate(game.players):
                player_name = player.name
                results["player_stats"][player_name]["games_played"] += 1
                results["player_stats"][player_name]["total_score"] += player.score
                
                # Check if player won
                if (i % 2 == 0 and winner == "Team 1 (Alice & Charlie)") or \
                   (i % 2 == 1 and winner == "Team 2 (Bob & David)"):
                    results["player_stats"][player_name]["games_won"] += 1
                
                # Check if player's team was set
                if team_set:
                    results["player_stats"][player_name]["team_sets"] += 1
            
            results["overall_stats"]["total_games"] += 1
            
        except Exception as e:
            print(f"Error in game {game_num}: {e}")
            continue
    
    # Calculate final statistics
    if results["overall_stats"]["total_games"] > 0:
        results["overall_stats"]["average_game_duration"] = total_duration / results["overall_stats"]["total_games"]
    
    return results


def save_results(results, output_dir):
    """Save training results to files."""
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"optimal_ai_training_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary_file = Path(output_dir) / f"optimal_ai_training_summary_{timestamp}.txt"
    
    with open(summary_file, 'w') as f:
        f.write("OPTIMAL CLASSIC AI TRAINING SESSION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        # Training configuration
        f.write("TRAINING CONFIGURATION:\n")
        f.write(f"Games: {results['training_config']['num_games']}\n")
        f.write(f"AI Types: {', '.join(results['training_config']['ai_types'])}\n")
        f.write(f"Risk Ratios: {', '.join(map(str, results['training_config']['risk_ratios']))}\n")
        f.write(f"Strategy: {results['training_config']['strategy']}\n")
        f.write(f"Timestamp: {results['training_config']['timestamp']}\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS:\n")
        overall = results['overall_stats']
        f.write(f"Total Games: {overall['total_games']}\n")
        f.write(f"Team 1 Wins (Alice & Charlie): {overall['team1_wins']} ({overall['team1_wins']/overall['total_games']*100:.1f}%)\n")
        f.write(f"Team 2 Wins (Bob & David): {overall['team2_wins']} ({overall['team2_wins']/overall['total_games']*100:.1f}%)\n")
        f.write(f"Team Sets: {overall['total_team_sets']}\n")
        f.write(f"Average Game Duration: {overall['average_game_duration']:.2f} seconds\n\n")
        
        # Player statistics
        f.write("PLAYER STATISTICS:\n")
        for player, stats in results['player_stats'].items():
            if stats['games_played'] > 0:
                f.write(f"\n{player}:\n")
                f.write(f"  Games Played: {stats['games_played']}\n")
                f.write(f"  Games Won: {stats['games_won']} ({stats['games_won']/stats['games_played']*100:.1f}%)\n")
                f.write(f"  Average Score: {stats['total_score']/stats['games_played']:.2f}\n")
                f.write(f"  Team Sets: {stats['team_sets']}\n")
    
    print(f"\n📊 Results saved to: {results_file}")
    print(f"📝 Summary saved to: {summary_file}")
    
    return results_file, summary_file


def print_summary(results):
    """Print a summary of the training results."""
    
    print("\n" + "=" * 60)
    print("🎯 OPTIMAL TRAINING SESSION COMPLETE!")
    print("=" * 60)
    
    # Overall statistics
    overall = results['overall_stats']
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total Games: {overall['total_games']}")
    print(f"   Team 1 Wins (Alice & Charlie): {overall['team1_wins']} ({overall['team1_wins']/overall['total_games']*100:.1f}%)")
    print(f"   Team 2 Wins (Bob & David): {overall['team2_wins']} ({overall['team2_wins']/overall['total_games']*100:.1f}%)")
    print(f"   Team Sets: {overall['total_team_sets']}")
    print(f"   Average Game Duration: {overall['average_game_duration']:.2f} seconds")
    
    # Player statistics
    print(f"\n👥 PLAYER PERFORMANCE:")
    for player, stats in results['player_stats'].items():
        if stats['games_played'] > 0:
            win_rate = stats['games_won'] / stats['games_played'] * 100
            avg_score = stats['total_score'] / stats['games_played']
            print(f"   {player}: {win_rate:.1f}% win rate, {avg_score:.1f} avg score")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    team1_win_rate = overall['team1_wins'] / overall['total_games'] * 100
    team2_win_rate = overall['team2_wins'] / overall['total_games'] * 100
    
    if abs(team1_win_rate - team2_win_rate) < 10:
        print(f"   ✅ BALANCED: Teams are well-matched (difference < 10%)")
    elif abs(team1_win_rate - team2_win_rate) < 20:
        print(f"   ⚠️  MODERATELY BALANCED: Some imbalance (difference {abs(team1_win_rate - team2_win_rate):.1f}%)")
    else:
        print(f"   ❌ IMBALANCED: Significant difference (difference {abs(team1_win_rate - team2_win_rate):.1f}%)")
    
    print("\n" + "=" * 60)


def main():
    """Main training function."""
    
    print("🧠 Optimal Classic AI Profile Training")
    print("=" * 40)
    print("This script uses the optimal parameters from analysis:")
    print("  • Alice: Conservative AI (risk ratio 0.15) - Optimal conservative")
    print("  • Bob: Balanced AI (risk ratio 0.45) - Optimal balanced")
    print("  • Charlie: Aggressive AI (risk ratio 0.75) - Moderate aggressive")
    print("  • David: Opportunistic AI (risk ratio 0.65) - Moderate opportunistic")
    print()
    print("Expected outcome: More balanced and competitive gameplay")
    print()
    
    # Configuration
    num_games = 3000  # More games for better statistics
    output_dir = "training_results"
    
    print(f"🎯 Training Configuration:")
    print(f"   Number of games: {num_games}")
    print(f"   Output directory: {output_dir}")
    print()
    
    # Run training
    try:
        results = run_training_games(num_games, output_dir)
        
        # Save results
        results_file, summary_file = save_results(results, output_dir)
        
        # Print summary
        print_summary(results)
        
        print(f"\n✅ Optimal training completed successfully!")
        print(f"📁 Check the results in: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main() 