#!/usr/bin/env python3
"""
Train Adaptive AI with Balanced Scoring System

This script combines the adaptive AI that learns from losses with
the balanced scoring system that adjusts team advantages for more
competitive gameplay.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

from euchre.game import EuchreGame
from euchre.ai.adaptive_ai import AdaptiveAIFactory
from euchre.scoring.balanced_scoring import create_scoring_system, ScoringMode
from euchre.models import Player, PlayerType


def create_adaptive_balanced_team():
    """Create a team of adaptive AI players with balanced scoring."""
    print("🧠 Creating Adaptive AI Team with Balanced Scoring...")
    
    # Create balanced team with different learning rates
    players = AdaptiveAIFactory.create_balanced_team()
    
    print("Created adaptive AI players:")
    for player in players:
        print(f"  {player.name}: {player.base_ai_type} AI (risk: {player.initial_risk_ratio:.2f}, learning: {player.learning_rate:.2f})")
    
    return players


def create_balanced_scoring_system():
    """Create a balanced scoring system."""
    print("⚖️  Creating Balanced Scoring System...")
    
    # Create balanced scoring with moderate handicaps
    scoring = create_scoring_system(
        mode=ScoringMode.BALANCED,
        handicap_threshold=0.6,
        max_handicap=2,
        performance_window=10,
        random_factor=0.05
    )
    
    print("Balanced scoring system created with:")
    print(f"  Mode: {scoring.config.mode.value}")
    print(f"  Handicap threshold: {scoring.config.handicap_threshold}")
    print(f"  Max handicap: {scoring.config.max_handicap}")
    print(f"  Performance window: {scoring.config.performance_window}")
    print(f"  Random factor: {scoring.config.random_factor}")
    
    return scoring


def run_adaptive_balanced_training(num_games=3000, output_dir="adaptive_balanced_results"):
    """Run training games with adaptive AI and balanced scoring."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create adaptive AI players
    players = create_adaptive_balanced_team()
    
    # Create balanced scoring system
    scoring_system = create_balanced_scoring_system()
    
    # Initialize results tracking
    results = {
        "training_config": {
            "num_games": num_games,
            "ai_types": ["conservative", "balanced", "aggressive", "opportunistic"],
            "initial_risk_ratios": [0.3, 0.5, 0.7, 0.6],
            "learning_rates": [0.15, 0.1, 0.15, 0.1],
            "scoring_mode": "balanced",
            "strategy": "Adaptive AI + Balanced Scoring for competitive gameplay",
            "timestamp": datetime.now().isoformat()
        },
        "game_results": [],
        "player_evolution": {
            "Alice": {"risk_ratios": [], "win_rates": [], "adaptations": []},
            "Bob": {"risk_ratios": [], "win_rates": [], "adaptations": []},
            "Charlie": {"risk_ratios": [], "win_rates": [], "adaptations": []},
            "David": {"risk_ratios": [], "win_rates": [], "adaptations": []}
        },
        "scoring_balance": {
            "total_adjustments": 0,
            "handicaps_applied": 0,
            "team_performance": {},
            "balance_effectiveness": []
        },
        "overall_stats": {
            "total_games": 0,
            "team1_wins": 0,  # Alice & Charlie
            "team2_wins": 0,  # Bob & David
            "total_team_sets": 0,
            "average_game_duration": 0.0,
            "total_adaptations": 0,
            "close_games": 0  # Games decided by 1-2 points
        }
    }
    
    print(f"\n🎯 Starting Adaptive AI + Balanced Scoring training with {num_games} games...")
    print("=" * 80)
    print("This system combines:")
    print("  • AI that learns from losses and adjusts strategies")
    print("  • Balanced scoring that adjusts team advantages")
    print("  • Goal: More competitive and balanced gameplay!")
    print("=" * 80)
    
    total_duration = 0
    last_status_check = 0
    
    for game_num in range(1, num_games + 1):
        if game_num % 300 == 0:
            print(f"Playing game {game_num}/{num_games}...")
            # Show current status
            _show_training_status(players, scoring_system, results)
        
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
            team1_score = game.players[0].score + game.players[2].score  # Alice & Charlie
            team2_score = game.players[1].score + game.players[3].score  # Bob & David
            
            # Determine winner
            if team1_score > team2_score:
                winner = "Team 1 (Alice & Charlie)"
                results["overall_stats"]["team1_wins"] += 1
                team1_won, team2_won = True, False
            else:
                winner = "Team 2 (Bob & David)"
                results["overall_stats"]["team2_wins"] += 1
                team1_won, team2_won = False, True
            
            # Check for team set
            team_set = False
            if game.scoring_manager:
                if team1_score < 3 or team2_score < 3:
                    team_set = True
                    results["overall_stats"]["total_team_sets"] += 1
            
            # Check if game was close (decided by 1-2 points)
            score_diff = abs(team1_score - team2_score)
            if score_diff <= 2:
                results["overall_stats"]["close_games"] += 1
            
            # Record game result
            game_result = {
                "game_id": f"adaptive_balanced_game_{game_num:06d}",
                "winner": winner,
                "team1_score": team1_score,
                "team2_score": team2_score,
                "score_difference": score_diff,
                "team_set": team_set,
                "duration_seconds": game_duration,
                "timestamp": datetime.now().isoformat()
            }
            results["game_results"].append(game_result)
            
            # Update player evolution tracking
            for i, player in enumerate(players):
                player_name = player.name
                
                # Get current adaptation status
                adaptation_summary = player.get_adaptation_summary()
                
                # Track risk ratio evolution
                results["player_evolution"][player_name]["risk_ratios"].append(
                    adaptation_summary["current_risk_ratio"]
                )
                
                # Track win rate evolution
                results["player_evolution"][player_name]["win_rates"].append(
                    adaptation_summary["memory"]["win_rate"]
                )
                
                # Track adaptations
                results["player_evolution"][player_name]["adaptations"].append(
                    adaptation_summary["memory"]["total_adjustments"]
                )
            
            # Count total adaptations
            total_adaptations = sum(
                player.memory.total_adjustments for player in players
            )
            results["overall_stats"]["total_adaptations"] = total_adaptations
            
            # Update scoring system performance tracking
            scoring_system.update_team_performance(
                team1_score, team2_score, team1_won, team2_won
            )
            
            # Update scoring balance tracking
            balance_summary = scoring_system.get_balance_summary()
            results["scoring_balance"]["total_adjustments"] = balance_summary["total_adjustments"]
            results["scoring_balance"]["team_performance"] = balance_summary["team_performance"]
            
            # Notify players of game end for learning
            for i, player in enumerate(players):
                # Determine if this player won
                player_won = (i % 2 == 0 and team1_won) or (i % 2 == 1 and team2_won)
                
                # Calculate tricks won/lost (simplified)
                tricks_won = game.players[i].score
                tricks_lost = 5 - tricks_won  # Assuming 5 tricks per round
                
                # Check if team was set
                was_set = team_set and not player_won
                set_opponent = team_set and player_won
                
                # Notify player for learning
                if hasattr(player, 'on_game_end'):
                    player.on_game_end(player_won, tricks_won, tricks_lost, was_set, set_opponent)
            
            results["overall_stats"]["total_games"] += 1
            
        except Exception as e:
            print(f"Error in game {game_num}: {e}")
            continue
    
    # Calculate final statistics
    if results["overall_stats"]["total_games"] > 0:
        results["overall_stats"]["average_game_duration"] = total_duration / results["overall_stats"]["total_games"]
    
    return results


def _show_training_status(players, scoring_system, results):
    """Show current training status."""
    print(f"\n📊 Training Status (Game {results['overall_stats']['total_games']}):")
    print("-" * 70)
    
    # Show player adaptation status
    print("🧠 Player Adaptations:")
    for player in players:
        summary = player.get_adaptation_summary()
        print(f"  {player.name}: Risk {summary['current_risk_ratio']:.2f} "
              f"(Win Rate: {summary['memory']['win_rate']:.1%}, "
              f"Adaptations: {summary['memory']['total_adjustments']})")
    
    # Show scoring balance status
    balance_summary = scoring_system.get_balance_summary()
    print(f"\n⚖️  Scoring Balance:")
    print(f"  Total Adjustments: {balance_summary['total_adjustments']}")
    print(f"  Team 1 Performance: {balance_summary['team_performance']['team1']['win_rate']:.1%}")
    print(f"  Team 2 Performance: {balance_summary['team_performance']['team2']['win_rate']:.1%}")
    
    print("-" * 70)


def save_results(results, output_dir):
    """Save training results to files."""
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"adaptive_balanced_training_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary_file = Path(output_dir) / f"adaptive_balanced_training_summary_{timestamp}.txt"
    
    with open(summary_file, 'w') as f:
        f.write("ADAPTIVE AI + BALANCED SCORING TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        # Training configuration
        f.write("TRAINING CONFIGURATION:\n")
        f.write(f"Games: {results['training_config']['num_games']}\n")
        f.write(f"AI Types: {', '.join(results['training_config']['ai_types'])}\n")
        f.write(f"Initial Risk Ratios: {', '.join(map(str, results['training_config']['initial_risk_ratios']))}\n")
        f.write(f"Learning Rates: {', '.join(map(str, results['training_config']['learning_rates']))}\n")
        f.write(f"Scoring Mode: {results['training_config']['scoring_mode']}\n")
        f.write(f"Strategy: {results['training_config']['strategy']}\n")
        f.write(f"Timestamp: {results['training_config']['timestamp']}\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS:\n")
        overall = results['overall_stats']
        f.write(f"Total Games: {overall['total_games']}\n")
        f.write(f"Team 1 Wins (Alice & Charlie): {overall['team1_wins']} ({overall['team1_wins']/overall['total_games']*100:.1f}%)\n")
        f.write(f"Team 2 Wins (Bob & David): {overall['team2_wins']} ({overall['team2_wins']/overall['total_games']*100:.1f}%)\n")
        f.write(f"Team Sets: {overall['total_team_sets']}\n")
        f.write(f"Total Adaptations: {overall['total_adaptations']}\n")
        f.write(f"Close Games (≤2 points): {overall['close_games']} ({overall['close_games']/overall['total_games']*100:.1f}%)\n")
        f.write(f"Average Game Duration: {overall['average_game_duration']:.2f} seconds\n\n")
        
        # Scoring balance
        f.write("SCORING BALANCE:\n")
        balance = results['scoring_balance']
        f.write(f"Total Adjustments: {balance['total_adjustments']}\n")
        f.write(f"Team Performance: {balance['team_performance']}\n\n")
        
        # Player evolution
        f.write("PLAYER EVOLUTION:\n")
        for player_name, evolution in results['player_evolution'].items():
            f.write(f"\n{player_name}:\n")
            if evolution['risk_ratios']:
                initial_risk = evolution['risk_ratios'][0]
                final_risk = evolution['risk_ratios'][-1]
                risk_change = final_risk - initial_risk
                f.write(f"  Risk Ratio: {initial_risk:.2f} → {final_risk:.2f} (Δ{risk_change:+.2f})\n")
            
            if evolution['win_rates']:
                final_win_rate = evolution['win_rates'][-1]
                f.write(f"  Final Win Rate: {final_win_rate:.1%}\n")
            
            if evolution['adaptations']:
                total_adaptations = evolution['adaptations'][-1]
                f.write(f"  Total Adaptations: {total_adaptations}\n")
    
    print(f"\n📊 Results saved to: {results_file}")
    print(f"📝 Summary saved to: {summary_file}")
    
    return results_file, summary_file


def print_summary(results):
    """Print a summary of the training results."""
    
    print("\n" + "=" * 80)
    print("🎯 ADAPTIVE AI + BALANCED SCORING TRAINING COMPLETE!")
    print("=" * 80)
    
    # Overall statistics
    overall = results['overall_stats']
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total Games: {overall['total_games']}")
    print(f"   Team 1 Wins (Alice & Charlie): {overall['team1_wins']} ({overall['team1_wins']/overall['total_games']*100:.1f}%)")
    print(f"   Team 2 Wins (Bob & David): {overall['team2_wins']} ({overall['team2_wins']/overall['total_games']*100:.1f}%)")
    print(f"   Team Sets: {overall['total_team_sets']}")
    print(f"   Total Adaptations: {overall['total_adaptations']}")
    print(f"   Close Games (≤2 points): {overall['close_games']} ({overall['close_games']/overall['total_games']*100:.1f}%)")
    print(f"   Average Game Duration: {overall['average_game_duration']:.2f} seconds")
    
    # Player evolution summary
    print(f"\n🧠 PLAYER EVOLUTION:")
    for player_name, evolution in results['player_evolution'].items():
        if evolution['risk_ratios']:
            initial_risk = evolution['risk_ratios'][0]
            final_risk = evolution['risk_ratios'][-1]
            risk_change = final_risk - initial_risk
            print(f"   {player_name}: Risk {initial_risk:.2f} → {final_risk:.2f} (Δ{risk_change:+.2f})")
    
    # Scoring balance summary
    balance = results['scoring_balance']
    print(f"\n⚖️  SCORING BALANCE:")
    print(f"   Total Adjustments: {balance['total_adjustments']}")
    
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
    
    if overall['total_adaptations'] > 0:
        print(f"   🧠 LEARNING: AI adapted {overall['total_adaptations']} times during training")
    else:
        print(f"   ⚠️  NO LEARNING: AI did not adapt during training")
    
    if balance['total_adjustments'] > 0:
        print(f"   ⚖️  BALANCING: Scoring system made {balance['total_adjustments']} adjustments")
    else:
        print(f"   ⚠️  NO BALANCING: Scoring system made no adjustments")
    
    close_game_rate = overall['close_games'] / overall['total_games'] * 100
    if close_game_rate > 30:
        print(f"   🎯 COMPETITIVE: {close_game_rate:.1f}% of games were close (≤2 points)")
    else:
        print(f"   ⚠️  LESS COMPETITIVE: Only {close_game_rate:.1f}% of games were close")
    
    print("\n" + "=" * 80)


def main():
    """Main training function."""
    
    print("🧠⚖️  Adaptive AI + Balanced Scoring Training System")
    print("=" * 50)
    print("This system combines two advanced features:")
    print("  1. Adaptive AI that learns from losses")
    print("  2. Balanced scoring that adjusts team advantages")
    print()
    print("Goal: Create more competitive and balanced gameplay!")
    print()
    
    # Configuration
    num_games = 3000  # Good number for learning and balancing to take effect
    output_dir = "adaptive_balanced_results"
    
    print(f"🎯 Training Configuration:")
    print(f"   Number of games: {num_games}")
    print(f"   Output directory: {output_dir}")
    print()
    
    # Run training
    try:
        results = run_adaptive_balanced_training(num_games, output_dir)
        
        # Save results
        results_file, summary_file = save_results(results, output_dir)
        
        # Print summary
        print_summary(results)
        
        print(f"\n✅ Adaptive AI + Balanced Scoring training completed successfully!")
        print(f"📁 Check the results in: {output_dir}")
        print(f"🧠 AI players have learned and adapted their strategies!")
        print(f"⚖️  Scoring system has balanced team advantages!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main() 