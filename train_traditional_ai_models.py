#!/usr/bin/env python3
"""
Traditional AI Model Training Script

This script trains each of the traditional AI models (aggressive, conservative, 
balanced, opportunistic) on a thousand games to improve their performance.

Each AI type will be trained in different team configurations to develop
diverse strategies and improve overall gameplay.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-12
Context: Training traditional AI models on thousands of games
"""

import os
import sys
import time
import click
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from euchre.ai_training_framework import AITrainingFramework, TrainingConfig, TrainingMode


def train_ai_model(ai_type: str, num_games: int = 1000, output_dir: str = "training_results") -> None:
    """Train a specific AI model on the specified number of games.
    
    Parameters
    ----------
    ai_type : str
        The AI type to train: "aggressive", "conservative", "balanced", "opportunistic"
    num_games : int
        Number of games to train on
    output_dir : str
        Output directory for training results
    """
    print(f"\n🎯 Training {ai_type.upper()} AI Model")
    print(f"📊 Games: {num_games}")
    print(f"📁 Output: {output_dir}")
    print("=" * 60)
    
    # Create training configuration for this AI type
    # Use MIXED_TEAMS mode but with all players having the same AI type
    config = TrainingConfig(
        mode=TrainingMode.MIXED_TEAMS,
        num_games=num_games,
        ai_types=[ai_type] * 4,  # All 4 players use the same AI type
        risk_ratios=[0.5] * 4,   # Balanced risk ratio
        output_dir=f"{output_dir}/{ai_type}_training",
        quiet_mode=True
    )
    
    # Create and run training framework
    framework = AITrainingFramework(config)
    
    start_time = time.time()
    results = framework.run_training_session()
    training_time = time.time() - start_time
    
    # Display results
    print(f"\n✅ {ai_type.upper()} AI Training Completed!")
    print(f"⏱️  Training time: {training_time/60:.1f} minutes")
    print(f"📊 Results saved to: {config.output_dir}/")
    
    # Show key statistics
    overall = results["overall_stats"]
    print(f"\n📈 TRAINING STATISTICS:")
    print(f"Total Games: {overall['total_games']}")
    print(f"Team 1 Wins: {overall['team1_wins']} ({overall['team1_wins']/overall['total_games']*100:.1f}%)")
    print(f"Team 2 Wins: {overall['team2_wins']} ({overall['team2_wins']/overall['total_games']*100:.1f}%)")
    print(f"Team Sets: {overall['total_team_sets']}")
    print(f"Total Reneges: {overall['total_reneges']}")
    
    return results


def train_mixed_teams(ai_types: list, num_games: int = 1000, output_dir: str = "training_results") -> None:
    """Train AI models in mixed team configurations.
    
    Parameters
    ----------
    ai_types : list
        List of AI types to use for mixed team training
    num_games : int
        Number of games to train on
    output_dir : str
        Output directory for training results
    """
    print(f"\n🎯 Training Mixed Team Configuration")
    print(f"🤖 AI Types: {', '.join(ai_types)}")
    print(f"📊 Games: {num_games}")
    print(f"📁 Output: {output_dir}")
    print("=" * 60)
    
    # Create training configuration for mixed teams
    config = TrainingConfig(
        mode=TrainingMode.MIXED_TEAMS,
        num_games=num_games,
        ai_types=ai_types,
        risk_ratios=[0.5] * 4,   # Balanced risk ratio
        output_dir=f"{output_dir}/mixed_teams_training",
        quiet_mode=True
    )
    
    # Create and run training framework
    framework = AITrainingFramework(config)
    
    start_time = time.time()
    results = framework.run_training_session()
    training_time = time.time() - start_time
    
    # Display results
    print(f"\n✅ Mixed Team Training Completed!")
    print(f"⏱️  Training time: {training_time/60:.1f} minutes")
    print(f"📊 Results saved to: {config.output_dir}/")
    
    # Show key statistics
    overall = results["overall_stats"]
    print(f"\n📈 TRAINING STATISTICS:")
    print(f"Total Games: {overall['total_games']}")
    print(f"Team 1 Wins: {overall['team1_wins']} ({overall['team1_wins']/overall['total_games']*100:.1f}%)")
    print(f"Team 2 Wins: {overall['team2_wins']} ({overall['team2_wins']/overall['total_games']*100:.1f}%)")
    print(f"Team Sets: {overall['total_team_sets']}")
    print(f"Total Reneges: {overall['total_reneges']}")
    
    return results


def main():
    """Main training function."""
    print("🚀 Traditional AI Model Training Program")
    print("=" * 60)
    
    # Configuration
    num_games = 1000
    output_dir = "training_results"
    
    # Traditional AI types
    ai_types = ["aggressive", "conservative", "balanced", "opportunistic"]
    
    print(f"🎯 Training each AI model on {num_games} games")
    print(f"🤖 AI Types: {', '.join(ai_types)}")
    print(f"📁 Output Directory: {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Track overall training time
    total_start_time = time.time()
    all_results = {}
    
    # Train each AI model individually
    for ai_type in ai_types:
        try:
            results = train_ai_model(ai_type, num_games, output_dir)
            all_results[ai_type] = results
        except Exception as e:
            print(f"❌ Error training {ai_type} AI: {e}")
            continue
    
    # Train mixed team configuration
    try:
        print(f"\n🎯 Training Mixed Team Configuration...")
        mixed_results = train_mixed_teams(ai_types, num_games, output_dir)
        all_results["mixed_teams"] = mixed_results
    except Exception as e:
        print(f"❌ Error training mixed teams: {e}")
    
    # Calculate total training time
    total_training_time = time.time() - total_start_time
    
    # Final summary
    print(f"\n🎉 ALL TRAINING COMPLETED!")
    print(f"⏱️  Total training time: {total_training_time/3600:.1f} hours")
    print(f"📊 Results saved to: {output_dir}/")
    
    # Summary of all training sessions
    print(f"\n📋 TRAINING SUMMARY:")
    for ai_type, results in all_results.items():
        if results and "overall_stats" in results:
            overall = results["overall_stats"]
            print(f"  {ai_type.upper()}: {overall['total_games']} games completed")
    
    print(f"\n✅ Training program finished successfully!")


if __name__ == "__main__":
    main() 