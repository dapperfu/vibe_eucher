#!/usr/bin/env python3
"""Demonstration script for generating AI player profiles with different playing styles."""

import sys
from pathlib import Path

# Add the euchre package to the path
sys.path.insert(0, str(Path(__file__).parent))

from euchre.ai_model.self_play_trainer import SelfPlayTrainer


def main():
    """Generate multiple AI player profiles with different playing styles."""
    print("🎯 AI Player Profile Generator")
    print("=" * 50)
    
    # Model configuration
    model_config = {
        'type': 'standard',
        'input_size': 128,
        'hidden_size': 256,
        'output_size': 64,
        'risk_embedding_size': 32,
        'use_risk_attention': True
    }
    
    # Initialize trainer
    trainer = SelfPlayTrainer(
        model_config=model_config,
        output_dir="trained_models",
        device="auto"
    )
    
    # Define the 5 distinct player profiles
    player_profiles = [
        {
            'name': 'Alice',
            'risk_profile': 'aggressive',
            'description': 'Aggressive player - high risk tolerance, leads with high cards'
        },
        {
            'name': 'Bob', 
            'risk_profile': 'conservative',
            'description': 'Conservative player - low risk tolerance, plays safe'
        },
        {
            'name': 'Charlie',
            'risk_profile': 'balanced',
            'description': 'Balanced player - moderate risk, adaptive strategy'
        },
        {
            'name': 'David',
            'risk_profile': 'ace_hunter',
            'description': 'Ace hunter - loves ordering up aces, strategic risk taker'
        },
        {
            'name': 'Eve',
            'risk_profile': 'trump_caller',
            'description': 'Trump caller - always calls trump, aggressive trump play'
        }
    ]
    
    print(f"🧠 Will generate {len(player_profiles)} distinct AI player profiles")
    print(f"📊 Each profile will be trained on 15,000 games")
    print(f"📁 Models will be saved to: trained_models/")
    print()
    
    # Train each profile
    results = {}
    for profile in player_profiles:
        print(f"🎯 Training {profile['name']} ({profile['risk_profile']})")
        print(f"📝 Style: {profile['description']}")
        print("-" * 40)
        
        try:
            # Train this profile
            profile_results = trainer.train_single_profile(
                player_name=profile['name'],
                risk_profile=profile['risk_profile'],
                num_games=15000,  # Minimum 10,000 games as requested
                save_interval=1000,
                evaluation_interval=2000
            )
            
            results[profile['name']] = profile_results
            
            print(f"✅ {profile['name']} training completed!")
            print(f"   Final win rate: {profile_results['final_win_rate']*100:.1f}%")
            print(f"   Games played: {profile_results['total_games']}")
            print(f"   Model saved: {profile_results['model_path']}")
            
        except Exception as e:
            print(f"❌ Failed to train {profile['name']}: {e}")
            results[profile['name']] = {'error': str(e)}
        
        print()
    
    # Summary
    print("=" * 50)
    print("🎉 PLAYER PROFILE GENERATION COMPLETED!")
    print("=" * 50)
    
    successful_profiles = [name for name, result in results.items() if 'error' not in result]
    failed_profiles = [name for name, result in results.items() if 'error' in result]
    
    print(f"✅ Successfully trained: {len(successful_profiles)} profiles")
    if successful_profiles:
        print(f"   {', '.join(successful_profiles)}")
    
    if failed_profiles:
        print(f"❌ Failed to train: {len(failed_profiles)} profiles")
        print(f"   {', '.join(failed_profiles)}")
    
    print(f"\n📁 All models saved to: trained_models/")
    print(f"🎮 Use these profiles in games with:")
    print(f"   python -m euchre.cli play --ai-profiles")
    print(f"   or")
    print(f"   make profiles")


if __name__ == "__main__":
    main() 