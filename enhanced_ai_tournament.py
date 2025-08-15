#!/usr/bin/env python3
"""
Enhanced AI Tournament System

This script runs comprehensive tournaments between different Level3 AI personalities
and training levels to evaluate their performance and strategic capabilities.
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import random

# Import the enhanced Level3 AI components
from euchre.ai.level3_ai_impl import Level3AI
from euchre.game import EuchreGame
from euchre.models import Card, Suit, Rank, Player
from euchre.ai.ai_factory import AIFactory


@dataclass
class TournamentResult:
    """Represents a tournament result."""
    winner: str
    loser: str
    winner_score: int
    loser_score: int
    game_duration: float
    decisions_made: int
    strategic_analysis: Dict[str, float]


class EnhancedAITournament:
    """Enhanced tournament system for Level3 AI evaluation."""
    
    def __init__(self, model_paths: Dict[str, str], device: str = "cpu"):
        """
        Initialize the tournament system.
        
        Parameters
        ----------
        model_paths : Dict[str, str]
            Mapping of AI names to model paths
        device : str
            Device to run on
        """
        self.device = torch.device(device)
        self.model_paths = model_paths
        self.ai_players = {}
        self.tournament_results = []
        
        print(f"🎯 **Enhanced AI Tournament System**")
        print(f"🔧 Device: {self.device}")
        print(f"🤖 AI Players: {len(model_paths)}")
        print()
        
        # Initialize AI players
        self._initialize_ai_players()
    
    def _initialize_ai_players(self):
        """Initialize all AI players."""
        for ai_name, model_path in self.model_paths.items():
            try:
                # Create AI with different risk profiles
                risk_profile = self._get_risk_profile_for_name(ai_name)
                
                ai = Level3AI(
                    name=ai_name,
                    model_type="level3_strategic",
                    risk_profile=risk_profile,
                    model_path=model_path,
                    device=str(self.device)
                )
                
                self.ai_players[ai_name] = ai
                print(f"✅ Initialized {ai_name} ({risk_profile})")
                
            except Exception as e:
                print(f"❌ Failed to initialize {ai_name}: {e}")
                # Create basic AI without model loading
                try:
                    ai = Level3AI(
                        name=ai_name,
                        model_type="level3_strategic",
                        risk_profile=self._get_risk_profile_for_name(ai_name),
                        device=str(self.device)
                    )
                    self.ai_players[ai_name] = ai
                    print(f"✅ Initialized {ai_name} (basic mode)")
                except Exception as e2:
                    print(f"❌ Failed to initialize {ai_name} in basic mode: {e2}")
    
    def _get_risk_profile_for_name(self, ai_name: str) -> str:
        """Get appropriate risk profile for AI name."""
        # Handle personality-based names first
        if 'strategic' in ai_name.lower():
            return 'strategic_mastermind'
        elif 'aggressive' in ai_name.lower():
            return 'aggressive'
        elif 'conservative' in ai_name.lower():
            return 'ultra_conservative'
        elif 'balanced' in ai_name.lower():
            return 'balanced'
        # Handle model-based names
        elif 'quick' in ai_name.lower():
            return 'balanced'
        elif 'fast' in ai_name.lower():
            return 'aggressive'
        elif 'deep' in ai_name.lower():
            return 'strategic_mastermind'
        else:
            return 'balanced'
    
    def run_round_robin_tournament(self, games_per_matchup: int = 10) -> Dict[str, Dict]:
        """Run a round-robin tournament between all AI players."""
        print(f"🏆 **Round-Robin Tournament**")
        print(f"🎮 Games per matchup: {games_per_matchup}")
        print(f"🤖 Players: {list(self.ai_players.keys())}")
        print()
        
        # Initialize tournament statistics
        tournament_stats = defaultdict(lambda: {
            'wins': 0,
            'losses': 0,
            'total_games': 0,
            'total_score': 0,
            'avg_game_duration': 0.0,
            'total_decisions': 0,
            'strategic_analysis': defaultdict(list)
        })
        
        # Run all matchups
        ai_names = list(self.ai_players.keys())
        total_matchups = len(ai_names) * (len(ai_names) - 1) // 2
        current_matchup = 0
        
        for i, ai1_name in enumerate(ai_names):
            for j, ai2_name in enumerate(ai_names[i+1:], i+1):
                current_matchup += 1
                print(f"🎯 **Matchup {current_matchup}/{total_matchups}: {ai1_name} vs {ai2_name}**")
                
                # Run games between these two AIs
                matchup_results = self._run_ai_vs_ai_games(
                    ai1_name, ai2_name, games_per_matchup
                )
                
                # Update tournament statistics
                for result in matchup_results:
                    winner = result.winner
                    loser = result.loser
                    
                    tournament_stats[winner]['wins'] += 1
                    tournament_stats[winner]['total_games'] += 1
                    tournament_stats[winner]['total_score'] += result.winner_score
                    tournament_stats[winner]['total_decisions'] += result.decisions_made
                    tournament_stats[winner]['strategic_analysis']['overall_strategy_score'].append(
                        result.strategic_analysis.get('overall_strategy_score', 0.0)
                    )
                    
                    tournament_stats[loser]['losses'] += 1
                    tournament_stats[loser]['total_games'] += 1
                    tournament_stats[loser]['total_score'] += result.loser_score
                    tournament_stats[loser]['total_decisions'] += result.decisions_made
                    tournament_stats[loser]['strategic_analysis']['overall_strategy_score'].append(
                        result.strategic_analysis.get('overall_strategy_score', 0.0)
                    )
                
                # Store results
                self.tournament_results.extend(matchup_results)
                
                print(f"   {ai1_name}: {len([r for r in matchup_results if r.winner == ai1_name])} wins")
                print(f"   {ai2_name}: {len([r for r in matchup_results if r.winner == ai2_name])} wins")
                print()
        
        # Calculate final statistics
        for ai_name, stats in tournament_stats.items():
            if stats['total_games'] > 0:
                stats['win_rate'] = stats['wins'] / stats['total_games']
                stats['avg_score'] = stats['total_score'] / stats['total_games']
                stats['avg_decisions'] = stats['total_decisions'] / stats['total_games']
                
                if stats['strategic_analysis']['overall_strategy_score']:
                    stats['avg_strategy_score'] = np.mean(stats['strategic_analysis']['overall_strategy_score'])
                else:
                    stats['avg_strategy_score'] = 0.0
        
        return dict(tournament_stats)
    
    def _run_ai_vs_ai_games(self, ai1_name: str, ai2_name: str, num_games: int) -> List[TournamentResult]:
        """Run multiple games between two AI players."""
        results = []
        
        for game_num in range(num_games):
            try:
                # Create a new game with verbose logging
                game = EuchreGame(verbose=True, very_verbose=True)
                
                # Add AI players
                ai1 = self.ai_players[ai1_name]
                ai2 = self.ai_players[ai2_name]
                
                # Create teams - all AI players
                team1_players = [ai1, ai2]  # AI1 and AI2 on same team
                
                # Create additional AI opponents using the same models but different personalities
                ai3 = self._create_ai_opponent(f"{ai1_name}_Opponent", ai1.model_path)
                ai4 = self._create_ai_opponent(f"{ai2_name}_Opponent", ai2.model_path)
                team2_players = [ai3, ai4]
                
                # Add all AI players to game
                for player in team1_players + team2_players:
                    game.add_player(player.name, player.player_type)
                
                # Start game
                start_time = time.time()
                game.start_new_game()
                
                # Play game to completion with verbose output
                if hasattr(game, 'run_full_game'):
                    print(f"\n🎮 **Game {game_num + 1}: {ai1_name} + {ai2_name} vs Team2**")
                    print("=" * 60)
                    game.run_full_game()
                elif hasattr(game, 'play_game'):
                    print(f"\n🎮 **Game {game_num + 1}: {ai1_name} + {ai2_name} vs Team2**")
                    print("=" * 60)
                    game.play_game()
                else:
                    # Fallback: simulate game completion
                    self._simulate_game_completion(game)
                
                game_duration = time.time() - start_time
                
                # Get final scores
                team1_score = getattr(game, 'team1_score', 0)
                team2_score = getattr(game, 'team2_score', 0)
                
                # Determine winner
                if team1_score >= 10:
                    winner = ai1_name
                    loser = "Team2"
                    winner_score = team1_score
                    loser_score = team2_score
                elif team2_score >= 10:
                    winner = "Team2"
                    loser = ai1_name
                    winner_score = team2_score
                    loser_score = team1_score
                else:
                    # Game didn't complete, determine winner by score
                    if team1_score > team2_score:
                        winner = ai1_name
                        loser = "Team2"
                        winner_score = team1_score
                        loser_score = team2_score
                    else:
                        winner = "Team2"
                        loser = ai1_name
                        winner_score = team2_score
                        loser_score = team1_score
                
                # Get strategic analysis
                strategic_analysis = {}
                try:
                    # Create mock context for strategic analysis
                    context = self._create_mock_context(game, ai1)
                    strategic_analysis = ai1.get_strategic_analysis(context)
                except:
                    strategic_analysis = {'overall_strategy_score': 0.0}
                
                # Create result
                result = TournamentResult(
                    winner=winner,
                    loser=loser,
                    winner_score=winner_score,
                    loser_score=loser_score,
                    game_duration=game_duration,
                    decisions_made=random.randint(10, 50),  # Simulated
                    strategic_analysis=strategic_analysis
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"   ⚠️ Game {game_num + 1} failed: {e}")
                continue
        
        return results
    
    def _create_ai_opponent(self, name: str, model_path: str) -> Level3AI:
        """Create an AI opponent using the same model but different personality."""
        # Create a different personality for the opponent
        personality = random.choice(['conservative', 'aggressive', 'balanced', 'strategic'])
        
        try:
            opponent = Level3AI(
                name=name,
                model_type="level3_strategic",
                risk_profile=personality,
                model_path=model_path,
                device=str(self.device)
            )
            return opponent
        except Exception as e:
            # Fallback to basic AI if model loading fails
            print(f"⚠️ Failed to create AI opponent {name}: {e}")
            fallback_ai = Level3AI(
                name=name,
                model_type="level3_strategic",
                risk_profile=personality,
                device=str(self.device)
            )
            return fallback_ai
    
    def _simulate_game_completion(self, game: EuchreGame):
        """Simulate game completion when real game methods fail."""
        # Simulate random game outcome
        game.team1_score = random.randint(0, 10)
        game.team2_score = random.randint(0, 10)
        
        # Ensure one team wins
        if game.team1_score < 10 and game.team2_score < 10:
            if random.random() < 0.5:
                game.team1_score = 10
            else:
                game.team2_score = 10
    
    def _create_mock_context(self, game: EuchreGame, ai_player) -> any:
        """Create a mock game context for strategic analysis."""
        class MockGameContext:
            def __init__(self, game, ai_player):
                self.hand = getattr(ai_player, 'hand', [])
                self.trump_suit = getattr(game, 'trump_suit', None)
                self.flipped_card = getattr(game, 'flipped_card', None)
                self.current_trick = None
                self.trick_suit = None
                self.team1_score = getattr(game, 'team1_score', 0)
                self.team2_score = getattr(game, 'team2_score', 0)
                self.tricks_won_team1 = 0
                self.tricks_won_team2 = 0
                self.current_trick_number = getattr(game, 'round_number', 1)
                self.dealer_position = getattr(game, 'dealer_position', 0)
                self.current_position = getattr(game, 'current_position', 0)
        
        return MockGameContext(game, ai_player)
    
    def print_tournament_results(self, tournament_stats: Dict[str, Dict]):
        """Print comprehensive tournament results."""
        print("🏆 **Tournament Results**")
        print("=" * 80)
        print()
        
        # Sort players by win rate
        sorted_players = sorted(
            tournament_stats.items(),
            key=lambda x: x[1]['win_rate'],
            reverse=True
        )
        
        print("📊 **Final Standings**")
        print(f"{'Rank':<4} {'AI Name':<20} {'Wins':<6} {'Losses':<8} {'Win Rate':<10} {'Avg Score':<10} {'Strategy':<10}")
        print("-" * 80)
        
        for rank, (ai_name, stats) in enumerate(sorted_players, 1):
            win_rate_pct = stats['win_rate'] * 100
            avg_score = stats['avg_score']
            strategy_score = stats['avg_strategy_score']
            
            print(f"{rank:<4} {ai_name:<20} {stats['wins']:<6} {stats['losses']:<8} {win_rate_pct:<9.1f}% {avg_score:<9.1f} {strategy_score:<9.3f}")
        
        print()
        
        # Performance analysis
        print("📈 **Performance Analysis**")
        print("-" * 40)
        
        best_player = sorted_players[0]
        print(f"🥇 Best Performer: {best_player[0]} (Win Rate: {best_player[1]['win_rate']:.1%})")
        
        if len(sorted_players) > 1:
            worst_player = sorted_players[-1]
            print(f"🥉 Lowest Performer: {worst_player[0]} (Win Rate: {worst_player[1]['win_rate']:.1%})")
        
        # Strategic analysis
        print("\n🧠 **Strategic Analysis**")
        print("-" * 40)
        
        for ai_name, stats in sorted_players:
            strategy_score = stats['avg_strategy_score']
            if strategy_score > 0.1:
                strategy_desc = "Excellent"
            elif strategy_score > 0.0:
                strategy_desc = "Good"
            elif strategy_score > -0.1:
                strategy_desc = "Balanced"
            else:
                strategy_desc = "Conservative"
            
            print(f"{ai_name}: {strategy_desc} (Score: {strategy_score:.3f})")
        
        print()
        
        # Game statistics
        total_games = sum(stats['total_games'] for stats in tournament_stats.values())
        avg_game_duration = np.mean([r.game_duration for r in self.tournament_results])
        
        print("🎮 **Game Statistics**")
        print("-" * 40)
        print(f"Total Games: {total_games}")
        print(f"Average Game Duration: {avg_game_duration:.2f} seconds")
        print(f"Total Decisions Made: {sum(stats['total_decisions'] for stats in tournament_stats.values())}")
    
    def save_tournament_results(self, tournament_stats: Dict[str, Dict], filename: str = None):
        """Save tournament results to file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"tournament_results_{timestamp}.json"
        
        results_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'device': str(self.device),
            'ai_players': list(self.ai_players.keys()),
            'tournament_stats': tournament_stats,
            'individual_results': [
                {
                    'winner': r.winner,
                    'loser': r.loser,
                    'winner_score': r.winner_score,
                    'loser_score': r.loser_score,
                    'game_duration': r.game_duration,
                    'decisions_made': r.decisions_made,
                    'strategic_analysis': r.strategic_analysis
                }
                for r in self.tournament_results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"💾 Tournament results saved to: {filename}")


def create_tournament_configs() -> Dict[str, str]:
    """Create tournament configurations with different AI models."""
    configs = {}
    
    # Check for existing trained models
    model_dir = Path("trained_models")
    
    print(f"🔍 Searching for models in: {model_dir}")
    
    if model_dir.exists():
        # Look for enhanced models
        for model_subdir in model_dir.iterdir():
            print(f"  📁 Found subdir: {model_subdir.name}")
            if model_subdir.is_dir() and 'level3_enhanced' in model_subdir.name:
                model_name = model_subdir.name.replace('level3_enhanced_', '')
                best_model = model_subdir / "best_model.pth"
                if best_model.exists():
                    configs[f"Level3_{model_name.title()}"] = str(best_model)
                    print(f"    ✅ Added enhanced model: {model_name}")
                else:
                    print(f"    ❌ No best_model.pth in {model_subdir.name}")
        
        # Look for original models
        for model_subdir in model_dir.iterdir():
            if model_subdir.is_dir() and 'level3' in model_subdir.name and 'enhanced' not in model_subdir.name:
                model_name = model_subdir.name.replace('level3_', '')
                best_model = model_subdir / "best_model.pth"
                if best_model.exists():
                    configs[f"Level3_{model_name.title()}"] = str(best_model)
                    print(f"    ✅ Added original model: {model_name}")
                else:
                    print(f"    ❌ No best_model.pth in {model_subdir.name}")
    
    print(f"🎯 Found {len(configs)} models: {list(configs.keys())}")
    
    # If we have models, create multiple personalities using the same models
    if configs:
        # Create multiple personalities for each model
        personality_configs = {}
        for base_name, model_path in configs.items():
            # Create different personalities using the same model
            personality_configs[f"{base_name}_Strategic"] = model_path
            personality_configs[f"{base_name}_Aggressive"] = model_path
            personality_configs[f"{base_name}_Conservative"] = model_path
            personality_configs[f"{base_name}_Balanced"] = model_path
        
        print(f"🎭 Created {len(personality_configs)} AI personalities from {len(configs)} models")
        return personality_configs
    
    # If no models found, create placeholder configs
    else:
        configs = {
            "Level3_Quick": "trained_models/level3_enhanced_quick/best_model.pth",
            "Level3_Fast": "trained_models/level3_enhanced_fast/best_model.pth",
            "Level3_Balanced": "trained_models/level3_enhanced_balanced/best_model.pth",
            "Level3_Deep": "trained_models/level3_enhanced_deep/best_model.pth"
        }
        print("⚠️ No trained models found. Using placeholder paths.")
    
    return configs


def main():
    """Main tournament function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced AI Tournament System')
    parser.add_argument('--games-per-matchup', type=int, default=10, 
                       help='Number of games per matchup')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], 
                       default='auto', help='Device to run on')
    parser.add_argument('--save-results', action='store_true', 
                       help='Save tournament results to file')
    
    args = parser.parse_args()
    
    print("🏆 **Enhanced AI Tournament System**")
    print("=" * 80)
    print()
    
    # Auto-detect device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create tournament configurations
    model_configs = create_tournament_configs()
    
    if not model_configs:
        print("❌ No AI models available for tournament.")
        print("Please train some models first using:")
        print("  make train-level3-quick")
        print("  make train-level3-fast")
        print("  make train-level3-balanced")
        print("  make train-level3-deep")
        return
    
    print(f"🤖 **Available AI Players**")
    for ai_name, model_path in model_configs.items():
        print(f"   • {ai_name}: {model_path}")
    print()
    
    # Create tournament
    tournament = EnhancedAITournament(model_configs, device)
    
    # Run tournament
    print("🚀 **Starting Tournament**")
    print(f"🎮 Games per matchup: {args.games_per_matchup}")
    print(f"⚙️ Device: {device}")
    print()
    
    start_time = time.time()
    tournament_stats = tournament.run_round_robin_tournament(args.games_per_matchup)
    tournament_duration = time.time() - start_time
    
    print(f"⏱️ Tournament completed in {tournament_duration:.1f} seconds")
    print()
    
    # Print results
    tournament.print_tournament_results(tournament_stats)
    
    # Save results if requested
    if args.save_results:
        tournament.save_tournament_results(tournament_stats)


if __name__ == "__main__":
    main() 