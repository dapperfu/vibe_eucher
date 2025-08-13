"""Full model evaluator for trained euchre AI models."""

import argparse
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..game import EuchreGame
from ..ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI
from ..models import Player, PlayerType, Card, Suit, Rank
from ..game_logger import GameLogger
from .euchre_nn import EuchreNN


class ModelEvaluator:
    """Comprehensive evaluator for trained euchre AI models."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize the model evaluator.
        
        Parameters
        ----------
        model_path : str
            Path to the trained model file
        device : str
            Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        self.device = self._get_device()
        
        # Load the trained model
        self.model = self._load_model()
        
        # Evaluation results storage
        self.evaluation_results = {
            'model_performance': {},
            'strategy_comparisons': {},
            'game_statistics': {},
            'detailed_games': []
        }
        
        # AI profile configurations for comparison
        self.ai_profiles = {
            'aggressive': {'class': AggressiveAI, 'risk_ratio': 0.8},
            'conservative': {'class': ConservativeAI, 'risk_ratio': 0.2},
            'balanced': {'class': BalancedAI, 'risk_ratio': 0.5},
            'opportunistic': {'class': OpportunisticAI, 'risk_ratio': 0.7}
        }
        
    def _get_device(self) -> torch.device:
        """Get the device to use for computation.
        
        Returns
        -------
        torch.device
            Device to use
        """
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _load_model(self) -> EuchreNN:
        """Load the trained model.
        
        Returns
        -------
        EuchreNN
            Loaded model
        """
        try:
            # Load model with CPU mapping to handle CUDA-trained models on CPU
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Create model instance
            model = EuchreNN(
                input_size=128,
                hidden_size=256,
                output_size=64,
                risk_embedding_size=32,
                use_risk_attention=True
            )
            
            # Load state dict
            model.load_state_dict(checkpoint)
            
            # Move to device (will be CPU if CUDA not available)
            model.to(self.device)
            model.eval()
            
            print(f"✅ Model loaded successfully from {self.model_path}")
            print(f"📱 Device: {self.device}")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _create_model_instance(self, config: Dict[str, Any]) -> nn.Module:
        """Create a model instance based on configuration.
        
        This is a placeholder - you'll need to implement this based on your actual model architecture.
        """
        # For now, create a simple placeholder model
        # In practice, you would reconstruct your actual model architecture here
        class PlaceholderModel(nn.Module):
            def __init__(self, input_size=128, hidden_size=256, output_size=64):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        return PlaceholderModel()
    
    def evaluate_model(self, num_games: int = 100, strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate the trained model against different AI strategies.
        
        Parameters
        ----------
        num_games : int
            Number of games to play for each strategy
        strategies : Optional[List[str]]
            List of strategies to test against (default: all available)
        
        Returns
        -------
        Dict[str, Any]
            Comprehensive evaluation results
        """
        if strategies is None:
            strategies = list(self.ai_profiles.keys())
        
        print(f"🚀 Starting model evaluation against {len(strategies)} strategies")
        print(f"🎮 Playing {num_games} games per strategy")
        print(f"📱 Using device: {self.device}")
        print("=" * 80)
        
        # Evaluate against each strategy
        for strategy in strategies:
            if strategy not in self.ai_profiles:
                print(f"⚠️  Unknown strategy: {strategy}, skipping...")
                continue
                
            print(f"\n🎯 Evaluating against {strategy.upper()} strategy...")
            strategy_results = self._evaluate_against_strategy(strategy, num_games)
            
            self.evaluation_results['strategy_comparisons'][strategy] = strategy_results
            
            # Print summary
            self._print_strategy_summary(strategy, strategy_results)
        
        # Generate comprehensive analysis
        self._generate_comprehensive_analysis()
        
        print("\n" + "=" * 80)
        print("✅ Model evaluation completed!")
        print("=" * 80)
        
        return self.evaluation_results
    
    def _evaluate_against_strategy(self, strategy: str, num_games: int) -> Dict[str, Any]:
        """Evaluate the model against a specific strategy."""
        strategy_config = self.ai_profiles[strategy]
        results = {
            'strategy': strategy,
            'total_games': num_games,
            'model_wins': 0,
            'strategy_wins': 0,
            'ties': 0,
            'model_scores': [],
            'strategy_scores': [],
            'game_lengths': [],
            'detailed_games': []
        }
        
        # Progress bar
        pbar = tqdm(range(num_games), desc=f"Playing vs {strategy}")
        
        for game_num in pbar:
            # Play a single game
            game_result = self._play_single_game(strategy, strategy_config)
            
            # Update results
            if game_result['winner'] == 'model':
                results['model_wins'] += 1
            elif game_result['winner'] == 'strategy':
                results['strategy_wins'] += 1
            else:
                results['ties'] += 1
            
            results['model_scores'].append(game_result['model_score'])
            results['strategy_scores'].append(game_result['strategy_score'])
            results['game_lengths'].append(game_result['game_length'])
            results['detailed_games'].append(game_result)
            
            # Update progress bar
            pbar.set_postfix({
                'Model Wins': results['model_wins'],
                'Strategy Wins': results['strategy_wins'],
                'Ties': results['ties']
            })
        
        pbar.close()
        
        # Calculate statistics
        results['model_win_rate'] = results['model_wins'] / num_games
        results['strategy_win_rate'] = results['strategy_wins'] / num_games
        results['tie_rate'] = results['ties'] / num_games
        
        results['model_avg_score'] = statistics.mean(results['model_scores'])
        results['strategy_avg_score'] = statistics.mean(results['strategy_scores'])
        results['avg_game_length'] = statistics.mean(results['game_lengths'])
        
        results['model_score_std'] = statistics.stdev(results['model_scores']) if len(results['model_scores']) > 1 else 0
        results['strategy_score_std'] = statistics.stdev(results['strategy_scores']) if len(results['strategy_scores']) > 1 else 0
        
        return results
    
    def _play_single_game(self, strategy: str, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Play a single game between the model and a strategy."""
        # Create game
        game = EuchreGame(quiet_mode=True)
        
        # Add AI players
        game.add_ai_player("North", strategy, strategy_config['risk_ratio'])
        game.add_ai_player("East", strategy, strategy_config['risk_ratio'])
        
        # Add model players (you'll need to implement this based on your model interface)
        game.add_model_player("South", self.model, self.device)
        game.add_model_player("West", self.model, self.device)
        
        # Start game
        game.start_new_game()
        
        # Play until completion
        round_num = 1
        while not game.is_game_over():
            results = game.play_round()
            round_num += 1
        
        # Determine winner
        model_score = game.game_state.team1_score  # Assuming model is team 1
        strategy_score = game.game_state.team2_score
        
        if model_score > strategy_score:
            winner = 'model'
        elif strategy_score > model_score:
            winner = 'strategy'
        else:
            winner = 'tie'
        
        return {
            'winner': winner,
            'model_score': model_score,
            'strategy_score': strategy_score,
            'game_length': round_num - 1,
            'strategy': strategy
        }
    
    def _print_strategy_summary(self, strategy: str, results: Dict[str, Any]):
        """Print a summary of results against a specific strategy."""
        print(f"  📊 Results vs {strategy.upper()}:")
        print(f"    🏆 Model Wins: {results['model_wins']}/{results['total_games']} ({results['model_win_rate']:.1%})")
        print(f"    🎯 Strategy Wins: {results['strategy_wins']}/{results['total_games']} ({results['strategy_win_rate']:.1%})")
        print(f"    🤝 Ties: {results['ties']}/{results['total_games']} ({results['tie_rate']:.1%})")
        print(f"    📈 Model Avg Score: {results['model_avg_score']:.2f} ± {results['model_score_std']:.2f}")
        print(f"    📉 Strategy Avg Score: {results['strategy_avg_score']:.2f} ± {results['strategy_score_std']:.2f}")
        print(f"    ⏱️  Avg Game Length: {results['avg_game_length']:.1f} rounds")
    
    def _generate_comprehensive_analysis(self):
        """Generate comprehensive analysis of all evaluation results."""
        print("\n📊 GENERATING COMPREHENSIVE ANALYSIS")
        print("-" * 50)
        
        # Overall model performance
        total_games = sum(r['total_games'] for r in self.evaluation_results['strategy_comparisons'].values())
        total_wins = sum(r['model_wins'] for r in self.evaluation_results['strategy_comparisons'].values())
        overall_win_rate = total_wins / total_games if total_games > 0 else 0
        
        self.evaluation_results['model_performance'] = {
            'overall_win_rate': overall_win_rate,
            'total_games': total_games,
            'total_wins': total_wins,
            'best_strategy': self._find_best_strategy(),
            'worst_strategy': self._find_worst_strategy()
        }
        
        print(f"🎯 Overall Model Performance: {overall_win_rate:.1%} win rate")
        print(f"📊 Total Games Played: {total_games}")
        print(f"🏆 Best Strategy: {self.evaluation_results['model_performance']['best_strategy']}")
        print(f"📉 Worst Strategy: {self.evaluation_results['model_performance']['worst_strategy']}")
    
    def _find_best_strategy(self) -> str:
        """Find the strategy the model performs best against."""
        if not self.evaluation_results['strategy_comparisons']:
            return "None"
        
        best_strategy = max(
            self.evaluation_results['strategy_comparisons'].items(),
            key=lambda x: x[1]['model_win_rate']
        )
        return best_strategy[0]
    
    def _find_worst_strategy(self) -> str:
        """Find the strategy the model performs worst against."""
        if not self.evaluation_results['strategy_comparisons']:
            return "None"
        
        worst_strategy = min(
            self.evaluation_results['strategy_comparisons'].items(),
            key=lambda x: x[1]['model_win_rate']
        )
        return worst_strategy[0]
    
    def generate_visualizations(self, output_dir: str = "evaluation_plots"):
        """Generate comprehensive visualizations of evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📊 Generating visualizations in {output_path}")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Win rates by strategy
        self._plot_win_rates(output_path)
        
        # 2. Score distributions
        self._plot_score_distributions(output_path)
        
        # 3. Game length analysis
        self._plot_game_lengths(output_path)
        
        # 4. Performance comparison
        self._plot_performance_comparison(output_path)
        
        print(f"✅ Visualizations saved to {output_path}")
    
    def _plot_win_rates(self, output_path: Path):
        """Plot win rates against different strategies."""
        strategies = list(self.evaluation_results['strategy_comparisons'].keys())
        model_win_rates = [self.evaluation_results['strategy_comparisons'][s]['model_win_rate'] for s in strategies]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategies, model_win_rates, alpha=0.8, color='skyblue')
        
        # Add value labels on bars
        for bar, rate in zip(bars, model_win_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        plt.xlabel('Strategy')
        plt.ylabel('Model Win Rate')
        plt.title('Model Performance Against Different Strategies')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "win_rates_by_strategy.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_score_distributions(self, output_path: Path):
        """Plot score distributions for model vs strategies."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        strategies = list(self.evaluation_results['strategy_comparisons'].keys())
        
        for i, strategy in enumerate(strategies):
            row, col = i // 2, i % 2
            results = self.evaluation_results['strategy_comparisons'][strategy]
            
            # Plot score distributions
            axes[row, col].hist(results['model_scores'], bins=15, alpha=0.7, 
                               label='Model Scores', density=True, color='skyblue')
            axes[row, col].hist(results['strategy_scores'], bins=15, alpha=0.7, 
                               label=f'{strategy.title()} Scores', density=True, color='lightcoral')
            
            axes[row, col].set_xlabel('Final Score')
            axes[row, col].set_ylabel('Density')
            axes[row, col].set_title(f'Score Distribution vs {strategy.title()}')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "score_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_game_lengths(self, output_path: Path):
        """Plot game length analysis."""
        strategies = list(self.evaluation_results['strategy_comparisons'].keys())
        avg_lengths = [self.evaluation_results['strategy_comparisons'][s]['avg_game_length'] for s in strategies]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategies, avg_lengths, alpha=0.8, color='lightgreen')
        
        # Add value labels
        for bar, length in zip(bars, avg_lengths):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{length:.1f}', ha='center', va='bottom')
        
        plt.xlabel('Strategy')
        plt.ylabel('Average Game Length (Rounds)')
        plt.title('Game Length by Strategy')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "game_lengths.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, output_path: Path):
        """Plot comprehensive performance comparison."""
        strategies = list(self.evaluation_results['strategy_comparisons'].keys())
        
        # Prepare data
        model_win_rates = [self.evaluation_results['strategy_comparisons'][s]['model_win_rate'] for s in strategies]
        model_avg_scores = [self.evaluation_results['strategy_comparisons'][s]['model_avg_score'] for s in strategies]
        strategy_avg_scores = [self.evaluation_results['strategy_comparisons'][s]['strategy_avg_score'] for s in strategies]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Win rates
        x = np.arange(len(strategies))
        width = 0.35
        
        ax1.bar(x - width/2, model_win_rates, width, label='Model Win Rate', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, [1-r for r in model_win_rates], width, label='Strategy Win Rate', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Win Rate Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Average scores
        ax2.bar(x - width/2, model_avg_scores, width, label='Model Avg Score', alpha=0.8, color='skyblue')
        ax2.bar(x + width/2, strategy_avg_scores, width, label='Strategy Avg Score', alpha=0.8, color='lightcoral')
        
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Average Score')
        ax2.set_title('Average Score Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results(self, output_file: str = "model_evaluation_results.json"):
        """Export evaluation results to JSON file."""
        print(f"\n💾 Exporting results to {output_file}")
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        export_data = convert_numpy_types(self.evaluation_results)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Results exported to {output_file}")
    
    def print_final_summary(self):
        """Print a comprehensive final summary."""
        print("\n" + "=" * 80)
        print("🏆 FINAL EVALUATION SUMMARY")
        print("=" * 80)
        
        # Overall performance
        perf = self.evaluation_results['model_performance']
        print(f"📊 Overall Performance:")
        print(f"  🎯 Win Rate: {perf['overall_win_rate']:.1%}")
        print(f"  🎮 Total Games: {perf['total_games']}")
        print(f"  🏆 Total Wins: {perf['total_wins']}")
        
        # Strategy breakdown
        print(f"\n📈 Strategy Breakdown:")
        for strategy, results in self.evaluation_results['strategy_comparisons'].items():
            print(f"  {strategy.upper()}: {results['model_win_rate']:.1%} win rate "
                  f"({results['model_wins']}/{results['total_games']} games)")
        
        # Best and worst
        print(f"\n🎯 Performance Analysis:")
        print(f"  🚀 Best Strategy: {perf['best_strategy']}")
        print(f"  📉 Worst Strategy: {perf['worst_strategy']}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if perf['overall_win_rate'] > 0.6:
            print(f"  ✅ Model shows strong performance across all strategies")
        elif perf['overall_win_rate'] > 0.5:
            print(f"  🤝 Model performs competitively but has room for improvement")
        else:
            print(f"  ⚠️  Model needs significant improvement")
        
        print("=" * 80)


def main():
    """Main entry point for the model evaluator."""
    parser = argparse.ArgumentParser(description="Evaluate trained euchre AI models")
    parser.add_argument("--model", required=True, help="Path to trained model file")
    parser.add_argument("--games", type=int, default=100, help="Number of games per strategy")
    parser.add_argument("--strategies", nargs="+", 
                       default=["aggressive", "conservative", "balanced", "opportunistic"],
                       help="Strategies to test against")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--output-dir", default="evaluation_plots", help="Output directory for plots")
    parser.add_argument("--export", default="model_evaluation_results.json", help="Export results file")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        print("🚀 Initializing Model Evaluator...")
        evaluator = ModelEvaluator(args.model, args.device)
    
    # Run evaluation
        results = evaluator.evaluate_model(args.games, args.strategies)
        
        # Generate visualizations
        if not args.no_plots:
            evaluator.generate_visualizations(args.output_dir)
        
        # Export results
        evaluator.export_results(args.export)
        
        # Print final summary
        evaluator.print_final_summary()
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 