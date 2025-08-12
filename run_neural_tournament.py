#!/usr/bin/env python3
"""Script to run neural network euchre tournaments."""

import sys
from pathlib import Path

# Add the euchre package to the path
sys.path.insert(0, str(Path(__file__).parent))

from euchre.neural_mass_game_runner import NeuralMassGameRunner


def main():
    """Run a neural network tournament."""
    print("🧠 Neural Network Euchre Tournament Runner")
    print("=" * 50)
    
    # Initialize the runner with demo models
    runner = NeuralMassGameRunner(
        models_dir="demo_models",
        output_dir="neural_tournament_results",
        max_workers=4  # Adjust based on your system
    )
    
    print(f"Available models: {runner.available_models}")
    print()
    
    # Example 1: Run Alice vs Bob for 1000 games
    print("🏆 Running Alice vs Bob tournament (1000 games)...")
    runner.run_model_vs_model("Alice", "Bob", 1000)
    print()
    
    # Example 2: Run Charlie vs David for 1000 games
    print("🏆 Running Charlie vs David tournament (1000 games)...")
    runner.run_model_vs_model("Charlie", "David", 1000)
    print()
    
    # Example 3: Run all combinations for 500 games each
    print("🏆 Running all model combinations (500 games each)...")
    runner.run_all_model_combinations(500)
    print()
    
    print("🎉 Tournament completed!")
    print(f"Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    main() 