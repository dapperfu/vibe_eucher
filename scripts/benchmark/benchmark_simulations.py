#!/usr/bin/env python3
"""Script to benchmark Monte Carlo simulation performance.

This script runs a benchmark to determine simulations per second on the current machine
and saves the results for configuring thinking time.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eucher.ai_players.benchmark_manager import BenchmarkManager


def main() -> None:
    """Run benchmark and display results."""
    parser = argparse.ArgumentParser(description="Benchmark Monte Carlo simulation performance")
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=1000,
        help="Number of simulations to run for benchmark (default: 1000)",
    )
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=None,
        help="Path to benchmark JSON file (default: benchmarks/performance.json)",
    )

    args = parser.parse_args()

    # Initialize benchmark manager
    benchmark_manager = BenchmarkManager(benchmark_file=args.benchmark_file)

    print("Running benchmark...")
    print(f"Number of simulations: {args.num_simulations}")
    print("This may take a minute...\n")

    # Run benchmark
    result = benchmark_manager.benchmark_and_save(num_simulations=args.num_simulations)

    # Display results
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Device: {result['device']}")
    print(f"Device Type: {result['device_type']}")
    print(f"Simulations/Second: {result['simulations_per_second']:.2f}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Benchmark File: {benchmark_manager.benchmark_file}")
    print("=" * 60)

    # Display thinking time configuration
    print("\nThinking Time Configuration:")
    print("-" * 60)
    config = benchmark_manager.get_thinking_times_config()
    thinking_times = {
        "fast": "0.5s",
        "quick": "1.0s",
        "normal": "10.0s",
        "thorough": "60.0s",
        "deep": "120.0s",
    }

    for name, seconds in thinking_times.items():
        num_sims = config[name]
        print(f"  {name:10s} ({seconds:6s}): {num_sims:6d} simulations")

    print("\nBenchmark saved successfully!")


if __name__ == "__main__":
    main()

