#!/usr/bin/env python3
"""Quick profiling script for Monte Carlo simulation.

This script runs a small number of games with profiling enabled to identify
performance bottlenecks.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.monte_carlo.euchergo_vs_eucher_zero import run_monte_carlo_simulation

if __name__ == "__main__":
    print("=" * 80)
    print("PROFILING MONTE CARLO SIMULATION")
    print("=" * 80)
    print("Running 10 games with profiling enabled...")
    print()
    
    profile_output = Path("profiles") / "monte_carlo_profile_quick.txt"
    profile_output.parent.mkdir(exist_ok=True)
    
    stats = run_monte_carlo_simulation(
        num_games=10,
        seed=42,
        profile=True,
        profile_output=profile_output,
    )
    
    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print(f"Profile saved to: {profile_output}")
    print(f"Timing stats saved to: {profile_output.parent / (profile_output.stem + '_timing.txt')}")
    print("\nCheck the profile files to identify bottlenecks.")


