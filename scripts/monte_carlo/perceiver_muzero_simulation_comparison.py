#!/usr/bin/env python3
"""Compare perceiver_muzero performance with different MCTS simulation counts.

This script runs a round-robin tournament comparing perceiver_muzero players
with 1, 16, 64, and 128 MCTS simulations to measure the impact of simulation count
on performance.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eucher.game import Game
from scripts.ai_tournament import (
    TournamentSimulator,
    generate_round_robin_matchups,
    run_matchup,
)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare perceiver_muzero with different MCTS simulation counts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comparison with 50 games per matchup (faster)
  python scripts/perceiver_muzero_simulation_comparison.py --num-games 50

  # Run with checkpoint support
  python scripts/perceiver_muzero_simulation_comparison.py --num-games 100 --checkpoint pmz_comparison.json
        """,
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=50,
        help="Number of games per matchup (default: 50, recommended: 50-100 for reasonable runtime)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON format). If not specified, uses pmz_simulation_comparison_<timestamp>.json",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per game in seconds (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file path. If specified, will save checkpoints and can resume from this file.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N games (default: 10). Only used if --checkpoint is specified.",
    )

    args = parser.parse_args()

    # Define player types: perceiver_muzero with different simulation counts
    player_types = [
        "perceiver_muzero_1",    # Fast mode (feed-forward only)
        "perceiver_muzero_16",   # 16 MCTS simulations
        "perceiver_muzero_64",   # 64 MCTS simulations
        "perceiver_muzero_128",  # 128 MCTS simulations (full)
    ]

    print("=" * 80)
    print("PerceiverMuZero MCTS Simulation Count Comparison")
    print("=" * 80)
    print(f"Player types: {', '.join(player_types)}")
    print(f"Games per matchup: {args.num_games}")
    print(f"Total matchups: {len(generate_round_robin_matchups(player_types))}")
    print(f"Total games: {len(generate_round_robin_matchups(player_types)) * args.num_games}")
    print("=" * 80)
    print()

    # Checkpoint handling
    checkpoint_path = None
    checkpoint_data = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_data = TournamentSimulator.load_checkpoint(checkpoint_path)
        if checkpoint_data:
            print(f"📂 Resuming from checkpoint: {checkpoint_path}")
            print(f"   Timestamp: {checkpoint_data.get('timestamp', 'unknown')}")
            print(f"   Completed matchups: {len(checkpoint_data.get('completed_matchups', []))}")
            print()

    # Run tournament
    tournament = TournamentSimulator()

    # Load checkpoint state if available
    current_matchup_idx = 0
    start_game = 0
    if checkpoint_data:
        tournament_state = checkpoint_data.get("tournament_state", {})
        if "matchups" in tournament_state:
            for k_str, v in tournament_state["matchups"].items():
                parts = k_str.split("_vs_")
                if len(parts) == 2:
                    matchup = tuple(sorted(parts))
                    tournament.matchups[matchup] = v
        if "player_type_stats" in tournament_state:
            tournament.player_type_stats = checkpoint_data["tournament_state"]["player_type_stats"]
        
        tournament.completed_matchups = set(
            tuple(m) for m in checkpoint_data.get("completed_matchups", [])
        )
        current_matchup_idx = checkpoint_data.get("current_matchup_idx", 0)
        start_game = checkpoint_data.get("current_matchup_games_completed", 0)

    matchups = generate_round_robin_matchups(player_types)

    print(f"Running {len(matchups)} matchups...")
    print()

    for idx, (player_type_0, player_type_1) in enumerate(matchups):
        matchup_str = f"{player_type_0} vs {player_type_1}"

        # Skip if already completed
        if (player_type_0, player_type_1) in tournament.completed_matchups:
            print(f"⏭️  Skipping completed matchup {idx + 1}/{len(matchups)}: {matchup_str}")
            continue

        # Skip if not the current matchup (when resuming)
        if idx < current_matchup_idx:
            continue

        tournament.current_matchup = (player_type_0, player_type_1)
        tournament.current_matchup_games_completed = start_game if idx == current_matchup_idx else 0

        with tqdm(
            total=args.num_games,
            desc=f"Matchup {idx + 1}/{len(matchups)}: {matchup_str}",
            unit="game",
        ) as pbar:
            pbar.update(start_game if idx == current_matchup_idx else 0)

            # Run games for this matchup
            print(f"\n▶️  Starting matchup {idx + 1}/{len(matchups)}: {matchup_str}")
            results = run_matchup(
                player_type_0,
                player_type_1,
                args.num_games,
                seed_base=args.seed or 0,
                progress_bar=pbar,
                timeout_per_game=args.timeout,
                start_game=start_game if idx == current_matchup_idx else 0,
            )
            print(f"✅ Completed matchup {idx + 1}/{len(matchups)}: {matchup_str} ({len(results)} games)")

            # Record results
            for result in results:
                tournament.record_game(
                    result["player_type_0"],
                    result["player_type_1"],
                    result["winner"],
                    result["scores"],
                    result["hands_played"],
                )

            # Mark matchup as completed
            tournament.completed_matchups.add((player_type_0, player_type_1))
            tournament.current_matchup = None
            tournament.current_matchup_games_completed = 0
            start_game = 0  # Reset for next matchup

            # Save checkpoint after each matchup
            if checkpoint_path:
                tournament.save_checkpoint(
                    checkpoint_path,
                    player_types,
                    matchups,
                    args.num_games,
                    args.seed,
                    idx + 1,  # Next matchup index
                )

    # Print results
    print("\n" + "=" * 80)
    print("TOURNAMENT RESULTS")
    print("=" * 80)
    print()

    # Player type rankings
    print("Player Type Rankings (by win rate):")
    print("-" * 80)
    
    player_stats = []
    for player_type in player_types:
        stats = tournament.player_type_stats.get(player_type, {})
        total_games = stats.get("total_games", 0)
        wins = stats.get("wins", 0)
        win_rate = (wins / total_games * 100) if total_games > 0 else 0.0
        avg_score = stats.get("total_score", 0) / total_games if total_games > 0 else 0.0
        
        player_stats.append({
            "type": player_type,
            "win_rate": win_rate,
            "wins": wins,
            "losses": total_games - wins,
            "total_games": total_games,
            "avg_score": avg_score,
        })
    
    # Sort by win rate
    player_stats.sort(key=lambda x: x["win_rate"], reverse=True)
    
    for i, stats in enumerate(player_stats, 1):
        sim_count = stats["type"].split("_")[-1] if "_" in stats["type"] else "?"
        print(f"{i}. {stats['type']:25s} | Win Rate: {stats['win_rate']:5.1f}% | "
              f"W-L: {stats['wins']:3d}-{stats['losses']:3d} | "
              f"Avg Score: {stats['avg_score']:5.1f} | Simulations: {sim_count}")
    
    print()

    # Head-to-head results
    print("Head-to-Head Results:")
    print("-" * 80)
    for matchup in matchups:
        matchup_key = tuple(sorted(matchup))
        matchup_stats = tournament.matchups.get(matchup_key, {})
        if matchup_stats:
            wins_0 = matchup_stats.get("wins_0", 0)
            wins_1 = matchup_stats.get("wins_1", 0)
            total = wins_0 + wins_1
            if total > 0:
                win_rate_0 = wins_0 / total * 100
                print(f"{matchup[0]:25s} vs {matchup[1]:25s}: "
                      f"{wins_0:3d}-{wins_1:3d} ({win_rate_0:5.1f}% - {100-win_rate_0:5.1f}%)")
    print()

    # Save results
    summary = tournament.get_summary()
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "comparison_type": "perceiver_muzero_simulation_count",
        "player_types_tested": player_types,
        "num_games_per_matchup": args.num_games,
        "seed": args.seed,
        **summary,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"pmz_simulation_comparison_{timestamp}.json")

    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"✅ Results saved to: {output_path}")

    # Analysis summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    if len(player_stats) >= 2:
        best = player_stats[0]
        worst = player_stats[-1]
        best_sims = best["type"].split("_")[-1] if "_" in best["type"] else "?"
        worst_sims = worst["type"].split("_")[-1] if "_" in worst["type"] else "?"
        
        print(f"Best performer: {best['type']} ({best_sims} sims) - {best['win_rate']:.1f}% win rate")
        print(f"Worst performer: {worst['type']} ({worst_sims} sims) - {worst['win_rate']:.1f}% win rate")
        print(f"Performance gap: {best['win_rate'] - worst['win_rate']:.1f} percentage points")
        
        # Check if more simulations = better performance
        sim_counts = []
        win_rates = []
        for stats in player_stats:
            sim_count = int(stats["type"].split("_")[-1]) if stats["type"].split("_")[-1].isdigit() else 1
            sim_counts.append(sim_count)
            win_rates.append(stats["win_rate"])
        
        # Calculate correlation
        if len(sim_counts) > 1:
            correlation = np.corrcoef(sim_counts, win_rates)[0, 1]
            print(f"\nCorrelation between simulation count and win rate: {correlation:.3f}")
            if correlation > 0.3:
                print("✅ More simulations appear to improve performance")
            elif correlation < -0.3:
                print("⚠️  More simulations appear to decrease performance (unexpected!)")
            else:
                print("➡️  Simulation count has weak/no correlation with performance")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

