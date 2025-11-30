#!/usr/bin/env python3
"""Benchmark HeuristicPlayer performance by running many games.

This script runs games with HeuristicPlayer and collects statistics on
decision points and outcomes to identify areas for improvement.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eucher.game import Game
from eucher.players.computer import AIPlayer, HeuristicPlayer, RandomPlayer
from eucher.players.computer.ai import AIDecisionMaker


class HeuristicBenchmark:
    """Collects statistics from HeuristicPlayer games."""

    def __init__(self) -> None:
        """Initialize the benchmark collector."""
        self.games: List[Dict[str, Any]] = []
        self.decisions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def run_game(
        self, player_config: List[Tuple[str, str]], seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a single game and collect statistics.

        Parameters
        ----------
        player_config : List[Tuple[str, str]]
            Player configuration.
        seed : Optional[int]
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Game statistics.
        """
        if seed is not None:
            random.seed(seed)

        game = Game(player_config)
        tricks_won_history: List[List[int]] = []

        # Track decisions (would need to wrap player methods to collect this)
        while True:
            continue_game = game.play_hand()
            tricks_won = getattr(game, "_current_tricks_won", [0, 0])
            tricks_won_history.append(tricks_won.copy())

            scores = game.get_scores()
            winner = game.get_winner()

            if winner is not None or not continue_game:
                break

        game_stats = {
            "winner": winner,
            "scores": game.get_scores(),
            "tricks_won_history": tricks_won_history,
            "hands_played": len(tricks_won_history),
        }

        self.games.append(game_stats)
        return game_stats

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics from collected games.

        Returns
        -------
        Dict[str, Any]
            Statistics summary.
        """
        if not self.games:
            return {}

        winners = [g["winner"] for g in self.games if g["winner"] is not None]
        hands_played = [g["hands_played"] for g in self.games]
        scores_team0 = [g["scores"][0] for g in self.games]
        scores_team1 = [g["scores"][1] for g in self.games]

        # Calculate win rates by team
        team0_wins = sum(1 for w in winners if w == 0)
        team1_wins = sum(1 for w in winners if w == 1)
        total_games = len(self.games)

        return {
            "total_games": total_games,
            "team0_wins": team0_wins,
            "team1_wins": team1_wins,
            "team0_win_rate": team0_wins / total_games if total_games > 0 else 0.0,
            "team1_win_rate": team1_wins / total_games if total_games > 0 else 0.0,
            "hands_played": {
                "mean": sum(hands_played) / len(hands_played) if hands_played else 0.0,
                "min": min(hands_played) if hands_played else 0,
                "max": max(hands_played) if hands_played else 0,
            },
            "scores_team0": {
                "mean": sum(scores_team0) / len(scores_team0) if scores_team0 else 0.0,
                "min": min(scores_team0) if scores_team0 else 0,
                "max": max(scores_team0) if scores_team0 else 0,
            },
            "scores_team1": {
                "mean": sum(scores_team1) / len(scores_team1) if scores_team1 else 0.0,
                "min": min(scores_team1) if scores_team1 else 0,
                "max": max(scores_team1) if scores_team1 else 0,
            },
        }

    def save_results(self, output_file: Path) -> None:
        """
        Save benchmark results to a file.

        Parameters
        ----------
        output_file : Path
            File to save results to.
        """
        results = {
            "statistics": self.get_statistics(),
            "games": self.games,
        }
        output_file.write_text(json.dumps(results, indent=2))


def compare_player_types(num_games: int = 1000) -> Dict[str, Dict[str, Any]]:
    """
    Compare different player types.

    Parameters
    ----------
    num_games : int
        Number of games to run for each configuration.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Statistics for each player configuration.
    """
    configurations = {
        "heuristic_vs_random": [
            ("H1", "heuristic"),
            ("R1", "random"),
            ("H2", "heuristic"),
            ("R2", "random"),
        ],
        "heuristic_vs_ai": [
            ("H1", "heuristic"),
            ("A1", "ai"),
            ("H2", "heuristic"),
            ("A2", "ai"),
        ],
        "heuristic_all": [
            ("H1", "heuristic"),
            ("H2", "heuristic"),
            ("H3", "heuristic"),
            ("H4", "heuristic"),
        ],
    }

    results = {}

    for config_name, player_config in configurations.items():
        print(f"\nRunning {num_games} games with {config_name}...")
        benchmark = HeuristicBenchmark()

        for i in range(num_games):
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{num_games} games...")
            benchmark.run_game(player_config, seed=i)

        stats = benchmark.get_statistics()
        results[config_name] = stats
        print(f"  Win rate (team 0): {stats['team0_win_rate']:.1%}")
        print(f"  Win rate (team 1): {stats['team1_win_rate']:.1%}")

    return results


def main() -> None:
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark HeuristicPlayer performance")
    parser.add_argument(
        "--num_games",
        type=int,
        default=1000,
        help="Number of games to run (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: print to stdout)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare HeuristicPlayer against other player types",
    )

    args = parser.parse_args()

    if args.compare:
        results = compare_player_types(args.num_games)
        output = json.dumps(results, indent=2)
    else:
        # Run single benchmark
        print(f"Running {args.num_games} games with HeuristicPlayer...")
        benchmark = HeuristicBenchmark()

        player_config = [
            ("H1", "heuristic"),
            ("H2", "heuristic"),
            ("H3", "heuristic"),
            ("H4", "heuristic"),
        ]

        for i in range(args.num_games):
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{args.num_games} games...")
            benchmark.run_game(player_config, seed=i)

        stats = benchmark.get_statistics()
        results = {"statistics": stats}
        output = json.dumps(results, indent=2)

        print("\nResults:")
        print(f"  Total games: {stats['total_games']}")
        print(f"  Team 0 win rate: {stats['team0_win_rate']:.1%}")
        print(f"  Team 1 win rate: {stats['team1_win_rate']:.1%}")

    if args.output:
        Path(args.output).write_text(output)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + output)


if __name__ == "__main__":
    main()


