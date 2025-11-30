#!/usr/bin/env python3
"""Monte Carlo simulation to find optimal risk factor for HeuristicPlayer2.

This script runs many games with different risk factor combinations to determine
the optimal risk factor for competitive play. Tests team vs team scenarios where
each team has players with the same risk factor.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eucher.game import Game
from eucher.players import Player
from eucher.players.computer.heuristic2 import HeuristicPlayer2


class RiskFactorComparison:
    """Collects statistics from risk factor comparison games."""

    def __init__(self) -> None:
        """Initialize the comparison collector."""
        self.matchups: Dict[str, Dict[str, any]] = defaultdict(
            lambda: {
                "games_played": 0,
                "team0_wins": 0,
                "team1_wins": 0,
                "team0_total_score": 0,
                "team1_total_score": 0,
                "hands_played": [],
                "team0_avg_score": 0.0,
                "team1_avg_score": 0.0,
            }
        )
        self.risk_factor_stats: Dict[float, Dict[str, any]] = defaultdict(
            lambda: {
                "total_games": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "total_score": 0,
                "games_as_team0": 0,
                "games_as_team1": 0,
                "win_rate": 0.0,
                "avg_score": 0.0,
            }
        )

    def record_game(
        self,
        team0_risk: float,
        team1_risk: float,
        winner: Optional[int],
        scores: Tuple[int, int],
        hands_played: int,
    ) -> None:
        """
        Record a game result.

        Parameters
        ----------
        team0_risk : float
            Risk factor for team 0 (players 0 and 2).
        team1_risk : float
            Risk factor for team 1 (players 1 and 3).
        winner : Optional[int]
            Winning team (0 or 1), or None if tie.
        scores : Tuple[int, int]
            Final scores (team0, team1).
        hands_played : int
            Number of hands played.
        """
        # Create matchup key
        matchup_key = f"{team0_risk:.2f}_vs_{team1_risk:.2f}"

        self.matchups[matchup_key]["games_played"] += 1
        self.matchups[matchup_key]["hands_played"].append(hands_played)

        if winner == 0:
            self.matchups[matchup_key]["team0_wins"] += 1
            self.risk_factor_stats[team0_risk]["wins"] += 1
            self.risk_factor_stats[team1_risk]["losses"] += 1
        elif winner == 1:
            self.matchups[matchup_key]["team1_wins"] += 1
            self.risk_factor_stats[team1_risk]["wins"] += 1
            self.risk_factor_stats[team0_risk]["losses"] += 1
        else:
            self.risk_factor_stats[team0_risk]["ties"] += 1
            self.risk_factor_stats[team1_risk]["ties"] += 1

        self.matchups[matchup_key]["team0_total_score"] += scores[0]
        self.matchups[matchup_key]["team1_total_score"] += scores[1]

        # Update risk factor stats
        self.risk_factor_stats[team0_risk]["total_games"] += 1
        self.risk_factor_stats[team1_risk]["total_games"] += 1
        self.risk_factor_stats[team0_risk]["total_score"] += scores[0]
        self.risk_factor_stats[team1_risk]["total_score"] += scores[1]
        self.risk_factor_stats[team0_risk]["games_as_team0"] += 1
        self.risk_factor_stats[team1_risk]["games_as_team1"] += 1

        # Update win rates
        for risk_factor in [team0_risk, team1_risk]:
            stats = self.risk_factor_stats[risk_factor]
            if stats["total_games"] > 0:
                stats["win_rate"] = stats["wins"] / stats["total_games"]
                stats["avg_score"] = stats["total_score"] / stats["total_games"]

        # Update matchup averages
        matchup = self.matchups[matchup_key]
        if matchup["games_played"] > 0:
            matchup["team0_avg_score"] = matchup["team0_total_score"] / matchup["games_played"]
            matchup["team1_avg_score"] = matchup["team1_total_score"] / matchup["games_played"]

    def get_summary(self) -> Dict:
        """
        Get summary statistics.

        Returns
        -------
        Dict
            Summary statistics.
        """
        # Find best risk factor
        best_risk = None
        best_win_rate = -1.0
        for risk_factor, stats in self.risk_factor_stats.items():
            if stats["win_rate"] > best_win_rate:
                best_win_rate = stats["win_rate"]
                best_risk = risk_factor

        return {
            "best_risk_factor": best_risk,
            "best_win_rate": best_win_rate,
            "risk_factor_stats": dict(self.risk_factor_stats),
            "matchups": dict(self.matchups),
        }

    def print_summary(self) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("RISK FACTOR ANALYSIS SUMMARY")
        print("=" * 80)

        # Sort risk factors by win rate
        sorted_risks = sorted(
            self.risk_factor_stats.items(),
            key=lambda x: x[1]["win_rate"],
            reverse=True,
        )

        print("\nRisk Factor Performance (sorted by win rate):")
        print("-" * 80)
        print(f"{'Risk':<10} {'Games':<10} {'Wins':<10} {'Losses':<10} {'Ties':<10} {'Win Rate':<12} {'Avg Score':<12}")
        print("-" * 80)

        for risk_factor, stats in sorted_risks:
            print(
                f"{risk_factor:<10.2f} {stats['total_games']:<10} "
                f"{stats['wins']:<10} {stats['losses']:<10} {stats['ties']:<10} "
                f"{stats['win_rate']:<12.4f} {stats['avg_score']:<12.2f}"
            )

        # Show top matchups
        print("\n" + "=" * 80)
        print("TOP MATCHUPS:")
        print("=" * 80)
        print(f"{'Matchup':<30} {'Games':<10} {'Team0 Wins':<15} {'Team1 Wins':<15} {'Team0 Avg':<12} {'Team1 Avg':<12}")
        print("-" * 80)

        sorted_matchups = sorted(
            self.matchups.items(),
            key=lambda x: x[1]["games_played"],
            reverse=True,
        )[:20]  # Top 20 matchups

        for matchup_key, stats in sorted_matchups:
            print(
                f"{matchup_key:<30} {stats['games_played']:<10} "
                f"{stats['team0_wins']:<15} {stats['team1_wins']:<15} "
                f"{stats['team0_avg_score']:<12.2f} {stats['team1_avg_score']:<12.2f}"
            )


def create_game_with_risk_factors(
    team0_risk: float, team1_risk: float, seed: Optional[int] = None
) -> Game:
    """
    Create a game with specified risk factors for each team.

    Parameters
    ----------
    team0_risk : float
        Risk factor for team 0 (players 0 and 2).
    team1_risk : float
        Risk factor for team 1 (players 1 and 3).
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Game
        Created game instance.
    """
    # Create game with placeholder config
    player_config = [
        ("Player 0", "heuristic2"),
        ("Player 1", "heuristic2"),
        ("Player 2", "heuristic2"),
        ("Player 3", "heuristic2"),
    ]
    game = Game(player_config, seed=seed)

    # Replace profiles with HeuristicPlayer2 instances with specific risk factors
    # Team 0: players 0 and 2
    game.players[0].profile = HeuristicPlayer2(risk_factor=team0_risk)
    game.players[2].profile = HeuristicPlayer2(risk_factor=team0_risk)
    # Team 1: players 1 and 3
    game.players[1].profile = HeuristicPlayer2(risk_factor=team1_risk)
    game.players[3].profile = HeuristicPlayer2(risk_factor=team1_risk)

    return game


def play_single_game(
    team0_risk: float, team1_risk: float, seed: Optional[int] = None
) -> Tuple[Optional[int], Tuple[int, int], int]:
    """
    Play a single game and return results.

    Parameters
    ----------
    team0_risk : float
        Risk factor for team 0.
    team1_risk : float
        Risk factor for team 1.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Tuple[Optional[int], Tuple[int, int], int]
        (winner, scores, hands_played)
    """
    game = create_game_with_risk_factors(team0_risk, team1_risk, seed=seed)

    hands_played = 0
    while True:
        hands_played += 1
        continue_game = game.play_hand()

        scores = game.get_scores()
        winner = game.get_winner()

        if winner is not None:
            return (winner, scores, hands_played)

        if not continue_game:
            # Game ended without winner (shouldn't happen, but handle it)
            if scores[0] > scores[1]:
                return (0, scores, hands_played)
            elif scores[1] > scores[0]:
                return (1, scores, hands_played)
            else:
                return (None, scores, hands_played)


def run_monte_carlo_simulation(
    risk_factors: List[float],
    games_per_matchup: int = 100,
    seed: Optional[int] = None,
) -> RiskFactorComparison:
    """
    Run Monte Carlo simulation comparing different risk factors.

    Parameters
    ----------
    risk_factors : List[float]
        List of risk factors to test.
    games_per_matchup : int
        Number of games to play for each matchup.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    RiskFactorComparison
        Comparison results.
    """
    comparison = RiskFactorComparison()

    # Generate all matchups (each risk factor vs each risk factor)
    matchups = []
    for risk0 in risk_factors:
        for risk1 in risk_factors:
            matchups.append((risk0, risk1))

    total_games = len(matchups) * games_per_matchup
    print(f"Running {total_games} games across {len(matchups)} matchups...")
    print(f"Risk factors: {risk_factors}")
    print(f"Games per matchup: {games_per_matchup}\n")

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Run simulations with progress bar
    game_counter = 0
    with tqdm(total=total_games, desc="Simulating games") as pbar:
        for risk0, risk1 in matchups:
            for game_num in range(games_per_matchup):
                # Use different seed for each game
                game_seed = None if seed is None else seed + game_counter
                winner, scores, hands_played = play_single_game(
                    risk0, risk1, seed=game_seed
                )
                comparison.record_game(risk0, risk1, winner, scores, hands_played)
                game_counter += 1
                pbar.update(1)

    return comparison


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation to find optimal risk factor"
    )
    parser.add_argument(
        "--risk-factors",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Risk factors to test (default: 0.0 0.25 0.5 0.75 1.0)",
    )
    parser.add_argument(
        "--games-per-matchup",
        type=int,
        default=100,
        help="Number of games per matchup (default: 100)",
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
        help="Output file for JSON results (default: print only)",
    )

    args = parser.parse_args()

    # Validate risk factors
    for risk in args.risk_factors:
        if not 0.0 <= risk <= 1.0:
            parser.error(f"Risk factor must be between 0.0 and 1.0, got {risk}")

    # Run simulation
    comparison = run_monte_carlo_simulation(
        args.risk_factors, args.games_per_matchup, args.seed
    )

    # Print summary
    comparison.print_summary()

    # Save results if output file specified
    if args.output:
        summary = comparison.get_summary()
        summary["timestamp"] = datetime.now().isoformat()
        summary["config"] = {
            "risk_factors": args.risk_factors,
            "games_per_matchup": args.games_per_matchup,
            "seed": args.seed,
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()





