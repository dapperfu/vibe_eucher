#!/usr/bin/env python3
"""Round-robin tournament simulation for Euchre AI player types.

This script runs a round-robin tournament where each AI player type competes
as a team (both players on a team are the same type). Each matchup runs
multiple Monte Carlo simulations to determine relative performance.
"""

import argparse
import json
import random
import sys
import threading
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eucher.game import Game


# Default player types to test
DEFAULT_PLAYER_TYPES = [
    "heuristic",
    "ai",
    "random",
    "ml_sklearn",
    "ml_pytorch",
    "perceiver_muzero",
    "euchre_zero",
    "pytorch_ai",
]


def check_player_type_available(player_type: str) -> bool:
    """
    Check if a player type is available (can be instantiated).

    Parameters
    ----------
    player_type : str
        Player type to check.

    Returns
    -------
    bool
        True if the player type is available, False otherwise.
    """
    if player_type in ["human"]:
        return False  # Skip human players for tournament

    # Try to create a test game with this player type
    try:
        # Create a minimal test game configuration
        test_config = [
            (f"test_{player_type}_0", player_type),
            (f"test_{player_type}_1", "random"),  # Use random for other players
            (f"test_{player_type}_2", "random"),
            (f"test_{player_type}_3", "random"),
        ]
        test_game = Game(test_config, seed=0)
        # If we get here, the player type is available
        return True
    except (ValueError, ImportError, Exception) as e:
        # Player type not available
        return False


def get_available_player_types(requested_types: Optional[List[str]] = None) -> List[str]:
    """
    Get list of available player types, filtering out unavailable ones.

    Parameters
    ----------
    requested_types : Optional[List[str]]
        List of player types to check. If None, uses DEFAULT_PLAYER_TYPES.

    Returns
    -------
    List[str]
        List of available player types.
    """
    if requested_types is None:
        requested_types = DEFAULT_PLAYER_TYPES

    available_types = []
    for player_type in requested_types:
        if check_player_type_available(player_type):
            available_types.append(player_type)
        else:
            print(f"Warning: Player type '{player_type}' not available, skipping")

    return available_types


class TournamentSimulator:
    """Manages round-robin tournament simulation and statistics collection."""

    def __init__(self) -> None:
        """Initialize the tournament simulator."""
        # Matchup statistics: (player_type_0, player_type_1) -> stats
        self.matchups: Dict[Tuple[str, str], Dict[str, any]] = defaultdict(
            lambda: {
                "games_played": 0,
                "team0_wins": 0,
                "team1_wins": 0,
                "team0_total_score": 0,
                "team1_total_score": 0,
                "hands_played": [],
                "scores_team0": [],
                "scores_team1": [],
            }
        )
        # Player type statistics
        self.player_type_stats: Dict[str, Dict[str, any]] = defaultdict(
            lambda: {
                "total_games": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "total_score": 0,
                "total_score_against": 0,
            }
        )

    def record_game(
        self,
        player_type_0: str,
        player_type_1: str,
        winner: Optional[int],
        scores: Tuple[int, int],
        hands_played: int,
    ) -> None:
        """
        Record a game result.

        Parameters
        ----------
        player_type_0 : str
            Player type for Team 0 (players 0 and 2).
        player_type_1 : str
            Player type for Team 1 (players 1 and 3).
        winner : Optional[int]
            Winning team (0 or 1), or None if tie.
        scores : Tuple[int, int]
            Final scores (team0, team1).
        hands_played : int
            Number of hands played.
        """
        # Create matchup key (sorted for consistency)
        matchup_key = tuple(sorted([player_type_0, player_type_1]))

        self.matchups[matchup_key]["games_played"] += 1
        self.matchups[matchup_key]["hands_played"].append(hands_played)
        self.matchups[matchup_key]["scores_team0"].append(scores[0])
        self.matchups[matchup_key]["scores_team1"].append(scores[1])

        # Determine which team is which based on matchup key order
        if matchup_key[0] == player_type_0:
            # player_type_0 is Team 0
            if winner == 0:
                self.matchups[matchup_key]["team0_wins"] += 1
            elif winner == 1:
                self.matchups[matchup_key]["team1_wins"] += 1
            self.matchups[matchup_key]["team0_total_score"] += scores[0]
            self.matchups[matchup_key]["team1_total_score"] += scores[1]
        else:
            # player_type_1 is Team 0 (reversed)
            if winner == 1:
                self.matchups[matchup_key]["team0_wins"] += 1
            elif winner == 0:
                self.matchups[matchup_key]["team1_wins"] += 1
            self.matchups[matchup_key]["team0_total_score"] += scores[1]
            self.matchups[matchup_key]["team1_total_score"] += scores[0]

        # Update player type statistics
        # For player_type_0
        self.player_type_stats[player_type_0]["total_games"] += 1
        if winner == 0:
            self.player_type_stats[player_type_0]["wins"] += 1
            self.player_type_stats[player_type_0]["total_score"] += scores[0]
            self.player_type_stats[player_type_0]["total_score_against"] += scores[1]
        elif winner == 1:
            self.player_type_stats[player_type_0]["losses"] += 1
            self.player_type_stats[player_type_0]["total_score"] += scores[0]
            self.player_type_stats[player_type_0]["total_score_against"] += scores[1]
        else:
            self.player_type_stats[player_type_0]["ties"] += 1
            self.player_type_stats[player_type_0]["total_score"] += scores[0]
            self.player_type_stats[player_type_0]["total_score_against"] += scores[1]

        # For player_type_1
        self.player_type_stats[player_type_1]["total_games"] += 1
        if winner == 1:
            self.player_type_stats[player_type_1]["wins"] += 1
            self.player_type_stats[player_type_1]["total_score"] += scores[1]
            self.player_type_stats[player_type_1]["total_score_against"] += scores[0]
        elif winner == 0:
            self.player_type_stats[player_type_1]["losses"] += 1
            self.player_type_stats[player_type_1]["total_score"] += scores[1]
            self.player_type_stats[player_type_1]["total_score_against"] += scores[0]
        else:
            self.player_type_stats[player_type_1]["ties"] += 1
            self.player_type_stats[player_type_1]["total_score"] += scores[1]
            self.player_type_stats[player_type_1]["total_score_against"] += scores[0]

    def get_player_type_rankings(self) -> List[Tuple[str, Dict[str, any]]]:
        """
        Get player types ranked by win rate.

        Returns
        -------
        List[Tuple[str, Dict[str, any]]]
            List of (player_type, stats) tuples sorted by win rate.
        """
        rankings = []
        for player_type, stats in self.player_type_stats.items():
            if stats["total_games"] > 0:
                win_rate = stats["wins"] / stats["total_games"]
                avg_score = stats["total_score"] / stats["total_games"]
                avg_score_against = stats["total_score_against"] / stats["total_games"]
                score_diff = avg_score - avg_score_against

                rankings.append(
                    (
                        player_type,
                        {
                            **stats,
                            "win_rate": win_rate,
                            "avg_score": avg_score,
                            "avg_score_against": avg_score_against,
                            "avg_score_diff": score_diff,
                        },
                    )
                )

        # Sort by win rate descending, then by score difference
        rankings.sort(key=lambda x: (x[1]["win_rate"], x[1]["avg_score_diff"]), reverse=True)
        return rankings

    def get_summary(self) -> Dict[str, any]:
        """
        Get summary statistics.

        Returns
        -------
        Dict[str, any]
            Summary statistics.
        """
        total_games = sum(m["games_played"] for m in self.matchups.values())
        total_matchups = len(self.matchups)

        # Convert matchup keys from tuples to strings for JSON serialization
        matchups_dict = {}
        for matchup_key, stats in self.matchups.items():
            matchup_str = f"{matchup_key[0]}_vs_{matchup_key[1]}"
            matchups_dict[matchup_str] = dict(stats)

        return {
            "total_games": total_games,
            "total_matchups": total_matchups,
            "player_type_rankings": [
                {"player_type": pt, **stats} for pt, stats in self.get_player_type_rankings()
            ],
            "matchups": matchups_dict,
        }


class TimeoutError(Exception):
    """Timeout exception for game execution."""
    pass


def run_single_game_with_timeout(
    player_config: List[Tuple[str, str]],
    seed: int,
    timeout_seconds: int = 300,  # 5 minutes per game
) -> Optional[Dict[str, any]]:
    """
    Run a single game with timeout protection using threading.

    Parameters
    ----------
    player_config : List[Tuple[str, str]]
        Player configuration.
    seed : int
        Random seed.
    timeout_seconds : int
        Maximum time per game in seconds.

    Returns
    -------
    Optional[Dict[str, any]]
        Game result or None if timeout/error.
    """
    result_container = {"result": None, "exception": None}

    def run_game():
        """Run the game in a separate thread."""
        try:
            # Create game with seed
            game = Game(player_config, seed=seed)

            # Play game
            hands_played = 0
            max_hands = 50  # Safety limit to prevent infinite loops
            while hands_played < max_hands:
                continue_game = game.play_hand()
                hands_played += 1
                if not continue_game:
                    break

                winner = game.get_winner()
                if winner is not None:
                    break

            scores = game.get_scores()
            winner = game.get_winner()

            result_container["result"] = {
                "winner": winner,
                "scores": scores,
                "hands_played": hands_played,
            }
        except Exception as e:
            result_container["exception"] = e

    # Run game in thread with timeout
    thread = threading.Thread(target=run_game, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread is still running - timed out
        print(f"\nGame timed out after {timeout_seconds} seconds (seed: {seed})")
        return None

    if result_container["exception"]:
        print(f"\nError in game (seed: {seed}): {result_container['exception']}")
        import traceback
        traceback.print_exc()
        return None

    return result_container["result"]


def run_matchup(
    player_type_0: str,
    player_type_1: str,
    num_games: int,
    seed_base: int = 0,
    progress_bar: Optional[tqdm] = None,
    timeout_per_game: int = 300,
) -> List[Dict[str, any]]:
    """
    Run games for a specific matchup.

    Parameters
    ----------
    player_type_0 : str
        Player type for Team 0 (players 0 and 2).
    player_type_1 : str
        Player type for Team 1 (players 1 and 3).
    num_games : int
        Number of games to run.
    seed_base : int
        Base seed for random number generation.
    progress_bar : Optional[tqdm]
        Optional progress bar to update.
    timeout_per_game : int
        Maximum seconds per game before timeout.

    Returns
    -------
    List[Dict[str, any]]
        List of game results.
    """
    results = []

    for game_num in range(num_games):
        try:
            # Create player configuration
            # Team 0: players 0 and 2 are player_type_0
            # Team 1: players 1 and 3 are player_type_1
            player_config = [
                (f"{player_type_0}_0", player_type_0),
                (f"{player_type_1}_1", player_type_1),
                (f"{player_type_0}_2", player_type_0),
                (f"{player_type_1}_3", player_type_1),
            ]

            # Run game with timeout
            game_result = run_single_game_with_timeout(
                player_config, seed_base + game_num, timeout_per_game
            )

            if game_result is not None:
                results.append(
                    {
                        "player_type_0": player_type_0,
                        "player_type_1": player_type_1,
                        "winner": game_result["winner"],
                        "scores": game_result["scores"],
                        "hands_played": game_result["hands_played"],
                    }
                )

            if progress_bar:
                progress_bar.update(1)

        except Exception as e:
            print(f"\nError running game {game_num} for matchup {player_type_0} vs {player_type_1}: {e}")
            import traceback
            traceback.print_exc()
            if progress_bar:
                progress_bar.update(1)
            continue

    return results


def generate_round_robin_matchups(player_types: List[str]) -> List[Tuple[str, str]]:
    """
    Generate all unique round-robin matchups.

    Parameters
    ----------
    player_types : List[str]
        List of player types to test.

    Returns
    -------
    List[Tuple[str, str]]
        List of (player_type_0, player_type_1) tuples for matchups.
    """
    # Generate all combinations of 2 player types (round-robin)
    matchups = list(combinations(player_types, 2))
    return matchups


def run_tournament(
    player_types: List[str],
    num_games_per_matchup: int = 200,
    seed: Optional[int] = None,
    timeout_per_game: int = 300,
) -> TournamentSimulator:
    """
    Run round-robin tournament simulation.

    Parameters
    ----------
    player_types : List[str]
        List of player types to compare.
    num_games_per_matchup : int
        Number of games to run per matchup.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    TournamentSimulator
        Tournament results.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    tournament = TournamentSimulator()

    # Generate all matchups
    matchups = generate_round_robin_matchups(player_types)
    total_games = len(matchups) * num_games_per_matchup

    print(f"Testing {len(player_types)} player types: {', '.join(player_types)}")
    print(f"Generated {len(matchups)} unique matchups (round-robin)")
    print(f"Running {num_games_per_matchup} games per matchup ({total_games} total games)")
    print()

    # Run all matchups
    with tqdm(total=total_games, desc="Running games") as pbar:
        for player_type_0, player_type_1 in matchups:
            matchup_str = f"{player_type_0} vs {player_type_1}"
            pbar.set_description(f"Matchup: {matchup_str}")

            results = run_matchup(
                player_type_0, player_type_1, num_games_per_matchup, 
                seed_base=seed or 0, progress_bar=pbar, timeout_per_game=timeout_per_game
            )

            # Record results
            for result in results:
                tournament.record_game(
                    result["player_type_0"],
                    result["player_type_1"],
                    result["winner"],
                    result["scores"],
                    result["hands_played"],
                )

    return tournament


def print_results(tournament: TournamentSimulator) -> None:
    """Print tournament results in a readable format."""
    summary = tournament.get_summary()

    print("\n" + "=" * 80)
    print("PLAYER TYPE RANKINGS (by win rate)")
    print("=" * 80)
    print()

    rankings = summary["player_type_rankings"]
    for rank, player_data in enumerate(rankings, 1):
        player_type = player_data["player_type"]
        stats = {k: v for k, v in player_data.items() if k != "player_type"}
        print(f"{rank}. {player_type.upper()}")
        print(f"   Win Rate: {stats['win_rate']:.1%} ({stats['wins']}/{stats['total_games']})")
        print(f"   Avg Score: {stats['avg_score']:.2f}")
        print(f"   Avg Score Against: {stats['avg_score_against']:.2f}")
        print(f"   Score Difference: {stats['avg_score_diff']:+.2f}")
        print(f"   Games Played: {stats['total_games']}")
        print()

    print("=" * 80)
    print("MATCHUP DETAILS")
    print("=" * 80)
    print()

    for matchup_key, matchup_stats in sorted(summary["matchups"].items()):
        games = matchup_stats["games_played"]
        if games == 0:
            continue

        team0_wins = matchup_stats["team0_wins"]
        team1_wins = matchup_stats["team1_wins"]
        team0_win_rate = team0_wins / games if games > 0 else 0.0
        team1_win_rate = team1_wins / games if games > 0 else 0.0

        avg_hands = (
            np.mean(matchup_stats["hands_played"]) if matchup_stats["hands_played"] else 0.0
        )

        print(f"{matchup_key}")
        print(f"  Games: {games}")
        print(f"  Team 0 Win Rate: {team0_win_rate:.1%} ({team0_wins} wins)")
        print(f"  Team 1 Win Rate: {team1_win_rate:.1%} ({team1_wins} wins)")
        print(f"  Avg Hands per Game: {avg_hands:.1f}")
        print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Round-robin tournament simulation for Euchre AI player types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run tournament with all available player types, 200 games per matchup
  python scripts/ai_tournament.py

  # Run tournament with specific player types
  python scripts/ai_tournament.py --types heuristic ai random --num-games 200

  # Save results to file
  python scripts/ai_tournament.py --num-games 200 --output tournament_results.json
        """,
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="Player types to test (default: all available types)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=200,
        help="Number of games per matchup (default: 200)",
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
        help="Output file for results (JSON format). If not specified, uses tournament_results_<timestamp>.json",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per game in seconds (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow player types (perceiver_muzero, euchre_zero) that may hang",
    )

    args = parser.parse_args()

    # Get available player types
    requested_types = args.types if args.types else DEFAULT_PLAYER_TYPES
    
    # Skip slow player types if requested
    if args.skip_slow:
        slow_types = {"perceiver_muzero", "euchre_zero"}
        requested_types = [pt for pt in requested_types if pt not in slow_types]
        print(f"Skipping slow player types: {', '.join(slow_types)}")
    
    player_types = get_available_player_types(requested_types)

    if not player_types:
        print("Error: No valid player types to test")
        sys.exit(1)

    print("=" * 80)
    print("AI Tournament - Round-Robin Monte Carlo Simulation")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Run tournament
    tournament = run_tournament(
        player_types=player_types, 
        num_games_per_matchup=args.num_games, 
        seed=args.seed,
        timeout_per_game=args.timeout,
    )

    # Print results
    print_results(tournament)

    # Save results
    summary = tournament.get_summary()
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "player_types_tested": player_types,
        "num_games_per_matchup": args.num_games,
        "seed": args.seed,
        **summary,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        # Generate default filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"tournament_results_{timestamp_str}.json")

    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

