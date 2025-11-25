#!/usr/bin/env python3
"""Monte Carlo comparison of different Euchre bot types.

This script runs many games between different combinations of bot types
to determine which bot performs best overall.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eucher.game import Game


# Available bot types to test
BOT_TYPES = ["heuristic", "ai", "random", "euchre_zero"]

# Try to import euchre_zero, skip if not available
EUCHRE_ZERO_AVAILABLE = True
try:
    from eucher.players.computer.euchre_zero.player import EuchreZeroPlayer
except ImportError:
    EUCHRE_ZERO_AVAILABLE = False
    BOT_TYPES = ["heuristic", "ai", "random"]


class BotComparison:
    """Collects statistics from bot comparison games."""

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
            }
        )
        self.bot_stats: Dict[str, Dict[str, any]] = defaultdict(
            lambda: {
                "total_games": 0,
                "wins": 0,
                "losses": 0,
                "total_score": 0,
                "games_as_team0": 0,
                "games_as_team1": 0,
            }
        )

    def record_game(
        self,
        team0_bots: Tuple[str, str],
        team1_bots: Tuple[str, str],
        winner: Optional[int],
        scores: Tuple[int, int],
        hands_played: int,
    ) -> None:
        """
        Record a game result.

        Parameters
        ----------
        team0_bots : Tuple[str, str]
            Bot types for team 0 (players 0 and 2).
        team1_bots : Tuple[str, str]
            Bot types for team 1 (players 1 and 3).
        winner : Optional[int]
            Winning team (0 or 1), or None if tie.
        scores : Tuple[int, int]
            Final scores (team0, team1).
        hands_played : int
            Number of hands played.
        """
        # Create matchup key (sorted for consistency)
        matchup_key = f"{team0_bots[0]}+{team0_bots[1]}_vs_{team1_bots[0]}+{team1_bots[1]}"

        self.matchups[matchup_key]["games_played"] += 1
        self.matchups[matchup_key]["hands_played"].append(hands_played)

        if winner == 0:
            self.matchups[matchup_key]["team0_wins"] += 1
        elif winner == 1:
            self.matchups[matchup_key]["team1_wins"] += 1

        self.matchups[matchup_key]["team0_total_score"] += scores[0]
        self.matchups[matchup_key]["team1_total_score"] += scores[1]

        # Update bot statistics
        for bot_type in team0_bots:
            self.bot_stats[bot_type]["total_games"] += 1
            self.bot_stats[bot_type]["games_as_team0"] += 1
            if winner == 0:
                self.bot_stats[bot_type]["wins"] += 1
                self.bot_stats[bot_type]["total_score"] += scores[0]
            elif winner == 1:
                self.bot_stats[bot_type]["losses"] += 1

        for bot_type in team1_bots:
            self.bot_stats[bot_type]["total_games"] += 1
            self.bot_stats[bot_type]["games_as_team1"] += 1
            if winner == 1:
                self.bot_stats[bot_type]["wins"] += 1
                self.bot_stats[bot_type]["total_score"] += scores[1]
            elif winner == 0:
                self.bot_stats[bot_type]["losses"] += 1

    def get_bot_rankings(self) -> List[Tuple[str, Dict[str, any]]]:
        """
        Get bots ranked by win rate.

        Returns
        -------
        List[Tuple[str, Dict[str, any]]]
            List of (bot_type, stats) tuples sorted by win rate.
        """
        rankings = []
        for bot_type, stats in self.bot_stats.items():
            if stats["total_games"] > 0:
                win_rate = stats["wins"] / stats["total_games"]
                avg_score = (
                    stats["total_score"] / stats["total_games"] if stats["total_games"] > 0 else 0.0
                )
                rankings.append(
                    (
                        bot_type,
                        {
                            **stats,
                            "win_rate": win_rate,
                            "avg_score": avg_score,
                        },
                    )
                )

        # Sort by win rate descending
        rankings.sort(key=lambda x: x[1]["win_rate"], reverse=True)
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

        return {
            "total_games": total_games,
            "total_matchups": total_matchups,
            "bot_rankings": [
                {"bot_type": bot, **stats} for bot, stats in self.get_bot_rankings()
            ],
            "matchups": dict(self.matchups),
        }

    def to_dict(self) -> Dict[str, any]:
        """
        Convert comparison to dictionary for serialization.

        Returns
        -------
        Dict[str, any]
            Dictionary representation of comparison.
        """
        return {
            "matchups": dict(self.matchups),
            "bot_stats": dict(self.bot_stats),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "BotComparison":
        """
        Create comparison from dictionary.

        Parameters
        ----------
        data : Dict[str, any]
            Dictionary representation of comparison.

        Returns
        -------
        BotComparison
            Reconstructed comparison object.
        """
        comparison = cls()
        comparison.matchups = defaultdict(
            lambda: {
                "games_played": 0,
                "team0_wins": 0,
                "team1_wins": 0,
                "team0_total_score": 0,
                "team1_total_score": 0,
                "hands_played": [],
            },
            data.get("matchups", {}),
        )
        comparison.bot_stats = defaultdict(
            lambda: {
                "total_games": 0,
                "wins": 0,
                "losses": 0,
                "total_score": 0,
                "games_as_team0": 0,
                "games_as_team1": 0,
            },
            data.get("bot_stats", {}),
        )
        return comparison


def run_matchup(
    team0_bots: Tuple[str, str],
    team1_bots: Tuple[str, str],
    num_games: int,
    seed_base: int = 0,
    progress_bar: Optional[tqdm] = None,
) -> List[Dict[str, any]]:
    """
    Run games for a specific matchup.

    Parameters
    ----------
    team0_bots : Tuple[str, str]
        Bot types for team 0.
    team1_bots : Tuple[str, str]
        Bot types for team 1.
    num_games : int
        Number of games to run.
    seed_base : int
        Base seed for random number generation.
    progress_bar : Optional[tqdm]
        Optional progress bar to update.

    Returns
    -------
    List[Dict[str, any]]
        List of game results.
    """
    results = []

    for game_num in range(num_games):
        try:
            # Create player configuration
            player_config = [
                (f"{team0_bots[0]}_0", team0_bots[0]),
                (f"{team1_bots[0]}_1", team1_bots[0]),
                (f"{team0_bots[1]}_2", team0_bots[1]),
                (f"{team1_bots[1]}_3", team1_bots[1]),
            ]

            # Create game with seed
            game = Game(player_config, seed=seed_base + game_num)

            # Play game
            hands_played = 0
            while True:
                try:
                    continue_game = game.play_hand()
                    hands_played += 1
                    if not continue_game:
                        break

                    winner = game.get_winner()
                    if winner is not None:
                        break
                except Exception as e:
                    print(f"\nError in game {game_num}: {e}")
                    break

            scores = game.get_scores()
            winner = game.get_winner()

            results.append(
                {
                    "team0_bots": team0_bots,
                    "team1_bots": team1_bots,
                    "winner": winner,
                    "scores": scores,
                    "hands_played": hands_played,
                }
            )

            if progress_bar:
                progress_bar.update(1)

        except Exception as e:
            print(f"\nError running game {game_num} for matchup {team0_bots} vs {team1_bots}: {e}")
            continue

    return results


def generate_matchups(bot_types: List[str]) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
    """
    Generate all unique matchups between bot types.

    Parameters
    ----------
    bot_types : List[str]
        List of bot types to test.

    Returns
    -------
    List[Tuple[Tuple[str, str], Tuple[str, str]]]
        List of (team0_bots, team1_bots) tuples.
    """
    matchups = []

    # Generate all combinations of 2 bots for a team
    team_combinations = list(combinations_with_replacement(bot_types, 2))

    # Generate all matchup combinations
    for team0 in team_combinations:
        for team1 in team_combinations:
            # Skip if both teams are identical (not interesting)
            if team0 == team1:
                continue

            matchups.append((team0, team1))

    # Also test same bot vs same bot (e.g., heuristic+heuristic vs ai+ai)
    for bot0 in bot_types:
        for bot1 in bot_types:
            if bot0 != bot1:
                matchups.append(((bot0, bot0), (bot1, bot1)))

    return matchups


def run_monte_carlo_comparison(
    bot_types: List[str],
    num_games_per_matchup: int = 100,
    seed: Optional[int] = None,
    checkpoint_file: Optional[Path] = None,
    checkpoint_interval: int = 1000,
) -> BotComparison:
    """
    Run Monte Carlo comparison of all bot types.

    Parameters
    ----------
    bot_types : List[str]
        List of bot types to compare.
    num_games_per_matchup : int
        Number of games to run per matchup.
    seed : Optional[int]
        Random seed for reproducibility.
    checkpoint_file : Optional[Path]
        Path to checkpoint file for saving/loading progress.
    checkpoint_interval : int
        Number of games between checkpoint saves.

    Returns
    -------
    BotComparison
        Comparison results.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    comparison = BotComparison()

    # Try to load checkpoint if it exists
    if checkpoint_file and checkpoint_file.exists():
        try:
            checkpoint_data = json.loads(checkpoint_file.read_text())
            comparison = BotComparison.from_dict(checkpoint_data.get("comparison", {}))
            print(f"Loaded checkpoint from {checkpoint_file}")
            print(f"  Resuming with {comparison.get_summary()['total_games']} games already completed")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            print("  Starting fresh")

    # Generate all matchups
    matchups = generate_matchups(bot_types)
    total_games = len(matchups) * num_games_per_matchup

    print(f"Testing {len(bot_types)} bot types: {', '.join(bot_types)}")
    print(f"Generated {len(matchups)} unique matchups")
    print(f"Running {num_games_per_matchup} games per matchup ({total_games} total games)")
    if checkpoint_file:
        print(f"Checkpoint file: {checkpoint_file} (saving every {checkpoint_interval} games)")
    print()

    # Track games played for checkpointing
    games_played_since_checkpoint = 0

    # Run all matchups
    with tqdm(total=total_games, desc="Running games", initial=comparison.get_summary()["total_games"]) as pbar:
        for matchup_idx, (team0_bots, team1_bots) in enumerate(matchups):
            matchup_str = f"{team0_bots[0]}+{team0_bots[1]} vs {team1_bots[0]}+{team1_bots[1]}"
            pbar.set_description(f"Matchup: {matchup_str}")

            # Check if this matchup is already complete
            matchup_key = f"{team0_bots[0]}+{team0_bots[1]}_vs_{team1_bots[0]}+{team1_bots[1]}"
            matchup_stats = comparison.matchups.get(matchup_key, {})
            games_completed = matchup_stats.get("games_played", 0)
            
            if games_completed >= num_games_per_matchup:
                # Skip already completed matchups
                pbar.update(num_games_per_matchup)
                continue

            # Run remaining games for this matchup
            remaining_games = num_games_per_matchup - games_completed
            results = run_matchup(
                team0_bots, team1_bots, remaining_games, 
                seed_base=(seed or 0) + games_completed, 
                progress_bar=pbar
            )

            # Record results
            for result in results:
                comparison.record_game(
                    result["team0_bots"],
                    result["team1_bots"],
                    result["winner"],
                    result["scores"],
                    result["hands_played"],
                )
                games_played_since_checkpoint += 1

                # Save checkpoint periodically
                if checkpoint_file and games_played_since_checkpoint >= checkpoint_interval:
                    try:
                        checkpoint_data = {
                            "timestamp": datetime.now().isoformat(),
                            "bot_types": bot_types,
                            "num_games_per_matchup": num_games_per_matchup,
                            "seed": seed,
                            "comparison": comparison.to_dict(),
                        }
                        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
                        games_played_since_checkpoint = 0
                    except Exception as e:
                        print(f"\nWarning: Failed to save checkpoint: {e}")

    # Final checkpoint save
    if checkpoint_file:
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "bot_types": bot_types,
                "num_games_per_matchup": num_games_per_matchup,
                "seed": seed,
                "comparison": comparison.to_dict(),
            }
            checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
            print(f"\nFinal checkpoint saved to {checkpoint_file}")
        except Exception as e:
            print(f"\nWarning: Failed to save final checkpoint: {e}")

    return comparison


def print_results(comparison: BotComparison) -> None:
    """Print comparison results in a readable format."""
    summary = comparison.get_summary()

    print("\n" + "=" * 80)
    print("BOT RANKINGS (by win rate)")
    print("=" * 80)
    print()

    rankings = summary["bot_rankings"]
    for rank, bot_data in enumerate(rankings, 1):
        bot_type = bot_data["bot_type"]
        stats = {k: v for k, v in bot_data.items() if k != "bot_type"}
        print(f"{rank}. {bot_type.upper()}")
        print(f"   Win Rate: {stats['win_rate']:.1%} ({stats['wins']}/{stats['total_games']})")
        print(f"   Avg Score: {stats['avg_score']:.2f}")
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
        description="Monte Carlo comparison of Euchre bot types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all bots with 100 games per matchup
  python scripts/monte_carlo_bot_comparison.py --num-games 100

  # Compare only heuristic and ai bots
  python scripts/monte_carlo_bot_comparison.py --bots heuristic ai --num-games 200

  # Save results to file
  python scripts/monte_carlo_bot_comparison.py --num-games 100 --output results.json
        """,
    )
    parser.add_argument(
        "--bots",
        nargs="+",
        choices=BOT_TYPES,
        default=BOT_TYPES,
        help=f"Bot types to compare (default: {', '.join(BOT_TYPES)})",
    )
    parser.add_argument(
        "--num-games",
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
        help="Output file for results (JSON format)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file for saving/loading progress (JSON format)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Number of games between checkpoint saves (default: 1000)",
    )

    args = parser.parse_args()

    # Filter out euchre_zero if not available
    bots_to_test = [b for b in args.bots if b != "euchre_zero" or EUCHRE_ZERO_AVAILABLE]

    if not bots_to_test:
        print("Error: No valid bot types to test")
        sys.exit(1)

    if "euchre_zero" in args.bots and not EUCHRE_ZERO_AVAILABLE:
        print("Warning: euchre_zero not available, skipping")

    print("=" * 80)
    print("Monte Carlo Bot Comparison")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Determine checkpoint file
    checkpoint_file = None
    if args.checkpoint:
        checkpoint_file = Path(args.checkpoint)
    elif args.output:
        # Auto-generate checkpoint file name from output file
        checkpoint_file = Path(args.output).with_suffix(".checkpoint.json")

    # Run comparison
    comparison = run_monte_carlo_comparison(
        bot_types=bots_to_test,
        num_games_per_matchup=args.num_games,
        seed=args.seed,
        checkpoint_file=checkpoint_file,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Print results
    print_results(comparison)

    # Save results if requested
    if args.output:
        summary = comparison.get_summary()
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "bot_types_tested": bots_to_test,
            "num_games_per_matchup": args.num_games,
            "seed": args.seed,
            **summary,
        }

        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

