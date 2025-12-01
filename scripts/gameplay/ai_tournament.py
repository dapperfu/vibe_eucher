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
    "weighted_heuristic",
    "random",
    "ml_sklearn",
    "ml_pytorch",
    "perceiver_muzero",
    "eucher_zero",
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
        # Checkpoint tracking
        self.completed_matchups: set = set()  # Set of completed matchup keys
        self.current_matchup: Optional[Tuple[str, str]] = None
        self.current_matchup_games_completed: int = 0

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

    def save_checkpoint(self, checkpoint_path: Path, player_types: List[str], 
                       matchups: List[Tuple[str, str]], num_games_per_matchup: int,
                       seed: Optional[int], current_matchup_idx: int) -> None:
        """
        Save tournament state to checkpoint file.

        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file.
        player_types : List[str]
            List of player types being tested.
        matchups : List[Tuple[str, str]]
            List of all matchups.
        num_games_per_matchup : int
            Number of games per matchup.
        seed : Optional[int]
            Random seed used.
        current_matchup_idx : int
            Current matchup index being processed.
        """
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "player_types": player_types,
            "matchups": [list(m) for m in matchups],  # Convert tuples to lists for JSON
            "num_games_per_matchup": num_games_per_matchup,
            "seed": seed,
            "current_matchup_idx": current_matchup_idx,
            "completed_matchups": [list(m) for m in self.completed_matchups],
            "current_matchup": list(self.current_matchup) if self.current_matchup else None,
            "current_matchup_games_completed": self.current_matchup_games_completed,
            "tournament_state": {
                "matchups": {
                    f"{k[0]}_vs_{k[1]}": dict(v) for k, v in self.matchups.items()
                },
                "player_type_stats": dict(self.player_type_stats),
            },
        }
        checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))
        print(f"\nCheckpoint saved to: {checkpoint_path}")

    @staticmethod
    def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, any]]:
        """
        Load tournament state from checkpoint file.

        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file.

        Returns
        -------
        Optional[Dict[str, any]]
            Checkpoint data if file exists, None otherwise.
        """
        if not checkpoint_path.exists():
            return None
        
        try:
            checkpoint_data = json.loads(checkpoint_path.read_text())
            # Convert matchup lists back to tuples
            checkpoint_data["matchups"] = [tuple(m) for m in checkpoint_data["matchups"]]
            checkpoint_data["completed_matchups"] = {tuple(m) for m in checkpoint_data.get("completed_matchups", [])}
            if checkpoint_data.get("current_matchup"):
                checkpoint_data["current_matchup"] = tuple(checkpoint_data["current_matchup"])
            return checkpoint_data
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None


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
        # Note: Daemon threads can't be forcefully killed, but they'll be cleaned up when main thread exits
        print(f"\n⚠️  Game timed out after {timeout_seconds} seconds (seed: {seed})")
        print(f"   Player config: {[p[1] for p in player_config]}")
        # Force garbage collection to help clean up
        import gc
        gc.collect()
        return None

    if result_container["exception"]:
        print(f"\n❌ Error in game (seed: {seed}): {result_container['exception']}")
        print(f"   Player config: {[p[1] for p in player_config]}")
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
    start_game: int = 0,
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
    start_game : int
        Game number to start from (for resuming).

    Returns
    -------
    List[Dict[str, any]]
        List of game results.
    """
    results = []
    timeout_count = 0
    consecutive_timeouts = 0
    max_consecutive_timeouts = 5  # Skip matchup if too many consecutive timeouts
    last_progress_time = None
    stall_timeout = 600  # 10 minutes without progress = stall

    for game_num in range(start_game, num_games):
        import time
        current_time = time.time()
        
        # Check for stall (no progress for too long)
        if last_progress_time is not None:
            time_since_progress = current_time - last_progress_time
            if time_since_progress > stall_timeout:
                print(f"\n⚠️  STALL DETECTED: No progress for {time_since_progress:.0f} seconds")
                print(f"   Last completed game: {game_num - 1}")
                print(f"   Current game: {game_num}")
                print(f"   Matchup: {player_type_0} vs {player_type_1}")
                print(f"   Skipping remaining games in this matchup")
                break
        
        last_progress_time = current_time
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

            # Log progress every 50 games
            if game_num > 0 and game_num % 50 == 0:
                print(f"\n  Progress: {game_num}/{num_games} games completed for {player_type_0} vs {player_type_1}")

            # Run game with timeout
            game_start_time = time.time()
            game_result = run_single_game_with_timeout(
                player_config, seed_base + game_num, timeout_per_game
            )
            game_duration = time.time() - game_start_time
            
            # Warn if games are taking too long (even if not timing out)
            if game_duration > timeout_per_game * 0.8:  # 80% of timeout
                print(f"\n⚠️  Slow game: {game_duration:.1f}s (game {game_num}, seed {seed_base + game_num})")
            
            # Force garbage collection periodically to prevent memory buildup
            if game_num > 0 and game_num % 25 == 0:
                import gc
                gc.collect()

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
                consecutive_timeouts = 0  # Reset counter on success
                last_progress_time = time.time()  # Update progress time
            else:
                timeout_count += 1
                consecutive_timeouts += 1
                # If too many consecutive timeouts, skip this matchup
                if consecutive_timeouts >= max_consecutive_timeouts:
                    print(f"\n⚠️  Skipping matchup {player_type_0} vs {player_type_1} after {consecutive_timeouts} consecutive timeouts")
                    print(f"   Completed {len(results)}/{num_games} games before skipping")
                    break

            if progress_bar:
                progress_bar.update(1)

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted at game {game_num}/{num_games} for {player_type_0} vs {player_type_1}")
            raise
        except Exception as e:
            print(f"\n❌ Error running game {game_num} for matchup {player_type_0} vs {player_type_1}: {e}")
            import traceback
            traceback.print_exc()
            consecutive_timeouts = 0  # Reset on exception (different from timeout)
            if progress_bar:
                progress_bar.update(1)
            continue

    if timeout_count > 0:
        print(f"\n⚠️  Matchup {player_type_0} vs {player_type_1}: {timeout_count} games timed out out of {num_games}")

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
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval: int = 10,
) -> TournamentSimulator:
    """
    Run round-robin tournament simulation with checkpoint support.

    Parameters
    ----------
    player_types : List[str]
        List of player types to compare.
    num_games_per_matchup : int
        Number of games to run per matchup.
    seed : Optional[int]
        Random seed for reproducibility.
    checkpoint_path : Optional[Path]
        Path to checkpoint file. If provided, will save checkpoints and can resume.
    checkpoint_interval : int
        Save checkpoint every N games (default: 10).

    Returns
    -------
    TournamentSimulator
        Tournament results.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    tournament = TournamentSimulator()
    matchups = generate_round_robin_matchups(player_types)
    current_matchup_idx = 0
    start_game = 0

    # Try to load checkpoint if it exists
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_data = TournamentSimulator.load_checkpoint(checkpoint_path)
        if checkpoint_data:
            print(f"Loading checkpoint from: {checkpoint_path}")
            print(f"Checkpoint timestamp: {checkpoint_data.get('timestamp', 'Unknown')}")
            
            # Restore tournament state
            tournament.completed_matchups = checkpoint_data["completed_matchups"]
            tournament.current_matchup = checkpoint_data.get("current_matchup")
            tournament.current_matchup_games_completed = checkpoint_data.get("current_matchup_games_completed", 0)
            
            # Restore statistics
            if "tournament_state" in checkpoint_data:
                state = checkpoint_data["tournament_state"]
                # Restore matchups
                for matchup_str, stats in state.get("matchups", {}).items():
                    parts = matchup_str.split("_vs_")
                    if len(parts) == 2:
                        matchup_key = tuple(sorted([parts[0], parts[1]]))
                        # Convert stats dict to proper format with lists
                        matchup_stats = {
                            "games_played": stats.get("games_played", 0),
                            "team0_wins": stats.get("team0_wins", 0),
                            "team1_wins": stats.get("team1_wins", 0),
                            "team0_total_score": stats.get("team0_total_score", 0),
                            "team1_total_score": stats.get("team1_total_score", 0),
                            "hands_played": stats.get("hands_played", []),
                            "scores_team0": stats.get("scores_team0", []),
                            "scores_team1": stats.get("scores_team1", []),
                        }
                        tournament.matchups[matchup_key] = matchup_stats
                # Restore player type stats
                for pt, stats in state.get("player_type_stats", {}).items():
                    tournament.player_type_stats[pt] = dict(stats)
            
            # Find where to resume
            current_matchup_idx = checkpoint_data.get("current_matchup_idx", 0)
            if tournament.current_matchup:
                start_game = tournament.current_matchup_games_completed
            
            completed_count = len(tournament.completed_matchups)
            print(f"Resuming: {completed_count}/{len(matchups)} matchups completed")
            if tournament.current_matchup:
                print(f"Resuming matchup: {tournament.current_matchup[0]} vs {tournament.current_matchup[1]}")
                print(f"  Games completed: {start_game}/{num_games_per_matchup}")
            print()

    # Calculate total games (accounting for already completed)
    total_games = len(matchups) * num_games_per_matchup
    completed_games = sum(
        tournament.matchups[m].get("games_played", 0) 
        for m in tournament.completed_matchups
    )
    if tournament.current_matchup:
        completed_games += tournament.current_matchup_games_completed
    
    remaining_games = total_games - completed_games

    if current_matchup_idx == 0 and start_game == 0:
        print(f"Testing {len(player_types)} player types: {', '.join(player_types)}")
        print(f"Generated {len(matchups)} unique matchups (round-robin)")
        print(f"Running {num_games_per_matchup} games per matchup ({total_games} total games)")
        if checkpoint_path:
            print(f"Checkpoints will be saved to: {checkpoint_path}")
            print(f"Checkpoint interval: every {checkpoint_interval} games")
        print()
    else:
        print(f"Remaining games: {remaining_games}")

    # Run matchups
    with tqdm(total=total_games, initial=completed_games, desc="Running games") as pbar:
        for idx in range(current_matchup_idx, len(matchups)):
            player_type_0, player_type_1 = matchups[idx]
            matchup_key = tuple(sorted([player_type_0, player_type_1]))
            matchup_str = f"{player_type_0} vs {player_type_1}"
            
            # Skip if already completed
            if matchup_key in tournament.completed_matchups:
                continue
            
            tournament.current_matchup = matchup_key
            pbar.set_description(f"Matchup: {matchup_str}")

            # Run games for this matchup
            print(f"\n▶️  Starting matchup {idx + 1}/{len(matchups)}: {matchup_str}")
            results = run_matchup(
                player_type_0, 
                player_type_1, 
                num_games_per_matchup, 
                seed_base=seed or 0, 
                progress_bar=pbar,
                timeout_per_game=timeout_per_game,
                start_game=start_game if idx == current_matchup_idx else 0,
            )
            print(f"✅ Completed matchup {idx + 1}/{len(matchups)}: {matchup_str} ({len(results)} games)")

            # Record results
            for game_idx, result in enumerate(results):
                tournament.record_game(
                    result["player_type_0"],
                    result["player_type_1"],
                    result["winner"],
                    result["scores"],
                    result["hands_played"],
                )
                
                # Update checkpoint tracking
                tournament.current_matchup_games_completed = start_game + game_idx + 1
                
                # Save checkpoint periodically during matchup
                if checkpoint_path and (game_idx + 1) % checkpoint_interval == 0:
                    tournament.save_checkpoint(
                        checkpoint_path, player_types, matchups, 
                        num_games_per_matchup, seed, idx
                    )
            
            # Mark matchup as completed
            tournament.completed_matchups.add(matchup_key)
            tournament.current_matchup = None
            tournament.current_matchup_games_completed = 0
            start_game = 0  # Reset for next matchup

            # Save checkpoint after each matchup completes
            if checkpoint_path:
                tournament.save_checkpoint(
                    checkpoint_path, player_types, matchups, 
                    num_games_per_matchup, seed, idx + 1
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

  # Run tournament with checkpoint support (can resume if interrupted)
  python scripts/ai_tournament.py --checkpoint tournament_checkpoint.json

  # Run tournament with specific player types and checkpoint
  python scripts/ai_tournament.py --types heuristic weighted_heuristic random --num-games 200 --checkpoint checkpoint.json

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
        default=120,
        help="Timeout per game in seconds (default: 120 = 2 minutes). Reduce if games hang frequently.",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow player types (perceiver_muzero, eucher_zero) that may hang",
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

    # Get available player types
    requested_types = args.types if args.types else DEFAULT_PLAYER_TYPES
    
    # Skip slow player types if requested
    if args.skip_slow:
        slow_types = {"perceiver_muzero", "eucher_zero"}
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

    # Set up checkpoint path
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Found existing checkpoint: {checkpoint_path}")
            print("Tournament will resume from checkpoint.")
        else:
            print(f"Checkpoint file will be created: {checkpoint_path}")
        print()

    # Run tournament
    tournament = run_tournament(
        player_types=player_types, 
        num_games_per_matchup=args.num_games, 
        seed=args.seed,
        timeout_per_game=args.timeout,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
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

