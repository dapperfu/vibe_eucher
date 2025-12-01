#!/usr/bin/env python3
"""Monte Carlo simulation comparing eucher_zero vs euchergo.

This script runs many games between eucher_zero and euchergo players,
with each bot type on a team (2v2). It tracks comprehensive statistics
including tricks won, games won, hands played, and other metrics.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eucher.game import Game
from eucher.game_stats import GameStatistics
from eucher.training.profiling import Profiler, TimingStats, get_timing_stats


class ComparisonStatistics:
    """Collects statistics from eucher_zero vs euchergo comparison games."""

    def __init__(self) -> None:
        """Initialize the comparison collector."""
        # Game-level statistics
        self.games: List[Dict] = []
        
        # Team-level statistics
        self.euchergo_team_stats: Dict[str, any] = {
            "games_won": 0,
            "games_lost": 0,
            "total_score": 0,
            "total_tricks_won": 0,
            "total_hands_played": 0,
            "hands_won": 0,
            "hands_lost": 0,
            "trump_makes": 0,
            "going_alone_attempts": 0,
            "going_alone_successes": 0,
        }
        
        self.eucher_zero_team_stats: Dict[str, any] = {
            "games_won": 0,
            "games_lost": 0,
            "total_score": 0,
            "total_tricks_won": 0,
            "total_hands_played": 0,
            "hands_won": 0,
            "hands_lost": 0,
            "trump_makes": 0,
            "going_alone_attempts": 0,
            "going_alone_successes": 0,
        }
        
        # Per-player statistics (aggregated across all games)
        self.euchergo_player_stats: Dict[int, Dict[str, any]] = defaultdict(
            lambda: {
                "tricks_won": 0,
                "trump_makes": 0,
                "points_contributed": 0,
                "going_alone_attempts": 0,
                "going_alone_successes": 0,
            }
        )
        
        self.eucher_zero_player_stats: Dict[int, Dict[str, any]] = defaultdict(
            lambda: {
                "tricks_won": 0,
                "trump_makes": 0,
                "points_contributed": 0,
                "going_alone_attempts": 0,
                "going_alone_successes": 0,
            }
        )
        
        # Per-hand statistics
        self.hand_results: List[Dict] = []

    def record_game(
        self,
        winner: Optional[int],
        scores: Tuple[int, int],
        game_stats: GameStatistics,
        players: List,
        euchergo_team: int,
    ) -> None:
        """
        Record a completed game.

        Parameters
        ----------
        winner : Optional[int]
            Winning team (0 or 1), or None if tie.
        scores : Tuple[int, int]
            Final scores (team0, team1).
        game_stats : GameStatistics
            Statistics from the game.
        players : List
            List of Player objects.
        euchergo_team : int
            Team ID (0 or 1) that euchergo is on.
        """
        eucher_zero_team = 1 - euchergo_team
        
        # Determine which team won
        if winner == euchergo_team:
            self.euchergo_team_stats["games_won"] += 1
            self.eucher_zero_team_stats["games_lost"] += 1
        elif winner == eucher_zero_team:
            self.eucher_zero_team_stats["games_won"] += 1
            self.euchergo_team_stats["games_lost"] += 1
        
        # Record scores
        self.euchergo_team_stats["total_score"] += scores[euchergo_team]
        self.eucher_zero_team_stats["total_score"] += scores[eucher_zero_team]
        
        # Record tricks won per team
        euchergo_tricks = game_stats.tricks_won_per_team.get(euchergo_team, 0)
        eucher_zero_tricks = game_stats.tricks_won_per_team.get(eucher_zero_team, 0)
        
        self.euchergo_team_stats["total_tricks_won"] += euchergo_tricks
        self.eucher_zero_team_stats["total_tricks_won"] += eucher_zero_tricks
        
        # Record hands played
        hands_played = game_stats.hands_played
        self.euchergo_team_stats["total_hands_played"] += hands_played
        self.eucher_zero_team_stats["total_hands_played"] += hands_played
        
        # Record per-player statistics
        for player in players:
            player_id = player.player_id
            player_team = player.team
            
            if player_team == euchergo_team:
                stats_dict = self.euchergo_player_stats[player_id]
            else:
                stats_dict = self.eucher_zero_player_stats[player_id]
            
            stats_dict["tricks_won"] += game_stats.tricks_won_per_player.get(player_id, 0)
            stats_dict["trump_makes"] += game_stats.trump_makers.get(player_id, 0)
            stats_dict["points_contributed"] += game_stats.points_contributed_per_player.get(player_id, 0)
            stats_dict["going_alone_attempts"] += game_stats.going_alone_attempts.get(player_id, 0)
            stats_dict["going_alone_successes"] += game_stats.going_alone_successes.get(player_id, 0)
        
        # Record trump makes per team
        for player_id, count in game_stats.trump_makers.items():
            player_team = players[player_id].team
            if player_team == euchergo_team:
                self.euchergo_team_stats["trump_makes"] += count
            else:
                self.eucher_zero_team_stats["trump_makes"] += count
        
        # Record going alone stats per team
        for player_id, attempts in game_stats.going_alone_attempts.items():
            player_team = players[player_id].team
            successes = game_stats.going_alone_successes.get(player_id, 0)
            if player_team == euchergo_team:
                self.euchergo_team_stats["going_alone_attempts"] += attempts
                self.euchergo_team_stats["going_alone_successes"] += successes
            else:
                self.eucher_zero_team_stats["going_alone_attempts"] += attempts
                self.eucher_zero_team_stats["going_alone_successes"] += successes
        
        # Store game data
        game_data = {
            "winner": winner,
            "euchergo_team": euchergo_team,
            "scores": scores,
            "hands_played": hands_played,
            "euchergo_tricks": euchergo_tricks,
            "eucher_zero_tricks": eucher_zero_tricks,
            "euchergo_score": scores[euchergo_team],
            "eucher_zero_score": scores[eucher_zero_team],
            "timestamp": datetime.now().isoformat(),
        }
        self.games.append(game_data)

    def get_summary(self) -> Dict[str, any]:
        """
        Get summary statistics.

        Returns
        -------
        Dict[str, any]
            Summary statistics dictionary.
        """
        total_games = len(self.games)
        
        # Calculate win rates
        euchergo_win_rate = (
            self.euchergo_team_stats["games_won"] / total_games
            if total_games > 0
            else 0.0
        )
        eucher_zero_win_rate = (
            self.eucher_zero_team_stats["games_won"] / total_games
            if total_games > 0
            else 0.0
        )
        
        # Calculate average scores
        euchergo_avg_score = (
            self.euchergo_team_stats["total_score"] / total_games
            if total_games > 0
            else 0.0
        )
        eucher_zero_avg_score = (
            self.eucher_zero_team_stats["total_score"] / total_games
            if total_games > 0
            else 0.0
        )
        
        # Calculate average tricks per game
        euchergo_avg_tricks = (
            self.euchergo_team_stats["total_tricks_won"] / total_games
            if total_games > 0
            else 0.0
        )
        eucher_zero_avg_tricks = (
            self.eucher_zero_team_stats["total_tricks_won"] / total_games
            if total_games > 0
            else 0.0
        )
        
        # Calculate average hands per game
        avg_hands_per_game = (
            self.euchergo_team_stats["total_hands_played"] / total_games
            if total_games > 0
            else 0.0
        )
        
        # Calculate going alone success rates
        euchergo_alone_success_rate = (
            self.euchergo_team_stats["going_alone_successes"]
            / self.euchergo_team_stats["going_alone_attempts"]
            if self.euchergo_team_stats["going_alone_attempts"] > 0
            else 0.0
        )
        eucher_zero_alone_success_rate = (
            self.eucher_zero_team_stats["going_alone_successes"]
            / self.eucher_zero_team_stats["going_alone_attempts"]
            if self.eucher_zero_team_stats["going_alone_attempts"] > 0
            else 0.0
        )
        
        return {
            "total_games": total_games,
            "euchergo": {
                **self.euchergo_team_stats,
                "win_rate": euchergo_win_rate,
                "avg_score": euchergo_avg_score,
                "avg_tricks_per_game": euchergo_avg_tricks,
                "going_alone_success_rate": euchergo_alone_success_rate,
            },
            "eucher_zero": {
                **self.eucher_zero_team_stats,
                "win_rate": eucher_zero_win_rate,
                "avg_score": eucher_zero_avg_score,
                "avg_tricks_per_game": eucher_zero_avg_tricks,
                "going_alone_success_rate": eucher_zero_alone_success_rate,
            },
            "avg_hands_per_game": avg_hands_per_game,
            "player_stats": {
                "euchergo": dict(self.euchergo_player_stats),
                "eucher_zero": dict(self.eucher_zero_player_stats),
            },
        }

    def to_dict(self) -> Dict[str, any]:
        """
        Convert statistics to dictionary for serialization.

        Returns
        -------
        Dict[str, any]
            Dictionary representation of statistics.
        """
        return {
            "games": self.games,
            "euchergo_team_stats": self.euchergo_team_stats,
            "eucher_zero_team_stats": self.eucher_zero_team_stats,
            "euchergo_player_stats": dict(self.euchergo_player_stats),
            "eucher_zero_player_stats": dict(self.eucher_zero_player_stats),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "ComparisonStatistics":
        """
        Create statistics from dictionary.

        Parameters
        ----------
        data : Dict[str, any]
            Dictionary representation of statistics.

        Returns
        -------
        ComparisonStatistics
            Reconstructed statistics object.
        """
        stats = cls()
        stats.games = data.get("games", [])
        stats.euchergo_team_stats = data.get("euchergo_team_stats", stats.euchergo_team_stats)
        stats.eucher_zero_team_stats = data.get("eucher_zero_team_stats", stats.eucher_zero_team_stats)
        stats.euchergo_player_stats = defaultdict(
            lambda: {
                "tricks_won": 0,
                "trump_makes": 0,
                "points_contributed": 0,
                "going_alone_attempts": 0,
                "going_alone_successes": 0,
            },
            data.get("euchergo_player_stats", {}),
        )
        stats.eucher_zero_player_stats = defaultdict(
            lambda: {
                "tricks_won": 0,
                "trump_makes": 0,
                "points_contributed": 0,
                "going_alone_attempts": 0,
                "going_alone_successes": 0,
            },
            data.get("eucher_zero_player_stats", {}),
        )
        return stats


def run_single_game(
    seed: int,
    euchergo_team: int = 0,
    euchergo_model_path: Optional[str] = None,
    eucher_zero_model_path: Optional[str] = None,
    euchergo_num_simulations: Optional[int] = None,
    eucher_zero_num_simulations: Optional[int] = None,
    euchergo_risk_factor: float = 0.0,
    eucher_zero_risk_factor: float = 0.0,
    timing_stats: Optional[TimingStats] = None,
) -> Tuple[Optional[int], Tuple[int, int], GameStatistics, List]:
    """
    Run a single game between euchergo and eucher_zero.

    Parameters
    ----------
    seed : int
        Random seed for the game.
    euchergo_team : int
        Team ID (0 or 1) for euchergo players.
    euchergo_model_path : Optional[str]
        Path to euchergo model checkpoint.
    eucher_zero_model_path : Optional[str]
        Path to eucher_zero model checkpoint.
    euchergo_num_simulations : Optional[int]
        Number of MCTS simulations for euchergo.
    eucher_zero_num_simulations : Optional[int]
        Number of MCTS simulations for eucher_zero.
    euchergo_risk_factor : float
        Risk factor for euchergo players.
    eucher_zero_risk_factor : float
        Risk factor for eucher_zero players.
    timing_stats : Optional[TimingStats]
        Optional timing statistics collector.

    Returns
    -------
    Tuple[Optional[int], Tuple[int, int], GameStatistics, List]
        (winner, scores, game_stats, players)
    """
    eucher_zero_team = 1 - euchergo_team
    
    # Import player classes
    from eucher.players.computer.euchergo.player import EucherGoPlayer
    from eucher.players.computer.euchergo.config import EucherGoConfig
    from eucher.players.computer.eucher_zero.player import EucherZeroPlayer
    from eucher.players.computer.eucher_zero.config import EucherZeroConfig
    
    # Create player configuration (placeholder - we'll replace profiles)
    # Team 0: players 0 and 2
    # Team 1: players 1 and 3
    if euchergo_team == 0:
        player_config = [
            ("EucherGo_0", "heuristic"),  # Placeholder
            ("EucherZero_1", "heuristic"),  # Placeholder
            ("EucherGo_2", "heuristic"),  # Placeholder
            ("EucherZero_3", "heuristic"),  # Placeholder
        ]
    else:
        player_config = [
            ("EucherZero_0", "heuristic"),  # Placeholder
            ("EucherGo_1", "heuristic"),  # Placeholder
            ("EucherZero_2", "heuristic"),  # Placeholder
            ("EucherGo_3", "heuristic"),  # Placeholder
        ]
    
    # Create game with placeholder config
    start_time = time.perf_counter()
    game = Game(player_config, seed=seed)
    if timing_stats:
        timing_stats.record("game_creation", time.perf_counter() - start_time)
    
    # Create and inject custom profiles
    profiles = []
    for i, (name, _) in enumerate(player_config):
        player_team = game.players[i].team
        
        if player_team == euchergo_team:
            # Create euchergo profile
            config = EucherGoConfig()
            if euchergo_num_simulations is not None:
                config.num_simulations = euchergo_num_simulations
            profile = EucherGoPlayer(
                model_path=euchergo_model_path,
                config=config,
                num_simulations=euchergo_num_simulations,
                risk_factor=euchergo_risk_factor,
                game=game,
            )
        else:
            # Create eucher_zero profile
            config = EucherZeroConfig()
            if eucher_zero_num_simulations is not None:
                config.num_simulations = eucher_zero_num_simulations
            profile = EucherZeroPlayer(
                model_path=eucher_zero_model_path,
                config=config,
                num_simulations=eucher_zero_num_simulations,
                risk_factor=eucher_zero_risk_factor,
                game=game,
            )
        profiles.append(profile)
    
    # Replace profiles
    for i, profile in enumerate(profiles):
        game.players[i].profile = profile
    
    # Play game
    hands_played = 0
    while True:
        try:
            hand_start = time.perf_counter()
            continue_game = game.play_hand()
            hand_time = time.perf_counter() - hand_start
            hands_played += 1
            
            if timing_stats:
                timing_stats.record("play_hand", hand_time)
            
            if not continue_game:
                break
            
            winner = game.get_winner()
            if winner is not None:
                break
        except Exception as e:
            print(f"\nError in game (seed {seed}): {e}")
            import traceback
            traceback.print_exc()
            break
    
    scores = game.get_scores()
    winner = game.get_winner()
    game_stats = game.stats
    
    if timing_stats:
        timing_stats.record("hands_per_game", hands_played)
    
    return winner, scores, game_stats, game.players


def run_monte_carlo_simulation(
    num_games: int,
    seed: Optional[int] = None,
    euchergo_model_path: Optional[str] = None,
    eucher_zero_model_path: Optional[str] = None,
    euchergo_num_simulations: Optional[int] = None,
    eucher_zero_num_simulations: Optional[int] = None,
    euchergo_risk_factor: float = 0.0,
    eucher_zero_risk_factor: float = 0.0,
    checkpoint_file: Optional[Path] = None,
    checkpoint_interval: int = 100,
    profile: bool = False,
    profile_output: Optional[Path] = None,
) -> ComparisonStatistics:
    """
    Run Monte Carlo simulation comparing euchergo vs eucher_zero.

    Parameters
    ----------
    num_games : int
        Number of games to run.
    seed : Optional[int]
        Base random seed for reproducibility.
    euchergo_model_path : Optional[str]
        Path to euchergo model checkpoint.
    eucher_zero_model_path : Optional[str]
        Path to eucher_zero model checkpoint.
    euchergo_num_simulations : Optional[int]
        Number of MCTS simulations for euchergo.
    eucher_zero_num_simulations : Optional[int]
        Number of MCTS simulations for eucher_zero.
    euchergo_risk_factor : float
        Risk factor for euchergo players.
    eucher_zero_risk_factor : float
        Risk factor for eucher_zero players.
    checkpoint_file : Optional[Path]
        Path to checkpoint file for saving/loading progress.
    checkpoint_interval : int
        Number of games between checkpoint saves.
    profile : bool
        Enable profiling to identify performance bottlenecks.
    profile_output : Optional[Path]
        Output file for profiling results.

    Returns
    -------
    ComparisonStatistics
        Comparison statistics.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    stats = ComparisonStatistics()
    timing_stats = TimingStats() if profile else None
    
    # Setup profiling if requested
    profiler_context = None
    if profile:
        profiler_context = Profiler(output_file=profile_output, sort_by="cumulative")
        profiler_context.__enter__()
    
    # Try to load checkpoint if it exists
    if checkpoint_file and checkpoint_file.exists():
        try:
            checkpoint_data = json.loads(checkpoint_file.read_text())
            stats = ComparisonStatistics.from_dict(checkpoint_data.get("statistics", {}))
            games_completed = len(stats.games)
            print(f"Loaded checkpoint from {checkpoint_file}")
            print(f"  Resuming with {games_completed} games already completed")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            print("  Starting fresh")
    
    # Alternate which team euchergo is on for fairness
    games_played_since_checkpoint = 0
    
    with tqdm(total=num_games, desc="Running games", initial=len(stats.games)) as pbar:
        for game_num in range(len(stats.games), num_games):
            # Alternate teams: even games = team 0, odd games = team 1
            euchergo_team = game_num % 2
            
            try:
                game_start = time.perf_counter()
                winner, scores, game_stats, players = run_single_game(
                    seed=(seed or 0) + game_num,
                    euchergo_team=euchergo_team,
                    euchergo_model_path=euchergo_model_path,
                    eucher_zero_model_path=eucher_zero_model_path,
                    euchergo_num_simulations=euchergo_num_simulations,
                    eucher_zero_num_simulations=eucher_zero_num_simulations,
                    euchergo_risk_factor=euchergo_risk_factor,
                    eucher_zero_risk_factor=eucher_zero_risk_factor,
                    timing_stats=timing_stats,
                )
                game_time = time.perf_counter() - game_start
                if timing_stats:
                    timing_stats.record("full_game", game_time)
                
                stats.record_game(winner, scores, game_stats, players, euchergo_team)
                games_played_since_checkpoint += 1
                
                # Save checkpoint periodically
                if checkpoint_file and games_played_since_checkpoint >= checkpoint_interval:
                    try:
                        checkpoint_data = {
                            "timestamp": datetime.now().isoformat(),
                            "num_games": num_games,
                            "seed": seed,
                            "statistics": stats.to_dict(),
                        }
                        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
                        games_played_since_checkpoint = 0
                    except Exception as e:
                        print(f"\nWarning: Failed to save checkpoint: {e}")
                
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError running game {game_num}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Final checkpoint save
    if checkpoint_file:
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "num_games": num_games,
                "seed": seed,
                "statistics": stats.to_dict(),
            }
            checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
            print(f"\nFinal checkpoint saved to {checkpoint_file}")
        except Exception as e:
            print(f"\nWarning: Failed to save final checkpoint: {e}")
    
    # Print timing statistics if profiling
    if timing_stats:
        timing_stats.print_summary()
        if profile_output:
            timing_file = profile_output.parent / f"{profile_output.stem}_timing.txt"
            timing_stats.save_summary(timing_file)
    
    # Exit profiler if active
    if profiler_context:
        profiler_context.__exit__(None, None, None)
    
    return stats


def print_results(stats: ComparisonStatistics) -> None:
    """Print comparison results in a readable format."""
    summary = stats.get_summary()
    
    print("\n" + "=" * 80)
    print("eucher_zero vs euchergo COMPARISON RESULTS")
    print("=" * 80)
    print()
    
    print(f"Total Games: {summary['total_games']}")
    print(f"Average Hands per Game: {summary['avg_hands_per_game']:.2f}")
    print()
    
    print("-" * 80)
    print("euchergo Team Statistics")
    print("-" * 80)
    euchergo = summary["euchergo"]
    print(f"Games Won: {euchergo['games_won']} ({euchergo['win_rate']:.1%})")
    print(f"Games Lost: {euchergo['games_lost']}")
    print(f"Total Score: {euchergo['total_score']}")
    print(f"Average Score per Game: {euchergo['avg_score']:.2f}")
    print(f"Total Tricks Won: {euchergo['total_tricks_won']}")
    print(f"Average Tricks per Game: {euchergo['avg_tricks_per_game']:.2f}")
    print(f"Trump Makes: {euchergo['trump_makes']}")
    print(f"Going Alone: {euchergo['going_alone_attempts']} attempts, "
          f"{euchergo['going_alone_successes']} successes "
          f"({euchergo['going_alone_success_rate']:.1%})")
    print()
    
    print("-" * 80)
    print("eucher_zero Team Statistics")
    print("-" * 80)
    eucher_zero = summary["eucher_zero"]
    print(f"Games Won: {eucher_zero['games_won']} ({eucher_zero['win_rate']:.1%})")
    print(f"Games Lost: {eucher_zero['games_lost']}")
    print(f"Total Score: {eucher_zero['total_score']}")
    print(f"Average Score per Game: {eucher_zero['avg_score']:.2f}")
    print(f"Total Tricks Won: {eucher_zero['total_tricks_won']}")
    print(f"Average Tricks per Game: {eucher_zero['avg_tricks_per_game']:.2f}")
    print(f"Trump Makes: {eucher_zero['trump_makes']}")
    print(f"Going Alone: {eucher_zero['going_alone_attempts']} attempts, "
          f"{eucher_zero['going_alone_successes']} successes "
          f"({eucher_zero['going_alone_success_rate']:.1%})")
    print()
    
    print("=" * 80)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo comparison of eucher_zero vs euchergo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 1000 games
  python scripts/monte_carlo/eucher_zero_vs_euchergo.py --num-games 1000

  # Run with custom model paths
  python scripts/monte_carlo/eucher_zero_vs_euchergo.py --num-games 500 \\
      --euchergo-model path/to/euchergo/model.pt \\
      --eucher-zero-model path/to/eucher_zero/model.pt

  # Run with checkpointing
  python scripts/monte_carlo/eucher_zero_vs_euchergo.py --num-games 1000 \\
      --checkpoint results.checkpoint.json
        """,
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games to run (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--euchergo-model",
        type=str,
        default=None,
        help="Path to euchergo model checkpoint",
    )
    parser.add_argument(
        "--eucher-zero-model",
        type=str,
        default=None,
        help="Path to eucher_zero model checkpoint",
    )
    parser.add_argument(
        "--euchergo-simulations",
        type=int,
        default=None,
        help="Number of MCTS simulations for euchergo",
    )
    parser.add_argument(
        "--eucher-zero-simulations",
        type=int,
        default=None,
        help="Number of MCTS simulations for eucher_zero",
    )
    parser.add_argument(
        "--euchergo-risk",
        type=float,
        default=0.0,
        help="Risk factor for euchergo players (default: 0.0)",
    )
    parser.add_argument(
        "--eucher-zero-risk",
        type=float,
        default=0.0,
        help="Risk factor for eucher_zero players (default: 0.0)",
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
        default=100,
        help="Number of games between checkpoint saves (default: 100)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling to identify performance bottlenecks",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default=None,
        help="Output file for profiling results (default: profiles/monte_carlo_profile.txt)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("eucher_zero vs euchergo Monte Carlo Comparison")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Number of games: {args.num_games}")
    print()
    
    # Determine checkpoint file
    checkpoint_file = None
    if args.checkpoint:
        checkpoint_file = Path(args.checkpoint)
    elif args.output:
        # Auto-generate checkpoint file name from output file
        checkpoint_file = Path(args.output).with_suffix(".checkpoint.json")
    
    # Determine profile output file
    profile_output = None
    if args.profile:
        if args.profile_output:
            profile_output = Path(args.profile_output)
        else:
            profile_dir = Path("profiles")
            profile_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            profile_output = profile_dir / f"monte_carlo_profile_{timestamp}.txt"
    
    # Run simulation
    stats = run_monte_carlo_simulation(
        num_games=args.num_games,
        seed=args.seed,
        euchergo_model_path=args.euchergo_model,
        eucher_zero_model_path=args.eucher_zero_model,
        euchergo_num_simulations=args.euchergo_simulations,
        eucher_zero_num_simulations=args.eucher_zero_simulations,
        euchergo_risk_factor=args.euchergo_risk,
        eucher_zero_risk_factor=args.eucher_zero_risk,
        checkpoint_file=checkpoint_file,
        checkpoint_interval=args.checkpoint_interval,
        profile=args.profile,
        profile_output=profile_output,
    )
    
    # Print results
    print_results(stats)
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate output file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("stats") / f"eucher_zero_vs_euchergo_{timestamp}.json"
        output_path.parent.mkdir(exist_ok=True)
    
    summary = stats.get_summary()
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "num_games": args.num_games,
        "seed": args.seed,
        "euchergo_model_path": args.euchergo_model,
        "eucher_zero_model_path": args.eucher_zero_model,
        "euchergo_num_simulations": args.euchergo_simulations,
        "eucher_zero_num_simulations": args.eucher_zero_simulations,
        "euchergo_risk_factor": args.euchergo_risk,
        "eucher_zero_risk_factor": args.eucher_zero_risk,
        **summary,
        "detailed_statistics": stats.to_dict(),
    }
    
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

