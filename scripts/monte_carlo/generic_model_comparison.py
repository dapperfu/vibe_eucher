#!/usr/bin/env python3
"""Generic Monte Carlo simulation comparing any two player models.

This script runs many games between two specified player models,
with each model type on a team (2v2). It tracks comprehensive statistics
including tricks won, games won, hands played, and other metrics.

The script uses the plugin system to instantiate players dynamically,
allowing comparison of any registered player types with configurable parameters.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eucher.game import Game
from eucher.game_stats import GameStatistics
from eucher.plugins.registry import get_registry
from eucher.training.profiling import Profiler, TimingStats, get_timing_stats


class ComparisonStatistics:
    """Collects statistics from model comparison games."""

    def __init__(self, model_a_name: str, model_b_name: str) -> None:
        """
        Initialize the comparison collector.

        Parameters
        ----------
        model_a_name : str
            Name of model A for labeling statistics.
        model_b_name : str
            Name of model B for labeling statistics.
        """
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        
        # Game-level statistics
        self.games: List[Dict[str, Any]] = []
        
        # Team-level statistics
        self.model_a_team_stats: Dict[str, Any] = {
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
        
        self.model_b_team_stats: Dict[str, Any] = {
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
        self.model_a_player_stats: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "tricks_won": 0,
                "trump_makes": 0,
                "points_contributed": 0,
                "going_alone_attempts": 0,
                "going_alone_successes": 0,
            }
        )
        
        self.model_b_player_stats: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "tricks_won": 0,
                "trump_makes": 0,
                "points_contributed": 0,
                "going_alone_attempts": 0,
                "going_alone_successes": 0,
            }
        )
        
        # Per-hand statistics
        self.hand_results: List[Dict[str, Any]] = []

    def record_game(
        self,
        winner: Optional[int],
        scores: Tuple[int, int],
        game_stats: GameStatistics,
        players: List[Any],
        model_a_team: int,
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
        players : List[Any]
            List of Player objects.
        model_a_team : int
            Team ID (0 or 1) that model A is on.
        """
        model_b_team = 1 - model_a_team
        
        # Determine which team won
        if winner == model_a_team:
            self.model_a_team_stats["games_won"] += 1
            self.model_b_team_stats["games_lost"] += 1
        elif winner == model_b_team:
            self.model_b_team_stats["games_won"] += 1
            self.model_a_team_stats["games_lost"] += 1
        
        # Record scores
        self.model_a_team_stats["total_score"] += scores[model_a_team]
        self.model_b_team_stats["total_score"] += scores[model_b_team]
        
        # Record tricks won per team
        model_a_tricks = game_stats.tricks_won_per_team.get(model_a_team, 0)
        model_b_tricks = game_stats.tricks_won_per_team.get(model_b_team, 0)
        
        self.model_a_team_stats["total_tricks_won"] += model_a_tricks
        self.model_b_team_stats["total_tricks_won"] += model_b_tricks
        
        # Record hands played
        hands_played = game_stats.hands_played
        self.model_a_team_stats["total_hands_played"] += hands_played
        self.model_b_team_stats["total_hands_played"] += hands_played
        
        # Record per-player statistics
        for player in players:
            player_id = player.player_id
            player_team = player.team
            
            if player_team == model_a_team:
                stats_dict = self.model_a_player_stats[player_id]
            else:
                stats_dict = self.model_b_player_stats[player_id]
            
            stats_dict["tricks_won"] += game_stats.tricks_won_per_player.get(player_id, 0)
            stats_dict["trump_makes"] += game_stats.trump_makers.get(player_id, 0)
            stats_dict["points_contributed"] += game_stats.points_contributed_per_player.get(player_id, 0)
            stats_dict["going_alone_attempts"] += game_stats.going_alone_attempts.get(player_id, 0)
            stats_dict["going_alone_successes"] += game_stats.going_alone_successes.get(player_id, 0)
        
        # Record trump makes per team
        for player_id, count in game_stats.trump_makers.items():
            player_team = players[player_id].team
            if player_team == model_a_team:
                self.model_a_team_stats["trump_makes"] += count
            else:
                self.model_b_team_stats["trump_makes"] += count
        
        # Record going alone stats per team
        for player_id, attempts in game_stats.going_alone_attempts.items():
            player_team = players[player_id].team
            successes = game_stats.going_alone_successes.get(player_id, 0)
            if player_team == model_a_team:
                self.model_a_team_stats["going_alone_attempts"] += attempts
                self.model_a_team_stats["going_alone_successes"] += successes
            else:
                self.model_b_team_stats["going_alone_attempts"] += attempts
                self.model_b_team_stats["going_alone_successes"] += successes
        
        # Store game data
        game_data = {
            "winner": winner,
            "model_a_team": model_a_team,
            "scores": scores,
            "hands_played": hands_played,
            "model_a_tricks": model_a_tricks,
            "model_b_tricks": model_b_tricks,
            "model_a_score": scores[model_a_team],
            "model_b_score": scores[model_b_team],
            "timestamp": datetime.now().isoformat(),
        }
        self.games.append(game_data)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns
        -------
        Dict[str, Any]
            Summary statistics dictionary.
        """
        total_games = len(self.games)
        
        # Calculate win rates
        model_a_win_rate = (
            self.model_a_team_stats["games_won"] / total_games
            if total_games > 0
            else 0.0
        )
        model_b_win_rate = (
            self.model_b_team_stats["games_won"] / total_games
            if total_games > 0
            else 0.0
        )
        
        # Calculate average scores
        model_a_avg_score = (
            self.model_a_team_stats["total_score"] / total_games
            if total_games > 0
            else 0.0
        )
        model_b_avg_score = (
            self.model_b_team_stats["total_score"] / total_games
            if total_games > 0
            else 0.0
        )
        
        # Calculate average tricks per game
        model_a_avg_tricks = (
            self.model_a_team_stats["total_tricks_won"] / total_games
            if total_games > 0
            else 0.0
        )
        model_b_avg_tricks = (
            self.model_b_team_stats["total_tricks_won"] / total_games
            if total_games > 0
            else 0.0
        )
        
        # Calculate average hands per game
        avg_hands_per_game = (
            self.model_a_team_stats["total_hands_played"] / total_games
            if total_games > 0
            else 0.0
        )
        
        # Calculate going alone success rates
        model_a_alone_success_rate = (
            self.model_a_team_stats["going_alone_successes"]
            / self.model_a_team_stats["going_alone_attempts"]
            if self.model_a_team_stats["going_alone_attempts"] > 0
            else 0.0
        )
        model_b_alone_success_rate = (
            self.model_b_team_stats["going_alone_successes"]
            / self.model_b_team_stats["going_alone_attempts"]
            if self.model_b_team_stats["going_alone_attempts"] > 0
            else 0.0
        )
        
        return {
            "total_games": total_games,
            self.model_a_name: {
                **self.model_a_team_stats,
                "win_rate": model_a_win_rate,
                "avg_score": model_a_avg_score,
                "avg_tricks_per_game": model_a_avg_tricks,
                "going_alone_success_rate": model_a_alone_success_rate,
            },
            self.model_b_name: {
                **self.model_b_team_stats,
                "win_rate": model_b_win_rate,
                "avg_score": model_b_avg_score,
                "avg_tricks_per_game": model_b_avg_tricks,
                "going_alone_success_rate": model_b_alone_success_rate,
            },
            "avg_hands_per_game": avg_hands_per_game,
            "player_stats": {
                self.model_a_name: dict(self.model_a_player_stats),
                self.model_b_name: dict(self.model_b_player_stats),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to dictionary for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of statistics.
        """
        return {
            "games": self.games,
            "model_a_team_stats": self.model_a_team_stats,
            "model_b_team_stats": self.model_b_team_stats,
            "model_a_player_stats": dict(self.model_a_player_stats),
            "model_b_player_stats": dict(self.model_b_player_stats),
            "model_a_name": self.model_a_name,
            "model_b_name": self.model_b_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComparisonStatistics":
        """
        Create statistics from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary representation of statistics.

        Returns
        -------
        ComparisonStatistics
            Reconstructed statistics object.
        """
        model_a_name = data.get("model_a_name", "model_a")
        model_b_name = data.get("model_b_name", "model_b")
        stats = cls(model_a_name, model_b_name)
        stats.games = data.get("games", [])
        stats.model_a_team_stats = data.get("model_a_team_stats", stats.model_a_team_stats)
        stats.model_b_team_stats = data.get("model_b_team_stats", stats.model_b_team_stats)
        stats.model_a_player_stats = defaultdict(
            lambda: {
                "tricks_won": 0,
                "trump_makes": 0,
                "points_contributed": 0,
                "going_alone_attempts": 0,
                "going_alone_successes": 0,
            },
            data.get("model_a_player_stats", {}),
        )
        stats.model_b_player_stats = defaultdict(
            lambda: {
                "tricks_won": 0,
                "trump_makes": 0,
                "points_contributed": 0,
                "going_alone_attempts": 0,
                "going_alone_successes": 0,
            },
            data.get("model_b_player_stats", {}),
        )
        return stats


def create_player_profile(
    plugin_name: str,
    game: Game,
    player_id: int,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a player profile using the plugin system.

    Parameters
    ----------
    plugin_name : str
        Name of the plugin to use (e.g., "euchergo", "eucher_zero", "perceiver_muzero").
    game : Game
        Game instance.
    player_id : int
        Player ID.
    config : Optional[Dict[str, Any]]
        Configuration dictionary to pass to the plugin factory as kwargs.

    Returns
    -------
    Any
        PlayerProfile instance.

    Raises
    ------
    ValueError
        If plugin is not found or cannot be instantiated.
    """
    registry = get_registry()
    plugin_metadata = registry.get(plugin_name)
    
    if plugin_metadata is None:
        raise ValueError(f"Plugin '{plugin_name}' not found. Available plugins: {registry.list_plugins()}")
    
    kwargs: Dict[str, Any] = config.copy() if config else {}
    
    # Add player_id for ML players that need it
    if plugin_metadata.requires_game or "ml" in plugin_name.lower():
        kwargs["player_id"] = player_id
    
    # Call factory with game instance if required
    if plugin_metadata.requires_game:
        return plugin_metadata.factory(game=game, **kwargs)
    else:
        return plugin_metadata.factory(game=None, **kwargs)


def run_single_game(
    seed: int,
    model_a_team: int,
    model_a_type: str,
    model_b_type: str,
    model_a_config: Optional[Dict[str, Any]] = None,
    model_b_config: Optional[Dict[str, Any]] = None,
    timing_stats: Optional[TimingStats] = None,
) -> Tuple[Optional[int], Tuple[int, int], GameStatistics, List[Any]]:
    """
    Run a single game between two model types.

    Parameters
    ----------
    seed : int
        Random seed for the game.
    model_a_team : int
        Team ID (0 or 1) for model A players.
    model_a_type : str
        Plugin name for model A (e.g., "euchergo", "eucher_zero").
    model_b_type : str
        Plugin name for model B (e.g., "euchergo", "eucher_zero").
    model_a_config : Optional[Dict[str, Any]]
        Configuration dictionary for model A (passed as kwargs to plugin factory).
    model_b_config : Optional[Dict[str, Any]]
        Configuration dictionary for model B (passed as kwargs to plugin factory).
    timing_stats : Optional[TimingStats]
        Optional timing statistics collector.

    Returns
    -------
    Tuple[Optional[int], Tuple[int, int], GameStatistics, List[Any]]
        (winner, scores, game_stats, players)
    """
    model_b_team = 1 - model_a_team
    
    # Create placeholder player config
    # Team 0: players 0 and 2
    # Team 1: players 1 and 3
    if model_a_team == 0:
        player_config = [
            ("ModelA_0", "heuristic"),  # Placeholder
            ("ModelB_1", "heuristic"),  # Placeholder
            ("ModelA_2", "heuristic"),  # Placeholder
            ("ModelB_3", "heuristic"),  # Placeholder
        ]
    else:
        player_config = [
            ("ModelB_0", "heuristic"),  # Placeholder
            ("ModelA_1", "heuristic"),  # Placeholder
            ("ModelB_2", "heuristic"),  # Placeholder
            ("ModelA_3", "heuristic"),  # Placeholder
        ]
    
    # Create game with placeholder config
    start_time = time.perf_counter()
    game = Game(player_config, seed=seed)
    if timing_stats:
        timing_stats.record("game_creation", time.perf_counter() - start_time)
    
    # Create and inject custom profiles using plugins
    profiles = []
    for i, (name, _) in enumerate(player_config):
        player_team = game.players[i].team
        
        if player_team == model_a_team:
            # Create model A profile
            profile = create_player_profile(
                plugin_name=model_a_type,
                game=game,
                player_id=i,
                config=model_a_config,
            )
        else:
            # Create model B profile
            profile = create_player_profile(
                plugin_name=model_b_type,
                game=game,
                player_id=i,
                config=model_b_config,
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
    model_a_type: str,
    model_b_type: str,
    model_a_config: Optional[Dict[str, Any]] = None,
    model_b_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    checkpoint_file: Optional[Path] = None,
    checkpoint_interval: int = 100,
    profile: bool = False,
    profile_output: Optional[Path] = None,
) -> ComparisonStatistics:
    """
    Run Monte Carlo simulation comparing two model types.

    Parameters
    ----------
    num_games : int
        Number of games to run.
    model_a_type : str
        Plugin name for model A.
    model_b_type : str
        Plugin name for model B.
    model_a_config : Optional[Dict[str, Any]]
        Configuration dictionary for model A (passed as kwargs to plugin factory).
    model_b_config : Optional[Dict[str, Any]]
        Configuration dictionary for model B (passed as kwargs to plugin factory).
    seed : Optional[int]
        Base random seed for reproducibility.
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
    
    stats = ComparisonStatistics(model_a_type, model_b_type)
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
    
    # Alternate which team model A is on for fairness
    games_played_since_checkpoint = 0
    
    with tqdm(total=num_games, desc="Running games", initial=len(stats.games)) as pbar:
        for game_num in range(len(stats.games), num_games):
            # Alternate teams: even games = team 0, odd games = team 1
            model_a_team = game_num % 2
            
            try:
                game_start = time.perf_counter()
                winner, scores, game_stats, players = run_single_game(
                    seed=(seed or 0) + game_num,
                    model_a_team=model_a_team,
                    model_a_type=model_a_type,
                    model_b_type=model_b_type,
                    model_a_config=model_a_config,
                    model_b_config=model_b_config,
                    timing_stats=timing_stats,
                )
                game_time = time.perf_counter() - game_start
                if timing_stats:
                    timing_stats.record("full_game", game_time)
                
                stats.record_game(winner, scores, game_stats, players, model_a_team)
                games_played_since_checkpoint += 1
                
                # Save checkpoint periodically
                if checkpoint_file and games_played_since_checkpoint >= checkpoint_interval:
                    try:
                        checkpoint_data = {
                            "timestamp": datetime.now().isoformat(),
                            "num_games": num_games,
                            "seed": seed,
                            "model_a_type": model_a_type,
                            "model_b_type": model_b_type,
                            "model_a_config": model_a_config,
                            "model_b_config": model_b_config,
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
                "model_a_type": model_a_type,
                "model_b_type": model_b_type,
                "model_a_config": model_a_config,
                "model_b_config": model_b_config,
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
    """
    Print comparison results in a readable format.

    Parameters
    ----------
    stats : ComparisonStatistics
        Statistics to print.
    """
    summary = stats.get_summary()
    
    print("\n" + "=" * 80)
    print(f"{stats.model_a_name} vs {stats.model_b_name} COMPARISON RESULTS")
    print("=" * 80)
    print()
    
    print(f"Total Games: {summary['total_games']}")
    print(f"Average Hands per Game: {summary['avg_hands_per_game']:.2f}")
    print()
    
    print("-" * 80)
    print(f"{stats.model_a_name} Team Statistics")
    print("-" * 80)
    model_a = summary[stats.model_a_name]
    print(f"Games Won: {model_a['games_won']} ({model_a['win_rate']:.1%})")
    print(f"Games Lost: {model_a['games_lost']}")
    print(f"Total Score: {model_a['total_score']}")
    print(f"Average Score per Game: {model_a['avg_score']:.2f}")
    print(f"Total Tricks Won: {model_a['total_tricks_won']}")
    print(f"Average Tricks per Game: {model_a['avg_tricks_per_game']:.2f}")
    print(f"Trump Makes: {model_a['trump_makes']}")
    print(f"Going Alone: {model_a['going_alone_attempts']} attempts, "
          f"{model_a['going_alone_successes']} successes "
          f"({model_a['going_alone_success_rate']:.1%})")
    print()
    
    print("-" * 80)
    print(f"{stats.model_b_name} Team Statistics")
    print("-" * 80)
    model_b = summary[stats.model_b_name]
    print(f"Games Won: {model_b['games_won']} ({model_b['win_rate']:.1%})")
    print(f"Games Lost: {model_b['games_lost']}")
    print(f"Total Score: {model_b['total_score']}")
    print(f"Average Score per Game: {model_b['avg_score']:.2f}")
    print(f"Total Tricks Won: {model_b['total_tricks_won']}")
    print(f"Average Tricks per Game: {model_b['avg_tricks_per_game']:.2f}")
    print(f"Trump Makes: {model_b['trump_makes']}")
    print(f"Going Alone: {model_b['going_alone_attempts']} attempts, "
          f"{model_b['going_alone_successes']} successes "
          f"({model_b['going_alone_success_rate']:.1%})")
    print()
    
    print("=" * 80)


def parse_config_json(config_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse a JSON configuration string.

    Parameters
    ----------
    config_str : Optional[str]
        JSON string to parse.

    Returns
    -------
    Optional[Dict[str, Any]]
        Parsed configuration dictionary, or None if config_str is None/empty.
    """
    if not config_str:
        return None
    try:
        return json.loads(config_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON configuration: {e}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generic Monte Carlo comparison of any two player models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 1000 games between euchergo and eucher_zero
  python scripts/monte_carlo/generic_model_comparison.py \\
      --model-a euchergo --model-b eucher_zero --num-games 1000

  # Run with custom model paths and parameters
  python scripts/monte_carlo/generic_model_comparison.py \\
      --model-a euchergo \\
      --model-b eucher_zero \\
      --num-games 500 \\
      --config-a '{"model_path": "path/to/model.pt", "num_simulations": 100}' \\
      --config-b '{"model_path": "path/to/model.pt", "num_simulations": 50}'

  # Run with checkpointing
  python scripts/monte_carlo/generic_model_comparison.py \\
      --model-a perceiver_muzero \\
      --model-b euchergo \\
      --num-games 1000 \\
      --checkpoint results.checkpoint.json

  # List available plugins
  python scripts/monte_carlo/generic_model_comparison.py --list-plugins
        """,
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default=None,
        help="Plugin name for model A (e.g., 'euchergo', 'eucher_zero', 'perceiver_muzero')",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default=None,
        help="Plugin name for model B (e.g., 'euchergo', 'eucher_zero', 'perceiver_muzero')",
    )
    parser.add_argument(
        "--config-a",
        type=str,
        default=None,
        help="JSON configuration for model A (e.g., '{\"model_path\": \"path\", \"num_simulations\": 100}')",
    )
    parser.add_argument(
        "--config-b",
        type=str,
        default=None,
        help="JSON configuration for model B (e.g., '{\"model_path\": \"path\", \"num_simulations\": 100}')",
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
    parser.add_argument(
        "--list-plugins",
        action="store_true",
        help="List all available player plugins and exit",
    )
    
    args = parser.parse_args()
    
    # List plugins if requested
    if args.list_plugins:
        registry = get_registry()
        plugins = registry.list_plugins()
        print("Available player plugins:")
        print("=" * 80)
        for plugin_name in sorted(plugins):
            metadata = registry.get(plugin_name)
            if metadata:
                print(f"  {plugin_name}")
                print(f"    Description: {metadata.description}")
                print(f"    Requires game: {metadata.requires_game}")
                print(f"    Supports kwargs: {metadata.supports_kwargs}")
                if metadata.model_name:
                    print(f"    Model: {metadata.model_name}")
                print()
        return
    
    # Validate required arguments
    if not args.model_a or not args.model_b:
        parser.error("--model-a and --model-b are required (use --list-plugins to see available plugins)")
    
    # Validate plugins exist
    registry = get_registry()
    if not registry.get(args.model_a):
        parser.error(f"Plugin '{args.model_a}' not found. Use --list-plugins to see available plugins.")
    if not registry.get(args.model_b):
        parser.error(f"Plugin '{args.model_b}' not found. Use --list-plugins to see available plugins.")
    
    print("=" * 80)
    print(f"{args.model_a} vs {args.model_b} Monte Carlo Comparison")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Number of games: {args.num_games}")
    print()
    
    # Parse configurations
    model_a_config = parse_config_json(args.config_a)
    model_b_config = parse_config_json(args.config_b)
    
    if model_a_config:
        print(f"Model A config: {json.dumps(model_a_config, indent=2)}")
    if model_b_config:
        print(f"Model B config: {json.dumps(model_b_config, indent=2)}")
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
        model_a_type=args.model_a,
        model_b_type=args.model_b,
        model_a_config=model_a_config,
        model_b_config=model_b_config,
        seed=args.seed,
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
        output_path = Path("stats") / f"{args.model_a}_vs_{args.model_b}_{timestamp}.json"
        output_path.parent.mkdir(exist_ok=True)
    
    summary = stats.get_summary()
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "num_games": args.num_games,
        "seed": args.seed,
        "model_a_type": args.model_a,
        "model_b_type": args.model_b,
        "model_a_config": model_a_config,
        "model_b_config": model_b_config,
        **summary,
        "detailed_statistics": stats.to_dict(),
    }
    
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

