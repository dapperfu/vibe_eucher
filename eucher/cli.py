"""Command-line interface for Euchre game."""

import json
import random
from datetime import datetime
from typing import List, Optional, Tuple

import click
from faker import Faker

from eucher.database import init_database
from eucher.db_queries import (
    filter_games_by_profile_type,
    get_game,
    get_game_statistics,
    get_games_by_date_range,
    get_hand,
    get_hands_by_game,
    get_most_common_trump_suits,
    get_player,
    get_player_games,
    get_player_statistics,
    get_player_win_rate,
    get_trick,
    get_tricks_by_hand,
    get_trick_win_statistics,
    list_games,
    list_players,
    search_games,
)
from eucher.db_serializers import (
    game_to_dict,
    hand_to_dict,
    player_to_dict,
    trick_to_dict,
)
from eucher.game import Game
from eucher.tui import TextTUI


# Available computer player types
COMPUTER_TYPES = ["simple", "heuristic", "ai", "random", "ml", "ml_sklearn", "ml_pytorch"]


@click.group()
def cli() -> None:
    """Euchre card game CLI."""
    pass


@cli.command()
@click.option(
    "--opponent-type",
    "--model",
    type=click.Choice(COMPUTER_TYPES, case_sensitive=False),
    default=None,
    help="Computer player type for all opponents (if not specified, types are randomly chosen)",
)
@click.option("--name", default="You", help="Human player name")
@click.option("--seed", type=int, default=None, help="Random seed for reproducible games")
def play(opponent_type: Optional[str], name: str, seed: Optional[int]) -> None:
    """
    Play a game of Euchre with 1 human player and 3 computer opponents.

    The human player will be Player 0, and 3 computer opponents will be
    randomly assigned names using Faker. If --opponent-type is not specified,
    each opponent will have a randomly selected computer player type.
    """
    # Initialize random seed if provided
    if seed is not None:
        random.seed(seed)
        # Seed numpy random if available
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
    
    # Initialize Faker for generating opponent names
    fake = Faker()
    # Seed Faker instance if seed is provided
    if seed is not None:
        fake.seed_instance(seed)

    # Generate 3 unique opponent names (first names only)
    opponent_names: List[str] = []
    while len(opponent_names) < 3:
        first_name = fake.first_name()
        if first_name not in opponent_names and first_name != name:
            opponent_names.append(first_name)

    # Determine opponent types
    if opponent_type:
        # Use specified type for all opponents (normalize case)
        opponent_type_normalized = opponent_type.lower()
        opponent_types = [opponent_type_normalized] * 3
    else:
        # Randomly select types for each opponent
        opponent_types = [random.choice(COMPUTER_TYPES).lower() for _ in range(3)]

    # Create player configuration: (name, profile_type)
    player_config: List[Tuple[str, str]] = [
        (name, "human"),  # Player 0: Human
    ]
    # Add 3 computer opponents
    for opp_name, opp_type in zip(opponent_names, opponent_types):
        player_config.append((opp_name, opp_type))

    # Display game setup
    click.echo("=" * 50)
    click.echo("Welcome to Euchre!")
    click.echo("=" * 50)
    click.echo(f"\nPlayers:")
    click.echo(f"  Player 0: {name} (Human)")
    for i, (opp_name, opp_type) in enumerate(zip(opponent_names, opponent_types), 1):
        click.echo(f"  Player {i}: {opp_name} ({opp_type})")
    click.echo()

    # Create game
    try:
        game = Game(player_config)
    except ValueError as e:
        click.echo(f"Error creating game: {e}", err=True)
        raise click.Abort()

    # Create and set TUI
    tui = TextTUI()
    game.set_tui(tui)

    # Play game
    while True:
        click.echo("\n" + "=" * 50)
        click.echo("New Hand")
        click.echo("=" * 50)

        continue_game = game.play_hand()

        # Display scores
        scores = game.get_scores()
        tui.display_scores(scores[0], scores[1])

        # Check for game over
        winner = game.get_winner()
        if winner is not None:
            tui.display_game_over(winner)
            # Display complete game log
            if hasattr(tui, "display_game_log"):
                tui.display_game_log()
            break

        if not continue_game:
            break

    # Display game log if game ended without winner
    if hasattr(tui, "display_game_log"):
        tui.display_game_log()

    click.echo("\nThanks for playing!")


# ============================================================================
# Database Query Commands
# ============================================================================


@cli.group()
@click.option(
    "--db",
    default="euchre.db",
    help="Path to database file",
    type=click.Path(exists=False),
)
@click.pass_context
def query(ctx: click.Context, db: str) -> None:
    """Query the game database."""
    # Initialize database connection
    init_database(db)
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db


@query.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def players(as_json: bool) -> None:
    """List all players in the database."""
    players_list = list_players()
    if as_json:
        click.echo(json.dumps([player_to_dict(p) for p in players_list], indent=2))
    else:
        click.echo(f"Total players: {len(players_list)}")
        for player in players_list:
            click.echo(f"  - {player.name} (created: {player.created_at})")


@query.command()
@click.argument("name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def player(name: str, as_json: bool) -> None:
    """Get information about a specific player."""
    player_obj = get_player(name)
    if player_obj is None:
        click.echo(f"Player '{name}' not found", err=True)
        raise click.Abort()

    if as_json:
        click.echo(json.dumps(player_to_dict(player_obj), indent=2))
    else:
        click.echo(f"Player: {player_obj.name}")
        click.echo(f"Created: {player_obj.created_at}")


@query.command()
@click.argument("name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def player_stats(name: str, as_json: bool) -> None:
    """Get statistics for a player."""
    stats = get_player_statistics(name)
    if as_json:
        click.echo(json.dumps(stats, indent=2))
    else:
        click.echo(f"Statistics for {name}:")
        click.echo(f"  Total games: {stats['total_games']}")
        click.echo(f"  Games won: {stats['games_won']}")
        click.echo(f"  Games lost: {stats['games_lost']}")
        click.echo(f"  Win rate: {stats['win_rate']:.2f}%")
        click.echo(f"  Total hands: {stats['total_hands']}")
        click.echo(f"  Profile types: {', '.join(stats['profile_types'])}")


@query.command()
@click.option("--limit", type=int, help="Limit number of results")
@click.option("--offset", type=int, default=0, help="Offset for pagination")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def games(limit: Optional[int], offset: int, as_json: bool) -> None:
    """List games in the database."""
    games_list = list_games(limit=limit, offset=offset)
    if as_json:
        click.echo(json.dumps([game_to_dict(g) for g in games_list], indent=2))
    else:
        click.echo(f"Total games: {len(games_list)}")
        for game in games_list:
            winner = f"Team {game.winner_team}" if game.winner_team is not None else "None"
            click.echo(
                f"  Game {game.id}: {game.started_at} | "
                f"Score: {game.final_team0_score}-{game.final_team1_score} | "
                f"Winner: {winner}"
            )


@query.command()
@click.argument("game_id", type=int)
@click.option("--include-hands", is_flag=True, help="Include hand data")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def game(game_id: int, include_hands: bool, as_json: bool) -> None:
    """Get information about a specific game."""
    game_obj = get_game(game_id)
    if game_obj is None:
        click.echo(f"Game {game_id} not found", err=True)
        raise click.Abort()

    if as_json:
        click.echo(json.dumps(game_to_dict(game_obj, include_hands=include_hands), indent=2))
    else:
        click.echo(f"Game {game_obj.id}")
        click.echo(f"Started: {game_obj.started_at}")
        click.echo(f"Ended: {game_obj.ended_at}")
        click.echo(
            f"Final Score: Team 0: {game_obj.final_team0_score}, Team 1: {game_obj.final_team1_score}"
        )
        click.echo(
            f"Winner: Team {game_obj.winner_team}"
            if game_obj.winner_team is not None
            else "Winner: None"
        )
        click.echo(f"Players: {len(game_obj.game_players)}")
        click.echo(f"Hands: {len(game_obj.hands)}")


@query.command()
@click.argument("name")
@click.option("--limit", type=int, help="Limit number of results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def player_games(name: str, limit: Optional[int], as_json: bool) -> None:
    """Get all games for a player."""
    games_list = get_player_games(name, limit=limit)
    if as_json:
        click.echo(json.dumps([game_to_dict(g) for g in games_list], indent=2))
    else:
        click.echo(f"Games for {name}: {len(games_list)}")
        for game in games_list:
            click.echo(f"  Game {game.id}: {game.started_at}")


@query.command()
@click.argument("hand_id", type=int)
@click.option("--include-tricks", is_flag=True, help="Include trick data")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def hand(hand_id: int, include_tricks: bool, as_json: bool) -> None:
    """Get information about a specific hand."""
    hand_obj = get_hand(hand_id)
    if hand_obj is None:
        click.echo(f"Hand {hand_id} not found", err=True)
        raise click.Abort()

    if as_json:
        click.echo(json.dumps(hand_to_dict(hand_obj, include_tricks=include_tricks), indent=2))
    else:
        click.echo(f"Hand {hand_obj.id} (Game {hand_obj.game.id}, Hand #{hand_obj.hand_number})")
        click.echo(f"Trump: {hand_obj.trump_suit}")
        click.echo(
            f"Tricks: Team 0: {hand_obj.tricks_won_team0}, Team 1: {hand_obj.tricks_won_team1}"
        )


@query.command()
@click.argument("game_id", type=int)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def game_hands(game_id: int, as_json: bool) -> None:
    """Get all hands for a game."""
    hands_list = get_hands_by_game(game_id)
    if as_json:
        click.echo(json.dumps([hand_to_dict(h) for h in hands_list], indent=2))
    else:
        click.echo(f"Hands for game {game_id}: {len(hands_list)}")
        for hand in hands_list:
            click.echo(f"  Hand {hand.id}: #{hand.hand_number} | Trump: {hand.trump_suit}")


@query.command()
@click.argument("trick_id", type=int)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def trick(trick_id: int, as_json: bool) -> None:
    """Get information about a specific trick."""
    trick_obj = get_trick(trick_id)
    if trick_obj is None:
        click.echo(f"Trick {trick_id} not found", err=True)
        raise click.Abort()

    if as_json:
        click.echo(json.dumps(trick_to_dict(trick_obj), indent=2))
    else:
        click.echo(
            f"Trick {trick_obj.id} (Hand {trick_obj.hand.id}, Trick #{trick_obj.trick_number})"
        )
        click.echo(f"Leader: Player {trick_obj.leader_player_id}")
        click.echo(f"Winner: Player {trick_obj.winner_player_id} (Team {trick_obj.winner_team})")


@query.command()
@click.argument("hand_id", type=int)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def hand_tricks(hand_id: int, as_json: bool) -> None:
    """Get all tricks for a hand."""
    tricks_list = get_tricks_by_hand(hand_id)
    if as_json:
        click.echo(json.dumps([trick_to_dict(t) for t in tricks_list], indent=2))
    else:
        click.echo(f"Tricks for hand {hand_id}: {len(tricks_list)}")
        for trick in tricks_list:
            click.echo(
                f"  Trick {trick.id}: #{trick.trick_number} | Winner: Player {trick.winner_player_id}"
            )


@query.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def stats(as_json: bool) -> None:
    """Get overall game statistics."""
    stats_dict = get_game_statistics()
    if as_json:
        click.echo(json.dumps(stats_dict, indent=2, default=str))
    else:
        click.echo("Game Statistics:")
        click.echo(f"  Total games: {stats_dict['total_games']}")
        click.echo(f"  Total players: {stats_dict['total_players']}")
        if stats_dict["average_game_duration"]:
            duration_min = stats_dict["average_game_duration"] / 60
            click.echo(f"  Average game duration: {duration_min:.2f} minutes")
        click.echo("  Profile types:")
        for profile_type, count in stats_dict["profile_types"].items():
            click.echo(f"    {profile_type}: {count}")


@query.command()
@click.argument("name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def win_rate(name: str, as_json: bool) -> None:
    """Get win rate for a player."""
    rate = get_player_win_rate(name)
    if as_json:
        click.echo(json.dumps({"player": name, "win_rate": rate}, indent=2))
    else:
        click.echo(f"Win rate for {name}: {rate:.2f}%")


@query.command()
@click.option("--limit", type=int, default=4, help="Number of results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def trump_suits(limit: int, as_json: bool) -> None:
    """Get most common trump suits."""
    suits = get_most_common_trump_suits(limit=limit)
    if as_json:
        click.echo(json.dumps([{"suit": suit, "count": count} for suit, count in suits], indent=2))
    else:
        click.echo("Most common trump suits:")
        for suit, count in suits:
            click.echo(f"  {suit}: {count}")


@query.command()
@click.argument("name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def trick_stats(name: str, as_json: bool) -> None:
    """Get trick win statistics for a player."""
    stats_dict = get_trick_win_statistics(name)
    if as_json:
        click.echo(json.dumps(stats_dict, indent=2))
    else:
        click.echo(f"Trick statistics for {name}:")
        click.echo(f"  Tricks won: {stats_dict['tricks_won']}")
        click.echo(f"  Tricks lost: {stats_dict['tricks_lost']}")
        click.echo(f"  Win rate: {stats_dict['win_rate']:.2f}%")


@query.command()
@click.argument("query_str")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def search(query_str: str, as_json: bool) -> None:
    """Search games by player name."""
    games_list = search_games(query_str)
    if as_json:
        click.echo(json.dumps([game_to_dict(g) for g in games_list], indent=2))
    else:
        click.echo(f"Games matching '{query_str}': {len(games_list)}")
        for game in games_list:
            click.echo(f"  Game {game.id}: {game.started_at}")


@query.command()
@click.argument("profile_type")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def filter_profile(profile_type: str, as_json: bool) -> None:
    """Filter games by profile type."""
    games_list = filter_games_by_profile_type(profile_type)
    if as_json:
        click.echo(json.dumps([game_to_dict(g) for g in games_list], indent=2))
    else:
        click.echo(f"Games with profile type '{profile_type}': {len(games_list)}")
        for game in games_list:
            click.echo(f"  Game {game.id}: {game.started_at}")


# ============================================================================
# AI Commands
# ============================================================================


@cli.group()
def ai() -> None:
    """AI and bot-related commands."""
    pass


@ai.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list(as_json: bool) -> None:
    """
    List all available bots and their personalities/training methods.
    
    Shows information about each bot type including how they were trained
    and their playing characteristics.
    """
    # Only list bots that are in COMPUTER_TYPES (the officially supported ones)
    bots_info = [
        {
            "name": "random",
            "display_name": "Random Player",
            "personality": "Completely random decision-making. No strategy or training.",
            "training_method": "None - makes random legal moves",
            "characteristics": [
                "No strategic thinking",
                "Random card selection",
                "Random trump decisions",
                "Good for testing game mechanics"
            ]
        },
        {
            "name": "heuristic",
            "display_name": "Heuristic Player",
            "personality": "Rule-based player using basic Euchre heuristics and strategies.",
            "training_method": "None - uses hardcoded rule-based logic",
            "characteristics": [
                "Follows basic Euchre rules and conventions",
                "Makes decisions based on card strength",
                "Considers trump suit when available",
                "Predictable but competent play"
            ]
        },
        {
            "name": "ai",
            "display_name": "AI Decision Maker",
            "personality": "Advanced rule-based AI with strategic decision-making capabilities.",
            "training_method": "None - uses sophisticated rule-based algorithms",
            "characteristics": [
                "Strategic trump selection",
                "Card counting and probability estimation",
                "Partner coordination awareness",
                "Strong baseline AI opponent"
            ]
        },
        {
            "name": "ml_sklearn",
            "display_name": "ML Player (sklearn)",
            "personality": "Machine learning player trained on collected game data using supervised learning.",
            "training_method": "Supervised learning - trained on collected game data using sklearn models (Random Forest, Gradient Boosting, or Neural Network)",
            "characteristics": [
                "Learns from historical game data",
                "Makes decisions based on patterns in training data",
                "Supports multiple model types (random_forest, gradient_boosting, neural_network)",
                "Requires training data collection before use"
            ]
        },
        {
            "name": "ml_pytorch",
            "display_name": "ML Player (PyTorch)",
            "personality": "Deep learning player using PyTorch neural networks for decision-making.",
            "training_method": "Supervised learning - trained on game data using PyTorch neural networks with 260 input features",
            "characteristics": [
                "Uses deep neural networks for decision-making",
                "Learns complex patterns from game state",
                "Supports risk/temperature tuning for play style",
                "Requires trained model files"
            ]
        }
    ]
    
    if as_json:
        click.echo(json.dumps(bots_info, indent=2))
    else:
        click.echo("Available Bots and Their Personalities")
        click.echo("=" * 70)
        click.echo()
        
        for bot in bots_info:
            click.echo(f"Bot: {bot['display_name']} ({bot['name']})")
            click.echo(f"  Personality: {bot['personality']}")
            click.echo(f"  Training Method: {bot['training_method']}")
            click.echo("  Characteristics:")
            for char in bot['characteristics']:
                click.echo(f"    - {char}")
            click.echo()


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
