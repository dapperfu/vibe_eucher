"""Command-line interface for Euchre game."""

import json
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

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
from eucher.plugins import get_registry
from eucher.tui import TextTUI


def get_computer_types() -> list[str]:
    """
    Get list of available computer player types from plugin registry.

    Returns
    -------
    list[str]
        List of available computer player type names.
    """
    # Plugins are auto-discovered via entry points on import
    registry = get_registry()
    plugin_names = registry.list_plugins()
    
    # Filter out "human" if it somehow got registered
    plugin_names = [name for name in plugin_names if name != "human"]
    
    # Sort for consistent ordering
    return sorted(plugin_names)


# Available computer player types (dynamically generated from plugins)
COMPUTER_TYPES = get_computer_types()


@click.group()
def cli() -> None:
    """Euchre card game CLI."""
    pass


def _play_server(
    name: str,
    port: int,
    seed: Optional[str],
    save_dir: Optional[str],
    assistant: Optional[str],
    xray: bool,
    record_decisions: Optional[str],
    playback_decisions: Optional[str],
) -> None:
    """Run game as server."""
    from eucher.network.server import EuchreServer
    import signal
    import sys

    # Create server
    server = EuchreServer(port=port)
    
    def signal_handler(sig: Any, frame: Any) -> None:
        """Handle shutdown signal."""
        click.echo("\nShutting down server...")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        server.start()
        
        # Wait for 3 clients to connect
        click.echo(f"\nWaiting for 3 clients to connect...")
        click.echo(f"Server player: {name} (Player 0)")
        
        while len(server.get_connected_players()) < 4:
            import time
            time.sleep(0.5)
            if not server.running:
                return
        
        click.echo("\nAll players connected! Starting game...")
        
        # Create player config with server as Player 0
        # Other players will be filled in as they connect
        player_config: List[Tuple[str, str]] = [
            (name, "human"),  # Server player
        ]
        
        # Get client names
        for player_id in [1, 2, 3]:
            client = server.get_client_by_player_id(player_id)
            if client and client.name:
                player_config.append((client.name, "human"))
            else:
                player_config.append((f"Player {player_id}", "human"))
        
        # Parse seed
        parsed_seed: Optional[Union[int, str]] = None
        if seed is not None:
            try:
                parsed_seed = int(seed)
            except ValueError:
                parsed_seed = seed
        
        # Create game
        game = Game(player_config, seed=parsed_seed)
        
        # Create TUI for server (handles both local and network players)
        from eucher.tui.server_network_tui import ServerNetworkTUI
        from eucher.decision_playback import DecisionPlayback
        
        decision_playback = None
        if record_decisions or playback_decisions:
            decision_playback = DecisionPlayback(
                record_file=record_decisions,
                playback_file=playback_decisions,
            )
            if hasattr(game, "game_uuid"):
                decision_playback.start_game(game.game_uuid)
            elif parsed_seed is not None:
                decision_playback.start_game(parsed_seed)
        
        tui = ServerNetworkTUI(
            server=server,
            local_player_id=0,
            assistant_helpers=None,  # TODO: Support assistants in network mode
            xray_mode=xray,
            decision_playback=decision_playback,
        )
        game.set_tui(tui)
        
        # Play game loop
        while True:
            click.echo("\n" + "=" * 50)
            click.echo("New Hand")
            click.echo("=" * 50)
            
            continue_game = game.play_hand()
            
            scores = game.get_scores()
            tui.display_scores(scores[0], scores[1])
            
            winner = game.get_winner()
            if winner is not None:
                tui.display_game_over(winner)
                if hasattr(tui, "display_game_log"):
                    tui.display_game_log()
                from eucher.game_stats import display_game_summary
                final_scores = game.get_scores()
                display_game_summary(game.stats, game.players, final_scores, winner)
                break
            
            if not continue_game:
                break
        
        # Save game if requested
        if save_dir is not None and hasattr(game, 'game_uuid'):
            try:
                from eucher.game_file import save_game
                save_path = save_game(game, Path(save_dir))
                click.echo(f"\nGame saved to: {save_path}")
            except Exception as e:
                click.echo(f"Error saving game: {e}", err=True)
        
        click.echo("\nThanks for playing!")
        
    except Exception as e:
        click.echo(f"Server error: {e}", err=True)
        raise
    finally:
        server.stop()


def _play_client(
    server_host: str,
    server_port: int,
    timeout: int,
    name: str,
    assistant: Optional[str],
    xray: bool,
) -> None:
    """Run game as client."""
    from eucher.network.client import EuchreClient
    from eucher.tui.network_tui import NetworkTUI
    from eucher.network.protocol import MessageType
    
    client = EuchreClient(server_host, server_port, timeout)
    
    try:
        # Connect to server
        if not client.connect():
            click.echo("Failed to connect to server", err=True)
            raise click.Abort()
        
        click.echo("Connected to server!")
        
        # Wait for position selection prompt
        while True:
            message = client.receive_message()
            if message is None:
                click.echo("Lost connection to server", err=True)
                raise click.Abort()
            
            msg_type = MessageType(message["type"])
            
            if msg_type == MessageType.CONNECTION_ACK:
                click.echo(message["data"].get("message", "Connected"))
            elif msg_type == MessageType.POSITION_SELECT:
                available = message["data"]["available_positions"]
                position = client.select_position(available)
                
                # Send position selection
                position_msg = {
                    "type": "POSITION_SELECT",
                    "data": {
                        "position": position,
                        "name": name,
                    },
                }
                client.send_message(position_msg)
            elif msg_type == MessageType.POSITION_ASSIGNED:
                client.player_id = message["player_id"]
                click.echo(f"\nAssigned to position {client.player_id}: {message['data']['name']}")
                break
            elif msg_type == MessageType.ERROR:
                click.echo(f"Error: {message['data'].get('message', 'Unknown error')}", err=True)
                raise click.Abort()
        
        # Create network TUI
        tui = NetworkTUI(
            client=client,
            player_id=client.player_id,
            assistant_helpers=None,  # TODO: Support assistants in network mode
            xray_mode=xray,
            decision_playback=None,  # TODO: Support decision playback in network mode
        )
        
        click.echo("\nWaiting for game to start...")
        
        # Wait for game start message
        game_started = False
        while not game_started:
            message = client.receive_message()
            if message is None:
                click.echo("Lost connection to server", err=True)
                break
            
            msg_type = MessageType(message["type"])
            
            if msg_type == MessageType.GAME_START:
                click.echo("Game starting!")
                game_started = True
            elif msg_type == MessageType.ERROR:
                click.echo(f"Error: {message['data'].get('message', 'Unknown error')}", err=True)
                break
        
        if not game_started:
            return
        
        # The NetworkTUI will handle decision requests and game state updates
        # Keep connection alive and handle messages
        click.echo("\nGame session active. Make decisions when prompted.")
        click.echo("Press Ctrl+C to disconnect.")
        
        try:
            while client.connected:
                # NetworkTUI methods will be called when server requests decisions
                # Just keep the connection alive and let NetworkTUI handle everything
                message = client.receive_message(timeout=0.5)
                if message:
                    msg_type = MessageType(message["type"])
                    if msg_type == MessageType.GAME_STATE:
                        tui._update_game_state(message["data"])
                    elif msg_type == MessageType.ERROR:
                        click.echo(f"Error: {message['data'].get('message', 'Unknown error')}", err=True)
        except KeyboardInterrupt:
            click.echo("\nDisconnecting...")
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
        
    except Exception as e:
        click.echo(f"Client error: {e}", err=True)
        raise
    finally:
        client.close()


@cli.command()
@click.option(
    "--opponent-type",
    "--model",
    type=str,
    default=None,
    help="Computer player type(s) for opponents. Can be a single type or comma-separated list (e.g., 'heuristic' or 'euchergo,eucher_zero'). With 2 types, assigns to the 2 opponents (Players 1 and 3); partner (Player 2) uses first type.",
)
@click.option("--name", default="You", help="Human player name")
@click.option("--seed", type=str, default=None, help="Random seed for reproducible games (integer or UUID string)")
@click.option("--save-dir", type=click.Path(file_okay=False, dir_okay=True), default=None, help="Directory to save game file")
@click.option("--assistant", type=str, default=None, help="Assistant bot type(s) to show decision recommendations (comma-separated, e.g., 'heuristic,euchergo'). Use 'list' to see available types)")
@click.option("--xray", is_flag=True, default=False, help="Enable xray mode to display all hidden information (all players' hands, kitty cards, etc.)")
@click.option("--record-decisions", type=click.Path(file_okay=True, dir_okay=False), default=None, help="Path to file where human decisions will be recorded for later playback")
@click.option("--playback-decisions", type=click.Path(file_okay=True, dir_okay=False), default=None, help="Path to file containing recorded decisions to replay")
@click.option("--server", is_flag=True, default=False, help="Run as server (host game for network multiplayer)")
@click.option("--client", type=str, default=None, help="Connect to server at specified IP address (e.g., --client=10.0.0.10)")
@click.option("--port", type=int, default=8765, help="Server port number (default: 8765)")
@click.option("--timeout", type=int, default=300, help="Client connection timeout in seconds (default: 300)")
def play(opponent_type: Optional[str], name: str, seed: Optional[str], save_dir: Optional[str], assistant: Optional[str], xray: bool, record_decisions: Optional[str], playback_decisions: Optional[str], server: bool, client: Optional[str], port: int, timeout: int) -> None:
    """
    Play a game of Euchre with 1 human player and 3 computer opponents.

    The human player will be Player 0, and 3 computer opponents will be
    randomly assigned names using Faker. If --opponent-type is not specified,
    each opponent will have a randomly selected computer player type.

    The seed can be an integer or a UUID string. If a UUID is provided, it will
    be used to seed the random number generator and the game will be saved as
    <uuid>.gz if --save-dir is provided.

    Use --assistant=<bot_type> to display decision recommendations from a bot.
    Use --assistant=<bot_type1>,<bot_type2> to display recommendations from multiple bots.
    Use --assistant=list to see available assistant bot types.

    Network multiplayer:
    Use --server to host a game (you are Player 0).
    Use --client=<ip> to connect to a server (you select your position).
    """
    # Handle network multiplayer modes
    if server:
        _play_server(name, port, seed, save_dir, assistant, xray, record_decisions, playback_decisions)
        return
    elif client is not None:
        _play_client(client, port, timeout, name, assistant, xray)
        return

    # Handle assistant list request
    if assistant == "list":
        registry = get_registry()
        all_metadata = registry.get_all_metadata()
        
        click.echo("Available assistant bot types:")
        click.echo()
        
        # Sort by name for consistent output
        sorted_plugins = sorted(all_metadata.items(), key=lambda x: x[0])
        
        for plugin_name, metadata in sorted_plugins:
            # Skip human profile
            if plugin_name == "human":
                continue
            
            display_name = metadata.display_name
            description = metadata.description
            
            # Format output
            if description:
                click.echo(f"  {plugin_name}")
                click.echo(f"    Display Name: {display_name}")
                click.echo(f"    Description: {description}")
            else:
                click.echo(f"  {plugin_name} ({display_name})")
            click.echo()
        
        return
    
    # Parse seed - can be integer or UUID string
    parsed_seed: Optional[Union[int, str]] = None
    if seed is not None:
        # Try to parse as integer first
        try:
            parsed_seed = int(seed)
        except ValueError:
            # If not an integer, treat as UUID string
            parsed_seed = seed
    
    # Initialize random seed if provided (for Faker compatibility)
    if parsed_seed is not None:
        if isinstance(parsed_seed, int):
            random.seed(parsed_seed)
            # Seed numpy random if available
            try:
                import numpy as np
                np.random.seed(parsed_seed)
            except ImportError:
                pass
        else:
            # For UUID, convert to numeric seed for Faker
            numeric_seed = Game._uuid_to_seed(parsed_seed)
            random.seed(numeric_seed)
            try:
                import numpy as np
                np.random.seed(numeric_seed)
            except ImportError:
                pass
    
    # Initialize Faker for generating opponent names
    fake = Faker()
    # Seed Faker instance if seed is provided
    if parsed_seed is not None:
        if isinstance(parsed_seed, int):
            fake.seed_instance(parsed_seed)
        else:
            numeric_seed = Game._uuid_to_seed(parsed_seed)
            fake.seed_instance(numeric_seed)

    # Generate 3 unique opponent names (first names only)
    opponent_names: List[str] = []
    while len(opponent_names) < 3:
        first_name = fake.first_name()
        if first_name not in opponent_names and first_name != name:
            opponent_names.append(first_name)

    # Determine opponent types
    # Euchre has 4 players: Player 0 (human), Player 1 (opponent), Player 2 (partner/teammate), Player 3 (opponent)
    if opponent_type:
        # Parse comma-separated list of opponent types
        type_list = [t.strip().lower() for t in opponent_type.split(",")]
        
        # Validate each type
        valid_types = [t.lower() for t in COMPUTER_TYPES]
        for opp_type in type_list:
            if opp_type not in valid_types:
                click.echo(f"Error: Invalid opponent type '{opp_type}'.", err=True)
                click.echo(f"Valid types: {', '.join(COMPUTER_TYPES)}", err=True)
                raise click.Abort()
        
        # Assign types to computer players:
        # - Player 1 (opponent): first type
        # - Player 2 (partner/teammate): first type (same as first opponent, or use first type if only 2 provided)
        # - Player 3 (opponent): second type (if provided) or cycles
        opponent_types = []
        if len(type_list) == 1:
            # Single type: all 3 computer players get the same type
            opponent_types = [type_list[0]] * 3
        elif len(type_list) == 2:
            # Two types: assign to the 2 opponents (Players 1 and 3)
            # Partner (Player 2) gets the first type
            opponent_types = [type_list[0], type_list[0], type_list[1]]
        else:
            # Three or more types: assign in order (Player 1, Player 2, Player 3), cycling if needed
            opponent_types = [type_list[i % len(type_list)] for i in range(3)]
    else:
        # Randomly select types for each computer player
        opponent_types = [random.choice(COMPUTER_TYPES).lower() for _ in range(3)]

    # Create player configuration: (name, profile_type)
    player_config: List[Tuple[str, str]] = [
        (name, "human"),  # Player 0: Human
    ]
    # Add 3 computer players: Player 1 (opponent), Player 2 (teammate), Player 3 (opponent)
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
        game = Game(player_config, seed=parsed_seed)
    except ValueError as e:
        click.echo(f"Error creating game: {e}", err=True)
        raise click.Abort()

    # Display game UUID if available
    if hasattr(game, 'game_uuid'):
        click.echo(f"Game UUID: {game.game_uuid}")
        click.echo(f"To replay this game, use: --seed {game.game_uuid}")
        click.echo()

    # Create assistant helpers if specified
    assistant_helpers = []
    if assistant is not None:
        # Parse comma-separated assistant types
        assistant_types = [t.strip() for t in assistant.split(",")]
        from eucher.assistant import AssistantHelper
        
        for assistant_type in assistant_types:
            try:
                helper = AssistantHelper(assistant_type, game=game)
                assistant_helpers.append(helper)
            except ValueError as e:
                click.echo(f"Warning: Invalid assistant type '{assistant_type}': {e}", err=True)
                click.echo("Skipping this assistant...")
            except Exception as e:
                click.echo(f"Warning: Failed to initialize assistant '{assistant_type}': {e}", err=True)
                click.echo("Skipping this assistant...")
        
        if assistant_helpers:
            assistant_names = [h.bot_type for h in assistant_helpers]
            click.echo(f"Assistant mode enabled: {', '.join(assistant_names)}")
            click.echo()
        else:
            click.echo("Warning: No valid assistants initialized. Continuing without assistants...")
            click.echo()

    # Create decision playback if requested
    decision_playback = None
    if record_decisions or playback_decisions:
        from eucher.decision_playback import DecisionPlayback

        decision_playback = DecisionPlayback(
            record_file=record_decisions,
            playback_file=playback_decisions,
        )
        # Start game recording/playback using game UUID or seed
        if hasattr(game, "game_uuid"):
            decision_playback.start_game(game.game_uuid)
        elif parsed_seed is not None:
            decision_playback.start_game(parsed_seed)
        
        if record_decisions:
            click.echo(f"Recording decisions to: {record_decisions}")
        if playback_decisions:
            click.echo(f"Playing back decisions from: {playback_decisions}")
        click.echo()

    # Create and set TUI
    tui = TextTUI(
        assistant_helpers=assistant_helpers if assistant_helpers else None,
        xray_mode=xray,
        decision_playback=decision_playback,
    )
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
            
            # Display game summary statistics
            from eucher.game_stats import display_game_summary
            final_scores = game.get_scores()
            display_game_summary(game.stats, game.players, final_scores, winner)
            break

        if not continue_game:
            break

    # Display game log if game ended without winner
    if hasattr(tui, "display_game_log"):
        tui.display_game_log()
    
    # Display game summary even if game ended without explicit winner
    winner = game.get_winner()
    if winner is None:
        from eucher.game_stats import display_game_summary
        final_scores = game.get_scores()
        display_game_summary(game.stats, game.players, final_scores, winner)

    # Save game if save_dir is provided
    if save_dir is not None and hasattr(game, 'game_uuid'):
        try:
            from eucher.game_file import save_game
            save_path = save_game(game, Path(save_dir))
            click.echo(f"\nGame saved to: {save_path}")
        except Exception as e:
            click.echo(f"Error saving game: {e}", err=True)

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


@cli.group()
def game() -> None:
    """Game file management commands."""
    pass


@game.command()
@click.argument("game_file", type=str)
@click.option("--seed", type=str, default=None, help="Override seed (integer or UUID)")
@click.option("--output-dir", type=click.Path(file_okay=False, dir_okay=True), default="games", help="Directory to save game file")
def save(game_file: str, seed: Optional[str], output_dir: str) -> None:
    """
    Save a game to a file.

    GAME_FILE can be a path to a game file or a UUID string.
    If a UUID is provided, it will look for <uuid>.gz in common directories.
    """
    from eucher.game_file import load_game, save_game
    from pathlib import Path

    try:
        # Load the game
        game = load_game(game_file, seed=seed)
        click.echo(f"Loaded game with UUID: {game.game_uuid}")

        # Save it
        output_path = Path(output_dir)
        saved_path = save_game(game, output_path)
        click.echo(f"Game saved to: {saved_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@game.command()
@click.argument("game_file", type=str)
@click.option("--seed", type=str, default=None, help="Override seed (integer or UUID)")
def load(game_file: str, seed: Optional[str]) -> None:
    """
    Load and display information about a saved game.

    GAME_FILE can be a path to a game file or a UUID string.
    """
    from eucher.game_file import load_game

    try:
        game = load_game(game_file, seed=seed)
        click.echo(f"Game UUID: {game.game_uuid}")
        click.echo(f"Players: {[p.name for p in game.players]}")
        click.echo(f"Dealer ID: {game.dealer_id}")
        click.echo(f"Scores: Team 0: {game.scores[0]}, Team 1: {game.scores[1]}")
        if game.trump_suit:
            click.echo(f"Trump Suit: {game.trump_suit.value}")
        if game.turned_card:
            click.echo(f"Turned Card: {game.turned_card}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("game_id", type=str)
@click.option("--data-dir", type=click.Path(file_okay=False, dir_okay=True), default=None, help="Directory containing training data")
def replay(game_id: str, data_dir: Optional[str]) -> None:
    """
    Replay a game from training data files.

    GAME_ID can be a UUID string (file name without extension) or a full file path.
    The script will look for <game_id>.gz or <game_id>.npz in the training data directory.
    """
    import sys
    from pathlib import Path
    
    # Import here to avoid circular dependencies
    from plugins.ml.ml_config import MLConfig
    from plugins.ml.ml_features import GameStateEncoder
    
    # Get data directory
    if data_dir:
        data_dir_path = Path(data_dir)
    else:
        config = MLConfig()
        data_dir_path = config.training_data_dir
    
    if not data_dir_path.exists():
        click.echo(f"Error: Data directory does not exist: {data_dir_path}", err=True)
        raise click.Abort()
    
    # First try to load as a full game replay (.gz)
    game_file = data_dir_path / f"{game_id}.gz"
    is_game_replay = game_file.exists()
    
    if not is_game_replay:
        # Try as training data file (.npz)
        game_file = data_dir_path / f"{game_id}.npz"
        if not game_file.exists():
            click.echo(f"Error: File not found: {game_id}.gz or {game_id}.npz", err=True)
            click.echo(f"Looking in: {data_dir_path}", err=True)
            raise click.Abort()
    
    if is_game_replay:
        # Load full game replay
        from eucher.game_file import load_game
        try:
            game = load_game(game_file)
            click.echo("=" * 80)
            click.echo(f"Game Replay: {game_id}")
            click.echo("=" * 80)
            click.echo()
            click.echo(f"Game UUID: {game.game_uuid}")
            click.echo(f"Players: {[p.name for p in game.players]}")
            click.echo(f"Dealer ID: {game.dealer_id}")
            click.echo(f"Scores: Team 0: {game.scores[0]}, Team 1: {game.scores[1]}")
            if game.trump_suit:
                click.echo(f"Trump Suit: {game.trump_suit.value}")
            if game.turned_card:
                click.echo(f"Turned Card: {game.turned_card}")
            click.echo()
            click.echo("Note: This is a game state snapshot. Full move-by-move replay")
            click.echo("requires additional game history data.")
            click.echo("=" * 80)
        except Exception as e:
            click.echo(f"Error loading game replay: {e}", err=True)
            raise click.Abort()
    else:
        # Load training data file
        try:
            import numpy as np
            loaded = np.load(game_file)
            X = loaded["X"]
            y = loaded["y"]
        except Exception as e:
            click.echo(f"Error loading file: {e}", err=True)
            raise click.Abort()
        
        # Load metadata
        metadata_file = game_file.with_suffix(".txt")
        dataset_type = "unknown"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = f.read()
                for line in metadata.split("\n"):
                    if line.startswith("Dataset:"):
                        dataset_type = line.split(":", 1)[1].strip()
                        break
        
        click.echo("=" * 80)
        click.echo(f"Training Data File: {game_id}")
        click.echo(f"Dataset Type: {dataset_type}")
        click.echo(f"File: {game_file.name}")
        click.echo("=" * 80)
        click.echo()
        
        click.echo(f"Total decisions: {len(y)}")
        click.echo()
        
        encoder = GameStateEncoder()
        
        # Display decisions
        for i, (features, decision) in enumerate(zip(X, y)):
            click.echo(f"Decision {i + 1}:")
            click.echo(f"  Decision value: {decision}")
            
            if dataset_type == "play_card" or dataset_type == "discard":
                card = encoder.decode_card_index(int(decision))
                if card:
                    click.echo(f"  Card: {card}")
                else:
                    click.echo(f"  Card: (invalid index {decision})")
            elif dataset_type == "call_trump":
                suit_map = {0: "Pass", 1: "Hearts", 2: "Diamonds", 3: "Clubs", 4: "Spades"}
                suit_name = suit_map.get(int(decision), f"Unknown ({decision})")
                click.echo(f"  Trump suit: {suit_name}")
            elif dataset_type == "order_up":
                decision_text = "Order up" if int(decision) == 1 else "Pass"
                click.echo(f"  Decision: {decision_text}")
            
            click.echo()
        
        click.echo("=" * 80)
        click.echo("Note: This shows training data decisions. For full game replays,")
        click.echo("look for corresponding .gz files with the same game_id.")
        click.echo("=" * 80)


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
            "name": "weighted_heuristic",
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
        },
        {
            "name": "perceiver_muzero",
            "display_name": "EucherPerceiverMuZero",
            "personality": "Advanced MuZero-style planning architecture with Perceiver-IO encoder for state representation.",
            "training_method": "Reinforcement learning - MuZero-style self-play with Perceiver-IO encoder, learned dynamics, and MCTS planning",
            "characteristics": [
                "Uses Perceiver-IO for flexible token-based state encoding",
                "MuZero-style planning with learned dynamics and prediction networks",
                "MCTS search with 128-800 simulations per decision",
                "Supports variant flags (screw dealer, 9-10 trade-in, go alone)",
                "Risk modulation via temperature scaling",
                "Learns game model implicitly through self-play",
                "GPU-accelerated training with multi-agent self-play"
            ]
        },
        {
            "name": "eucher_zero",
            "display_name": "EucherZero",
            "personality": "AlphaZero/MuZero-inspired reinforcement learning system for Euchre.",
            "training_method": "Reinforcement learning - self-play with representation learning, learned dynamics, and MCTS",
            "characteristics": [
                "Search-enhanced RL architecture",
                "Perfect memory and hidden card inference",
                "Belief-based dynamics for hidden information",
                "MCTS with belief sampling",
                "Tunable risk profiles",
                "Self-play training on GPU"
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
