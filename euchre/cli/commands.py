"""CLI commands for the euchre game."""

import click
from typing import List, Optional
from ..game import EuchreGame
from ..models import PlayerType
from ..ai.ai_factory import AIFactory


class GameCommands:
    """CLI commands for managing euchre games."""
    
    @staticmethod
    def play_game(player_name: Optional[str], ai_names: tuple, verbose: bool = False, very_verbose: bool = False) -> None:
        """Start a new euchre game.
        
        Parameters
        ----------
        player_name : Optional[str]
            Human player name (None for AI-only game)
        ai_names : tuple
            Names for AI opponents
        verbose : bool
            Enable verbose logging
        very_verbose : bool
            Enable very verbose logging
        """
        try:
            # Create players list
            players = []
            
            # Check if this is a human game or AI-only game
            if player_name is not None:
                # Human game - prompt for name if not specified
                if player_name == "":
                    player_name = click.prompt("Enter your name", default="Player")
                
                click.echo(f"Welcome to Euchre, {player_name}!")
                
                # Add human player
                from ..models import Player, PlayerType
                human_player = Player(player_name, PlayerType.HUMAN)
                players.append(human_player)
                
                # Add AI players
                GameCommands._add_ai_players_to_list(players, ai_names)
            else:
                click.echo("Starting AI-only Euchre game...")
                GameCommands._add_ai_players_to_list(players, ai_names)
            
            # Create game with players
            game = EuchreGame(players, verbose=verbose, very_verbose=very_verbose)
            
            # Start and play the game
            game.start_new_game()
            click.echo("Game started! Dealing cards...")
            
            # Show human player's hand if this is a human game
            if player_name is not None:
                GameCommands._show_human_game_info(game, player_name)
                game.run_interactive_game()
            else:
                # AI-only game - just run it
                game.run_full_game()
                
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
    
    @staticmethod
    def _add_ai_players_to_list(players: List, ai_names: tuple) -> None:
        """Add AI players to the players list.
        
        Parameters
        ----------
        players : List
            List to add AI players to
        ai_names : tuple
            AI player names
        """
        ai_names_list = list(ai_names)
        while len(players) < 4:
            if ai_names_list:
                name = ai_names_list.pop(0)
                ai_player = AIFactory.create_ai_player(name, "balanced", 0.5)
                players.append(ai_player)
            else:
                # Use default AI names if not enough provided
                default_names = ["Alice", "Bob", "Charlie", "David"]
                used_names = [p.name for p in players]
                for default_name in default_names:
                    if default_name not in used_names:
                        ai_player = AIFactory.create_ai_player(default_name, "balanced", 0.5)
                        players.append(ai_player)
                        break
                else:
                    # Fallback if all default names are used
                    ai_player = AIFactory.create_ai_player(f"AI_{len(players)}", "balanced", 0.5)
                    players.append(ai_player)
    
    @staticmethod
    def _show_human_game_info(game: EuchreGame, player_name: str) -> None:
        """Show information for human players.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        player_name : str
            Human player name
        """
        human_hand = game.get_player_hand(player_name)
        click.echo(f"\nYour hand:")
        for i, card in enumerate(human_hand, 1):
            click.echo(f"  {i}. {card}")
        
        # Show trump information
        if game.trump_suit:
            click.echo(f"\nTrump suit: {game.trump_suit.name}")
            # Get trump caller from game state manager
            trump_caller = game.game_state_manager.get_trump_caller()
            if trump_caller:
                click.echo(f"Trump called by: {trump_caller.name}")
                # Determine team based on player index
                caller_index = next(i for i, p in enumerate(game.players) if p.name == trump_caller.name)
                team_name = "Team 1" if caller_index % 2 == 0 else "Team 2"
                click.echo(f"Team: {team_name}")
    
    @staticmethod
    def ai_profiles_game(ai_profiles: tuple, risk_ratios: tuple, verbose: bool = False, very_verbose: bool = False) -> None:
        """Run a game with different AI profiles and risk ratios.
        
        Parameters
        ----------
        ai_profiles : tuple
            AI profiles for each player
        risk_ratios : tuple
            Risk ratios for each player
        verbose : bool
            Enable verbose logging
        very_verbose : bool
            Enable very verbose logging
        """
        try:
            click.echo("Starting AI vs AI euchre game with custom profiles...")
            
            # Create players list
            players = []
            
            # Convert risk ratios to floats
            risk_values = []
            for ratio in risk_ratios:
                try:
                    risk_values.append(float(ratio))
                except ValueError:
                    click.echo(f"Warning: Invalid risk ratio '{ratio}', using 0.5")
                    risk_values.append(0.5)
            
            # Ensure we have 4 values
            while len(risk_values) < 4:
                risk_values.append(0.5)
            while len(ai_profiles) < 4:
                ai_profiles = ai_profiles + ("balanced",)
            
            # Add AI players with profiles
            player_names = ["Alice", "Bob", "Charlie", "David"]
            for i, (name, profile, risk) in enumerate(zip(player_names, ai_profiles, risk_values)):
                ai_player = AIFactory.create_ai_player(name, profile, risk)
                players.append(ai_player)
                click.echo(f"  {name}: {profile} AI (risk: {risk:.1f})")
            
            # Create game with players
            game = EuchreGame(players, verbose=verbose, very_verbose=very_verbose)
            
            # Start and play the game
            game.start_new_game()
            click.echo("Game started! Playing rounds...")
            
            # Play the game
            game.run_full_game()
            
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
    
    @staticmethod
    def ai_vs_ai_game(verbose: bool = False, very_verbose: bool = False) -> None:
        """Run a basic AI vs AI game.
        
        Parameters
        ----------
        verbose : bool
            Enable verbose logging
        very_verbose : bool
            Enable very verbose logging
        """
        try:
            click.echo("Starting AI vs AI euchre game...")
            
            # Create AI players
            ai_players = AIFactory.create_mixed_ai_players()
            
            # Create game with players
            game = EuchreGame(ai_players, verbose=verbose, very_verbose=very_verbose)
            
            click.echo(f"Players: {', '.join(p.name for p in game.players)}")
            
            # Start and play the game
            game.start_new_game()
            game.run_full_game()
            
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
    
    @staticmethod
    def tournament_game(num_games: int, verbose: bool = False, very_verbose: bool = False) -> None:
        """Run a tournament between AI players.
        
        Parameters
        ----------
        num_games : int
            Number of games to play
        verbose : bool
            Enable verbose logging
        very_verbose : bool
            Enable very verbose logging
        """
        try:
            click.echo(f"Starting tournament with {num_games} games...")
            
            # Create AI players with different styles
            ai_players = AIFactory.create_mixed_ai_players()
            
            # Create tournament game
            game = EuchreGame(ai_players, verbose=verbose, very_verbose=very_verbose)
            
            click.echo(f"Tournament players: {', '.join(p.name for p in game.players)}")
            
            # Run tournament
            game.run_tournament(num_games)
            
        except ValueError as e:
            click.echo(f"Error: {e}", err=True) 