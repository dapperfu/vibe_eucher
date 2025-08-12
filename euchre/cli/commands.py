"""CLI commands for the euchre game."""

import click
from typing import List, Optional
from ..game import EuchreGame
from ..models import PlayerType
from ..ai.ai_factory import AIFactory


class GameCommands:
    """CLI commands for managing euchre games."""
    
    @staticmethod
    def play_game(player_name: Optional[str], ai_names: tuple) -> None:
        """Start a new euchre game.
        
        Parameters
        ----------
        player_name : Optional[str]
            Human player name (None for AI-only game)
        ai_names : tuple
            Names for AI opponents
        """
        try:
            # Create game
            game = EuchreGame()
            
            # Check if this is a human game or AI-only game
            if player_name is not None:
                GameCommands._setup_human_game(game, player_name, ai_names)
            else:
                GameCommands._setup_ai_only_game(game, ai_names)
            
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
    def _setup_human_game(game: EuchreGame, player_name: str, ai_names: tuple) -> None:
        """Set up a game with a human player.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        player_name : str
            Human player name
        ai_names : tuple
            AI opponent names
        """
        # Human game - prompt for name if not specified
        if player_name == "":
            player_name = click.prompt("Enter your name", default="Player")
        
        click.echo(f"Welcome to Euchre, {player_name}!")
        
        # Add human player
        game.add_player(player_name, PlayerType.HUMAN)
        
        # Add AI players
        GameCommands._add_ai_players(game, ai_names)
    
    @staticmethod
    def _setup_ai_only_game(game: EuchreGame, ai_names: tuple) -> None:
        """Set up an AI-only game.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        ai_names : tuple
            AI player names
        """
        click.echo("Starting AI-only Euchre game...")
        
        # Add AI players
        GameCommands._add_ai_players(game, ai_names)
    
    @staticmethod
    def _add_ai_players(game: EuchreGame, ai_names: tuple) -> None:
        """Add AI players to the game.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        ai_names : tuple
            AI player names
        """
        ai_names_list = list(ai_names)
        while len(game.players) < 4:
            if ai_names_list:
                game.add_ai_player(ai_names_list.pop(0), "balanced", 0.5)
            else:
                # Use default AI names if not enough provided
                default_names = ["Alice", "Bob", "Charlie", "David"]
                used_names = [p.name for p in game.players]
                for default_name in default_names:
                    if default_name not in used_names:
                        game.add_ai_player(default_name, "balanced", 0.5)
                        break
                else:
                    # Fallback if all default names are used
                    game.add_ai_player(f"AI_{len(game.players)}", "balanced", 0.5)
    
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
        if game.game_state and game.game_state.trump_suit:
            click.echo(f"\nTrump suit: {game.game_state.trump_suit.value.title()}")
            if game.trump_caller:
                click.echo(f"Trump called by: {game.trump_caller.name}")
                if game.trump_caller_team is not None:
                    team_name = "Team 1" if game.trump_caller_team == 0 else "Team 2"
                    click.echo(f"Team: {team_name}")
    
    @staticmethod
    def ai_profiles_game(ai_profiles: tuple, risk_ratios: tuple, enable_logging: bool) -> None:
        """Run a game with different AI profiles and risk ratios.
        
        Parameters
        ----------
        ai_profiles : tuple
            AI profiles for each player
        risk_ratios : tuple
            Risk ratios for each player
        enable_logging : bool
            Whether to enable game logging
        """
        try:
            click.echo("Starting AI vs AI euchre game with custom profiles...")
            
            # Create game with logging
            game = EuchreGame(enable_logging=enable_logging)
            
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
                game.add_ai_player(name, profile, risk)
                click.echo(f"  {name}: {profile} AI (risk: {risk:.1f})")
            
            # Start and play the game
            game.start_new_game()
            click.echo(f"Game started! Trump: {game.game_state.trump_suit.name.title()}")
            click.echo("Playing rounds...")
            
            # Play the game
            game.run_full_game()
            
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
    
    @staticmethod
    def ai_vs_ai_game() -> None:
        """Run a basic AI vs AI game."""
        try:
            click.echo("Starting AI vs AI euchre game...")
            
            # Create game
            game = EuchreGame()
            
            # Add AI players
            ai_players = AIFactory.create_mixed_ai_players()
            for player in ai_players:
                game.players.append(player)
            
            click.echo(f"Players: {', '.join(p.name for p in game.players)}")
            
            # Start and play the game
            game.start_new_game()
            game.run_full_game()
            
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
    
    @staticmethod
    def tournament_game(num_games: int) -> None:
        """Run a tournament between AI players.
        
        Parameters
        ----------
        num_games : int
            Number of games to play
        """
        try:
            click.echo(f"Starting tournament with {num_games} games...")
            
            # Create tournament game
            game = EuchreGame()
            
            # Add AI players with different styles
            ai_players = AIFactory.create_mixed_ai_players()
            for player in ai_players:
                game.players.append(player)
            
            click.echo(f"Tournament players: {', '.join(p.name for p in game.players)}")
            
            # Run tournament
            game.run_tournament(num_games)
            
        except ValueError as e:
            click.echo(f"Error: {e}", err=True) 