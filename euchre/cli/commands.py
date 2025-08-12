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
    def ai_vs_ai_game(verbose: bool = False, very_verbose: bool = False, dealer_method: str = "black_jack") -> None:
        """Run AI vs AI euchre game.
        
        Parameters
        ----------
        verbose : bool
            Enable verbose logging
        very_verbose : bool
            Enable very verbose logging
        dealer_method : str
            Dealer selection method
        """
        try:
            click.echo(f"Starting AI vs AI euchre game with {dealer_method} dealer selection...")
            
            # Create AI players with different profiles
            players = []
            ai_types = ["aggressive", "conservative", "balanced", "opportunistic"]
            risk_ratios = [0.7, 0.3, 0.5, 0.6]
            
            for i, (name, ai_type, risk) in enumerate(zip(["Alice", "Bob", "Charlie", "David"], ai_types, risk_ratios)):
                ai_player = AIFactory.create_ai_player(name, ai_type, risk)
                players.append(ai_player)
            
            # Create game
            game = EuchreGame(players, verbose=verbose, very_verbose=very_verbose)
            game.dealer_selection_method = dealer_method
            
            # Start and run the game
            game.start_new_game()
            game.run_full_game()
            
        except Exception as e:
            click.echo(f"Error during AI vs AI game: {e}", err=True)
    
    @staticmethod
    def human_vs_ai_game(player_name: str, your_position: int, 
                         partner_ai_type: str, opponent1_ai_type: str, opponent2_ai_type: str,
                         partner_risk: float, opponent1_risk: float, opponent2_risk: float,
                         verbose: bool = False, very_verbose: bool = False) -> None:
        """Play euchre as a human against AI opponents with a specified AI partner.
        
        Parameters
        ----------
        player_name : str
            Your player name
        your_position : int
            Your position (0=Alice, 1=Bob, 2=Charlie, 3=David)
        partner_ai_type : str
            AI type for your partner
        opponent1_ai_type : str
            AI type for first opponent
        opponent2_ai_type : str
            AI type for second opponent
        partner_risk : float
            Risk ratio for your partner (0.0-1.0)
        opponent1_risk : float
            Risk ratio for first opponent (0.0-1.0)
        opponent2_risk : float
            Risk ratio for second opponent (0.0-1.0)
        verbose : bool
            Enable verbose logging
        very_verbose : bool
            Enable very verbose logging
        """
        try:
            click.echo(f"🎮 Welcome to Human vs AI Euchre, {player_name}!")
            click.echo(f"📍 Your position: {['Alice', 'Bob', 'Charlie', 'David'][your_position]}")
            click.echo(f"🤝 Your partner: {partner_ai_type} AI (risk: {partner_risk})")
            click.echo(f"👥 Opponents: {opponent1_ai_type} AI (risk: {opponent1_risk}) and {opponent2_ai_type} AI (risk: {opponent2_risk})")
            click.echo("=" * 60)
            
            # Create players list
            players = []
            
            # Define player positions and types
            positions = ["Alice", "Bob", "Charlie", "David"]
            ai_types = [None, None, None, None]  # Will be filled based on your position
            risk_ratios = [0.5, 0.5, 0.5, 0.5]  # Will be filled based on your position
            
            # Set AI types and risk ratios based on your position
            if your_position == 0:  # Alice - you are human
                ai_types = [None, opponent1_ai_type, partner_ai_type, opponent2_ai_type]
                risk_ratios = [0.5, opponent1_risk, partner_risk, opponent2_risk]
            elif your_position == 1:  # Bob - you are human
                ai_types = [opponent1_ai_type, None, opponent2_ai_type, partner_ai_type]
                risk_ratios = [opponent1_risk, 0.5, opponent2_risk, partner_risk]
            elif your_position == 2:  # Charlie - you are human
                ai_types = [partner_ai_type, opponent1_ai_type, None, opponent2_ai_type]
                risk_ratios = [partner_risk, opponent1_risk, 0.5, opponent2_risk]
            elif your_position == 3:  # David - you are human
                ai_types = [opponent1_ai_type, partner_ai_type, opponent2_ai_type, None]
                risk_ratios = [opponent1_risk, partner_risk, opponent2_risk, 0.5]
            
            # Create players
            for i, (name, ai_type, risk) in enumerate(zip(positions, ai_types, risk_ratios)):
                if i == your_position:
                    # This is the human player - use position name internally for consistency
                    from ..models import Player, PlayerType
                    human_player = Player(name, PlayerType.HUMAN)  # Use position name (Alice, Bob, etc.)
                    players.append(human_player)
                    click.echo(f"👤 {name}: {player_name} (Human)")
                else:
                    # This is an AI player
                    ai_player = AIFactory.create_ai_player(name, ai_type, risk)
                    players.append(ai_player)
                    click.echo(f"🤖 {name}: {ai_type} AI (risk: {risk})")
            
            # Create game
            game = EuchreGame(players, verbose=verbose, very_verbose=very_verbose)
            
            # Manually initialize tricks_won with actual player names
            game.tricks_won = {player.name: 0 for player in players}
            
            # Start the game
            game.start_new_game()
            click.echo("\n🎯 Game started! Dealing cards...")
            
            # Show game setup - use the position name for consistency
            GameCommands._show_human_vs_ai_game_info(game, positions[your_position], your_position)
            
            # Run the game using the proper full game method
            # This will handle the complete game flow including trump selection and tricks
            game.run_full_game()
            
        except Exception as e:
            click.echo(f"❌ Error during Human vs AI game: {e}", err=True)
    
    @staticmethod
    def _show_human_vs_ai_game_info(game: EuchreGame, player_name: str, your_position: int) -> None:
        """Show information for human vs AI game.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        player_name : str
            Human player name
        your_position : int
            Human player position
        """
        positions = ["Alice", "Bob", "Charlie", "David"]
        click.echo(f"\n📋 Game Setup:")
        click.echo(f"  Dealer: {game.game_state_manager.get_dealer().name}")
        click.echo(f"  Top Card: {game.top_card.unicode_str() if game.top_card else 'Not dealt yet'}")
        
        # Show your hand - use the actual player name, not the position name
        your_hand = game.get_player_hand(player_name)
        if your_hand:
            click.echo(f"\n🃏 Your hand ({positions[your_position]}):")
            for i, card in enumerate(your_hand):
                click.echo(f"  {i+1}. {card.unicode_str()}")
        
        click.echo(f"\n🎮 You are playing as {positions[your_position]}")
        click.echo(f"🤝 Your partner is {positions[(your_position + 2) % 4]}")
        click.echo(f"👥 Your opponents are {positions[(your_position + 1) % 4]} and {positions[(your_position + 3) % 4]}")
        click.echo("\n" + "=" * 60)
    
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