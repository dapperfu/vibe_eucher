"""CLI commands for the euchre game."""

import click
from typing import List, Optional
from ..game import EuchreGame
from ..models import PlayerType, Suit
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
            
            # Start the game (human or AI)
            if player_name is not None:
                # Human game - use CLI-based interactive flow
                GameCommands._play_interactive_round(game, player_name, 0)
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
        click.echo(f"\n🃏 Your hand ({player_name}):")
        for i, card in enumerate(human_hand, 1):
            click.echo(f"  {i}. {card.unicode_str()}")
        
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
            
            # Run interactive human vs AI game
            GameCommands._run_interactive_human_vs_ai_game(game, positions[your_position], your_position)
            
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
        
        # Show top card status
        if game.top_card and not getattr(game, '_top_card_picked_up', False):
            click.echo(f"  Top Card: {game.top_card.unicode_str()}")
        else:
            click.echo(f"  Top Card: Picked up by player")
        
        # Debug: Show kitty (remaining cards in deck)
        if hasattr(game, 'deck') and hasattr(game.deck, 'size'):
            click.echo(f"  Kitty: {game.deck.size} cards remaining")
            if game.deck.size <= 10:  # Show actual cards if few remain
                remaining_cards = game.deck.get_remaining_cards()
                click.echo(f"  Remaining cards: {[card.unicode_str() for card in remaining_cards]}")
        else:
            click.echo(f"  Kitty: Deck info not available")
        
        click.echo(f"\n🎮 You are playing as {positions[your_position]}")
        click.echo(f"🤝 Your partner is {positions[(your_position + 2) % 4]}")
        click.echo(f"👥 Your opponents are {positions[(your_position + 1) % 4]} and {positions[(your_position + 3) % 4]}")
        click.echo("\n" + "=" * 60)
    
    @staticmethod
    def _run_interactive_human_vs_ai_game(game: EuchreGame, player_position: str, your_position: int) -> None:
        """Run an interactive human vs AI game with proper human decision-making.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        player_position : str
            Human player position name (Alice, Bob, Charlie, David)
        your_position : int
            Human player position index (0-3)
        """
        click.echo("\n🎯 Starting Interactive Human vs AI Game!")
        click.echo("=" * 60)
        
        # Game is already started in human_vs_ai_game, don't start it again
        # game.start_new_game()  # REMOVED: This was causing duplicate card dealing
        
        # Play the first round
        GameCommands._play_interactive_round(game, player_position, your_position)
        
        # Continue rounds until game is over
        while not game.scoring_manager.is_game_over(game.players):
            game.round_number += 1
            click.echo(f"\n🔄 Starting Round {game.round_number}")
            click.echo("=" * 60)
            
            # Note: Don't call game._start_new_round() here - the interactive flow manages rounds itself
            # The main game flow's _start_new_round() is designed for AI vs AI games, not human vs AI
            
            # Play the round
            GameCommands._play_interactive_round(game, player_position, your_position)
        
        # Show final results
        GameCommands._show_final_game_results(game)
    
    @staticmethod
    def _play_interactive_round(game: EuchreGame, player_position: str, your_position: int) -> None:
        """Play a single round with human interaction.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        player_position : str
            Human player position name
        your_position : int
            Human player position index
        """
        click.echo(f"\n🎴 Round {game.round_number}")
        click.echo(f"Dealer: {game.game_state_manager.get_dealer().name}")
        
        # Handle new rounds (except the first round which was already dealt)
        if game.round_number > 1:
            # Rotate dealer for new rounds
            current_dealer = game.game_state_manager.get_dealer()
            dealer_index = next(i for i, p in enumerate(game.players) if p.name == current_dealer.name)
            next_dealer_index = (dealer_index + 1) % 4
            next_dealer = game.players[next_dealer_index]
            game.game_state_manager.set_dealer(next_dealer)
            
            # Reset round state
            game.current_trick = None
            game.tricks_won = {player.name: 0 for player in game.players}
            game.trump_suit = None
            game._top_card_picked_up = False
            
            # Deal new cards for this round
            game.deck.reset_and_shuffle()
            game._deal_cards()
            
            click.echo(f"🔄 New dealer: {next_dealer.name}")
            click.echo(f"🎴 New cards dealt for Round {game.round_number}")
        
        # Debug: Show all player hands at the beginning of the round
        click.echo(f"\n🔍 Debug: Player hands at start of round:")
        for player in game.players:
            click.echo(f"  {player.name}: {len(player.hand)} cards - {[card.unicode_str() for card in player.hand]}")
        
        # Debug: Show kitty (remaining cards in deck)
        if hasattr(game, 'deck') and hasattr(game.deck, 'size'):
            click.echo(f"🔍 Debug: Kitty - {game.deck.size} cards remaining")
            if game.deck.size <= 10:  # Show actual cards if few remain
                remaining_cards = game.deck.get_remaining_cards()
                click.echo(f"  Remaining cards: {[card.unicode_str() for card in remaining_cards]}")
        
        # Trump selection phase (hand and top card shown during this phase)
        GameCommands._handle_trump_selection(game, player_position, your_position)
        
        # Debug: Check all player hands after returning from trump selection
        click.echo(f"\n🔍 Debug: Player hands AFTER returning from trump selection:")
        for player in game.players:
            click.echo(f"  {player.name}: {len(player.hand)} cards - {[card.unicode_str() for card in player.hand]}")
        
        # Debug: Show all player hands after trump selection
        click.echo(f"\n🔍 Debug: Player hands after trump selection:")
        for player in game.players:
            click.echo(f"  {player.name}: {len(player.hand)} cards - {[card.unicode_str() for card in player.hand]}")
        
        # Now show the final hand after trump selection
        GameCommands._show_human_game_info(game, player_position)
        
        # Play 5 tricks
        previous_trick_winner = None
        for trick_number in range(1, 6):
            click.echo(f"\n--- Trick {trick_number} ---")
            previous_trick_winner = GameCommands._play_interactive_trick(game, player_position, your_position, trick_number, previous_trick_winner)
        
        # Score the round
        GameCommands._score_round(game)
    
    @staticmethod
    def _handle_trump_selection(game: EuchreGame, player_position: str, your_position: int) -> None:
        """Handle the trump selection phase with human input.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        player_position : str
            Human player position name
        your_position : int
            Human player position index
        """
        click.echo(f"\n🎯 Trump Selection Phase")
        click.echo("=" * 40)
        
        # First round of trump selection
        click.echo("First round - players can order up the top card")
        
        # Determine player order for trump selection
        dealer = game.game_state_manager.get_dealer()
        dealer_index = next(i for i, p in enumerate(game.players) if p.name == dealer.name)
        
        # Start with player after dealer
        current_index = (dealer_index + 1) % 4
        
        # Go through each player for trump selection
        for i in range(4):
            current_player = game.players[current_index]
            
            if current_player.name == player_position:
                # Human player's turn
                click.echo(f"\n🤔 {player_position}'s turn to decide on trump")
                if game.top_card and not getattr(game, '_top_card_picked_up', False):
                    click.echo(f"Top card: {game.top_card.unicode_str()}")
                else:
                    click.echo("Top card: Already picked up")
                
                # Show current hand
                your_hand = game.get_player_hand(player_position)
                click.echo(f"Your hand:")
                for j, card in enumerate(your_hand):
                    click.echo(f"  {j+1}. {card.unicode_str()}")
                
                # Get human decision
                while True:
                    try:
                        choice = click.prompt(
                            "Do you want to order up the top card? (y/n)",
                            type=click.Choice(['y', 'n', 'yes', 'no']),
                            default='n'
                        )
                        if choice in ['y', 'yes']:
                            # Human orders up the top card
                            trump_suit = game.top_card.suit
                            game.trump_suit = trump_suit
                            game.game_state_manager.set_trump_suit(trump_suit, current_player)
                            click.echo(f"🎯 {player_position} orders up {trump_suit.name} as trump!")
                            
                            # Human gets to discard one card and pick up the top card
                            GameCommands._handle_human_discard_and_pickup(game, player_position)
                            return
                        else:
                            click.echo(f"😴 {player_position} passes")
                            break
                    except Exception as e:
                        click.echo(f"Invalid input: {e}")
            else:
                # AI player's turn
                # For now, use simple AI logic - can be enhanced later
                if hasattr(current_player, 'choose_trump_action'):
                    action = current_player.choose_trump_action(game.top_card)
                else:
                    # Simple AI logic: order up if you have good cards in that suit
                    action = GameCommands._simple_ai_trump_decision(current_player, game.top_card)
                
                if action == 'order_up':
                    trump_suit = game.top_card.suit
                    game.trump_suit = trump_suit
                    game.game_state_manager.set_trump_suit(trump_suit, current_player)
                    click.echo(f"🎯 {current_player.name} orders up {trump_suit.name} as trump!")
                    
                    # When someone orders up, the DEALER must discard and pick up the top card
                    dealer = game.game_state_manager.get_dealer()
                    if dealer.name == player_position:
                        # Human is dealer - they need to discard and pick up
                        GameCommands._handle_human_discard_and_pickup(game, player_position)
                    else:
                        # AI is dealer - handle AI discard and pickup
                        GameCommands._handle_ai_discard_and_pickup(game, dealer)
                    
                    return
                else:
                    click.echo(f"😴 {current_player.name} passes")
            
            current_index = (current_index + 1) % 4
        
        # If no one ordered up, go to second round
        click.echo(f"\n🔄 Second round - players can call any suit as trump")
        
        # Second round: players can call any suit (except the top card suit)
        top_suit = game.top_card.suit
        available_suits = [suit for suit in Suit if suit != top_suit]
        
        # Start with player after dealer
        current_index = (dealer_index + 1) % 4
        
        # Go through each player for second round trump selection
        for i in range(4):
            current_player = game.players[current_index]
            
            if current_player.name == player_position:
                # Human player's turn
                click.echo(f"\n🤔 {player_position}'s turn to call trump")
                click.echo(f"Available suits (excluding {top_suit.name}):")
                for j, suit in enumerate(available_suits):
                    click.echo(f"  {j+1}. {suit.name}")
                
                # Get human decision
                while True:
                    try:
                        choice = click.prompt(
                            "Do you want to call a trump suit? (y/n)",
                            type=click.Choice(['y', 'n', 'yes', 'no']),
                            default='n'
                        )
                        if choice in ['y', 'yes']:
                            # Human calls a trump suit
                            suit_choice = click.prompt(
                                f"Which suit do you want as trump? (1-{len(available_suits)})",
                                type=int,
                                default=1
                            )
                            if 1 <= suit_choice <= len(available_suits):
                                chosen_suit = available_suits[suit_choice - 1]
                                game.trump_suit = chosen_suit
                                game.game_state_manager.set_trump_suit(chosen_suit, current_player)
                                click.echo(f"🎯 {player_position} calls {chosen_suit.name} as trump!")
                                
                                # Debug: Check all player hands immediately after trump selection
                                click.echo(f"\n🔍 Debug: Player hands IMMEDIATELY after trump selection:")
                                for player in game.players:
                                    click.echo(f"  {player.name}: {len(player.hand)} cards - {[card.unicode_str() for card in player.hand]}")
                                
                                return
                            else:
                                click.echo(f"Please enter a number between 1 and {len(available_suits)}")
                        else:
                            click.echo(f"😴 {player_position} passes")
                            break
                    except Exception as e:
                        click.echo(f"Invalid input: {e}")
            else:
                # AI player's turn
                # For now, use simple AI logic - can be enhanced later
                if hasattr(current_player, 'should_call_trump'):
                    trump_suit = current_player.should_call_trump(top_card)
                    if trump_suit and trump_suit != top_suit:
                        game.trump_suit = trump_suit
                        game.game_state_manager.set_trump_suit(trump_suit, current_player)
                        click.echo(f"🎯 {current_player.name} calls {trump_suit.name} as trump!")
                        return
                
                click.echo(f"😴 {current_player.name} passes")
            
            current_index = (current_index + 1) % 4
        
        # If no one called trump, dealer must pick (Screw the Dealer!)
        click.echo(f"\n👑 {dealer.name} must pick a trump suit (Screw the Dealer!)")
        GameCommands._handle_dealer_trump_selection(game, player_position, your_position)
    
    @staticmethod
    def _handle_human_discard_and_pickup(game: EuchreGame, player_position: str) -> None:
        """Handle human player discarding a card and picking up the top card.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        player_position : str
            Human player position name
        """
        click.echo(f"\n🔄 {player_position}, you need to discard one card and pick up the top card")
        
        # Find the human player object directly
        human_player = None
        for player in game.players:
            if player.name == player_position:
                human_player = player
                break
        
        if not human_player:
            click.echo(f"❌ Error: Could not find player {player_position}")
            return
        
        # Debug: Show all player hands before the discard/pickup process
        click.echo(f"\n🔍 Debug: Player hands BEFORE discard/pickup:")
        for player in game.players:
            click.echo(f"  {player.name}: {len(player.hand)} cards - {[card.unicode_str() for card in player.hand]}")
        
        # Show current hand
        click.echo(f"Your current hand:")
        for i, card in enumerate(human_player.hand):
            click.echo(f"  {i+1}. {card.unicode_str()}")
        
        if game.top_card and not getattr(game, '_top_card_picked_up', False):
            click.echo(f"Top card to pick up: {game.top_card.unicode_str()}")
        else:
            click.echo("Top card: Already picked up")
        
        # Get discard choice
        while True:
            try:
                discard_choice = click.prompt(
                    "Which card do you want to discard? (1-5)",
                    type=int,
                    default=1
                )
                if 1 <= discard_choice <= len(human_player.hand):
                    discarded_card = human_player.hand[discard_choice - 1]
                    human_player.hand.remove(discarded_card)
                    human_player.hand.append(game.top_card)
                    
                    click.echo(f"🗑️  Discarded: {discarded_card.unicode_str()}")
                    if game.top_card:
                        click.echo(f"🆕 Picked up: {game.top_card.unicode_str()}")
                    else:
                        click.echo("🆕 Picked up: Top card")
                    
                    # Mark the top card as picked up
                    game._top_card_picked_up = True
                    
                    # Debug: Check all player hands after the update
                    click.echo(f"\n🔍 Debug: Player hands after pickup:")
                    for player in game.players:
                        click.echo(f"  {player.name}: {len(player.hand)} cards - {[card.unicode_str() for card in player.hand]}")
                    
                    # Ensure all AI players still have 5 cards
                    for player in game.players:
                        if player.player_type.name == "AI" and len(player.hand) != 5:
                            click.echo(f"⚠️  Warning: {player.name} has {len(player.hand)} cards instead of 5")
                    
                    break
                else:
                    click.echo(f"Please enter a number between 1 and {len(human_player.hand)}")
            except Exception as e:
                click.echo(f"Invalid input: {e}")
    
    @staticmethod
    def _handle_ai_discard_and_pickup(game: EuchreGame, dealer: 'Player') -> None:
        """Handle AI dealer discarding a card and picking up the top card.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        dealer : Player
            The AI dealer who needs to discard and pick up
        """
        click.echo(f"\n🔄 {dealer.name} (AI dealer) needs to discard one card and pick up the top card")
        
        # AI logic: discard the lowest value card
        if dealer.hand:
            # Simple AI logic: discard the lowest card
            lowest_card = min(dealer.hand, key=lambda c: c.rank.value)
            dealer.hand.remove(lowest_card)
            dealer.hand.append(game.top_card)
            
            click.echo(f"🤖 {dealer.name} discarded: {lowest_card.unicode_str()}")
            click.echo(f"🤖 {dealer.name} picked up: {game.top_card.unicode_str()}")
            
            # Mark the top card as picked up
            game._top_card_picked_up = True
            
            # Debug: Check all player hands after the update
            click.echo(f"\n🔍 Debug: Player hands after AI dealer pickup:")
            for player in game.players:
                click.echo(f"  {player.name}: {len(player.hand)} cards - {[card.unicode_str() for card in player.hand]}")
            
            # Debug: Show kitty (remaining cards in deck) after AI dealer pickup
            if hasattr(game, 'deck') and hasattr(game.deck, 'size'):
                click.echo(f"🔍 Debug: Kitty after AI dealer pickup - {game.deck.size} cards remaining")
                if game.deck.size <= 10:  # Show actual cards if few remain
                    remaining_cards = game.deck.get_remaining_cards()
                    click.echo(f"  Remaining cards: {[card.unicode_str() for card in remaining_cards]}")
    
    @staticmethod
    def _handle_second_trump_round(game: EuchreGame, player_position: str, your_position: int) -> None:
        """Handle the second round of trump selection where dealer must pick a suit.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        player_position : str
            Human player position name
        your_position : int
            Human player position index
        """
        dealer = game.game_state_manager.get_dealer()
        
        if dealer.name == player_position:
            # Human is dealer and must pick a suit
            click.echo(f"\n👑 {player_position}, you are the dealer and must pick a trump suit")
            
            # Show available suits (excluding the top card suit)
            top_suit = game.top_card.suit
            available_suits = [suit for suit in Suit if suit != top_suit]
            
            click.echo(f"Available suits (excluding {top_suit.name}):")
            for i, suit in enumerate(available_suits):
                click.echo(f"  {i+1}. {suit.name}")
            
            # Get human choice
            while True:
                try:
                    suit_choice = click.prompt(
                        f"Which suit do you want as trump? (1-{len(available_suits)})",
                        type=int,
                        default=1
                    )
                    if 1 <= suit_choice <= len(available_suits):
                        chosen_suit = available_suits[suit_choice - 1]
                        game.trump_suit = chosen_suit
                        game.game_state_manager.set_trump_suit(chosen_suit, dealer)
                        click.echo(f"🎯 {player_position} chooses {chosen_suit.name} as trump!")
                        break
                    else:
                        click.echo(f"Please enter a number between 1 and {len(available_suits)}")
                except Exception as e:
                    click.echo(f"Invalid input: {e}")
        else:
            # AI dealer picks a suit
            # Simple AI logic for now
            top_suit = game.top_card.suit
            available_suits = [suit for suit in Suit if suit != top_suit]
            
            if available_suits:
                chosen_suit = available_suits[0]  # Simple: pick first available
                game.trump_suit = chosen_suit
                game.game_state_manager.set_trump_suit(chosen_suit, dealer)
                click.echo(f"🎯 {dealer.name} chooses {chosen_suit.name} as trump!")
    
    @staticmethod
    def _simple_ai_trump_decision(player, top_card) -> str:
        """Simple AI logic for trump decision.
        
        Parameters
        ----------
        player
            The AI player
        top_card
            The top card for trump selection
            
        Returns
        -------
        str
            'order_up' or 'pass'
        """
        # Simple logic: order up if you have at least 2 cards in that suit
        suit_count = sum(1 for card in player.hand if card.suit == top_card.suit)
        return 'order_up' if suit_count >= 2 else 'pass'
    
    @staticmethod
    def _play_interactive_trick(game: EuchreGame, player_position: str, your_position: int, trick_number: int, previous_trick_winner: Optional[str] = None) -> str:
        """Play a single trick with human interaction.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        player_position : str
            Human player position name
        your_position : int
            Human player position index
        trick_number : int
            Current trick number
        previous_trick_winner : Optional[str]
            Name of the player who won the previous trick (None for first trick)
            
        Returns
        -------
        str
            Name of the player who won this trick
        """
        click.echo(f"\n🎴 Playing Trick {trick_number}")
        
        # Start new trick
        game.trick_manager.start_new_trick()
        game.current_trick = game.trick_manager.get_current_trick()
        
        # Determine starting player
        if trick_number == 1:
            # First trick: player after dealer
            dealer_index = next(i for i, p in enumerate(game.players) if p.name == game.game_state_manager.get_dealer().name)
            current_player_index = (dealer_index + 1) % 4
            click.echo(f"First trick - starting with {game.players[current_player_index].name}")
        else:
            # Subsequent tricks: winner of previous trick
            if previous_trick_winner:
                current_player_index = next(i for i, p in enumerate(game.players) if p.name == previous_trick_winner)
                click.echo(f"Starting trick with {previous_trick_winner} (winner of previous trick)")
            else:
                # Fallback to player after dealer
                dealer_index = next(i for i, p in enumerate(game.players) if p.name == game.game_state_manager.get_dealer().name)
                current_player_index = (dealer_index + 1) % 4
                click.echo(f"Fallback - starting with {game.players[current_player_index].name}")
        
        # Play cards for this trick
        for _ in range(4):
            current_player = game.players[current_player_index]
            
            if current_player.name == player_position:
                # Human player's turn
                GameCommands._handle_human_card_play(game, player_position, your_position)
            else:
                # AI player's turn
                GameCommands._handle_ai_card_play(game, current_player)
            
            # Move to next player
            current_player_index = (current_player_index + 1) % 4
        
        # Complete the trick
        winner = game.trick_manager.complete_trick(game.trump_suit)
        game.tricks_won[winner.name] += 1
        
        click.echo(f"🏆 Trick {trick_number} won by {winner.name}!")
        click.echo(f"Tricks won so far: Alice: {game.tricks_won['Alice']}, Bob: {game.tricks_won['Bob']}, Charlie: {game.tricks_won['Charlie']}, David: {game.tricks_won['David']}")
        
        return winner.name
    
    @staticmethod
    def _handle_human_card_play(game: EuchreGame, player_position: str, your_position: int) -> None:
        """Handle human player playing a card.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        player_position : str
            Human player position name
        your_position : int
            Human player position index
        """
        click.echo(f"\n🤔 {player_position}'s turn to play a card")
        
        # Show current trick state
        current_trick = game.trick_manager.get_current_trick()
        if current_trick and current_trick.cards_played:
            click.echo("Cards played so far:")
            for i, (player, card) in enumerate(current_trick.cards_played):
                click.echo(f"  {player.name}: {card.unicode_str()}")
        
        # Get valid cards to play (enforcing follow suit rules)
        valid_cards = GameCommands._get_valid_cards_for_player(game, player_position, current_trick)
        
        if not valid_cards:
            click.echo("❌ No valid cards to play!")
            return
        
        # Show valid cards only
        click.echo(f"Your valid plays:")
        for i, card in enumerate(valid_cards):
            click.echo(f"  {i+1}. {card.unicode_str()}")
        
        # Get card choice
        while True:
            try:
                card_choice = click.prompt(
                    f"Which card do you want to play? (1-{len(valid_cards)})",
                    type=int,
                    default=1
                )
                if 1 <= card_choice <= len(valid_cards):
                    chosen_card = valid_cards[card_choice - 1]
                    
                    # Find the human player object directly
                    human_player = None
                    for player in game.players:
                        if player.name == player_position:
                            human_player = player
                            break
                    
                    if not human_player:
                        click.echo(f"❌ Error: Could not find player {player_position}")
                        return
                    
                    # Remove card from hand directly
                    human_player.hand.remove(chosen_card)
                    
                    # Play the card
                    game.trick_manager.play_card(human_player, chosen_card)
                    
                    click.echo(f"🎴 {player_position} plays {chosen_card.unicode_str()}")
                    break
                else:
                    click.echo(f"Please enter a number between 1 and {len(valid_cards)}")
            except Exception as e:
                click.echo(f"Invalid input: {e}")
    
    @staticmethod
    def _handle_ai_card_play(game: EuchreGame, ai_player) -> None:
        """Handle AI player playing a card.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        ai_player
            The AI player
        """
        # For now, use simple AI logic
        current_trick = game.trick_manager.get_current_trick()
        lead_suit = current_trick.lead_suit if current_trick else None
        
        if hasattr(ai_player, 'choose_card_to_play'):
            card = ai_player.choose_card_to_play(lead_suit, game.trump_suit)
        else:
            card = GameCommands._simple_ai_card_choice(ai_player, current_trick, game.trump_suit)
        
        # Play the card (don't remove from hand yet - let trick manager handle it)
        game.trick_manager.play_card(ai_player, card)
        
        click.echo(f"🤖 {ai_player.name} plays {card.unicode_str()}")
        
        # Show current trick state
        current_trick = game.trick_manager.get_current_trick()
        if current_trick and current_trick.cards_played:
            click.echo("  Cards played so far:")
            for player, card in current_trick.cards_played:
                click.echo(f"    {player.name}: {card.unicode_str()}")
    
    @staticmethod
    def _get_valid_cards_for_player(game: EuchreGame, player_position: str, current_trick) -> List['Card']:
        """Get valid cards a player can play following suit rules.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        player_position : str
            The player's position name
        current_trick
            The current trick
            
        Returns
        -------
        List[Card]
            List of valid cards to play
        """
        # Find the player object directly
        player_obj = None
        for player in game.players:
            if player.name == player_position:
                player_obj = player
                break
        
        if not player_obj:
            return []
        
        if not current_trick or not current_trick.lead_suit:
            # Leading - can play any card
            return player_obj.hand.copy()  # Return a copy to avoid reference issues
        
        # Must follow suit if possible
        lead_suit = current_trick.lead_suit
        cards_of_lead_suit = [card for card in player_obj.hand if card.suit == lead_suit]
        
        if cards_of_lead_suit:
            # Can follow suit - must play one of these
            return cards_of_lead_suit
        else:
            # Can't follow suit - can play any card
            return player_obj.hand.copy()  # Return a copy to avoid reference issues
    
    @staticmethod
    def _simple_ai_card_choice(player, current_trick, trump_suit) -> 'Card':
        """Simple AI logic for card choice.
        
        Parameters
        ----------
        player
            The AI player
        current_trick
            The current trick
        trump_suit
            The trump suit
            
        Returns
        -------
        Card
            The chosen card
        """
        # Simple logic: play first card if leading, otherwise play highest card of lead suit
        if not current_trick or not current_trick.lead_suit:
            # Leading - play first card
            return player.hand[0]
        else:
            # Must follow suit if possible
            lead_suit = current_trick.lead_suit
            cards_of_suit = [card for card in player.hand if card.suit == lead_suit]
            if cards_of_suit:
                return max(cards_of_suit, key=lambda c: c.rank.value)
            else:
                # Can't follow suit - play first card
                return player.hand[0]
    
    @staticmethod
    def _score_round(game: EuchreGame) -> None:
        """Score the current round.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        """
        # Determine which team called trump
        trump_caller = game.game_state_manager.trump_caller
        if trump_caller:
            trump_caller_team = 0 if game.players.index(trump_caller) % 2 == 0 else 1
        else:
            # If no trump caller, use dealer's team
            dealer = game.game_state_manager.get_dealer()
            trump_caller_team = 0 if game.players.index(dealer) % 2 == 0 else 1
        
        # Get the round scores (don't add them again - they're already added by the main game logic)
        team1_score, team2_score = game.scoring_manager.score_round(game.players, trump_caller_team)
        
        # Display round results
        click.echo(f"\n📊 Round {game.round_number} Complete!")
        click.echo(f"Final trick counts: Alice: {game.tricks_won['Alice']}, Bob: {game.tricks_won['Bob']}, Charlie: {game.tricks_won['Charlie']}, David: {game.tricks_won['David']}")
        click.echo(f"Team 1 (Alice & Charlie): {team1_score} points")
        click.echo(f"Team 2 (Bob & David): {team2_score} points")
        
        # Show current game scores (these are already calculated by the main game logic)
        team1_total = sum(game.players[i].score for i in range(0, 4, 2))
        team2_total = sum(game.players[i].score for i in range(1, 4, 2))
        click.echo(f"Game Score - Team 1: {team1_total}, Team 2: {team2_total}")
    
    @staticmethod
    def _show_final_game_results(game: EuchreGame) -> None:
        """Show the final game results.
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        """
        click.echo("\n🎉 GAME OVER! 🎉")
        
        # Get final scores
        team1_total = sum(game.players[i].score for i in range(0, 4, 2))
        team2_total = sum(game.players[i].score for i in range(1, 4, 2))
        
        if team1_total > team2_total:
            click.echo("🏆 Team 1 (Alice & Charlie) wins!")
        else:
            click.echo("🏆 Team 2 (Bob & David) wins!")
        
        click.echo(f"Final Score - Team 1: {team1_total}, Team 2: {team2_total}")
        
        # Check if team gets set
        trump_caller = game.game_state_manager.trump_caller
        if trump_caller:
            trump_caller_team = 0 if game.players.index(trump_caller) % 2 == 0 else 1
            if game.scoring_manager.is_team_set(game.players, trump_caller_team):
                click.echo("\n🚨 TEAM SET! The trump calling team lost after calling trump!")
    
    @staticmethod
    def _handle_dealer_trump_selection(game: EuchreGame, player_position: str, your_position: str) -> None:
        """Handle the dealer being forced to pick a trump suit (Screw the Dealer!).
        
        Parameters
        ----------
        game : EuchreGame
            The game instance
        player_position : str
            Human player position name
        your_position : str
            Human player position name (for consistency)
        """
        dealer = game.game_state_manager.get_dealer()
        
        # Get available suits (excluding the top card suit)
        top_suit = game.top_card.suit if game.top_card else None
        available_suits = [suit for suit in Suit if suit != top_suit]
        
        if dealer.player_type.name == "HUMAN":
            # Human dealer - prompt for suit choice
            click.echo(f"\n👑 {dealer.name}, you must pick a trump suit (Screw the Dealer!)")
            click.echo("Available suits:")
            for i, suit in enumerate(available_suits, 1):
                click.echo(f"  {i}. {suit.name}")
            
            while True:
                try:
                    suit_choice = click.prompt(
                        f"Which suit do you want to call as trump? (1-{len(available_suits)})",
                        type=int
                    )
                    if 1 <= suit_choice <= len(available_suits):
                        chosen_suit = available_suits[suit_choice - 1]
                        game.trump_suit = chosen_suit
                        game.game_state_manager.set_trump_suit(chosen_suit, dealer)
                        click.echo(f"🎯 {dealer.name} calls {chosen_suit.name} as trump!")
                        return
                    else:
                        click.echo(f"Please enter a number between 1 and {len(available_suits)}")
                except Exception as e:
                    click.echo(f"Invalid input: {e}")
        else:
            # AI dealer - use AI logic to pick trump
            click.echo(f"\n🤖 {dealer.name} (AI dealer) must pick a trump suit")
            
            # Simple AI logic: pick the suit with the most cards in hand
            suit_counts = {}
            for suit in available_suits:
                suit_counts[suit] = sum(1 for card in dealer.hand if card.suit == suit)
            
            # Pick the suit with the most cards, or random if tied
            best_suit = max(suit_counts, key=suit_counts.get)
            
            game.trump_suit = best_suit
            game.game_state_manager.set_trump_suit(best_suit, dealer)
            click.echo(f"🎯 {dealer.name} calls {best_suit.name} as trump!")
            return 