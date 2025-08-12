"""Command-line interface for the euchre card game."""

import click
from typing import List
from .game import EuchreGame
from .models import PlayerType, Suit


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Euchre - A CLI card game with AI opponents.
    
    Play the classic euchre card game against AI opponents.
    """
    pass


@main.command()
@click.option("--player-name", "-n", default=None, help="Your player name (omit for AI-only game)")
@click.option("--ai-names", "-a", multiple=True, 
              default=["Alice", "Bob", "Charlie", "David"], 
              help="Names for AI opponents")
def play(player_name: str, ai_names: tuple) -> None:
    """Start a new euchre game."""
    try:
        from .game import EuchreGame
        
        # Create game
        game = EuchreGame()
        
        # Check if this is a human game or AI-only game
        if player_name is not None:
            # Human game - prompt for name if not specified
            if player_name == "":
                player_name = click.prompt("Enter your name", default="Player")
            
            click.echo(f"Welcome to Euchre, {player_name}!")
            
            # Add human player
            game.add_player(player_name, PlayerType.HUMAN)
            
            # Add AI players with proper names
            ai_names_list = list(ai_names)
            while len(game.players) < 4:
                if len(ai_names_list) > 0:
                    game.add_player(ai_names_list.pop(0), PlayerType.AI)
                else:
                    # Use default AI names if not enough provided
                    default_names = ["Alice", "Bob", "Charlie", "David"]
                    used_names = [p.name for p in game.players]
                    for default_name in default_names:
                        if default_name not in used_names:
                            game.add_player(default_name, PlayerType.AI)
                            break
                    else:
                        # Fallback if all default names are used
                        game.add_player(f"AI_{len(game.players)}", PlayerType.AI)
        else:
            # AI-only game - no prompts
            click.echo("Starting AI-only Euchre game...")
            
            # Add AI players with proper names
            ai_names_list = list(ai_names)
            while len(game.players) < 4:
                if len(ai_names_list) > 0:
                    game.add_player(ai_names_list.pop(0), PlayerType.AI)
                else:
                    # Use default AI names if not enough provided
                    default_names = ["Alice", "Bob", "Charlie", "David"]
                    used_names = [p.name for p in game.players]
                    for default_name in default_names:
                        if default_name not in used_names:
                            game.add_player(default_name, PlayerType.AI)
                            break
                    else:
                        # Fallback if all default names are used
                        game.add_player(f"AI_{len(game.players)}", PlayerType.AI)
        
        click.echo(f"Players: {', '.join(p.name for p in game.players)}")
        
        game.start_new_game()
        click.echo("Game started! Dealing cards...")
        
        # Show human player's hand if this is a human game
        if player_name is not None:
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
                        
            # Check if team gets set
            if game.is_team_set():
                click.echo("\n🚨 TEAM SET! The trump calling team lost after calling trump!")
                
            # Play the full game
            click.echo("\nStarting the game...")
            
            # Play rounds until game ends
            round_num = 1
            while not game.is_game_over():
                click.echo(f"\n=== ROUND {round_num} ===")
                
                # Play the round
                results = game.play_round()
                
                # Show round results
                click.echo(f"Round {round_num} complete!")
                for i, player in enumerate(game.players):
                    click.echo(f"  {player.name}: {results[i]} tricks")
                
                # Show current scores
                if game.game_state:
                    click.echo(f"Team 1: {game.game_state.team1_score}, Team 2: {game.game_state.team2_score}")
                
                round_num += 1
            
            # Show final result
            winner = game.get_winner()
            click.echo(f"\n🎉 GAME OVER! {winner} wins! 🎉")
            
            if game.game_state:
                click.echo(f"Final Score - Team 1: {game.game_state.team1_score}, Team 2: {game.game_state.team2_score}")
            
            # Check if team gets set
            if game.is_team_set():
                click.echo("\n🚨 TEAM SET! The trump calling team lost after calling trump!")
            
            # Show log filename
            log_filename = game.get_log_filename()
            if log_filename:
                click.echo(f"\nGame log saved to: {log_filename}")
        else:
            # AI-only game - just run it
            game.run_full_game()
        
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option("--ai-profiles", "-p", multiple=True, 
              default=["balanced", "balanced", "balanced", "balanced"],
              help="AI profiles: aggressive, conservative, balanced, opportunistic")
@click.option("--risk-ratios", "-r", multiple=True, 
              default=["0.5", "0.5", "0.5", "0.5"],
              help="Risk ratios (0.0-1.0) for each AI player")
@click.option("--enable-logging", "-l", is_flag=True, default=True,
              help="Enable game logging to file")
def ai_profiles(ai_profiles: tuple, risk_ratios: tuple, enable_logging: bool) -> None:
    """Run a game with different AI profiles and risk ratios."""
    try:
        from .game import EuchreGame
        
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
        
        # Start the game
        game.start_new_game()
        
        click.echo(f"Game started! Trump: {game.game_state.trump_suit.name.title()}")
        click.echo("Playing rounds...")
        
        # Play rounds until game ends
        round_num = 1
        while not game.is_game_over():
            click.echo(f"Playing round {round_num}...")
            results = game.play_round()
            click.echo(f"Round {round_num} complete. Trick counts: {results}")
            round_num += 1
            
        # Show final result
        winner = game.get_winner()
        click.echo(f"\nGame Over! {winner} wins!")
        click.echo(f"Final Score - Team 1: {game.game_state.team1_score}, Team 2: {game.game_state.team2_score}")
        
        # Show log filename
        log_filename = game.get_log_filename()
        if log_filename:
            click.echo(f"\nGame log saved to: {log_filename}")
            click.echo("You can review the detailed game log in this file.")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option("--num-games", "-n", default=1000, help="Number of games to run")
@click.option("--config", "-c", default="aggressive_vs_conservative", 
              help="Team configuration to use")
@click.option("--max-workers", "-w", default=None, type=int, 
              help="Maximum number of parallel workers")
@click.option("--output-dir", "-o", default="games", help="Output directory for game results")
def run_mass_games(num_games: int, config: str, max_workers: int, output_dir: str) -> None:
    """Run thousands of euchre games in parallel."""
    try:
        from .mass_game_runner import MassGameRunner
        
        click.echo(f"Starting mass game runner with {num_games} games...")
        click.echo(f"Configuration: {config}")
        click.echo(f"Output directory: {output_dir}")
        
        runner = MassGameRunner(output_dir=output_dir, max_workers=max_workers)
        runner.run_games(num_games, config)
        
        click.echo(f"\nMass game run completed!")
        click.echo(f"Results saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option("--games-per-config", "-g", default=1000, help="Games per configuration")
@click.option("--max-workers", "-w", default=None, type=int, 
              help="Maximum number of parallel workers")
@click.option("--output-dir", "-o", default="games", help="Output directory for game results")
def run_all_configs(games_per_config: int, max_workers: int, output_dir: str) -> None:
    """Run games for all team configurations."""
    try:
        from .mass_game_runner import MassGameRunner
        
        click.echo(f"Starting mass game runner for all configurations...")
        click.echo(f"Games per configuration: {games_per_config}")
        click.echo(f"Output directory: {output_dir}")
        
        runner = MassGameRunner(output_dir=output_dir, max_workers=max_workers)
        runner.run_all_configurations(games_per_config)
        
        click.echo(f"\nAll configuration runs completed!")
        click.echo(f"Results saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option("--games-dir", "-d", default="games", help="Directory containing game results")
@click.option("--max-games", "-m", default=None, type=int, help="Maximum games to analyze")
@click.option("--export-excel", "-e", default="euchre_analysis.xlsx", help="Excel export filename")
@click.option("--generate-plots", "-p", is_flag=True, help="Generate visualization plots")
@click.option("--plots-dir", default="analysis_plots", help="Directory for generated plots")
def analyze_games(games_dir: str, max_games: int, export_excel: str, 
                  generate_plots: bool, plots_dir: str) -> None:
    """Analyze game results and generate statistics."""
    try:
        from .game_analyzer import GameAnalyzer
        
        click.echo(f"Loading games from: {games_dir}")
        
        analyzer = GameAnalyzer(games_dir=games_dir)
        df = analyzer.load_games(max_games=max_games)
        
        if df is None or len(df) == 0:
            click.echo("No games found to analyze.")
            return
        
        click.echo(f"Analyzing {len(df)} games...")
        
        # Print summary
        analyzer.print_summary()
        
        # Export to Excel
        click.echo(f"Exporting analysis to: {export_excel}")
        analyzer.export_analysis(export_excel)
        
        # Generate plots if requested
        if generate_plots:
            click.echo(f"Generating plots in: {plots_dir}")
            analyzer.generate_visualizations(plots_dir)
        
        click.echo("Analysis completed successfully!")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option("--games-dir", "-d", default="games", help="Directory containing game results")
@click.option("--older-than-days", "-o", default=7, type=int, help="Remove files older than N days")
def cleanup_games(games_dir: str, older_than_days: int) -> None:
    """Clean up old game result files."""
    try:
        from .mass_game_runner import MassGameRunner
        
        click.echo(f"Cleaning up games in: {games_dir}")
        click.echo(f"Removing files older than: {older_than_days} days")
        
        runner = MassGameRunner(games_dir)
        removed_count = runner.cleanup_old_games(older_than_days)
        
        click.echo(f"Cleanup completed! Removed {removed_count} old game files.")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@main.command()
def logged_game() -> None:
    """Run a full AI vs AI euchre game with logging (no ncurses)."""
    try:
        from .game import EuchreGame
        from .models import PlayerType
        
        click.echo("Starting logged AI vs AI euchre game...")
        
        # Create game with logging enabled
        game = EuchreGame(enable_logging=True)
        
        # Add AI players
        game.add_player("North", PlayerType.AI)
        game.add_player("East", PlayerType.AI)
        game.add_player("South", PlayerType.AI)
        game.add_player("West", PlayerType.AI)
        
        # Start the game
        game.start_new_game()
        
        click.echo(f"Game started! Trump: {game.game_state.trump_suit.name.title()}")
        click.echo("Playing rounds...")
        
        # Play rounds until game ends
        round_num = 1
        while not game.is_game_over():
            click.echo(f"Playing round {round_num}...")
            results = game.play_round()
            click.echo(f"Round {round_num} complete. Trick counts: {results}")
            round_num += 1
            
        # Show final result
        winner = game.get_winner()
        click.echo(f"\nGame Over! {winner} wins!")
        click.echo(f"Final Score - Team 1: {game.game_state.team1_score}, Team 2: {game.game_state.team2_score}")
        
        # Show log filename
        log_filename = game.get_log_filename()
        if log_filename:
            click.echo(f"\nGame log saved to: {log_filename}")
            click.echo("You can review the detailed game log in this file.")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@main.command()
def ncurses() -> None:
    """Run a full AI vs AI euchre game with ncurses display."""
    try:
        from .ncurses_game import NcursesGame
        click.echo("Starting AI vs AI euchre game with ncurses...")
        click.echo("Press Ctrl+C to exit early")
        
        game = NcursesGame()
        game.run()
        
    except ImportError:
        click.echo("Error: curses module not available. This command requires a terminal that supports ncurses.", err=True)
    except KeyboardInterrupt:
        click.echo("\nGame interrupted by user.")


@main.command()
def ai_vs_ai() -> None:
    """Run a full AI vs AI euchre game with ncurses display."""
    try:
        from .ncurses_game import NcursesGame
        click.echo("Starting AI vs AI euchre game...")
        click.echo("Press Ctrl+C to exit early")
        
        game = NcursesGame()
        game.run()
        
    except ImportError:
        click.echo("Error: curses module not available. This command requires a terminal that supports ncurses.", err=True)
    except KeyboardInterrupt:
        click.echo("\nGame interrupted by user.")


@main.command()
def rules() -> None:
    """Show the rules of euchre."""
    click.echo("""
Euchre Rules:

1. **Objective**: Be the first team to score 10 points
2. **Players**: 4 players in teams of 2 (sitting across from each other)
3. **Cards**: 9, 10, J, Q, K, A of each suit (24 cards total)
4. **Dealing**: 5 cards to each player, 4 cards left in deck
5. **Trump**: The top card of the deck can be "ordered up" to set trump suit
6. **Gameplay**: Players take turns playing cards, following suit if possible
7. **Scoring**: 
   - Win 3-4 tricks: 1 point
   - Win all 5 tricks: 2 points
   - Win 5 tricks after being "euchred": 4 points

This is a simplified CLI version with basic AI opponents.
""")


@main.command()
def demo() -> None:
    """Run a quick demo of the game."""
    click.echo("Running Euchre demo...")
    
    # Create a demo game
    game = EuchreGame()
    game.add_player("You", PlayerType.HUMAN)
    game.add_player("Alice", PlayerType.AI)
    game.add_player("Bob", PlayerType.AI)
    game.add_player("Charlie", PlayerType.AI)
    
    try:
        game.start_new_game()
        
        # Show some game state
        click.echo(f"Dealer: {game.game_state.get_dealer().name}")
        click.echo(f"Current player: {game.game_state.get_current_player().name}")
        
        # Show human player's hand
        human_hand = game.get_player_hand("You")
        click.echo(f"\nYour hand:")
        for i, card in enumerate(human_hand, 1):
            click.echo(f"  {i}. {card}")
            
        click.echo("\nDemo completed! Use 'euchre play' to start a real game.")
        
    except Exception as e:
        click.echo(f"Demo error: {e}", err=True)


@click.command()
@click.option('--num-games', default=100000, help='Number of training games')
@click.option('--players', nargs=4, default=['Alice', 'Bob', 'Charlie', 'David'], 
              help='Player names for training')
@click.option('--risk-profiles', nargs=4, 
              default=['balanced', 'balanced', 'balanced', 'balanced'],
              help='Risk profiles for each player')
@click.option('--output-dir', default='trained_models', help='Output directory for models')
@click.option('--save-interval', default=1000, help='Save models every N games')
@click.option('--eval-interval', default=5000, help='Evaluate performance every N games')
def train_self_play(num_games, players, risk_profiles, output_dir, save_interval, eval_interval):
    """Train AI models using self-play."""
    try:
        from .ai_model.self_play_trainer import SelfPlayTrainer
        
        # Model configuration
        model_config = {
            'type': 'standard',
            'input_size': 128,
            'hidden_size': 256,
            'output_size': 64,
            'risk_embedding_size': 32,
            'use_risk_attention': True
        }
        
        # Create trainer
        trainer = SelfPlayTrainer(model_config, output_dir)
        
        # Start training
        results = trainer.train_self_play(
            num_games=num_games,
            players=players,
            risk_profiles=risk_profiles,
            save_interval=save_interval,
            evaluation_interval=eval_interval
        )
        
        click.echo("🎯 Self-play training completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Training failed: {e}")
        raise


@click.command()
@click.option('--models-dir', default='trained_models', help='Directory containing trained models')
def list_players(models_dir):
    """List available trained AI players."""
    try:
        from .ai_model.player_manager import PlayerManager
        
        manager = PlayerManager(models_dir)
        players = manager.get_available_players()
        
        if not players:
            click.echo("No trained players found.")
            return
        
        click.echo(f"Available trained players ({len(players)}):")
        for player in players:
            info = manager.get_player_info(player)
            if info:
                click.echo(f"  🎮 {player}")
                click.echo(f"    Training games: {info.get('training_games', 'Unknown')}")
                click.echo(f"    Last modified: {info.get('last_modified', 'Unknown')}")
            else:
                click.echo(f"  🎮 {player} (info unavailable)")
        
    except Exception as e:
        click.echo(f"❌ Failed to list players: {e}")
        raise


@click.command()
@click.option('--models-dir', default='trained_models', help='Directory containing trained models')
@click.option('--player1', required=True, help='First player name')
@click.option('--player2', required=True, help='Second player name')
@click.option('--player3', required=True, help='Third player name')
@click.option('--player4', required=True, help='Fourth player name')
@click.option('--risk1', default='balanced', help='Risk profile for player 1')
@click.option('--risk2', default='balanced', help='Risk profile for player 2')
@click.option('--risk3', default='balanced', help='Risk profile for player 3')
@click.option('--risk4', default='balanced', help='Risk profile for player 4')
@click.option('--enable-logging/--no-logging', default=True, help='Enable game logging')
def play_trained_players(models_dir, player1, player2, player3, player4, 
                        risk1, risk2, risk3, risk4, enable_logging):
    """Play a game with trained AI players."""
    try:
        from .ai_model.player_manager import PlayerManager
        
        # Create player manager
        manager = PlayerManager(models_dir)
        
        # Create player configurations
        player_configs = [
            {'name': player1, 'type': 'model', 'risk_profile': risk1},
            {'name': player2, 'type': 'model', 'risk_profile': risk2},
            {'name': player3, 'type': 'model', 'risk_profile': risk3},
            {'name': player4, 'type': 'model', 'risk_profile': risk4}
        ]
        
        # Create game
        game = manager.create_game_with_players(player_configs, enable_logging)
        
        if game is None:
            click.echo("❌ Failed to create game")
            return
        
        click.echo(f"🎮 Game created with players:")
        click.echo(f"  North: {player1} ({risk1})")
        click.echo(f"  East: {player2} ({risk2})")
        click.echo(f"  South: {player3} ({risk3})")
        click.echo(f"  West: {player4} ({risk4})")
        
        # Start game
        game.start_new_game()
        
        # Play until completion
        round_num = 1
        while not game.is_game_over():
            click.echo(f"\n🔄 Round {round_num}")
            results = game.play_round()
            round_num += 1
        
        # Show final results
        click.echo(f"\n🏁 Game completed!")
        click.echo(f"Team 1 (North/South): {game.game_state.team1_score}")
        click.echo(f"Team 2 (East/West): {game.game_state.team2_score}")
        
        if game.game_state.team1_score > game.game_state.team2_score:
            click.echo("🏆 Team 1 wins!")
        elif game.game_state.team2_score > game.game_state.team1_score:
            click.echo("🏆 Team 2 wins!")
        else:
            click.echo("🤝 Game is a tie!")
        
        # Show log file if logging was enabled
        if enable_logging and game.logger:
            log_file = game.get_log_filename()
            if log_file:
                click.echo(f"📝 Game log saved to: {log_file}")
        
    except Exception as e:
        click.echo(f"❌ Game failed: {e}")
        raise


@click.command()
@click.option('--models-dir', default='trained_models', help='Directory containing trained models')
@click.option('--num-games', default=100, help='Number of games to play')
@click.option('--config-name', help='Name for this configuration')
def tournament(models_dir, num_games, config_name):
    """Run a tournament between trained players."""
    try:
        from .ai_model.player_manager import PlayerManager
        from .ai_model.self_play_trainer import SelfPlayTrainer
        import time
        
        # Create player manager
        manager = PlayerManager(models_dir)
        available_players = manager.get_available_players()
        
        if len(available_players) < 4:
            click.echo(f"❌ Need at least 4 players, only {len(available_players)} available")
            return
        
        if not config_name:
            config_name = f"tournament_{int(time.time())}"
        
        click.echo(f"🏆 Starting tournament: {config_name}")
        click.echo(f"🎮 Players: {', '.join(available_players[:4])}")
        click.echo(f"📊 Games per matchup: {num_games}")
        
        # Create tournament trainer
        model_config = {
            'type': 'standard',
            'input_size': 128,
            'hidden_size': 256,
            'output_size': 64,
            'risk_embedding_size': 32,
            'use_risk_attention': True
        }
        
        trainer = SelfPlayTrainer(model_config, models_dir)
        
        # Run tournament
        results = trainer.train_self_play(
            num_games=num_games,
            players=available_players[:4],
            risk_profiles=['balanced'] * 4,
            save_interval=num_games,
            evaluation_interval=num_games
        )
        
        click.echo("🏆 Tournament completed!")
        
    except Exception as e:
        click.echo(f"❌ Tournament failed: {e}")
        raise


@click.command()
@click.option('--device', default='cpu', help='Device to run benchmark on (cpu/cuda)')
@click.option('--save-results', is_flag=True, default=True, help='Save benchmark results to file')
def benchmark(device, save_results):
    """Run performance benchmark comparing float vs integer models."""
    try:
        from .ai_model.benchmark_runner import run_benchmark
        
        click.echo(f"🚀 Running benchmark on {device}...")
        results = run_benchmark(device=device, save_results=save_results)
        
        click.echo("✅ Benchmark completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}")
        raise


@click.command()
@click.option('--num-games', '-n', default=10000, help='Number of games to run')
@click.option('--model1', '-m1', required=True, help='First model name (plays North/South)')
@click.option('--model2', '-m2', required=True, help='Second model name (plays East/West)')
@click.option('--max-workers', '-w', default=None, type=int, help='Maximum number of parallel workers')
@click.option('--output-dir', '-o', default='neural_games', help='Output directory for game results')
def run_neural_games(num_games, model1, model2, max_workers, output_dir):
    """Run thousands of euchre games with neural network AI players."""
    try:
        from .neural_mass_game_runner import NeuralMassGameRunner
        
        click.echo(f"🧠 Starting neural network mass game runner with {num_games} games...")
        click.echo(f"Team 1 (North/South): {model1}")
        click.echo(f"Team 2 (East/West): {model2}")
        click.echo(f"Output directory: {output_dir}")
        
        runner = NeuralMassGameRunner(output_dir=output_dir, max_workers=max_workers)
        runner.run_model_vs_model(model1, model2, num_games)
        
        click.echo(f"\n🧠 Neural network mass game run completed!")
        click.echo(f"Results saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@click.command()
@click.option('--num-games', '-n', default=1000, help='Number of games per combination')
@click.option('--max-workers', '-w', default=None, type=int, help='Maximum number of parallel workers')
@click.option('--output-dir', '-o', default='neural_games', help='Output directory for game results')
def run_all_neural_combinations(num_games, max_workers, output_dir):
    """Run games for all possible neural network model combinations."""
    try:
        from .neural_mass_game_runner import NeuralMassGameRunner
        
        click.echo(f"🧠 Starting neural network mass game runner for all model combinations...")
        click.echo(f"Games per combination: {num_games}")
        click.echo(f"Output directory: {output_dir}")
        
        runner = NeuralMassGameRunner(output_dir=output_dir, max_workers=max_workers)
        runner.run_all_model_combinations(num_games)
        
        click.echo(f"\n🧠 All neural network combination runs completed!")
        click.echo(f"Results saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@click.command()
@click.option('--models-dir', '-d', default='demo_models', help='Directory containing trained models')
def list_neural_models(models_dir):
    """List available neural network models."""
    try:
        from .ai_model.player_manager import PlayerManager
        
        click.echo(f"🔍 Looking for models in: {models_dir}")
        
        manager = PlayerManager(models_dir)
        available_models = manager.get_available_players()
        
        if not available_models:
            click.echo("❌ No models found")
            return
        
        click.echo(f"📋 Found {len(available_models)} models:")
        for model_name in available_models:
            info = manager.get_player_info(model_name)
            if info:
                click.echo(f"  🧠 {model_name}")
                click.echo(f"    📁 File: {info['model_file']}")
                click.echo(f"    📊 Training games: {info.get('training_games', 'Unknown')}")
                click.echo(f"    🕒 Last modified: {info.get('last_modified', 'Unknown')}")
            else:
                click.echo(f"  🧠 {model_name} (info unavailable)")
            click.echo()
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@click.command()
@click.option('--num-games', default=200000, help='Number of games to play')
@click.option('--integer-players', default='Integer_Alice,Integer_Bob', help='Comma-separated integer player names')
@click.option('--float-players', default='Float_Charlie,Float_David', help='Comma-separated float player names')
@click.option('--risk-profiles', default='balanced,aggressive,conservative,opportunistic', help='Comma-separated risk profiles')
@click.option('--save-interval', default=1000, help='How often to save checkpoints')
@click.option('--evaluation-interval', default=5000, help='How often to evaluate performance')
def train_integer_vs_float(num_games, integer_players, float_players, risk_profiles, save_interval, evaluation_interval):
    """Train integer models against float models in a mixed training scenario."""
    try:
        from .ai_model.self_play_trainer import SelfPlayTrainer
        
        # Parse comma-separated strings
        integer_player_list = [p.strip() for p in integer_players.split(',')]
        float_player_list = [p.strip() for p in float_players.split(',')]
        risk_profile_list = [p.strip() for p in risk_profiles.split(',')]
        
        click.echo(f"🚀 Starting Integer vs Float training for {num_games:,} games")
        click.echo(f"📊 Integer players: {', '.join(integer_player_list)}")
        click.echo(f"📊 Float players: {', '.join(float_player_list)}")
        click.echo(f"🎯 Risk profiles: {', '.join(risk_profile_list)}")
        
        # Create trainer
        model_config = {
            'type': 'standard',
            'input_size': 128,
            'hidden_size': 256,
            'output_size': 64,
            'risk_embedding_size': 32,
            'use_risk_attention': True
        }
        
        trainer = SelfPlayTrainer(model_config, output_dir="trained_models")
        
        # Start training
        results = trainer.train_integer_vs_float(
            num_games=num_games,
            integer_players=integer_player_list,
            float_players=float_player_list,
            risk_profiles=risk_profile_list,
            save_interval=save_interval,
            evaluation_interval=evaluation_interval
        )
        
        click.echo("✅ Integer vs Float training completed successfully!")
        click.echo(f"🏆 Final results: Integer {results['integer_win_rate']*100:.1f}% vs Float {results['float_win_rate']*100:.1f}%")
        
    except Exception as e:
        click.echo(f"❌ Training failed: {e}")
        raise


@click.command()
@click.option('--num-games', default=15000, help='Number of games per profile (minimum 10000)')
@click.option('--output-dir', default='trained_models', help='Output directory for trained models')
@click.option('--save-interval', default=1000, help='How often to save checkpoints')
@click.option('--evaluation-interval', default=2000, help='How often to evaluate performance')
@click.option('--device', default='auto', help='Device to use for training (cpu/cuda/mps/auto)')
@click.option('--gpu-memory-fraction', default=0.9, help='GPU memory fraction to use (0.1-1.0)')
@click.option('--enable-amp', is_flag=True, default=True, help='Enable automatic mixed precision for faster training')
def generate_player_profiles(num_games, output_dir, save_interval, evaluation_interval, device, gpu_memory_fraction, enable_amp):
    """Generate multiple AI player profiles with different playing styles."""
    try:
        from .ai_model.self_play_trainer import SelfPlayTrainer
        
        # Ensure minimum games requirement
        if num_games < 10000:
            click.echo("⚠️  Setting minimum games to 10,000 for proper training")
            num_games = 10000
        
        click.echo(f"🎯 Generating AI Player Profiles")
        click.echo(f"📊 Games per profile: {num_games:,}")
        click.echo(f"📁 Output directory: {output_dir}")
        click.echo(f"💾 Save interval: {save_interval}")
        click.echo(f"📈 Evaluation interval: {evaluation_interval}")
        click.echo(f"🖥️  Device: {device}")
        click.echo(f"💾 GPU memory fraction: {gpu_memory_fraction}")
        click.echo(f"⚡ AMP enabled: {enable_amp}")
        click.echo("=" * 60)
        
        # Check GPU availability
        if device == "auto" or device == "cuda":
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                click.echo(f"🚀 GPU detected: {gpu_name}")
                click.echo(f"💾 GPU Memory: {gpu_memory:.1f} GB")
                
                # Set GPU memory fraction
                if gpu_memory_fraction != 0.9:
                    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
                    click.echo(f"💾 GPU memory fraction set to {gpu_memory_fraction}")
            else:
                click.echo("⚠️  CUDA requested but not available, using CPU")
                device = "cpu"
        
        # Enable automatic mixed precision if requested
        if enable_amp and device in ["auto", "cuda"]:
            try:
                import torch.cuda.amp
                click.echo("⚡ Automatic Mixed Precision (AMP) enabled for faster training")
            except ImportError:
                click.echo("⚠️  AMP not available, continuing without it")
        
        # Define the 5 distinct player profiles with different playing styles
        player_profiles = [
            {
                'name': 'Alice',
                'risk_profile': 'aggressive',
                'description': 'Aggressive player - high risk tolerance, leads with high cards'
            },
            {
                'name': 'Bob', 
                'risk_profile': 'conservative',
                'description': 'Conservative player - low risk tolerance, plays safe'
            },
            {
                'name': 'Charlie',
                'risk_profile': 'balanced',
                'description': 'Balanced player - moderate risk, adaptive strategy'
            },
            {
                'name': 'David',
                'risk_profile': 'ace_hunter',
                'description': 'Ace hunter - loves ordering up aces, strategic risk taker'
            },
            {
                'name': 'Eve',
                'risk_profile': 'trump_caller',
                'description': 'Trump caller - always calls trump, aggressive trump play'
            }
        ]
        
        # Model configuration
        model_config = {
            'type': 'standard',
            'input_size': 128,
            'hidden_size': 256,
            'output_size': 64,
            'risk_embedding_size': 32,
            'use_risk_attention': True
        }
        
        # Create trainer
        trainer = SelfPlayTrainer(model_config, output_dir, device)
        
        # Train each profile
        results = {}
        for profile in player_profiles:
            click.echo(f"\n🧠 Training {profile['name']} ({profile['risk_profile']})")
            click.echo(f"📝 Style: {profile['description']}")
            click.echo("-" * 40)
            
            try:
                # Train this profile
                profile_results = trainer.train_single_profile(
                    player_name=profile['name'],
                    risk_profile=profile['risk_profile'],
                    num_games=num_games,
                    save_interval=save_interval,
                    evaluation_interval=evaluation_interval
                )
                
                results[profile['name']] = profile_results
                
                click.echo(f"✅ {profile['name']} training completed!")
                click.echo(f"   Final win rate: {profile_results['final_win_rate']*100:.1f}%")
                click.echo(f"   Games played: {profile_results['total_games']}")
                click.echo(f"   Model saved: {profile_results['model_path']}")
                
            except Exception as e:
                click.echo(f"❌ Failed to train {profile['name']}: {e}")
                results[profile['name']] = {'error': str(e)}
        
        # Summary
        click.echo("\n" + "=" * 60)
        click.echo("🎉 PLAYER PROFILE GENERATION COMPLETED!")
        click.echo("=" * 60)
        
        successful_profiles = [name for name, result in results.items() if 'error' not in result]
        failed_profiles = [name for name, result in results.items() if 'error' in result]
        
        click.echo(f"✅ Successfully trained: {len(successful_profiles)} profiles")
        if successful_profiles:
            click.echo(f"   {', '.join(successful_profiles)}")
        
        if failed_profiles:
            click.echo(f"❌ Failed to train: {len(failed_profiles)} profiles")
            click.echo(f"   {', '.join(failed_profiles)}")
        
        click.echo(f"\n📁 All models saved to: {output_dir}")
        click.echo(f"🎮 Use these profiles in games with: euchre play --ai-profiles")
        
    except Exception as e:
        click.echo(f"❌ Profile generation failed: {e}")
        raise


@main.command()
@click.option("--mode", "-m", 
              type=click.Choice(["same_model", "team_vs_team", "mixed_teams", "round_robin"]),
              default="same_model",
              help="Training mode: same_model, team_vs_team, mixed_teams, round_robin")
@click.option("--games", "-g", default=100, help="Number of games to run")
@click.option("--ai-types", "-t", multiple=True, 
              default=["balanced", "balanced", "balanced", "balanced"],
              help="AI types for each player")
@click.option("--risk-ratios", "-r", multiple=True, 
              default=["0.5", "0.5", "0.5", "0.5"],
              help="Risk ratios for each AI player")
@click.option("--output-dir", "-o", default="training_results", help="Output directory for results")
def train(mode: str, games: int, ai_types: tuple, risk_ratios: tuple, output_dir: str) -> None:
    """Run AI training sessions with different team configurations."""
    try:
        from .ai_training_framework import AITrainingFramework, TrainingConfig, TrainingMode
        
        # Convert string mode to enum
        mode_enum = TrainingMode(mode)
        
        # Convert risk ratios to floats
        risk_ratios_float = [float(r) for r in risk_ratios]
        
        # Create training configuration
        config = TrainingConfig(
            mode=mode_enum,
            num_games=games,
            ai_types=list(ai_types),
            risk_ratios=risk_ratios_float,
            output_dir=output_dir,
            quiet_mode=True  # Training should be quiet
        )
        
        # Create and run training framework
        framework = AITrainingFramework(config)
        results = framework.run_training_session()
        
        # Display summary
        click.echo(f"\n🎯 TRAINING COMPLETED! 🎯")
        click.echo(f"Mode: {mode}")
        click.echo(f"Games: {games}")
        click.echo(f"Results saved to: {output_dir}/")
        
        # Show key statistics
        overall = results["overall_stats"]
        click.echo(f"\n📊 OVERALL STATISTICS:")
        click.echo(f"Team 1 Wins: {overall['team1_wins']} ({overall['team1_wins']/overall['total_games']*100:.1f}%)")
        click.echo(f"Team 2 Wins: {overall['team2_wins']} ({overall['team2_wins']/overall['total_games']*100:.1f}%)")
        click.echo(f"Team Sets: {overall['total_team_sets']}")
        click.echo(f"Total Reneges: {overall['total_reneges']}")
        
    except ImportError as e:
        click.echo(f"Error: Could not import training framework: {e}", err=True)
    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)


@main.command()
@click.option("--model", "-m", 
              type=click.Choice(["gold_mixed_conservative_balanced", "gold_conservative_dominance", 
                                "silver_balanced_team", "silver_conservative_balanced",
                                "bronze_risk_optimized", "bronze_mixed_personalities"]),
              default="gold_mixed_conservative_balanced",
              help="Fine-tuned model to use")
@click.option("--player-names", "-n", multiple=True, 
              default=["Alice", "Bob", "Charlie", "David"],
              help="Names for the AI players")
def play_fine_tuned(model: str, player_names: tuple) -> None:
    """Play a game using fine-tuned AI models."""
    try:
        from .fine_tuned_models import create_fine_tuned_team
        
        # Create fine-tuned team
        ai_players = create_fine_tuned_team(model, list(player_names))
        
        # Create game
        game = EuchreGame()
        
        # Add AI players
        for player in ai_players:
            game.players.append(player)
        
        click.echo(f"🎯 Using Fine-Tuned Model: {model}")
        click.echo(f"Players: {', '.join(p.name for p in game.players)}")
        
        # Start and run the game
        game.start_new_game()
        click.echo("Game started! Dealing cards...")
        
        # Run the full game
        game.run_full_game()
        
    except ImportError as e:
        click.echo(f"Error: Could not import fine-tuned models: {e}", err=True)
    except Exception as e:
        click.echo(f"Error during game: {e}", err=True)


@main.command()
def list_models() -> None:
    """List all available fine-tuned AI models."""
    try:
        from .fine_tuned_models import list_available_models
        list_available_models()
    except ImportError as e:
        click.echo(f"Error: Could not import fine-tuned models: {e}", err=True)


@main.command()
@click.option("--use-case", "-u", 
              type=click.Choice(["Maximum performance", "Balanced competition", "Educational purposes", 
                                "Risk management", "Team coordination"]),
              default="Maximum performance",
              help="Use case to find optimal model for")
def find_optimal_model(use_case: str) -> None:
    """Find the optimal fine-tuned model for a specific use case."""
    try:
        from .fine_tuned_models import get_optimal_model_for_use_case
        
        model = get_optimal_model_for_use_case(use_case)
        
        if model:
            click.echo(f"🎯 OPTIMAL MODEL FOR: {use_case}")
            click.echo("=" * 50)
            click.echo(f"Model: {model.name}")
            click.echo(f"Tier: {model.tier.value.title()}")
            click.echo(f"Description: {model.description}")
            click.echo(f"Expected Win Rate: {model.expected_win_rate:.1f}%")
            click.echo(f"Team Sets: {model.team_sets_percentage:.1f}%")
            click.echo(f"AI Types: {', '.join(model.ai_types)}")
            click.echo(f"Risk Ratios: {', '.join(map(str, model.risk_ratios))}")
            click.echo(f"Use Cases: {', '.join(model.use_cases)}")
            
            click.echo(f"\nTo use this model:")
            click.echo(f"euchre play-fine-tuned --model {model.name.lower().replace(' ', '_')}")
        else:
            click.echo(f"No optimal model found for use case: {use_case}")
            
    except ImportError as e:
        click.echo(f"Error: Could not import fine-tuned models: {e}", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option("--mode", "-m", 
              type=click.Choice(["same_model", "team_vs_team", "mixed_teams"]),
              default="mixed_teams",
              help="Training mode")
@click.option("--games", "-g", default=100, help="Number of games to run")
@click.option("--model", "-t", 
              type=click.Choice(["gold_mixed_conservative_balanced", "gold_conservative_dominance", 
                                "silver_balanced_team", "silver_conservative_balanced"]),
              default="gold_mixed_conservative_balanced",
              help="Fine-tuned model to use")
@click.option("--output-dir", "-o", default="adaptive_training_results", help="Output directory for results")
def train_adaptive(mode: str, games: int, model: str, output_dir: str) -> None:
    """Run training sessions with adaptive AI profiles using fine-tuned models."""
    try:
        from .ai_training_framework import AITrainingFramework, TrainingConfig, TrainingMode
        from .fine_tuned_models import FineTunedModelLibrary
        
        # Get the fine-tuned model configuration
        library = FineTunedModelLibrary()
        fine_tuned_model = library.get_model(model)
        
        if not fine_tuned_model:
            click.echo(f"Error: Unknown model: {model}", err=True)
            return
        
        # Convert string mode to enum
        mode_enum = TrainingMode(mode)
        
        # Create training configuration using fine-tuned parameters
        config = TrainingConfig(
            mode=mode_enum,
            num_games=games,
            ai_types=fine_tuned_model.ai_types,
            risk_ratios=fine_tuned_model.risk_ratios,
            output_dir=output_dir,
            quiet_mode=True
        )
        
        click.echo(f"🎯 ADAPTIVE TRAINING WITH FINE-TUNED MODEL")
        click.echo(f"Model: {fine_tuned_model.name}")
        click.echo(f"Expected Win Rate: {fine_tuned_model.expected_win_rate:.1f}%")
        click.echo(f"AI Types: {', '.join(fine_tuned_model.ai_types)}")
        click.echo(f"Risk Ratios: {', '.join(map(str, fine_tuned_model.risk_ratios))}")
        click.echo(f"Games: {games}")
        click.echo("-" * 60)
        
        # Create and run training framework
        framework = AITrainingFramework(config)
        results = framework.run_training_session()
        
        # Display summary
        click.echo(f"\n🎯 ADAPTIVE TRAINING COMPLETED! 🎯")
        click.echo(f"Model: {fine_tuned_model.name}")
        click.echo(f"Mode: {mode}")
        click.echo(f"Games: {games}")
        click.echo(f"Results saved to: {output_dir}/")
        
        # Show key statistics
        overall = results["overall_stats"]
        click.echo(f"\n📊 OVERALL STATISTICS:")
        click.echo(f"Team 1 Wins: {overall['team1_wins']} ({overall['team1_wins']/overall['total_games']*100:.1f}%)")
        click.echo(f"Team 2 Wins: {overall['team2_wins']} ({overall['team2_wins']/overall['total_games']*100:.1f}%)")
        click.echo(f"Team Sets: {overall['total_team_sets']}")
        click.echo(f"Total Reneges: {overall['total_reneges']}")
        
        # Compare with expected performance
        actual_win_rate = overall['team1_wins'] / overall['total_games'] * 100
        expected_win_rate = fine_tuned_model.expected_win_rate
        
        click.echo(f"\n📈 PERFORMANCE COMPARISON:")
        click.echo(f"Expected Win Rate: {expected_win_rate:.1f}%")
        click.echo(f"Actual Win Rate: {actual_win_rate:.1f}%")
        click.echo(f"Difference: {actual_win_rate - expected_win_rate:+.1f}%")
        
        if actual_win_rate >= expected_win_rate:
            click.echo("✅ Performance meets or exceeds expectations!")
        else:
            click.echo("⚠️  Performance below expectations - may need further tuning")
        
    except ImportError as e:
        click.echo(f"Error: Could not import required modules: {e}", err=True)
    except Exception as e:
        click.echo(f"Error during adaptive training: {e}", err=True)


@click.command()
@click.option('--player-name', default='You', help='Your player name')
@click.option('--partner-ai', default='balanced', help='AI partner type: conservative, balanced, aggressive, ace_hunter, trump_caller')
@click.option('--opponent1-ai', default='balanced', help='First opponent AI type')
@click.option('--opponent2-ai', default='balanced', help='Second opponent AI type')
@click.option('--opponent3-ai', default='balanced', help='Third opponent AI type')
@click.option('--show-ai-hands', is_flag=True, default=False, help='Show AI player hands (for debugging)')
def single_player(player_name, partner_ai, opponent1_ai, opponent2_ai, opponent3_ai, show_ai_hands):
    """Play a single player euchre game against AI opponents."""
    from euchre.game import EuchreGame
    from euchre.ai_profiles import create_ai_profile
    
    click.echo(f"🎮 Starting Single Player Euchre Game")
    click.echo(f"👤 Player: {player_name}")
    click.echo(f"🤝 Partner AI: {partner_ai}")
    click.echo(f"👥 Opponents: {opponent1_ai}, {opponent2_ai}, {opponent3_ai}")
    click.echo("=" * 60)
    
    # Create game
    game = EuchreGame(enable_logging=True, quiet_mode=False)
    
    # Add human player
    from euchre.player import Player, PlayerType
    human_player = Player(player_name, PlayerType.HUMAN)
    game.add_player(human_player)
    
    # Add AI partner (same team as human)
    partner = create_ai_profile(f"Partner_{partner_ai.title()}", partner_ai)
    game.add_player(partner)
    
    # Add AI opponents (opposite team)
    opponent1 = create_ai_profile(f"Opponent1_{opponent1_ai.title()}", opponent1_ai)
    opponent2 = create_ai_profile(f"Opponent2_{opponent2_ai.title()}", opponent2_ai)
    game.add_player(opponent1)
    game.add_player(opponent2)
    
    # Start the game
    game.start_new_game()
    
    # Run the interactive game
    game.run_interactive_game(show_ai_hands=show_ai_hands)


# Add commands to the main group
main.add_command(train_self_play)
main.add_command(list_players)
main.add_command(play_trained_players)
main.add_command(tournament)
main.add_command(benchmark)
main.add_command(run_neural_games)
main.add_command(run_all_neural_combinations)
main.add_command(list_neural_models)
main.add_command(train_integer_vs_float)
main.add_command(generate_player_profiles)
main.add_command(play_fine_tuned)
main.add_command(list_models)
main.add_command(find_optimal_model)
main.add_command(train_adaptive)
main.add_command(single_player)


if __name__ == "__main__":
    main() 