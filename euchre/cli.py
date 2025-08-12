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
@click.option("--player-name", "-n", default="Player", help="Your player name")
@click.option("--ai-names", "-a", multiple=True, 
              default=["Alice", "Bob", "Charlie"], 
              help="Names for AI opponents")
def play(player_name: str, ai_names: tuple) -> None:
    """Start a new euchre game."""
    click.echo(f"Welcome to Euchre, {player_name}!")
    
    # Create game
    game = EuchreGame()
    
    # Add human player
    game.add_player(player_name, PlayerType.HUMAN)
    
    # Add AI players
    ai_names_list = list(ai_names)
    while len(game.players) < 4:
        if len(ai_names_list) > 0:
            game.add_player(ai_names_list.pop(0), PlayerType.AI)
        else:
            game.add_player(f"AI_{len(game.players)}", PlayerType.AI)
    
    click.echo(f"Players: {', '.join(p.name for p in game.players)}")
    
    try:
        game.start_new_game()
        click.echo("Game started! Dealing cards...")
        
        # Show human player's hand
        human_hand = game.get_player_hand(player_name)
        click.echo(f"\nYour hand:")
        for i, card in enumerate(human_hand, 1):
            click.echo(f"  {i}. {card}")
            
        # Simple game loop for demonstration
        click.echo("\nGame is running... (This is a skeleton implementation)")
        click.echo("In a full implementation, you would play through the game here.")
        
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
        player_names = ["North", "East", "South", "West"]
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


if __name__ == "__main__":
    main() 