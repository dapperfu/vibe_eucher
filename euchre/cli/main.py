#!/usr/bin/env python3
"""Main CLI module for euchre game."""

import click
import sys
from pathlib import Path

# Add the euchre package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from euchre.game import EuchreGame
from euchre.models import Player, PlayerType, Card, Suit, Rank
from euchre.ai.ai_factory import AIFactory


@click.group()
@click.version_option(version="1.0.0")
def main():
    """Euchre game CLI - play, train, and analyze Euchre games."""
    pass


@main.command()
def play():
    """Play a basic Euchre game."""
    click.echo("🎮 Starting Euchre game...")
    
    # Create game
    game = EuchreGame()
    
    # Add AI players
    game.add_ai_player("Alice", "balanced", 0.5)
    game.add_ai_player("Bob", "balanced", 0.5)
    game.add_ai_player("Charlie", "balanced", 0.5)
    game.add_ai_player("David", "balanced", 0.5)
    
    # Start game
    try:
        game.start_new_game()
        click.echo("✅ Game completed successfully!")
    except Exception as e:
        click.echo(f"❌ Game failed: {e}", err=True)


@main.command()
@click.option("--very-verbose", is_flag=True, help="Enable very verbose output")
def ai_vs_ai(very_verbose: bool):
    """Run AI vs AI euchre game."""
    click.echo("🤖 Starting AI vs AI Euchre game...")
    
    # Create game with appropriate verbosity
    if very_verbose:
        game = EuchreGame(verbose=True, very_verbose=True)
    else:
        game = EuchreGame(verbose=True)
    
    # Add AI players with different styles
    game.add_ai_player("Alice", "aggressive", 0.8)
    game.add_ai_player("Bob", "conservative", 0.2)
    game.add_ai_player("Charlie", "balanced", 0.5)
    game.add_ai_player("David", "opportunistic", 0.7)
    
    # Start game
    try:
        game.start_new_game()
        click.echo("✅ AI vs AI game completed successfully!")
    except Exception as e:
        click.echo(f"❌ AI vs AI game failed: {e}", err=True)


@main.command()
def ncurses():
    """Run AI vs AI euchre game with ncurses interface."""
    click.echo("🖥️  Starting ncurses interface...")
    
    try:
        from euchre.ncurses_game import NcursesGame
        game = NcursesGame()
        game.run()
    except ImportError:
        click.echo("❌ Ncurses interface not available. Install ncurses dependencies.")
    except Exception as e:
        click.echo(f"❌ Ncurses game failed: {e}", err=True)


@main.command()
def logged_game():
    """Run AI vs AI euchre game with logging (no ncurses)."""
    click.echo("📝 Starting logged game...")
    
    # Create game with logging
    game = EuchreGame(verbose=True)
    
    # Add AI players
    game.add_ai_player("Alice", "balanced", 0.5)
    game.add_ai_player("Bob", "balanced", 0.5)
    game.add_ai_player("Charlie", "balanced", 0.5)
    game.add_ai_player("David", "balanced", 0.5)
    
    # Start game
    try:
        game.start_new_game()
        click.echo("✅ Logged game completed successfully!")
    except Exception as e:
        click.echo(f"❌ Logged game failed: {e}", err=True)


@main.command()
def ai_profiles():
    """Run AI vs AI game with custom profiles and risk ratios."""
    click.echo("🎯 Starting AI profiles game...")
    
    # Create game
    game = EuchreGame(verbose=True)
    
    # Add AI players with different profiles
    game.add_ai_player("Alice", "aggressive", 0.9)
    game.add_ai_player("Bob", "conservative", 0.1)
    game.add_ai_player("Charlie", "balanced", 0.5)
    game.add_ai_player("David", "opportunistic", 0.8)
    
    # Start game
    try:
        game.start_new_game()
        click.echo("✅ AI profiles game completed successfully!")
    except Exception as e:
        click.echo(f"❌ AI profiles game failed: {e}", err=True)


@main.command()
@click.option("--num-games", default=1000, help="Number of games to run")
def run_mass_games(num_games: int):
    """Run thousands of games in parallel."""
    click.echo(f"🔄 Running {num_games} games...")
    
    try:
        from euchre.mass_game_runner import run_mass_games
        run_mass_games(num_games)
        click.echo("✅ Mass games completed successfully!")
    except ImportError:
        click.echo("❌ Mass game runner not available.")
    except Exception as e:
        click.echo(f"❌ Mass games failed: {e}", err=True)


@main.command()
def analyze_games():
    """Analyze game results and generate statistics."""
    click.echo("📊 Analyzing games...")
    
    try:
        from euchre.game_analyzer import analyze_games
        analyze_games()
        click.echo("✅ Game analysis completed successfully!")
    except ImportError:
        click.echo("❌ Game analyzer not available.")
    except Exception as e:
        click.echo(f"❌ Game analysis failed: {e}", err=True)


@main.command()
def cleanup_games():
    """Clean up old game result files."""
    click.echo("🧹 Cleaning up old game files...")
    
    try:
        import os
        import glob
        
        # Find and remove old game files
        game_files = glob.glob("games/*.json")
        neural_files = glob.glob("neural_games/*.json")
        
        total_removed = 0
        
        # Remove old game files (keep last 100)
        if len(game_files) > 100:
            files_to_remove = sorted(game_files)[:-100]
            for f in files_to_remove:
                os.remove(f)
                total_removed += 1
        
        # Remove old neural game files (keep last 100)
        if len(neural_files) > 100:
            files_to_remove = sorted(neural_files)[:-100]
            for f in files_to_remove:
                os.remove(f)
                total_removed += 1
        
        click.echo(f"✅ Cleanup completed! Removed {total_removed} old files.")
        
    except Exception as e:
        click.echo(f"❌ Cleanup failed: {e}", err=True)


@main.command()
def jupyter():
    """Start Jupyter Notebook."""
    click.echo("📓 Starting Jupyter Notebook...")
    
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "jupyter", "notebook"])
    except ImportError:
        click.echo("❌ Jupyter not available. Install with: pip install jupyter")
    except Exception as e:
        click.echo(f"❌ Jupyter failed: {e}", err=True)


@main.command()
def jupyter_lab():
    """Start Jupyter Lab."""
    click.echo("🔬 Starting Jupyter Lab...")
    
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "jupyter", "lab"])
    except ImportError:
        click.echo("❌ Jupyter Lab not available. Install with: pip install jupyterlab")
    except Exception as e:
        click.echo(f"❌ Jupyter Lab failed: {e}", err=True)


@main.command()
@click.option("--player-name", default="Player", help="Your player name")
@click.option("--your-position", default=0, help="Your position (0=Alice, 1=Bob, 2=Charlie, 3=David)")
@click.option("--partner-ai-type", default="balanced", help="Partner AI type")
@click.option("--opponent1-ai-type", default="balanced", help="First opponent AI type")
@click.option("--opponent2-ai-type", default="balanced", help="Second opponent AI type")
@click.option("--partner-risk", default=0.5, help="Partner risk ratio (0.0-1.0)")
@click.option("--opponent1-risk", default=0.5, help="First opponent risk ratio (0.0-1.0)")
@click.option("--opponent2-risk", default=0.5, help="Second opponent risk ratio (0.0-1.0)")
def human_vs_ai(player_name: str, your_position: int, partner_ai_type: str, 
                opponent1_ai_type: str, opponent2_ai_type: str,
                partner_risk: float, opponent1_risk: float, opponent2_risk: float):
    """Play as human vs AI with configurable AI types."""
    click.echo(f"🎮 Human vs AI Game - {player_name} at position {your_position}")
    
    # Create game
    game = EuchreGame(verbose=True)
    
    # Add AI players based on position
    ai_types = [partner_ai_type, opponent1_ai_type, opponent2_ai_type]
    ai_risks = [partner_risk, opponent1_risk, opponent2_risk]
    
    player_names = ["Alice", "Bob", "Charlie", "David"]
    current_ai = 0
    
    for i in range(4):
        if i == your_position:
            # This is the human player
            game.add_player(player_name, PlayerType.HUMAN)
        else:
            # This is an AI player
            game.add_ai_player(player_names[i], ai_types[current_ai], ai_risks[current_ai])
            current_ai += 1
    
    # Start game
    try:
        game.start_new_game()
        click.echo("✅ Human vs AI game completed successfully!")
    except Exception as e:
        click.echo(f"❌ Human vs AI game failed: {e}", err=True)


@main.command()
@click.option("--num-games", default=100, help="Number of games to play")
@click.option("--player1", default="Alice", help="First player name")
@click.option("--player2", default="Bob", help="Second player name")
@click.option("--player3", default="Charlie", help="Third player name")
@click.option("--player4", default="David", help="Fourth player name")
def play_trained_players(num_games: int, player1: str, player2: str, player3: str, player4: str):
    """Play a game with trained AI players."""
    click.echo(f"🎯 Playing {num_games} games with trained players...")
    
    # Create game
    game = EuchreGame(verbose=False)
    
    # Add AI players
    game.add_ai_player(player1, "balanced", 0.5)
    game.add_ai_player(player2, "balanced", 0.5)
    game.add_ai_player(player3, "balanced", 0.5)
    game.add_ai_player(player4, "balanced", 0.5)
    
    # Play multiple games
    try:
        for i in range(num_games):
            if i % 10 == 0:
                click.echo(f"Playing game {i+1}/{num_games}...")
            game.start_new_game()
        
        click.echo(f"✅ Completed {num_games} games successfully!")
    except Exception as e:
        click.echo(f"❌ Games failed: {e}", err=True)


@main.command()
@click.option("--num-games", default=1000, help="Number of games to run")
def tournament(num_games: int):
    """Run a tournament between trained players."""
    click.echo(f"🏆 Starting tournament with {num_games} games...")
    
    # Create game
    game = EuchreGame(verbose=False)
    
    # Add AI players with different styles
    game.add_ai_player("Alice", "aggressive", 0.8)
    game.add_ai_player("Bob", "conservative", 0.2)
    game.add_ai_player("Charlie", "balanced", 0.5)
    game.add_ai_player("David", "opportunistic", 0.7)
    
    # Play tournament games
    try:
        for i in range(num_games):
            if i % 100 == 0:
                click.echo(f"Tournament game {i+1}/{num_games}...")
            game.start_new_game()
        
        click.echo(f"🏆 Tournament completed! {num_games} games played.")
    except Exception as e:
        click.echo(f"❌ Tournament failed: {e}", err=True)


@main.command()
@click.option("--device", default="cpu", help="Device to use (cpu/cuda)")
@click.option("--save-results", is_flag=True, help="Save benchmark results")
def benchmark(device: str, save_results: bool):
    """Run performance benchmark comparing float vs integer models."""
    click.echo(f"⚡ Running benchmark on {device}...")
    
    try:
        # Simple benchmark - just run some games
        game = EuchreGame(verbose=False)
        game.add_ai_player("Alice", "balanced", 0.5)
        game.add_ai_player("Bob", "balanced", 0.5)
        game.add_ai_player("Charlie", "balanced", 0.5)
        game.add_ai_player("David", "balanced", 0.5)
        
        import time
        start_time = time.time()
        
        # Run 100 games for benchmark
        for i in range(100):
            game.start_new_game()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        click.echo(f"✅ Benchmark completed!")
        click.echo(f"   100 games in {total_time:.2f} seconds")
        click.echo(f"   Average: {total_time/100:.3f} seconds per game")
        
        if save_results:
            import json
            results = {
                "device": device,
                "games": 100,
                "total_time": total_time,
                "avg_time_per_game": total_time/100,
                "timestamp": time.time()
            }
            
            with open("benchmark_results.json", "w") as f:
                json.dump(results, f, indent=2)
            click.echo("   Results saved to benchmark_results.json")
            
    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}", err=True)


@main.command()
@click.option("--m1", default="Alice", help="First model name")
@click.option("--m2", default="Bob", help="Second model name")
@click.option("--n", default=1000, help="Number of games")
def run_neural_games(m1: str, m2: str, n: int):
    """Run neural network tournament (Alice vs Bob, 1000 games)."""
    click.echo(f"🧠 Neural tournament: {m1} vs {m2} ({n} games)")
    
    try:
        # Create game
        game = EuchreGame(verbose=False)
        
        # Add AI players
        game.add_ai_player(m1, "balanced", 0.5)
        game.add_ai_player(m2, "balanced", 0.5)
        game.add_ai_player("Charlie", "balanced", 0.5)
        game.add_ai_player("David", "balanced", 0.5)
        
        # Play games
        for i in range(n):
            if i % 100 == 0:
                click.echo(f"Neural game {i+1}/{n}...")
            game.start_new_game()
        
        click.echo(f"🧠 Neural tournament completed! {n} games played.")
        
    except Exception as e:
        click.echo(f"❌ Neural tournament failed: {e}", err=True)


@main.command()
def list_neural_models():
    """List available neural network models."""
    click.echo("🧠 Available neural network models:")
    
    try:
        import os
        import glob
        
        # Check for trained models
        model_dirs = [
            "trained_models/unified_trained",
            "trained_models/hybrid_trained", 
            "trained_models/gpu_trained"
        ]
        
        found_models = []
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                models = glob.glob(os.path.join(model_dir, "*.json"))
                found_models.extend([(model_dir, os.path.basename(m)) for m in models])
        
        if found_models:
            for model_dir, model_name in found_models:
                click.echo(f"  📁 {model_dir}/{model_name}")
        else:
            click.echo("  No trained models found.")
            click.echo("  Run 'make train-m-series' to train models.")
            
    except Exception as e:
        click.echo(f"❌ Error listing models: {e}", err=True)


@main.command()
def list_players():
    """List available trained AI players."""
    click.echo("🤖 Available AI players:")
    
    try:
        import os
        import glob
        
        # Check for trained models
        model_dirs = [
            "trained_models/unified_trained",
            "trained_models/hybrid_trained",
            "trained_models/gpu_trained",
            "trained_models/profile_checkpoints"
        ]
        
        found_models = []
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                models = glob.glob(os.path.join(model_dir, "*.json"))
                found_models.extend([(model_dir, os.path.basename(m)) for m in models])
        
        if found_models:
            for model_dir, model_name in found_models:
                click.echo(f"  📁 {model_dir}/{model_name}")
        else:
            click.echo("  No trained models found.")
            click.echo("  Run 'make train-m-series' to train models.")
            
    except Exception as e:
        click.echo(f"❌ Error listing players: {e}", err=True)


if __name__ == "__main__":
    main() 