#!/usr/bin/env python3
"""Main CLI module for euchre game."""

import click
import sys
from pathlib import Path
from typing import Optional

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
        game.run_full_game()
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
        game.run_full_game()
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
@click.option("--position", default=2, help="Human player position (0=North, 1=East, 2=South, 3=West)")
def ncurses_human_vs_ai(position: int):
    """Run human vs AI euchre game with ncurses interface."""
    click.echo(f"🎮 Starting human vs AI ncurses game (you at position {position})...")
    click.echo("Positions: 0=North, 1=East, 2=South (default), 3=West")
    
    try:
        from euchre.ncurses_game import NcursesGame
        game = NcursesGame(human_player_position=position)
        game.run()
    except ImportError:
        click.echo("❌ Ncurses interface not available. Install ncurses dependencies.")
    except Exception as e:
        click.echo(f"❌ Human vs AI ncurses game failed: {e}", err=True)


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
        game.run_full_game()
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
@click.argument("player_name", default="You")
@click.option("--your-position", default=0, help="Your position (0-3)")
@click.option("--partner-ai-type", default="level1_balanced", help="Partner AI type")
@click.option("--opponent1-ai-type", default="level1_aggressive", help="First opponent AI type")
@click.option("--opponent2-ai-type", default="level1_conservative", help="Second opponent AI type")
@click.option("--partner-risk", default=0.5, help="Partner risk ratio (0.0-1.0)")
@click.option("--opponent1-risk", default=0.5, help="First opponent risk ratio (0.0-1.0)")
@click.option("--opponent2-risk", default=0.5, help="Second opponent risk ratio (0.0-1.0)")
@click.option("--partner-model-path", help="Path to trained Level 2 model for partner")
@click.option("--opponent1-model-path", help="Path to trained Level 2 model for opponent1")
@click.option("--opponent2-model-path", help="Path to trained Level 2 model for opponent2")
def human_vs_ai(player_name: str, your_position: int, partner_ai_type: str, 
                opponent1_ai_type: str, opponent2_ai_type: str,
                partner_risk: float, opponent1_risk: float, opponent2_risk: float,
                partner_model_path: Optional[str] = None, opponent1_model_path: Optional[str] = None, 
                opponent2_model_path: Optional[str] = None):
    """Play as human vs AI with configurable AI types (including Level 1 and Level 2 models)."""
    click.echo(f"🎮 Human vs AI Game - {player_name} at position {your_position}")
    
    # Show available AI types
    from euchre.ai.ai_factory import AIFactory
    available_types = AIFactory.get_available_ai_types()
    click.echo(f"Available AI types: {', '.join(available_types)}")
    
    # Validate AI types
    for ai_type in [partner_ai_type, opponent1_ai_type, opponent2_ai_type]:
        if ai_type.lower() not in available_types:
            click.echo(f"❌ Unknown AI type: {ai_type}")
            click.echo(f"Available types: {', '.join(available_types)}")
            return
    
    # Create game
    game = EuchreGame(verbose=True)
    
    # Add AI players based on position
    ai_types = [partner_ai_type, opponent1_ai_type, opponent2_ai_type]
    ai_risks = [partner_risk, opponent1_risk, opponent2_risk]
    model_paths = [partner_model_path, opponent1_model_path, opponent2_model_path]
    
    player_names = ["Alice", "Bob", "Charlie", "David"]
    current_ai = 0
    
    for i in range(4):
        if i == your_position:
            # This is the human player
            game.add_player(player_name, PlayerType.HUMAN)
        else:
            # This is an AI player
            ai_type = ai_types[current_ai]
            risk = ai_risks[current_ai]
            model_path = model_paths[current_ai]
            
            # Check if this is a Level 1 AI type
            if AIFactory.is_level1_type(ai_type):
                click.echo(f"🤖 Adding Level 1 {ai_type} AI player: {player_names[i]}")
            # Check if this is a Level 2 model
            elif AIFactory.is_level2_type(ai_type):
                click.echo(f"🤖 Adding Level 2 {ai_type} AI player: {player_names[i]}")
                if not model_path:
                    click.echo("⚠️  No model path provided - using untrained Level 2 model")
            else:
                click.echo(f"🤖 Adding {ai_type} AI player: {player_names[i]}")
            
            game.add_ai_player(player_names[i], ai_type, risk)
            current_ai += 1
    
    # Start game
    try:
        game.start_new_game()
        click.echo("✅ Human vs AI game completed successfully!")
    except Exception as e:
        click.echo(f"❌ Human vs AI game failed: {e}", err=True)


@main.command()
@click.argument("player_name", default="You")
@click.option("--your-position", default=0, help="Your position (0-3)")
@click.option("--level1-model", default="level1_balanced",
              help="Level 1 AI model to play against")
@click.option("--ai-risk", default=0.5, help="AI risk ratio (0.0-1.0)")
def human_vs_level1(player_name: str, your_position: int, level1_model: str,
                    ai_risk: float = 0.5):
    """Play as human vs Level 1 AI models specifically."""
    click.echo(f"🧠 Human vs Level 1 AI Game")
    click.echo(f"👤 Player: {player_name} at position {your_position}")
    click.echo(f"🤖 Level 1 Model: {level1_model}")
    click.echo(f"🎯 AI Risk Level: {ai_risk}")
    click.echo(f"Positions: 0=North, 1=East, 2=South (default), 3=West")
    
    # Create game
    game = EuchreGame(verbose=True)
    
    # Add players
    player_names = ["Alice", "Bob", "Charlie", "David"]
    
    for i in range(4):
        if i == your_position:
            # This is the human player
            game.add_player(player_name, PlayerType.HUMAN)
            click.echo(f"👤 Added human player: {player_name} at position {i}")
        else:
            # This is a Level 1 AI player
            if AIFactory.is_level1_type(ai_type):
                ai_name = player_names[i]
                game.add_ai_player(ai_name, level1_model, ai_risk)
                click.echo(f"🤖 Added Level 1 {level1_model} AI: {ai_name} at position {i}")
    
    # Start game
    try:
        click.echo("\n🎮 Starting game...")
        game.start_new_game()
        click.echo("✅ Human vs Level 1 AI game completed successfully!")
    except Exception as e:
        click.echo(f"❌ Game failed: {e}", err=True)


@main.command()
@click.argument("player_name", default="You")
@click.option("--your-position", default=0, help="Your position (0-3)")
@click.option("--level2-model", default="level2_strategic",
              help="Level 2 AI model to play against")
@click.option("--model-path", help="Path to trained Level 2 model file (.pth)")
@click.option("--ai-risk", default=0.5, help="AI risk ratio (0.0-1.0)")
def human_vs_level2(player_name: str, your_position: int, level2_model: str,
                    model_path: Optional[str] = None, ai_risk: float = 0.5):
    """Play as human vs Level 2 AI models specifically."""
    click.echo(f"🧠 Human vs Level 2 AI Game")
    click.echo(f"👤 Player: {player_name} at position {your_position}")
    click.echo(f"🤖 Level 2 Model: {level2_model}")
    
    if not model_path:
        click.echo("⚠️  Using untrained Level 2 model")
    
    click.echo(f"🎯 AI Risk Level: {ai_risk}")
    
    # Create game
    game = EuchreGame(verbose=True)
    
    # Add players
    player_names = ["Alice", "Bob", "Charlie", "David"]
    
    for i in range(4):
        if i == your_position:
            # This is the human player
            game.add_player(player_name, PlayerType.HUMAN)
            click.echo(f"👤 Added human player: {player_name} at position {i}")
        else:
            # This is a Level 2 AI player
            ai_name = player_names[i]
            game.add_ai_player(ai_name, level2_model, ai_risk)
            click.echo(f"🤖 Added Level 2 {level2_model} AI: {ai_name} at position {i}")
    
    # Start game
    try:
        click.echo("\n🎮 Starting game...")
        game.start_new_game()
        click.echo("✅ Human vs Level 2 AI game completed successfully!")
    except Exception as e:
        click.echo(f"❌ Game failed: {e}", err=True)


@main.command()
def list_ai_types():
    """List all available AI types including M-Series models."""
    from euchre.ai.ai_factory import AIFactory
    
    click.echo("🤖 Available AI Types:")
    click.echo("=" * 40)
    
    # Traditional AI types
    click.echo("📚 Traditional Rule-Based AI:")
    traditional_types = ["aggressive", "conservative", "balanced", "opportunistic"]
    for ai_type in traditional_types:
        click.echo(f"  • {ai_type}")
    
    # Level 1 AI types
    click.echo("\n🤖 Level 1 Rule-Based AI Models:")
    level1_types = ["level1_aggressive", "level1_conservative", "level1_balanced", "level1_opportunistic"]
    for ai_type in level1_types:
        click.echo(f"  • {ai_type}")
    click.echo("💡 Level 1 models use traditional rule-based logic with configurable risk profiles")
    
    # Level 2 AI types
    if hasattr(AIFactory, 'LEVEL2_AVAILABLE') and AIFactory.LEVEL2_AVAILABLE:
        click.echo("\n🧠 Level 2 Neural AI Models:")
        level2_types = ["level2_strategic", "level2_aggressive", "level2_balanced", "level2_intuitive"]
        for ai_type in level2_types:
            click.echo(f"  • {ai_type}")
        click.echo("\n💡 Level 2 models can be trained and loaded from .pth files")
    else:
        click.echo("\n❌ Level 2 models not available")
        click.echo("   Install PyTorch and Level 2 dependencies to enable")
    
    click.echo("\n🎮 Usage Examples:")
    click.echo("  • Play vs Level 1: euchre human-vs-level1 --level1-model level1_aggressive")
    click.echo("  • Play vs Level 2: euchre human-vs-level2 --level2-model level2_strategic")
    click.echo("  • Mix AI types: euchre human-vs-ai --partner-ai-type level1_balanced --opponent1-ai-type level2_aggressive")


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
            click.echo("  Run 'make train-level2' to train models.")
            
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
            click.echo("  Run 'make train-level2' to train models.")
            
    except Exception as e:
        click.echo(f"❌ Error listing players: {e}", err=True)


if __name__ == "__main__":
    main() 