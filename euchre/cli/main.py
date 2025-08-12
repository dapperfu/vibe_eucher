"""Main CLI entry point for the euchre game."""

import click
from .commands import GameCommands


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging (INFO level)")
@click.option("--very-verbose", "-vv", is_flag=True, help="Enable very verbose logging (DEBUG level)")
@click.version_option(version="0.1.0")
@click.pass_context
def main(ctx: click.Context, verbose: bool, very_verbose: bool) -> None:
    """Euchre - A CLI card game with AI opponents.
    
    Play the classic euchre card game against AI opponents.
    """
    # Store verbosity flags in context
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['very_verbose'] = very_verbose


@main.command()
@click.option("--player-name", "-n", default=None, help="Your player name (omit for AI-only game)")
@click.option("--ai-names", "-a", multiple=True, 
              default=["Alice", "Bob", "Charlie", "David"], 
              help="Names for AI opponents")
@click.pass_context
def play(ctx: click.Context, player_name: str, ai_names: tuple) -> None:
    """Start a new euchre game."""
    verbose = ctx.obj.get('verbose', False)
    very_verbose = ctx.obj.get('very_verbose', False)
    GameCommands.play_game(player_name, ai_names, verbose, very_verbose)


@main.command()
@click.option("--ai-profiles", "-p", multiple=True, 
              default=["balanced", "balanced", "balanced", "balanced"],
              help="AI profiles: aggressive, conservative, balanced, opportunistic")
@click.option("--risk-ratios", "-r", multiple=True, 
              default=["0.5", "0.5", "0.5", "0.5"],
              help="Risk ratios (0.0-1.0) for each AI player")
@click.pass_context
def ai_profiles(ctx: click.Context, ai_profiles: tuple, risk_ratios: tuple) -> None:
    """Run a game with different AI profiles and risk ratios."""
    verbose = ctx.obj.get('verbose', False)
    very_verbose = ctx.obj.get('very_verbose', False)
    GameCommands.ai_profiles_game(ai_profiles, risk_ratios, verbose, very_verbose)


@main.command()
@click.option("--dealer-method", "-d", 
              type=click.Choice(["black_jack", "high_card"]), 
              default="black_jack",
              help="Dealer selection method: black_jack or high_card")
@click.pass_context
def ai_vs_ai(ctx: click.Context, dealer_method: str) -> None:
    """Run AI vs AI euchre game."""
    verbose = ctx.obj.get('verbose', False)
    very_verbose = ctx.obj.get('very_verbose', False)
    GameCommands.ai_vs_ai_game(verbose, very_verbose, dealer_method)


@main.command()
@click.option("--player-name", "-n", default="You", help="Your player name")
@click.option("--your-position", "-p", 
              type=click.Choice(["0", "1", "2", "3"]), 
              default="0", 
              help="Your position: 0=Alice, 1=Bob, 2=Charlie, 3=David")
@click.option("--partner-ai-type", "-pa", 
              type=click.Choice(["aggressive", "conservative", "balanced", "opportunistic"]),
              default="balanced",
              help="AI type for your partner")
@click.option("--opponent1-ai-type", "-o1", 
              type=click.Choice(["aggressive", "conservative", "balanced", "opportunistic"]),
              default="balanced",
              help="AI type for first opponent")
@click.option("--opponent2-ai-type", "-o2", 
              type=click.Choice(["aggressive", "conservative", "balanced", "opportunistic"]),
              default="balanced",
              help="AI type for second opponent")
@click.option("--partner-risk", "-pr", 
              type=float, default=0.5,
              help="Risk ratio for your partner (0.0-1.0)")
@click.option("--opponent1-risk", "-or1", 
              type=float, default=0.5,
              help="Risk ratio for first opponent (0.0-1.0)")
@click.option("--opponent2-risk", "-or2", 
              type=float, default=0.5,
              help="Risk ratio for second opponent (0.0-1.0)")
@click.pass_context
def human_vs_ai(ctx: click.Context, player_name: str, your_position: str, 
                partner_ai_type: str, opponent1_ai_type: str, opponent2_ai_type: str,
                partner_risk: float, opponent1_risk: float, opponent2_risk: float) -> None:
    """Play euchre as a human against AI opponents with a specified AI partner."""
    verbose = ctx.obj.get('verbose', False)
    very_verbose = ctx.obj.get('very_verbose', False)
    GameCommands.human_vs_ai_game(
        player_name, int(your_position), 
        partner_ai_type, opponent1_ai_type, opponent2_ai_type,
        partner_risk, opponent1_risk, opponent2_risk,
        verbose, very_verbose
    )


@main.command()
def ncurses() -> None:
    """Run AI vs AI euchre game with ncurses interface."""
    try:
        from ..ncurses_game import run_ncurses_game
        run_ncurses_game()
    except ImportError:
        click.echo("Ncurses interface not available. Install required dependencies.")
    except Exception as e:
        click.echo(f"Error running ncurses game: {e}")


@main.command()
@click.pass_context
def logged_game(ctx: click.Context) -> None:
    """Run AI vs AI euchre game with logging (no ncurses)."""
    verbose = ctx.obj.get('verbose', False)
    very_verbose = ctx.obj.get('very_verbose', False)
    GameCommands.ai_vs_ai_game(verbose, very_verbose)


@main.command()
@click.option("--num-games", "-n", default=100, help="Number of games to play")
@click.pass_context
def tournament(ctx: click.Context, num_games: int) -> None:
    """Run a tournament between trained players."""
    verbose = ctx.obj.get('verbose', False)
    very_verbose = ctx.obj.get('very_verbose', False)
    GameCommands.tournament_game(num_games, verbose, very_verbose)


@main.command()
@click.option("--num-games", "-n", default=1000, help="Number of games to run")
@click.option("--model1", "-m1", default="Alice", help="First model name")
@click.option("--model2", "-m2", default="Bob", help="Second model name")
def run_neural_games(num_games: int, model1: str, model2: str) -> None:
    """Run neural network tournament (Alice vs Bob, 1000 games)."""
    try:
        from ..neural_mass_game_runner import run_neural_tournament
        run_neural_tournament(model1, model2, num_games)
    except ImportError:
        click.echo("Neural game runner not available.")
    except Exception as e:
        click.echo(f"Error running neural tournament: {e}")


@main.command()
def list_players() -> None:
    """List available trained AI players."""
    try:
        from ..ai.ai_factory import AIFactory
        ai_types = AIFactory.get_available_ai_types()
        click.echo("Available AI types:")
        for ai_type in ai_types:
            click.echo(f"  - {ai_type}")
    except Exception as e:
        click.echo(f"Error listing players: {e}")


@main.command()
def list_neural_models() -> None:
    """List available neural network models."""
    try:
        import os
        import glob
        from pathlib import Path
        
        click.echo("=== Available Neural Network Models ===\n")
        
        # Check for models in different directories
        model_dirs = [
            "models",
            "trained_models", 
            "trained_models/mixed_training_checkpoints",
            "trained_models/profile_checkpoints"
        ]
        
        total_models = 0
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                click.echo(f"📁 {model_dir}/")
                
                # Look for .pth files (PyTorch models)
                pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
                if pth_files:
                    for pth_file in sorted(pth_files):
                        file_size = os.path.getsize(pth_file)
                        size_mb = file_size / (1024 * 1024)
                        click.echo(f"  🧠 {os.path.basename(pth_file)} ({size_mb:.1f} MB)")
                        total_models += 1
                
                # Look for .json files (trained player profiles)
                json_files = glob.glob(os.path.join(model_dir, "*.json"))
                if json_files:
                    for json_file in sorted(json_files):
                        file_size = os.path.getsize(json_file)
                        size_mb = file_size / (1024 * 1024)
                        click.echo(f"  📊 {os.path.basename(json_file)} ({size_mb:.1f} MB)")
                        total_models += 1
                
                # Look for checkpoint files in subdirectories
                checkpoint_patterns = [
                    "*_checkpoint_*.json",
                    "*_checkpoint_*.pth"
                ]
                
                for pattern in checkpoint_patterns:
                    checkpoint_files = glob.glob(os.path.join(model_dir, pattern))
                    if checkpoint_files:
                        for checkpoint_file in sorted(checkpoint_files):
                            file_size = os.path.getsize(checkpoint_file)
                            size_mb = file_size / (1024 * 1024)
                            rel_path = os.path.relpath(checkpoint_file, model_dir)
                            click.echo(f"  🔄 {rel_path} ({size_mb:.1f} MB)")
                            total_models += 1
                
                click.echo()
        
        if total_models == 0:
            click.echo("❌ No neural network models found.")
            click.echo("\nTo create models, use:")
            click.echo("  euchre generate-player-profiles")
            click.echo("  euchre train-self-play")
        else:
            click.echo(f"✅ Found {total_models} neural network models/checkpoints")
            
            click.echo("\n📋 Model Types:")
            click.echo("  🧠 .pth files: PyTorch neural network models")
            click.echo("  📊 .json files: Trained player profiles")
            click.echo("  🔄 checkpoints: Training checkpoints")
            
            click.echo("\n🚀 Usage Examples:")
            click.echo("  euchre run-neural-games -m1 Alice -m2 Bob")
            click.echo("  euchre play-trained-players -p1 Alice -p2 Bob")
            click.echo("  euchre benchmark --device cpu")
            
    except Exception as e:
        click.echo(f"Error listing neural models: {e}")


@main.command()
@click.option("--device", default="cpu", help="Device to use (cpu/cuda)")
@click.option("--save-results", is_flag=True, help="Save benchmark results")
def benchmark(device: str, save_results: bool) -> None:
    """Run performance benchmark comparing float vs integer models."""
    try:
        from ..ai_model.benchmark_runner import run_benchmark
        run_benchmark(device=device, save_results=save_results)
    except ImportError:
        click.echo("Benchmark runner not available.")
    except Exception as e:
        click.echo(f"Error running benchmark: {e}")


@main.command()
@click.option("--num-games", default=100000, help="Number of games to play")
@click.option("--players", "-p", multiple=True, 
              default=["Alice", "Bob", "Charlie", "David"],
              help="Player names for self-play training")
def train_self_play(num_games: int, players: tuple) -> None:
    """Train AI models using self-play."""
    try:
        from ..ai_training_framework import train_self_play
        train_self_play(num_games, list(players))
    except ImportError:
        click.echo("Self-play training not available.")
    except Exception as e:
        click.echo(f"Error running self-play training: {e}")


@main.command()
@click.option("--num-games", default=200000, help="Number of games to play")
def train_integer_vs_float(num_games: int) -> None:
    """Train integer models against float models."""
    try:
        from ..ai_training_framework import train_integer_vs_float
        train_integer_vs_float(num_games)
    except ImportError:
        click.echo("Integer vs float training not available.")
    except Exception as e:
        click.echo(f"Error running integer vs float training: {e}")


@main.command()
@click.option("--num-games", default=15000, help="Number of games to play")
@click.option("--output-dir", default="trained_models", help="Output directory for models")
@click.option("--device", default="auto", help="Device to use (cpu/cuda/auto)")
@click.option("--enable-amp", is_flag=True, help="Enable automatic mixed precision")
def generate_player_profiles(num_games: int, output_dir: str, device: str, enable_amp: bool) -> None:
    """Generate 5 distinct AI player profiles with different playing styles."""
    try:
        from ..ai_training_framework import generate_player_profiles
        generate_player_profiles(num_games, output_dir, device, enable_amp)
    except ImportError:
        click.echo("Player profile generation not available.")
    except Exception as e:
        click.echo(f"Error generating player profiles: {e}")


@main.command()
@click.option("--num-games", default=15000, help="Number of games to play")
@click.option("--output-dir", default="trained_models", help="Output directory for models")
def generate_profiles_cpu(num_games: int, output_dir: str) -> None:
    """Generate AI player profiles using CPU only."""
    generate_player_profiles(num_games, output_dir, "cpu", False)


@main.command()
@click.option("--num-games", default=15000, help="Number of games to play")
@click.option("--output-dir", default="trained_models", help="Output directory for models")
@click.option("--gpu-memory-fraction", default=0.9, help="GPU memory fraction to use")
def generate_profiles_gpu(num_games: int, output_dir: str, gpu_memory_fraction: float) -> None:
    """Generate AI player profiles with GPU acceleration."""
    generate_player_profiles(num_games, output_dir, "cuda", True)


@main.command()
@click.option("--num-games", default=100, help="Number of games to play")
@click.option("--player1", default="Alice", help="First player name")
@click.option("--player2", default="Bob", help="Second player name")
@click.option("--player3", default="Charlie", help="Third player name")
@click.option("--player4", default="David", help="Fourth player name")
def play_trained_players(num_games: int, player1: str, player2: str, player3: str, player4: str) -> None:
    """Play a game with trained AI players."""
    try:
        from ..ai_training_framework import play_trained_players
        play_trained_players(num_games, [player1, player2, player3, player4])
    except ImportError:
        click.echo("Trained player gameplay not available.")
    except Exception as e:
        click.echo(f"Error playing with trained players: {e}")


@main.command()
@click.option("--num-games", default=1000, help="Number of games to run")
def run_mass_games(num_games: int) -> None:
    """Run thousands of games in parallel."""
    try:
        from ..mass_game_runner import run_mass_games
        run_mass_games(num_games)
    except ImportError:
        click.echo("Mass game runner not available.")
    except Exception as e:
        click.echo(f"Error running mass games: {e}")


@main.command()
def analyze_games() -> None:
    """Analyze game results and generate statistics."""
    try:
        from ..game_analyzer import analyze_games
        analyze_games()
    except ImportError:
        click.echo("Game analyzer not available.")
    except Exception as e:
        click.echo(f"Error analyzing games: {e}")


@main.command()
def cleanup_games() -> None:
    """Clean up old game result files."""
    try:
        from ..game_analyzer import cleanup_games
        cleanup_games()
    except ImportError:
        click.echo("Game cleanup not available.")
    except Exception as e:
        click.echo(f"Error cleaning up games: {e}")


@main.command()
def jupyter() -> None:
    """Start Jupyter Notebook."""
    try:
        import subprocess
        subprocess.run(["jupyter", "notebook"], cwd="notebooks")
    except Exception as e:
        click.echo(f"Error starting Jupyter: {e}")


@main.command()
def jupyter_lab() -> None:
    """Start Jupyter Lab."""
    try:
        import subprocess
        subprocess.run(["jupyter", "lab"], cwd="notebooks")
    except Exception as e:
        click.echo(f"Error starting Jupyter Lab: {e}")


if __name__ == "__main__":
    main() 