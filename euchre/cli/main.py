"""Main CLI entry point for the euchre game."""

import click
from .commands import GameCommands


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
    GameCommands.play_game(player_name, ai_names)


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
    GameCommands.ai_profiles_game(ai_profiles, risk_ratios, enable_logging)


@main.command()
def ai_vs_ai() -> None:
    """Run AI vs AI euchre game."""
    GameCommands.ai_vs_ai_game()


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
def logged_game() -> None:
    """Run AI vs AI euchre game with logging (no ncurses)."""
    GameCommands.ai_vs_ai_game()


@main.command()
@click.option("--num-games", "-n", default=100, help="Number of games to play")
def tournament(num_games: int) -> None:
    """Run a tournament between trained players."""
    GameCommands.tournament_game(num_games)


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
        # This would list available neural models
        click.echo("Neural models not yet implemented.")
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