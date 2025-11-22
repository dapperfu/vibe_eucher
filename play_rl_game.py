"""Script to play a single game with 4 RL AI players using models/rl_model.pth.

This script creates a game with 4 reinforcement learning agents, all using
the same trained RL model with a specified risk factor.
"""

from pathlib import Path
from typing import Optional

from src.game import Game
from src.ml_config import MLConfig
from src.tui import TextTUI


def play_rl_game(risk_factor: float = 0.5, model_path: Optional[Path] = None) -> None:
    """
    Play a single game with 4 RL AI players.

    Parameters
    ----------
    risk_factor : float
        Risk factor for both trump selection and gameplay decisions (0.0-1.0).
        Lower values = more conservative, higher values = more aggressive.
    model_path : Optional[Path]
        Path to the RL model file. If None, uses default from MLConfig.
    """
    if model_path is None:
        config = MLConfig()
        model_path = config.models_dir / "rl_model.pth"

    # Check if model exists
    if not model_path.exists():
        print(f"Error: RL model not found at {model_path}")
        print("Please train the RL model first using:")
        print("  python scripts/train_models.py --mode train_rl")
        return

    # Load and display training statistics if available
    try:
        import torch
        from src.ml_models_rl import RLAgent

        agent = RLAgent()
        stats = agent.load(model_path)
        if stats:
            print("=" * 60)
            print("RL Model Training Statistics")
            print("=" * 60)
            print(f"Games played: {stats.get('games_played', 'N/A')}")
            print(f"Final epsilon: {stats.get('final_epsilon', 'N/A'):.4f}" if isinstance(stats.get('final_epsilon'), (int, float)) else f"Final epsilon: {stats.get('final_epsilon', 'N/A')}")
            if 'avg_reward' in stats:
                print(f"Average reward: {stats['avg_reward']:.2f}")
            if 'avg_loss' in stats:
                print(f"Average loss: {stats['avg_loss']:.4f}")
            print("=" * 60)
            print()
    except Exception as e:
        print(f"Note: Could not load training statistics: {e}")
        print()

    # Create game with 4 RL players
    player_config = [
        ("RL Player 1", "ml_rl"),
        ("RL Player 2", "ml_rl"),
        ("RL Player 3", "ml_rl"),
        ("RL Player 4", "ml_rl"),
    ]

    # Create game with specified risk factor
    game = Game(
        player_config,
        trump_selection_risk=risk_factor,
        gameplay_risk=risk_factor,
    )

    # Create and set TUI for display
    tui = TextTUI()
    game.set_tui(tui)

    print(f"Starting game with 4 RL AI players (risk factor: {risk_factor})")
    print(f"Using model: {model_path}")
    print()

    # Play game
    hand_num = 1
    while True:
        print("\n" + "=" * 60)
        print(f"Hand {hand_num}")
        print("=" * 60)

        continue_game = game.play_hand()

        # Scores are already displayed in the hand log, no need to display again

        # Check for game over
        winner = game.get_winner()
        if winner is not None:
            tui.display_game_over(winner)
            print(f"\nGame over! Team {winner} wins!")
            break

        if not continue_game:
            break

        hand_num += 1

    print("\nGame complete!")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Play a game with 4 RL AI players")
    parser.add_argument(
        "--risk-factor",
        type=float,
        default=0.5,
        help="Risk factor for both trump selection and gameplay (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to RL model file (default: models/rl_model.pth)",
    )

    args = parser.parse_args()

    # Validate risk factor
    if not 0.0 <= args.risk_factor <= 1.0:
        print("Error: Risk factor must be between 0.0 and 1.0")
        return

    play_rl_game(risk_factor=args.risk_factor, model_path=args.model_path)


if __name__ == "__main__":
    main()

