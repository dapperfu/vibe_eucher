"""Script to run a demo game automatically."""

from src.game import Game
from src.tui import TextTUI


def main() -> None:
    """Run a demo game with automated players."""
    # Create a game with mixed player types for variety
    player_config = [
        ("Alice", "ai"),
        ("Bob", "simple"),
        ("Charlie", "random"),
        ("Diana", "ai"),
    ]

    # Create game
    game = Game(player_config)

    # Create and set TUI
    tui = TextTUI()
    game.set_tui(tui)

    # Play game
    while True:
        print("\n" + "=" * 50)
        print("New Hand")
        print("=" * 50)

        continue_game = game.play_hand()

        # Display scores
        scores = game.get_scores()
        tui.display_scores(scores[0], scores[1])

        # Check for game over
        winner = game.get_winner()
        if winner is not None:
            tui.display_game_over(winner)
            # Display complete game log
            if hasattr(tui, "display_game_log"):
                tui.display_game_log()
            break

        if not continue_game:
            break

    # Display game log if game ended without winner
    if hasattr(tui, "display_game_log"):
        tui.display_game_log()

    print("\nThanks for playing!")


if __name__ == "__main__":
    main()

