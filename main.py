"""Main entry point for Euchre game."""

from src.game import Game
from src.ui import TextUI


def main() -> None:
    """Run the Euchre game."""
    print("=" * 50)
    print("Welcome to Euchre!")
    print("=" * 50)

    # Configure players
    player_config = []
    print("\nConfigure players:")
    for i in range(4):
        name = input(f"Player {i + 1} name: ").strip() or f"Player {i + 1}"
        human_input = input(f"Is {name} human? (y/n): ").strip().lower()
        is_human = human_input == "y"
        player_config.append((name, is_human))

    # Create game
    game = Game(player_config)

    # Create and set UI
    ui = TextUI()
    game.set_ui(ui)

    # Play game
    while True:
        print("\n" + "=" * 50)
        print("New Hand")
        print("=" * 50)

        continue_game = game.play_hand()

        # Display scores
        scores = game.get_scores()
        ui.display_scores(scores[0], scores[1])

        # Check for game over
        winner = game.get_winner()
        if winner is not None:
            ui.display_game_over(winner)
            break

        if not continue_game:
            break

    print("\nThanks for playing!")


if __name__ == "__main__":
    main()

