"""Main entry point for Euchre game."""

from src.game import Game
from src.tui import TextTUI


def main() -> None:
    """Run the Euchre game."""
    print("=" * 50)
    print("Welcome to Euchre!")
    print("=" * 50)

    # Configure players
    player_config = []
    print("\nConfigure players:")
    print("Profile types: 'human', 'simple' (rule-based), 'ai'")
    for i in range(4):
        name = input(f"Player {i + 1} name: ").strip() or f"Player {i + 1}"
        profile_input = (
            input(f"{name} profile type (human/simple/ai) [ai]: ").strip().lower() or "ai"
        )
        if profile_input not in ["human", "simple", "ai"]:
            print(f"Invalid profile type, defaulting to 'ai'")
            profile_input = "ai"
        player_config.append((name, profile_input))

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
            break

        if not continue_game:
            break

    print("\nThanks for playing!")


if __name__ == "__main__":
    main()

