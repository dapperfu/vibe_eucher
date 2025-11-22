"""Main entry point for Euchre game."""

from src.game import Game
from src.tui import TextTUI


def main() -> None:
    """Run the Euchre game."""
    print("Welcome to euchre, press any key to start")
    input()

    # Configure players
    player_config = []
    print("\nConfigure players:")
    print(
        "Profile types: 'human', 'simple'/'heuristic' (rule-based), 'ai', 'random', 'ml'/'ml_sklearn' (sklearn ML), 'ml_pytorch' (PyTorch ML)"
    )
    for i in range(4):
        name = input(f"Player {i + 1} name: ").strip() or f"Player {i + 1}"
        profile_input = (
            input(f"{name} profile type (human/simple/ai/random/ml) [ai]: ").strip().lower() or "ai"
        )
        valid_types = ["human", "simple", "heuristic", "ai", "random", "ml", "ml_sklearn", "ml_pytorch"]
        if profile_input not in valid_types:
            print(f"Invalid profile type, defaulting to 'ai'")
            profile_input = "ai"
        player_config.append((name, profile_input))
    
    # Display players after configuration
    print("\nPlayers:")
    for i, (name, profile_type) in enumerate(player_config):
        print(f"  Player {i}: {name} ({profile_type})")

    # Create game
    game = Game(player_config)

    # Create and set TUI
    tui = TextTUI()
    game.set_tui(tui)

    # Play game
    while True:
        print("\nNew Hand")

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

