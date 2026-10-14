"""Main entry point for Euchre game."""

from eucher.game import Game, parse_seed_text, seeded_random_training_command
from eucher.tui import TextTUI


def main() -> None:
    """Run the Euchre game."""
    print("Welcome to euchre, press any key to start")
    input()

    # Configure players
    player_config = []
    print("\nConfigure players:")
    print(
        "Profile types: 'human', 'heuristic' (rule-based), 'ai', 'random', 'ml_sklearn' (sklearn ML), 'ml_pytorch' (PyTorch ML)"
    )
    for i in range(4):
        name = input(f"Player {i + 1} name: ").strip() or f"Player {i + 1}"
        profile_input = (
            input(f"{name} profile type (human/heuristic/ai/random/ml_sklearn) [ai]: ").strip().lower() or "ai"
        )
        valid_types = ["human", "heuristic", "ai", "random", "ml_sklearn", "ml_pytorch"]
        if profile_input not in valid_types:
            print(f"Invalid profile type, defaulting to 'ai'")
            profile_input = "ai"
        player_config.append((name, profile_input))
    
    # Display players after configuration
    print("\nPlayers:")
    for i, (name, profile_type) in enumerate(player_config):
        print(f"  Player {i}: {name} ({profile_type})")

    seed_text = input("Random seed (integer or UUID, Enter for a new game): ").strip()
    parsed_seed = parse_seed_text(seed_text)

    # Create game
    game = Game(player_config, seed=parsed_seed)
    print(f"\nSeed: {game.seed_label}")
    print(f"To play these same hands again, enter seed {game.seed_label}")
    print("To train on these hands against random AI:")
    print(f"  {seeded_random_training_command(game.seed_label)}")

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

