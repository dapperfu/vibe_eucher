"""Script to collect training data by running games with Rich TUI."""

import argparse
import sys
from pathlib import Path

# Add project root to path so eucher package can be imported
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eucher.players.computer.ml.ml_config import MLConfig
from eucher.training.self_play import SelfPlayTrainer
from eucher.training.data_collection_progress import DataCollectionProgress


def main() -> None:
    """Collect training data by running games."""
    parser = argparse.ArgumentParser(description="Collect training data for ML models")
    parser.add_argument(
        "--num_games",
        type=int,
        default=100,
        help="Number of games to play",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save training data (default: training_data/)",
    )
    parser.add_argument(
        "--data_prefix",
        type=str,
        default="training",
        help="Prefix for data filenames (used only with --use-prefix-naming)",
    )
    parser.add_argument(
        "--use-prefix-naming",
        action="store_true",
        help="Use prefix-based naming instead of UUID-based naming (default: UUID)",
    )
    parser.add_argument(
        "--file-extension",
        type=str,
        default="npz",
        choices=["npz", "ngz"],
        help="File extension to use (default: npz)",
    )
    parser.add_argument(
        "--refresh_rate",
        type=float,
        default=2.0,
        help="Display refresh rate in updates per second (default: 2.0)",
    )

    args = parser.parse_args()

    config = MLConfig()
    output_dir = Path(args.output_dir) if args.output_dir else config.training_data_dir

    # Create trainer
    trainer = SelfPlayTrainer(output_dir=output_dir)

    # Get default combinations
    combinations = trainer._get_default_combinations()
    num_combinations = len(combinations)
    games_per_combination = args.num_games // num_combinations
    remaining_games = args.num_games % num_combinations  # Handle remainder

    # Create progress display
    progress = DataCollectionProgress(
        total_games=args.num_games,
        num_combinations=num_combinations,
        refresh_rate=args.refresh_rate,
    )

    # Start live display
    live_display = progress.start_live_display()
    live_display.__enter__()

    try:
        # Track previous sample counts to calculate deltas
        prev_order_up = 0
        prev_call_trump = 0
        prev_play_card = 0
        prev_discard = 0

        # Run games for each combination
        for combo_idx, combo in enumerate(combinations):
            combo_name = f"{combo[0][1]}/{combo[1][1]}/{combo[2][1]}/{combo[3][1]}"
            progress.set_combination(combo_idx, combo_name)

            # Add extra game to first few combinations if there's a remainder
            combo_games = games_per_combination + (1 if combo_idx < remaining_games else 0)
            
            for game_num in range(combo_games):
                # Run game
                trainer.run_game_with_collection(combo)

                # Calculate samples collected in this game
                collector = trainer.data_collector
                order_up_delta = len(collector.order_up_data) - prev_order_up
                call_trump_delta = len(collector.call_trump_data) - prev_call_trump
                play_card_delta = len(collector.play_card_data) - prev_play_card
                discard_delta = len(collector.discard_data) - prev_discard

                # Update progress
                progress.update_game_completed(
                    order_up_samples=order_up_delta,
                    call_trump_samples=call_trump_delta,
                    play_card_samples=play_card_delta,
                    discard_samples=discard_delta,
                )

                # Update previous counts
                prev_order_up = len(collector.order_up_data)
                prev_call_trump = len(collector.call_trump_data)
                prev_play_card = len(collector.play_card_data)
                prev_discard = len(collector.discard_data)

                # Update live display
                if game_num % 5 == 0:  # Update every 5 games to reduce overhead
                    progress.update_live_display(live_display)

        # Final update
        progress.update_live_display(live_display)

        # Save collected data
        saved_files = trainer.data_collector.save_data(
            prefix=args.data_prefix if args.use_prefix_naming else None,
            use_uuid_naming=not args.use_prefix_naming,
            file_extension=args.file_extension,
        )

    finally:
        # Close live display
        live_display.__exit__(None, None, None)
        progress.print_summary()

    # Print file locations
    from rich.console import Console
    console = Console()
    console.print(f"\n[green]✓[/green] Data saved to: [bold]{output_dir}[/bold]")
    if args.use_prefix_naming:
        console.print(f"[green]✓[/green] Files: [bold]{args.data_prefix}_*.{args.file_extension}[/bold]")
    else:
        console.print(f"[green]✓[/green] Saved {len(saved_files)} files with UUID-based naming")
        console.print(f"[green]✓[/green] File extension: [bold].{args.file_extension}[/bold]")


if __name__ == "__main__":
    main()
