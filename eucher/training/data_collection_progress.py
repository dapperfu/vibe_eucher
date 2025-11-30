"""Rich-based progress display for data collection."""

import time
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table


class DataCollectionProgress:
    """Rich-based progress display for data collection with live updates."""

    def __init__(
        self,
        total_games: int,
        num_combinations: int = 6,
        refresh_rate: float = 2.0,
    ) -> None:
        """
        Initialize the data collection progress display.

        Parameters
        ----------
        total_games : int
            Total number of games to collect.
        num_combinations : int
            Number of player combinations to run.
        refresh_rate : float
            Refresh rate in updates per second.
        """
        self.total_games = total_games
        self.num_combinations = num_combinations
        self.refresh_rate = refresh_rate
        self.console = Console()

        # Tracking
        self.games_completed = 0
        self.current_combination = 0
        self.current_combination_games = 0
        self.start_time = time.time()

        # Data collection stats
        self.order_up_samples = 0
        self.call_trump_samples = 0
        self.play_card_samples = 0
        self.discard_samples = 0

        # Create main progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.main_task_id = self.progress.add_task(
            "Collecting training data...", total=total_games
        )

        # Combination progress
        self.combo_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        )
        self.combo_task_id = self.combo_progress.add_task(
            "Current combination...", total=total_games // num_combinations
        )

    def update_game_completed(
        self,
        order_up_samples: int = 0,
        call_trump_samples: int = 0,
        play_card_samples: int = 0,
        discard_samples: int = 0,
    ) -> None:
        """
        Update progress after a game is completed.

        Parameters
        ----------
        order_up_samples : int
            Number of order_up samples collected in this game.
        call_trump_samples : int
            Number of call_trump samples collected in this game.
        play_card_samples : int
            Number of play_card samples collected in this game.
        discard_samples : int
            Number of discard samples collected in this game.
        """
        self.games_completed += 1
        self.current_combination_games += 1
        self.order_up_samples += order_up_samples
        self.call_trump_samples += call_trump_samples
        self.play_card_samples += play_card_samples
        self.discard_samples += discard_samples

        # Update progress bars
        self.progress.update(self.main_task_id, completed=self.games_completed)
        self.combo_progress.update(
            self.combo_task_id, completed=self.current_combination_games
        )

    def set_combination(self, combination_idx: int, combination_name: str) -> None:
        """
        Set the current player combination.

        Parameters
        ----------
        combination_idx : int
            Index of the current combination (0-based).
        combination_name : str
            Description of the combination.
        """
        self.current_combination = combination_idx
        self.current_combination_games = 0
        # Calculate games per combo (handle remainder)
        games_per_combo = self.total_games // self.num_combinations
        remaining = self.total_games % self.num_combinations
        if combination_idx < remaining:
            games_per_combo += 1
        self.combo_progress.update(
            self.combo_task_id,
            total=games_per_combo,
            description=f"Combination {combination_idx + 1}/{self.num_combinations}: {combination_name}",
        )

    def _create_stats_table(self) -> Table:
        """
        Create a table with data collection statistics.

        Returns
        -------
        Table
            Rich table with statistics.
        """
        table = Table(title="Data Collection Statistics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="yellow", no_wrap=True)
        table.add_column("Value", style="green", justify="right")

        # Games
        table.add_row("Games Completed", f"{self.games_completed}/{self.total_games}")
        table.add_row("Combination", f"{self.current_combination + 1}/{self.num_combinations}")

        # Samples collected
        table.add_row("", "")  # Separator
        table.add_row("Order Up Samples", f"{self.order_up_samples:,}")
        table.add_row("Call Trump Samples", f"{self.call_trump_samples:,}")
        table.add_row("Play Card Samples", f"{self.play_card_samples:,}")
        table.add_row("Discard Samples", f"{self.discard_samples:,}")

        total_samples = (
            self.order_up_samples
            + self.call_trump_samples
            + self.play_card_samples
            + self.discard_samples
        )
        table.add_row("Total Samples", f"[bold]{total_samples:,}[/bold]")

        # Performance
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            games_per_sec = self.games_completed / elapsed
            samples_per_sec = total_samples / elapsed if total_samples > 0 else 0
            table.add_row("", "")  # Separator
            table.add_row("Games/Second", f"{games_per_sec:.2f}")
            table.add_row("Samples/Second", f"{samples_per_sec:.1f}")

        return table

    def _create_layout(self) -> "Layout":
        """
        Create the dashboard layout.

        Returns
        -------
        Layout
            Rich layout with all components.
        """
        from rich.layout import Layout

        layout = Layout()

        # Top section: Progress bars
        progress_layout = Layout()
        progress_layout.split_column(
            Layout(self.progress, size=3),
            Layout(self.combo_progress, size=3),
        )

        # Bottom section: Statistics table
        stats_layout = Layout(self._create_stats_table())

        # Combine everything
        layout.split_column(
            Layout(progress_layout, size=6),
            Layout(stats_layout),
        )

        return layout

    def start_live_display(self) -> Live:
        """
        Start live updating display.

        Returns
        -------
        Live
            Rich Live context manager for continuous updates.
        """
        return Live(self._create_layout(), console=self.console, refresh_per_second=self.refresh_rate)

    def update_live_display(self, live_display: Live) -> None:
        """
        Update the live display with current status.

        Parameters
        ----------
        live_display : Live
            The live display context manager.
        """
        live_display.update(self._create_layout())

    def print_summary(self) -> None:
        """Print final data collection summary."""
        elapsed = time.time() - self.start_time

        summary_table = Table(title="Data Collection Summary", show_header=True, header_style="bold green")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Games", str(self.total_games))
        summary_table.add_row("Total Time", self._format_time(elapsed))

        if elapsed > 0:
            games_per_sec = self.total_games / elapsed
            summary_table.add_row("Average Games/Second", f"{games_per_sec:.2f}")

        # Total samples
        total_samples = (
            self.order_up_samples
            + self.call_trump_samples
            + self.play_card_samples
            + self.discard_samples
        )
        summary_table.add_row("", "")  # Separator
        summary_table.add_row("Order Up Samples", f"{self.order_up_samples:,}")
        summary_table.add_row("Call Trump Samples", f"{self.call_trump_samples:,}")
        summary_table.add_row("Play Card Samples", f"{self.play_card_samples:,}")
        summary_table.add_row("Discard Samples", f"{self.discard_samples:,}")
        summary_table.add_row("Total Samples", f"[bold]{total_samples:,}[/bold]")

        if elapsed > 0 and total_samples > 0:
            samples_per_sec = total_samples / elapsed
            summary_table.add_row("Average Samples/Second", f"{samples_per_sec:.1f}")

        panel = Panel(summary_table, title="[bold green]Data Collection Complete[/bold green]", border_style="green")
        self.console.print(panel)

    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds to human-readable string.

        Parameters
        ----------
        seconds : float
            Time in seconds.

        Returns
        -------
        str
            Formatted time string.
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

