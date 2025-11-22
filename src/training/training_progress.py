"""Rich-based progress bar and status display for training."""

import time
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from src.training.convergence_tracker import ConvergenceTracker
from src.training.training_orchestrator import TrainingOrchestrator
from src.training.training_scheduler import TrainingScheduler


class TrainingProgressDisplay:
    """Rich-based progress display for training with convergence metrics."""

    def __init__(
        self,
        orchestrator: TrainingOrchestrator,
        trump_selection_risk: float = 0.5,
        gameplay_risk: float = 0.5,
        show_progress_bar: bool = True,
    ) -> None:
        """
        Initialize the progress display.

        Parameters
        ----------
        orchestrator : TrainingOrchestrator
            Training orchestrator to monitor.
        trump_selection_risk : float
            Risk factor for trump selection decisions.
        gameplay_risk : float
            Risk factor for gameplay decisions.
        show_progress_bar : bool
            Whether to show the progress bar (True) or just status updates (False).
        """
        self.orchestrator = orchestrator
        self.trump_selection_risk = trump_selection_risk
        self.gameplay_risk = gameplay_risk
        self.show_progress_bar = show_progress_bar
        self.console = Console()

        # Create progress bar
        if show_progress_bar:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            )
            self.task_id = self.progress.add_task("Training...", total=None)
        else:
            self.progress = None
            self.task_id = None

        self.start_time = time.time()

    def create_status_table(self) -> Table:
        """
        Create a status table with training metrics.

        Returns
        -------
        Table
            Rich table with training status.
        """
        status = self.orchestrator.get_status()
        scheduler_progress = status["scheduler_progress"]

        table = Table(title="Training Status", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        # Basic stats
        table.add_row("Games Played", str(status["current_game"]))
        table.add_row("Training Mode", scheduler_progress["mode"].replace("_", " ").title())

        # Time-based progress
        if scheduler_progress["mode"] == "time_based":
            elapsed = scheduler_progress["elapsed_seconds"]
            remaining = scheduler_progress["remaining_seconds"]
            progress_pct = scheduler_progress["progress_percent"]

            elapsed_str = self._format_time(elapsed)
            table.add_row("Elapsed Time", elapsed_str)

            if remaining is not None:
                remaining_str = self._format_time(remaining)
                table.add_row("Remaining Time", remaining_str)

            if progress_pct is not None:
                table.add_row("Progress", f"{progress_pct:.1f}%")

        # Convergence stats
        if self.orchestrator.convergence_tracker:
            conv_stats = status["convergence_stats"]
            table.add_row("", "")  # Separator

            overall_wr = conv_stats["overall_win_rate"]
            window_wr = conv_stats["window_win_rate"]
            target_wr = conv_stats["target_win_rate"]
            converged = conv_stats["converged"]

            # Color win rates based on target
            overall_color = "green" if overall_wr >= target_wr else "yellow"
            window_color = "green" if window_wr >= target_wr else "yellow"

            table.add_row("Overall Win Rate", f"[{overall_color}]{overall_wr:.2%}[/{overall_color}]")
            table.add_row(
                f"Window Win Rate (last {conv_stats['window_size']} games)",
                f"[{window_color}]{window_wr:.2%}[/{window_color}]",
            )
            table.add_row("Target Win Rate", f"{target_wr:.2%}")
            table.add_row(
                "Converged",
                "[green]Yes[/green]" if converged else "[red]No[/red]",
            )

        # Risk factors
        table.add_row("", "")  # Separator
        table.add_row("Trump Selection Risk", f"{self.trump_selection_risk:.2f}")
        table.add_row("Gameplay Risk", f"{self.gameplay_risk:.2f}")

        # Games per second
        elapsed_total = time.time() - self.start_time
        if elapsed_total > 0:
            games_per_sec = status["current_game"] / elapsed_total
            table.add_row("Games/Second", f"{games_per_sec:.2f}")

        return table

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

    def update(self) -> None:
        """Update the progress display."""
        status = self.orchestrator.get_status()
        scheduler_progress = status["scheduler_progress"]

        if self.progress and self.task_id is not None:
            # Update progress bar
            if scheduler_progress["mode"] == "time_based" and scheduler_progress["progress_percent"] is not None:
                self.progress.update(
                    self.task_id,
                    completed=scheduler_progress["progress_percent"],
                    description=f"Training (Game {status['current_game']})",
                )
            else:
                # Convergence-based or no progress percentage
                self.progress.update(
                    self.task_id,
                    description=f"Training (Game {status['current_game']})",
                )

    def display_status(self) -> None:
        """Display current training status."""
        status_table = self.create_status_table()
        panel = Panel(status_table, title="[bold]Training Progress[/bold]", border_style="blue")
        self.console.print(panel)

    def start_live_display(self) -> "Live":
        """
        Start live updating display.

        Returns
        -------
        Live
            Rich Live context manager for continuous updates.
        """
        if self.progress:
            # Create layout with progress bar and status table
            from rich.layout import Layout

            layout = Layout()
            layout.split_column(
                Layout(self.progress, size=3),
                Layout(self.create_status_table()),
            )

            return Live(layout, console=self.console, refresh_per_second=4)
        else:
            # Just status table
            return Live(self.create_status_table(), console=self.console, refresh_per_second=4)

    def update_live_display(self, live_display: "Live") -> None:
        """
        Update the live display with current status.

        Parameters
        ----------
        live_display : Live
            The live display context manager.
        """
        if self.progress:
            from rich.layout import Layout

            layout = Layout()
            layout.split_column(
                Layout(self.progress, size=3),
                Layout(self.create_status_table()),
            )
            live_display.update(layout)
        else:
            live_display.update(self.create_status_table())

    def print_summary(self) -> None:
        """Print final training summary."""
        status = self.orchestrator.get_status()
        elapsed = time.time() - self.start_time

        summary_table = Table(title="Training Summary", show_header=True, header_style="bold green")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Games", str(status["current_game"]))
        summary_table.add_row("Total Time", self._format_time(elapsed))

        if status["current_game"] > 0:
            games_per_sec = status["current_game"] / elapsed
            summary_table.add_row("Average Games/Second", f"{games_per_sec:.2f}")

        if self.orchestrator.convergence_tracker:
            conv_stats = status["convergence_stats"]
            summary_table.add_row("Final Overall Win Rate", f"{conv_stats['overall_win_rate']:.2%}")
            summary_table.add_row("Final Window Win Rate", f"{conv_stats['window_win_rate']:.2%}")
            summary_table.add_row("Converged", "Yes" if conv_stats["converged"] else "No")

        panel = Panel(summary_table, title="[bold green]Training Complete[/bold green]", border_style="green")
        self.console.print(panel)

