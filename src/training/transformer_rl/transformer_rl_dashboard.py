"""Rich-based dashboard for Transformer RL training with comprehensive metrics display."""

import time
from typing import Dict, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table


class TransformerRLDashboard:
    """Rich-based dashboard for Transformer RL training with live metrics display."""

    def __init__(
        self,
        num_hands: int,
        refresh_rate: float = 2.0,
    ) -> None:
        """
        Initialize the Transformer RL training dashboard.

        Parameters
        ----------
        num_hands : int
            Total number of hands to train on.
        refresh_rate : float
            Refresh rate in updates per second.
        """
        self.num_hands = num_hands
        self.refresh_rate = refresh_rate
        self.console = Console()

        # Training metrics storage
        self.current_hand = 0
        self.hands_won = 0
        self.hands_played = 0
        self.stage = "Unknown"
        self.policy_loss: Optional[float] = None
        self.value_loss: Optional[float] = None
        self.entropy: Optional[float] = None
        self.total_loss: Optional[float] = None
        self.start_time = time.time()

        # Create progress bar
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
        self.task_id = self.progress.add_task("Training Transformer RL...", total=num_hands)

    def update(
        self,
        hand_num: int,
        total_hands: int,
        hands_won: int,
        hands_played: int,
        stage: str,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        entropy: Optional[float] = None,
        total_loss: Optional[float] = None,
    ) -> None:
        """
        Update dashboard with new metrics.

        Parameters
        ----------
        hand_num : int
            Current hand number.
        total_hands : int
            Total number of hands.
        hands_won : int
            Number of hands won.
        hands_played : int
            Number of hands played.
        stage : str
            Current curriculum stage.
        policy_loss : Optional[float]
            Policy loss from last update.
        value_loss : Optional[float]
            Value loss from last update.
        entropy : Optional[float]
            Entropy from last update.
        total_loss : Optional[float]
            Total loss from last update.
        """
        self.current_hand = hand_num
        self.hands_won = hands_won
        self.hands_played = hands_played
        self.stage = stage
        if policy_loss is not None:
            self.policy_loss = policy_loss
        if value_loss is not None:
            self.value_loss = value_loss
        if entropy is not None:
            self.entropy = entropy
        if total_loss is not None:
            self.total_loss = total_loss

        # Update progress bar
        self.progress.update(self.task_id, completed=hand_num + 1)

    def _create_metrics_table(self) -> Table:
        """
        Create a table with current training metrics.

        Returns
        -------
        Table
            Rich table with metrics.
        """
        table = Table(title="Training Metrics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="yellow", no_wrap=True)
        table.add_column("Value", style="green", justify="right")

        # Basic stats
        win_rate = (self.hands_won / self.hands_played * 100) if self.hands_played > 0 else 0.0
        table.add_row("Hands Played", f"{self.hands_played}")
        table.add_row("Hands Won", f"{self.hands_won}")
        table.add_row("Win Rate", f"{win_rate:.2f}%")
        # Convert stage to string if it's an enum
        stage_str = str(self.stage) if hasattr(self.stage, 'name') else self.stage
        table.add_row("Current Stage", stage_str)

        # Training losses (if available)
        if self.policy_loss is not None:
            table.add_row("Policy Loss", f"{self.policy_loss:.6f}")
        if self.value_loss is not None:
            table.add_row("Value Loss", f"{self.value_loss:.6f}")
        if self.entropy is not None:
            table.add_row("Entropy", f"{self.entropy:.6f}")
        if self.total_loss is not None:
            table.add_row("Total Loss", f"{self.total_loss:.6f}")

        # Time stats
        elapsed = time.time() - self.start_time
        table.add_row("Elapsed Time", f"{elapsed:.1f}s")
        if self.hands_played > 0:
            avg_time_per_hand = elapsed / self.hands_played
            remaining_hands = self.num_hands - self.current_hand - 1
            estimated_remaining = avg_time_per_hand * remaining_hands
            table.add_row("Est. Remaining", self._format_time(estimated_remaining))

        return table

    def start_live_display(self) -> Live:
        """
        Start live updating display.

        Parameters
        ----------
        refresh_rate : float
            Refresh rate in updates per second.

        Returns
        -------
        Live
            Rich Live context manager for continuous updates.
        """
        # Create layout with progress bar and status table
        layout = Layout()
        layout.split_column(
            Layout(self.progress, size=3),
            Layout(self._create_metrics_table()),
        )

        return Live(layout, console=self.console, refresh_per_second=self.refresh_rate)

    def update_live_display(self, live_display: Live) -> None:
        """
        Update the live display with current status.

        Parameters
        ----------
        live_display : Live
            The live display context manager.
        """
        from rich.layout import Layout

        layout = Layout()
        layout.split_column(
            Layout(self.progress, size=3),
            Layout(self._create_metrics_table()),
        )
        live_display.update(layout)

    def print_summary(self) -> None:
        """Print final training summary."""
        win_rate = (self.hands_won / self.hands_played * 100) if self.hands_played > 0 else 0.0
        elapsed = time.time() - self.start_time

        summary = Table(title="Training Summary", show_header=True, header_style="bold green")
        summary.add_column("Metric", style="cyan", no_wrap=True)
        summary.add_column("Value", style="yellow", justify="right")

        summary.add_row("Total Hands", f"{self.hands_played}")
        summary.add_row("Hands Won", f"{self.hands_won}")
        summary.add_row("Win Rate", f"{win_rate:.2f}%")
        # Convert stage to string if it's an enum
        stage_str = str(self.stage) if hasattr(self.stage, 'name') else self.stage
        summary.add_row("Final Stage", stage_str)
        summary.add_row("Total Time", f"{elapsed:.1f}s")
        if self.hands_played > 0:
            summary.add_row("Avg Time/Hand", f"{elapsed / self.hands_played:.2f}s")

        self.console.print()
        self.console.print(summary)

