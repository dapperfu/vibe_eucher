"""Rich-based dashboard for EuchrePerceiverMuZero training with comprehensive metrics display."""

import time
from typing import Dict, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


class PerceiverMuZeroDashboard:
    """Rich-based dashboard for EuchrePerceiverMuZero training with live metrics display."""

    def __init__(
        self,
        num_games: int,
        refresh_rate: float = 2.0,
    ) -> None:
        """
        Initialize the PerceiverMuZero training dashboard.

        Parameters
        ----------
        num_games : int
            Total number of games to train on (or estimated for duration-based training).
        refresh_rate : float
            Refresh rate in updates per second.
        """
        self.num_games = num_games
        self.refresh_rate = refresh_rate
        self.console = Console()

        # Training metrics storage
        self.current_game = 0
        self.current_iteration = 0
        self.games_completed = 0
        self.examples_collected = 0
        self.replay_buffer_size = 0

        # Loss metrics
        self.policy_loss: Optional[float] = None
        self.value_loss: Optional[float] = None
        self.reward_loss: Optional[float] = None
        self.entropy: Optional[float] = None
        self.total_loss: Optional[float] = None

        # Performance metrics
        self.avg_game_time: Optional[float] = None
        self.avg_mcts_time: Optional[float] = None

        self.start_time = time.time()
        self.last_update_time = time.time()

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
        self.task_id = self.progress.add_task(
            "Training EuchrePerceiverMuZero...", total=num_games
        )

    def update(
        self,
        game_count: int,
        iteration: int,
        examples_collected: int,
        replay_buffer_size: int,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        reward_loss: Optional[float] = None,
        entropy: Optional[float] = None,
        total_loss: Optional[float] = None,
        game_time: Optional[float] = None,
        mcts_time: Optional[float] = None,
    ) -> None:
        """
        Update dashboard with new metrics.

        Parameters
        ----------
        game_count : int
            Current game count.
        iteration : int
            Current training iteration.
        examples_collected : int
            Number of training examples collected.
        replay_buffer_size : int
            Current replay buffer size.
        policy_loss : Optional[float]
            Policy loss from last update.
        value_loss : Optional[float]
            Value loss from last update.
        reward_loss : Optional[float]
            Reward loss from last update.
        entropy : Optional[float]
            Entropy from last update.
        total_loss : Optional[float]
            Total loss from last update.
        game_time : Optional[float]
            Time taken for last game.
        mcts_time : Optional[float]
            Average MCTS time per decision.
        """
        self.current_game = game_count
        self.current_iteration = iteration
        self.games_completed = game_count
        self.examples_collected = examples_collected
        self.replay_buffer_size = replay_buffer_size

        if policy_loss is not None:
            self.policy_loss = policy_loss
        if value_loss is not None:
            self.value_loss = value_loss
        if reward_loss is not None:
            self.reward_loss = reward_loss
        if entropy is not None:
            self.entropy = entropy
        if total_loss is not None:
            self.total_loss = total_loss
        if game_time is not None:
            self.avg_game_time = game_time
        if mcts_time is not None:
            self.avg_mcts_time = mcts_time

        # Update progress bar
        self.progress.update(self.task_id, completed=game_count)

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
        table.add_row("Games Completed", f"{self.games_completed}")
        table.add_row("Iterations", f"{self.current_iteration}")
        table.add_row("Examples Collected", f"{self.examples_collected}")
        table.add_row("Replay Buffer Size", f"{self.replay_buffer_size}")

        # Loss metrics (if available)
        if self.policy_loss is not None:
            table.add_row("Policy Loss", f"{self.policy_loss:.6f}")
        if self.value_loss is not None:
            table.add_row("Value Loss", f"{self.value_loss:.6f}")
        if self.reward_loss is not None:
            table.add_row("Reward Loss", f"{self.reward_loss:.6f}")
        if self.entropy is not None:
            table.add_row("Entropy", f"{self.entropy:.6f}")
        if self.total_loss is not None:
            table.add_row("Total Loss", f"{self.total_loss:.6f}")

        # Performance metrics
        if self.avg_game_time is not None:
            table.add_row("Avg Game Time", f"{self.avg_game_time:.2f}s")
        if self.avg_mcts_time is not None:
            table.add_row("Avg MCTS Time", f"{self.avg_mcts_time:.3f}s")

        # Time stats
        elapsed = time.time() - self.start_time
        table.add_row("Elapsed Time", f"{elapsed:.1f}s")
        if self.games_completed > 0:
            avg_time_per_game = elapsed / self.games_completed
            remaining_games = max(0, self.num_games - self.current_game)
            estimated_remaining = avg_time_per_game * remaining_games
            table.add_row("Est. Remaining", f"{estimated_remaining:.1f}s")
            table.add_row("Avg Time/Game", f"{avg_time_per_game:.2f}s")

        return table

    def _create_config_table(self, config) -> Table:
        """
        Create a table with configuration information.

        Parameters
        ----------
        config
            Configuration object.

        Returns
        -------
        Table
            Rich table with config info.
        """
        table = Table(title="Configuration", show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", justify="right")

        table.add_row("MCTS Simulations", f"{config.num_simulations}")
        table.add_row("MCTS Depth", f"{config.max_depth}")
        table.add_row("Batch Size", f"{config.batch_size}")
        table.add_row("Learning Rate", f"{config.learning_rate:.2e}")
        table.add_row("Replay Buffer Size", f"{config.replay_buffer_size}")
        table.add_row("Device", str(config.device))

        return table

    def start_live_display(self, config=None) -> Live:
        """
        Start live updating display.

        Parameters
        ----------
        config : Optional
            Configuration object to display.

        Returns
        -------
        Live
            Rich Live context manager for continuous updates.
        """
        # Create layout with progress bar and status tables
        layout = Layout()

        if config:
            layout.split_column(
                Layout(self.progress, size=3),
                Layout(self._create_metrics_table()),
                Layout(self._create_config_table(config), size=8),
            )
        else:
            layout.split_column(
                Layout(self.progress, size=3),
                Layout(self._create_metrics_table()),
            )

        return Live(layout, console=self.console, refresh_per_second=self.refresh_rate)

    def update_live_display(self, live_display: Live, config=None) -> None:
        """
        Update the live display with current status.

        Parameters
        ----------
        live_display : Live
            The live display context manager.
        config : Optional
            Configuration object to display.
        """
        layout = Layout()

        if config:
            layout.split_column(
                Layout(self.progress, size=3),
                Layout(self._create_metrics_table()),
                Layout(self._create_config_table(config), size=8),
            )
        else:
            layout.split_column(
                Layout(self.progress, size=3),
                Layout(self._create_metrics_table()),
            )

        live_display.update(layout)

    def print_summary(self) -> None:
        """Print final training summary."""
        elapsed = time.time() - self.start_time

        summary = Table(title="Training Summary", show_header=True, header_style="bold green")
        summary.add_column("Metric", style="cyan", no_wrap=True)
        summary.add_column("Value", style="yellow", justify="right")

        summary.add_row("Total Games", f"{self.games_completed}")
        summary.add_row("Total Iterations", f"{self.current_iteration}")
        summary.add_row("Examples Collected", f"{self.examples_collected}")
        summary.add_row("Total Time", f"{elapsed:.1f}s")
        if self.games_completed > 0:
            summary.add_row("Avg Time/Game", f"{elapsed / self.games_completed:.2f}s")

        if self.total_loss is not None:
            summary.add_row("Final Total Loss", f"{self.total_loss:.6f}")
        if self.policy_loss is not None:
            summary.add_row("Final Policy Loss", f"{self.policy_loss:.6f}")
        if self.value_loss is not None:
            summary.add_row("Final Value Loss", f"{self.value_loss:.6f}")

        self.console.print()
        self.console.print(summary)

