"""Rich-based dashboard for EucherGo training with comprehensive metrics display."""

import time
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


class EucherGoDashboard:
    """Rich-based dashboard for EucherGo training with live metrics display."""

    def __init__(
        self,
        num_iterations: Optional[int] = None,
        num_games_per_iteration: int = 10,
        refresh_rate: float = 2.0,
    ) -> None:
        """
        Initialize the EucherGo training dashboard.

        Parameters
        ----------
        num_iterations : Optional[int]
            Total number of training iterations (None for duration-based training).
        num_games_per_iteration : int
            Number of games per training iteration.
        refresh_rate : float
            Refresh rate in updates per second.
        """
        self.num_iterations = num_iterations
        self.num_games_per_iteration = num_games_per_iteration
        self.refresh_rate = refresh_rate
        self.console = Console()

        # Training metrics storage
        self.current_iteration = 0
        self.games_played = 0
        self.total_games = 0

        # Loss metrics
        self.policy_loss: Optional[float] = None
        self.value_loss: Optional[float] = None
        self.total_loss: Optional[float] = None

        # MCTS metrics
        self.avg_mcts_time: Optional[float] = None
        self.avg_simulations_per_move: Optional[int] = None

        # Performance metrics
        self.avg_game_time: Optional[float] = None
        self.avg_iteration_time: Optional[float] = None

        # Checkpoint info
        self.last_checkpoint_iteration: Optional[int] = None
        self.checkpoint_path: Optional[str] = None

        self.start_time = time.time()
        self.last_update_time = time.time()

        # Create progress bar
        total_games = num_iterations * num_games_per_iteration if num_iterations else None
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})" if total_games else "({task.completed})"),
            TimeElapsedColumn(),
            TimeRemainingColumn() if total_games else TextColumn(""),
            console=self.console,
        )
        if total_games:
            self.task_id = self.progress.add_task(
                "Training EucherGo...", total=total_games
            )
        else:
            self.task_id = self.progress.add_task(
                "Training EucherGo...", total=None
            )

    def update(
        self,
        iteration: int,
        games_played: int,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        total_loss: Optional[float] = None,
        avg_mcts_time: Optional[float] = None,
        avg_simulations_per_move: Optional[int] = None,
        avg_game_time: Optional[float] = None,
        avg_iteration_time: Optional[float] = None,
        checkpoint_iteration: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Update dashboard with new metrics.

        Parameters
        ----------
        iteration : int
            Current training iteration.
        games_played : int
            Total games played so far.
        policy_loss : Optional[float]
            Policy loss from last update.
        value_loss : Optional[float]
            Value loss from last update.
        total_loss : Optional[float]
            Total loss from last update.
        avg_mcts_time : Optional[float]
            Average MCTS time per decision.
        avg_simulations_per_move : Optional[int]
            Average number of MCTS simulations per move.
        avg_game_time : Optional[float]
            Average time per game.
        avg_iteration_time : Optional[float]
            Average time per iteration.
        checkpoint_iteration : Optional[int]
            Last checkpoint iteration number.
        checkpoint_path : Optional[str]
            Path to last checkpoint.
        """
        self.current_iteration = iteration
        self.games_played = games_played
        self.total_games = games_played

        if policy_loss is not None:
            self.policy_loss = policy_loss
        if value_loss is not None:
            self.value_loss = value_loss
        if total_loss is not None:
            self.total_loss = total_loss
        if avg_mcts_time is not None:
            self.avg_mcts_time = avg_mcts_time
        if avg_simulations_per_move is not None:
            self.avg_simulations_per_move = avg_simulations_per_move
        if avg_game_time is not None:
            self.avg_game_time = avg_game_time
        if avg_iteration_time is not None:
            self.avg_iteration_time = avg_iteration_time
        if checkpoint_iteration is not None:
            self.last_checkpoint_iteration = checkpoint_iteration
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path

        # Update progress bar
        self.progress.update(self.task_id, completed=games_played)
        self.last_update_time = time.time()

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
        table.add_row("Iteration", f"{self.current_iteration}")
        table.add_row("Games Played", f"{self.games_played}")
        table.add_row("Games/Iteration", f"{self.num_games_per_iteration}")

        # Loss metrics (if available)
        if self.policy_loss is not None:
            table.add_row("Policy Loss", f"{self.policy_loss:.6f}")
        if self.value_loss is not None:
            table.add_row("Value Loss", f"{self.value_loss:.6f}")
        if self.total_loss is not None:
            table.add_row("Total Loss", f"{self.total_loss:.6f}")

        # MCTS metrics
        if self.avg_mcts_time is not None:
            table.add_row("Avg MCTS Time", f"{self.avg_mcts_time:.3f}s")
        if self.avg_simulations_per_move is not None:
            table.add_row("Avg Simulations/Move", f"{self.avg_simulations_per_move}")

        # Performance metrics
        if self.avg_game_time is not None:
            table.add_row("Avg Game Time", f"{self.avg_game_time:.2f}s")
        if self.avg_iteration_time is not None:
            table.add_row("Avg Iteration Time", f"{self.avg_iteration_time:.2f}s")

        # Time stats
        elapsed = time.time() - self.start_time
        table.add_row("Elapsed Time", f"{elapsed:.1f}s")
        if self.games_played > 0:
            avg_time_per_game = elapsed / self.games_played
            table.add_row("Avg Time/Game", f"{avg_time_per_game:.2f}s")
            if self.num_iterations:
                remaining_iterations = max(0, self.num_iterations - self.current_iteration)
                remaining_games = remaining_iterations * self.num_games_per_iteration
                estimated_remaining = avg_time_per_game * remaining_games
                table.add_row("Est. Remaining", self._format_time(estimated_remaining))

        # Checkpoint info
        if self.last_checkpoint_iteration is not None:
            table.add_row("Last Checkpoint", f"Iter {self.last_checkpoint_iteration}")

        return table

    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds to human-readable h:m:s format.

        Parameters
        ----------
        seconds : float
            Time in seconds.

        Returns
        -------
        str
            Formatted time string (e.g., "4h 51m 20s" or "51m 20s" or "20s").
        """
        if seconds < 0:
            return "0s"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)

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

        table.add_row("MCTS Simulations", f"{getattr(config, 'num_simulations', 'N/A')}")
        batch_size = getattr(config, 'batch_size', None)
        table.add_row("Batch Size", f"{batch_size}" if batch_size is not None else "N/A")
        learning_rate = getattr(config, 'learning_rate', None)
        if learning_rate is not None:
            table.add_row("Learning Rate", f"{learning_rate:.2e}")
        else:
            table.add_row("Learning Rate", "N/A")
        table.add_row("Games/Iteration", f"{self.num_games_per_iteration}")
        device = getattr(config, 'device', None)
        table.add_row("Device", str(device) if device is not None else "N/A")

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
            # Create a horizontal layout for the two tables
            tables_layout = Layout()
            tables_layout.split_row(
                Layout(self._create_metrics_table()),
                Layout(self._create_config_table(config)),
            )
            # Stack progress bar on top of the side-by-side tables
            layout.split_column(
                Layout(self.progress, size=3),
                tables_layout,
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
            # Create a horizontal layout for the two tables
            tables_layout = Layout()
            tables_layout.split_row(
                Layout(self._create_metrics_table()),
                Layout(self._create_config_table(config)),
            )
            # Stack progress bar on top of the side-by-side tables
            layout.split_column(
                Layout(self.progress, size=3),
                tables_layout,
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

        summary.add_row("Total Iterations", f"{self.current_iteration}")
        summary.add_row("Total Games", f"{self.games_played}")
        summary.add_row("Total Time", f"{elapsed:.1f}s")
        if self.games_played > 0:
            summary.add_row("Avg Time/Game", f"{elapsed / self.games_played:.2f}s")
        if self.avg_iteration_time is not None:
            summary.add_row("Avg Time/Iteration", f"{self.avg_iteration_time:.2f}s")

        if self.total_loss is not None:
            summary.add_row("Final Total Loss", f"{self.total_loss:.6f}")
        if self.policy_loss is not None:
            summary.add_row("Final Policy Loss", f"{self.policy_loss:.6f}")
        if self.value_loss is not None:
            summary.add_row("Final Value Loss", f"{self.value_loss:.6f}")

        if self.last_checkpoint_iteration is not None:
            summary.add_row("Last Checkpoint", f"Iter {self.last_checkpoint_iteration}")

        self.console.print()
        self.console.print(summary)

