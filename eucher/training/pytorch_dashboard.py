"""Rich-based dashboard for PyTorch training with comprehensive metrics display."""

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text


class PyTorchTrainingDashboard:
    """Rich-based dashboard for PyTorch training with live metrics display."""

    def __init__(
        self,
        num_epochs: int,
        num_batches_per_epoch: Optional[int] = None,
        model_type: str = "GAN",
        refresh_rate: float = 2.0,
    ) -> None:
        """
        Initialize the PyTorch training dashboard.

        Parameters
        ----------
        num_epochs : int
            Total number of epochs to train.
        num_batches_per_epoch : Optional[int]
            Number of batches per epoch (for batch-level progress).
        model_type : str
            Type of model being trained ("GAN", "RL", "Supervised").
        refresh_rate : float
            Refresh rate in updates per second.
        """
        self.num_epochs = num_epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.model_type = model_type
        self.refresh_rate = refresh_rate
        self.console = Console()

        # Training metrics storage
        self.metrics_history: Dict[str, deque] = {
            "epoch": deque(maxlen=100),
            "g_loss": deque(maxlen=100),
            "d_loss": deque(maxlen=100),
            "loss": deque(maxlen=100),
            "g_lr": deque(maxlen=100),
            "d_lr": deque(maxlen=100),
            "lr": deque(maxlen=100),
            "reward": deque(maxlen=100),
            "epsilon": deque(maxlen=100),
            "q_value": deque(maxlen=100),
        }

        # Current epoch/batch tracking
        self.current_epoch = 0
        self.current_batch = 0
        self.start_time = time.time()

        # Create progress bars
        self.epoch_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.epoch_task_id = self.epoch_progress.add_task("Training...", total=num_epochs)

        if num_batches_per_epoch:
            self.batch_progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console,
            )
            self.batch_task_id = self.batch_progress.add_task("Batch...", total=num_batches_per_epoch)
        else:
            self.batch_progress = None
            self.batch_task_id = None

    def update_epoch(self, epoch: int, **metrics: float) -> None:
        """
        Update dashboard with new epoch metrics.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed).
        **metrics : float
            Additional metrics to display (loss, learning_rate, etc.).
        """
        self.current_epoch = epoch
        self.epoch_progress.update(self.epoch_task_id, completed=epoch + 1)

        # Store metrics
        self.metrics_history["epoch"].append(epoch)
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

    def update_batch(self, batch: int, **metrics: float) -> None:
        """
        Update dashboard with new batch metrics.

        Parameters
        ----------
        batch : int
            Current batch number (0-indexed).
        **metrics : float
            Additional metrics for this batch.
        """
        if self.batch_progress and self.batch_task_id is not None:
            self.current_batch = batch
            self.batch_progress.update(self.batch_task_id, completed=batch + 1)

    def reset_batch_progress(self) -> None:
        """Reset batch progress for new epoch."""
        if self.batch_progress and self.batch_task_id is not None:
            self.batch_progress.reset(self.batch_task_id)

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
        table.add_column("Current", style="green", justify="right")
        table.add_column("Average", style="blue", justify="right")
        table.add_column("Min", style="magenta", justify="right")
        table.add_column("Max", style="red", justify="right")

        # Helper to get stats from history
        def get_stats(history_key: str) -> Tuple[float, float, float]:
            """Get current, average, min, max from history."""
            if not self.metrics_history[history_key]:
                return 0.0, 0.0, 0.0, 0.0
            values = list(self.metrics_history[history_key])
            current = values[-1] if values else 0.0
            avg = sum(values) / len(values) if values else 0.0
            min_val = min(values) if values else 0.0
            max_val = max(values) if values else 0.0
            return current, avg, min_val, max_val

        # Model-specific metrics
        if self.model_type == "GAN":
            # Generator loss
            g_curr, g_avg, g_min, g_max = get_stats("g_loss")
            table.add_row(
                "Generator Loss",
                f"{g_curr:.4f}",
                f"{g_avg:.4f}",
                f"{g_min:.4f}",
                f"{g_max:.4f}",
            )

            # Discriminator loss
            d_curr, d_avg, d_min, d_max = get_stats("d_loss")
            table.add_row(
                "Discriminator Loss",
                f"{d_curr:.4f}",
                f"{d_avg:.4f}",
                f"{d_min:.4f}",
                f"{d_max:.4f}",
            )

            # Generator learning rate
            if self.metrics_history["g_lr"]:
                g_lr_curr, g_lr_avg, _, _ = get_stats("g_lr")
                table.add_row("Generator LR", f"{g_lr_curr:.6f}", f"{g_lr_avg:.6f}", "-", "-")

            # Discriminator learning rate
            if self.metrics_history["d_lr"]:
                d_lr_curr, d_lr_avg, _, _ = get_stats("d_lr")
                table.add_row("Discriminator LR", f"{d_lr_curr:.6f}", f"{d_lr_avg:.6f}", "-", "-")

        elif self.model_type == "RL":
            # Training loss
            loss_curr, loss_avg, loss_min, loss_max = get_stats("loss")
            table.add_row(
                "Training Loss",
                f"{loss_curr:.4f}",
                f"{loss_avg:.4f}",
                f"{loss_min:.4f}",
                f"{loss_max:.4f}",
            )

            # Learning rate
            if self.metrics_history["lr"]:
                lr_curr, lr_avg, _, _ = get_stats("lr")
                table.add_row("Learning Rate", f"{lr_curr:.6f}", f"{lr_avg:.6f}", "-", "-")

            # Reward
            if self.metrics_history["reward"]:
                reward_curr, reward_avg, reward_min, reward_max = get_stats("reward")
                table.add_row(
                    "Reward",
                    f"{reward_curr:.2f}",
                    f"{reward_avg:.2f}",
                    f"{reward_min:.2f}",
                    f"{reward_max:.2f}",
                )

            # Epsilon (exploration rate)
            if self.metrics_history["epsilon"]:
                eps_curr, eps_avg, eps_min, eps_max = get_stats("epsilon")
                table.add_row(
                    "Epsilon",
                    f"{eps_curr:.4f}",
                    f"{eps_avg:.4f}",
                    f"{eps_min:.4f}",
                    f"{eps_max:.4f}",
                )

            # Q-value
            if self.metrics_history["q_value"]:
                q_curr, q_avg, q_min, q_max = get_stats("q_value")
                table.add_row(
                    "Q-Value",
                    f"{q_curr:.4f}",
                    f"{q_avg:.4f}",
                    f"{q_min:.4f}",
                    f"{q_max:.4f}",
                )

        else:  # Supervised or generic
            # Training loss
            loss_curr, loss_avg, loss_min, loss_max = get_stats("loss")
            table.add_row(
                "Training Loss",
                f"{loss_curr:.4f}",
                f"{loss_avg:.4f}",
                f"{loss_min:.4f}",
                f"{loss_max:.4f}",
            )

            # Learning rate
            if self.metrics_history["lr"]:
                lr_curr, lr_avg, _, _ = get_stats("lr")
                table.add_row("Learning Rate", f"{lr_curr:.6f}", f"{lr_avg:.6f}", "-", "-")

        return table

    def _create_status_table(self) -> Table:
        """
        Create a table with training status information.

        Returns
        -------
        Table
            Rich table with status.
        """
        table = Table(title="Training Status", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)

        table.add_row("Model Type", self.model_type)
        table.add_row("Epoch", f"{self.current_epoch + 1}/{self.num_epochs}")
        if self.num_batches_per_epoch:
            table.add_row("Batch", f"{self.current_batch + 1}/{self.num_batches_per_epoch}")
        table.add_row("Elapsed Time", elapsed_str)

        # Estimated time remaining
        if self.current_epoch > 0:
            time_per_epoch = elapsed / (self.current_epoch + 1)
            remaining_epochs = self.num_epochs - (self.current_epoch + 1)
            remaining_time = time_per_epoch * remaining_epochs
            remaining_str = self._format_time(remaining_time)
            table.add_row("Est. Remaining", remaining_str)

        # Epochs per second
        if elapsed > 0:
            epochs_per_sec = (self.current_epoch + 1) / elapsed
            table.add_row("Epochs/Second", f"{epochs_per_sec:.3f}")

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

    def _create_layout(self) -> Layout:
        """
        Create the dashboard layout.

        Returns
        -------
        Layout
            Rich layout with all dashboard components.
        """
        layout = Layout()

        # Top section: Progress bars
        progress_layout = Layout()
        progress_layout.split_column(
            Layout(self.epoch_progress, size=3),
            Layout(self.batch_progress, size=3) if self.batch_progress else Layout(),
        )

        # Bottom section: Metrics and status tables
        bottom_layout = Layout()
        bottom_layout.split_row(
            Layout(self._create_metrics_table()),
            Layout(self._create_status_table()),
        )

        # Combine everything
        layout.split_column(
            Layout(progress_layout, size=6 if self.batch_progress else 3),
            Layout(bottom_layout),
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
        """Print final training summary."""
        elapsed = time.time() - self.start_time

        summary_table = Table(title="Training Summary", show_header=True, header_style="bold green")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Epochs", str(self.num_epochs))
        summary_table.add_row("Total Time", self._format_time(elapsed))

        if elapsed > 0:
            epochs_per_sec = self.num_epochs / elapsed
            summary_table.add_row("Average Epochs/Second", f"{epochs_per_sec:.3f}")

        # Final metrics
        if self.model_type == "GAN":
            if self.metrics_history["g_loss"]:
                final_g_loss = list(self.metrics_history["g_loss"])[-1]
                avg_g_loss = sum(self.metrics_history["g_loss"]) / len(self.metrics_history["g_loss"])
                summary_table.add_row("Final Generator Loss", f"{final_g_loss:.4f}")
                summary_table.add_row("Average Generator Loss", f"{avg_g_loss:.4f}")

            if self.metrics_history["d_loss"]:
                final_d_loss = list(self.metrics_history["d_loss"])[-1]
                avg_d_loss = sum(self.metrics_history["d_loss"]) / len(self.metrics_history["d_loss"])
                summary_table.add_row("Final Discriminator Loss", f"{final_d_loss:.4f}")
                summary_table.add_row("Average Discriminator Loss", f"{avg_d_loss:.4f}")

        elif self.model_type == "RL" or self.model_type == "Supervised":
            if self.metrics_history["loss"]:
                final_loss = list(self.metrics_history["loss"])[-1]
                avg_loss = sum(self.metrics_history["loss"]) / len(self.metrics_history["loss"])
                summary_table.add_row("Final Loss", f"{final_loss:.4f}")
                summary_table.add_row("Average Loss", f"{avg_loss:.4f}")

        panel = Panel(summary_table, title="[bold green]Training Complete[/bold green]", border_style="green")
        self.console.print(panel)

