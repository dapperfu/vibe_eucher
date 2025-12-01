"""Rich-based dashboard for ReinforcementEucher training."""

import time
from collections import defaultdict
from typing import Dict, List, Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text


class ReinforcementEucherDashboard:
    """Rich-based dashboard for ReinforcementEucher training.

    Parameters
    ----------
    training_stage : str
        Current training stage ("trick_only" or "full_hand").
    """

    def __init__(self, training_stage: str = "trick_only") -> None:
        """Initialize dashboard.

        Parameters
        ----------
        training_stage : str
            Current training stage.
        """
        self.console = Console()
        self.training_stage = training_stage
        self.start_time = time.time()

        # Training state
        self.current_stage = training_stage
        self.game_count = 0
        self.trick_count = 0
        self.is_training = False

        # Metrics
        self.policy_loss_history: List[float] = []
        self.value_loss_history: List[float] = []
        self.entropy_history: List[float] = []
        self.recent_policy_losses: List[float] = []
        self.recent_value_losses: List[float] = []
        self.current_policy_loss: Optional[float] = None
        self.current_value_loss: Optional[float] = None
        self.current_entropy: Optional[float] = None
        self.current_lr: Optional[float] = None

        # Stage-specific metrics
        if training_stage == "trick_only":
            self.trick_win_rate_history: List[float] = []
            self.tricks_won = 0
            self.tricks_lost = 0
        else:
            self.win_rate_history: List[float] = []
            self.games_won = 0
            self.games_lost = 0

        # Create progress bars
        self.stage_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console,
        )
        self.stage_task_id = self.stage_progress.add_task(
            f"Training Stage: {training_stage}", total=None
        )

        self.training_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.training_task_id = self.training_progress.add_task(
            "Training Progress", total=1000
        )

    def update_stage(self, stage: str, current: int, total: Optional[int] = None) -> None:
        """Update stage information.

        Parameters
        ----------
        stage : str
            Current stage.
        current : int
            Current count.
        total : Optional[int]
            Total count (if known).
        """
        self.current_stage = stage
        if stage == "trick_only":
            self.trick_count = current
        else:
            self.game_count = current

        self.stage_progress.update(
            self.stage_task_id,
            description=f"Training Stage: {stage}",
            completed=current,
            total=total,
        )

    def update_training_progress(self, current: int, total: Optional[int] = None) -> None:
        """Update training progress.

        Parameters
        ----------
        current : int
            Current progress.
        total : Optional[int]
            Total progress.
        """
        if total is None:
            total = 1000
        self.training_progress.update(
            self.training_task_id,
            completed=current,
            total=total,
        )

    def update_training_status(self, is_training: bool) -> None:
        """Update training status.

        Parameters
        ----------
        is_training : bool
            Whether currently training.
        """
        self.is_training = is_training

    def record_losses(
        self,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Record training losses.

        Parameters
        ----------
        policy_loss : float
            Policy loss.
        value_loss : float
            Value loss.
        entropy : float
            Entropy value.
        learning_rate : Optional[float]
            Current learning rate.
        """
        self.current_policy_loss = policy_loss
        self.current_value_loss = value_loss
        self.current_entropy = entropy
        self.policy_loss_history.append(policy_loss)
        self.value_loss_history.append(value_loss)
        self.entropy_history.append(entropy)

        self.recent_policy_losses.append(policy_loss)
        self.recent_value_losses.append(value_loss)

        if len(self.recent_policy_losses) > 50:
            self.recent_policy_losses.pop(0)
        if len(self.recent_value_losses) > 50:
            self.recent_value_losses.pop(0)

        if learning_rate is not None:
            self.current_lr = learning_rate

    def record_trick_outcome(self, won: bool) -> None:
        """Record trick outcome.

        Parameters
        ----------
        won : bool
            Whether trick was won.
        """
        if won:
            self.tricks_won += 1
        else:
            self.tricks_lost += 1

        total = self.tricks_won + self.tricks_lost
        if total > 0:
            win_rate = (self.tricks_won / total) * 100
            self.trick_win_rate_history.append(win_rate)
            if len(self.trick_win_rate_history) > 100:
                self.trick_win_rate_history.pop(0)

    def record_game_outcome(self, won: bool) -> None:
        """Record game outcome.

        Parameters
        ----------
        won : bool
            Whether game was won.
        """
        if won:
            self.games_won += 1
        else:
            self.games_lost += 1

        total = self.games_won + self.games_lost
        if total > 0:
            win_rate = (self.games_won / total) * 100
            self.win_rate_history.append(win_rate)
            if len(self.win_rate_history) > 100:
                self.win_rate_history.pop(0)

    def _create_sparkline(self, data: List[float], width: int = 30, style: str = "cyan") -> Text:
        """Create a simple ASCII sparkline from data.

        Parameters
        ----------
        data : List[float]
            Data points to visualize.
        width : int
            Width of sparkline in characters.
        style : str
            Style for the sparkline.

        Returns
        -------
        Text
            Rich text with sparkline.
        """
        if len(data) < 2:
            return Text("", style=style)

        # Normalize data to 0-1 range
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val > min_val else 1.0

        # Map data to sparkline characters
        spark_chars = "▁▂▃▄▅▆▇█"
        sparkline_str = ""
        for val in data[-width:]:
            normalized = (val - min_val) / range_val if range_val > 0 else 0.5
            char_idx = int(normalized * (len(spark_chars) - 1))
            sparkline_str += spark_chars[char_idx]

        return Text(sparkline_str, style=style)

    def _create_training_metrics_table(self) -> Table:
        """Create training metrics table.

        Returns
        -------
        Table
            Rich table with training metrics.
        """
        table = Table(title="[bold cyan]Training Metrics[/bold cyan]", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", no_wrap=True, width=22)
        table.add_column("Value", style="green")

        # Policy loss
        if self.current_policy_loss is not None:
            loss_str = f"{self.current_policy_loss:.4f}"
            if len(self.recent_policy_losses) >= 2:
                if self.recent_policy_losses[-1] < self.recent_policy_losses[-2]:
                    trend = "📉"
                    style = "green"
                else:
                    trend = "📈"
                    style = "red"
                loss_str = Text(f"{loss_str} {trend}", style=style)
            table.add_row("🎯 Policy Loss", loss_str)
            if len(self.recent_policy_losses) >= 3:
                sparkline = self._create_sparkline(self.recent_policy_losses[-20:], width=25, style="cyan")
                table.add_row("", sparkline)
        else:
            table.add_row("🎯 Policy Loss", "[dim]N/A[/dim]")

        # Value loss
        if self.current_value_loss is not None:
            loss_str = f"{self.current_value_loss:.4f}"
            table.add_row("💎 Value Loss", loss_str)
            if len(self.recent_value_losses) >= 3:
                sparkline = self._create_sparkline(self.recent_value_losses[-20:], width=25, style="yellow")
                table.add_row("", sparkline)
        else:
            table.add_row("💎 Value Loss", "[dim]N/A[/dim]")

        # Entropy
        if self.current_entropy is not None:
            entropy_str = f"{self.current_entropy:.4f}"
            table.add_row("🔀 Entropy", entropy_str)
        else:
            table.add_row("🔀 Entropy", "[dim]N/A[/dim]")

        # Learning rate
        if self.current_lr is not None:
            lr_str = f"{self.current_lr:.2e}"
            table.add_row("📈 Learning Rate", lr_str)
        else:
            table.add_row("📈 Learning Rate", "[dim]N/A[/dim]")

        # Average losses
        if len(self.recent_policy_losses) >= 1:
            avg_policy = sum(self.recent_policy_losses[-10:]) / min(len(self.recent_policy_losses), 10)
            avg_value = sum(self.recent_value_losses[-10:]) / min(len(self.recent_value_losses), 10)
            table.add_row("📊 Avg Policy (10)", f"{avg_policy:.4f}")
            table.add_row("📊 Avg Value (10)", f"{avg_value:.4f}")

        # Training status
        status_emoji = "🔄" if self.is_training else "⏸️"
        status_text = "Training" if self.is_training else "Collecting"
        table.add_row("", "")  # Separator
        table.add_row(f"{status_emoji} Status", status_text)

        return table

    def _create_stage_stats_table(self) -> Table:
        """Create stage-specific statistics table.

        Returns
        -------
        Table
            Rich table with stage statistics.
        """
        table = Table(title="[bold yellow]Stage Statistics[/bold yellow]", show_header=True, header_style="bold yellow")
        table.add_column("Stat", style="yellow", no_wrap=True, width=22)
        table.add_column("Value", style="magenta")

        if self.current_stage == "trick_only":
            table.add_row("🎯 Tricks Trained", f"[bold]{self.trick_count}[/bold]")
            total_tricks = self.tricks_won + self.tricks_lost
            if total_tricks > 0:
                win_rate = (self.tricks_won / total_tricks) * 100
                table.add_row("🏆 Tricks Won", f"{self.tricks_won}")
                table.add_row("💔 Tricks Lost", f"{self.tricks_lost}")
                table.add_row("📊 Win Rate", f"{win_rate:.1f}%")
                
                # Win rate trend
                if len(self.trick_win_rate_history) >= 3:
                    sparkline = self._create_sparkline(self.trick_win_rate_history[-30:], width=25, style="green")
                    table.add_row("📈 Win Rate Trend", sparkline)
            else:
                table.add_row("🏆 Win Rate", "[dim]No tricks completed[/dim]")
        else:
            table.add_row("🎮 Games Trained", f"[bold]{self.game_count}[/bold]")
            total_games = self.games_won + self.games_lost
            if total_games > 0:
                win_rate = (self.games_won / total_games) * 100
                table.add_row("🏆 Games Won", f"{self.games_won}")
                table.add_row("💔 Games Lost", f"{self.games_lost}")
                table.add_row("📊 Win Rate", f"{win_rate:.1f}%")
                
                # Win rate trend
                if len(self.win_rate_history) >= 3:
                    sparkline = self._create_sparkline(self.win_rate_history[-30:], width=25, style="green")
                    table.add_row("📈 Win Rate Trend", sparkline)
            else:
                table.add_row("🏆 Win Rate", "[dim]No games completed[/dim]")

        # Performance metrics
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            if self.current_stage == "trick_only" and self.trick_count > 0:
                tricks_per_hour = (self.trick_count / elapsed) * 3600
                table.add_row("⚡ Tricks/Hour", f"[bold]{tricks_per_hour:.1f}[/bold]")
            elif self.game_count > 0:
                games_per_hour = (self.game_count / elapsed) * 3600
                table.add_row("⚡ Games/Hour", f"[bold]{games_per_hour:.1f}[/bold]")

        return table

    def _create_performance_panel(self) -> Panel:
        """Create performance metrics panel.

        Returns
        -------
        Panel
            Rich panel with performance info.
        """
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        content_lines = []
        content_lines.append(Text("⏱️  Performance Metrics", style="bold yellow"))
        content_lines.append("")

        # Time information
        content_lines.append(Text(f"⏰ Elapsed: {hours}h {minutes}m {seconds}s", style="cyan"))
        
        if elapsed > 0:
            if self.current_stage == "trick_only" and self.trick_count > 0:
                tricks_per_hour = (self.trick_count / elapsed) * 3600
                content_lines.append(Text(f"⚡ Speed: {tricks_per_hour:.1f} tricks/hour", style="green"))
            elif self.game_count > 0:
                games_per_hour = (self.game_count / elapsed) * 3600
                content_lines.append(Text(f"⚡ Speed: {games_per_hour:.1f} games/hour", style="green"))
        
        content_lines.append("")
        content_lines.append(Text("📊 Current Status", style="bold"))
        content_lines.append(Text(f"   Stage: {self.current_stage}", style="cyan"))
        if self.current_stage == "trick_only":
            content_lines.append(Text(f"   Tricks: {self.trick_count}", style="cyan"))
        else:
            content_lines.append(Text(f"   Games: {self.game_count}", style="cyan"))
        
        if self.is_training:
            content_lines.append(Text("   Status: 🔄 Training", style="green"))
        else:
            content_lines.append(Text("   Status: ⏸️  Collecting", style="yellow"))

        return Panel(
            Group(*content_lines),
            title="[bold yellow]Performance[/bold yellow]",
            border_style="yellow",
        )

    def _create_loss_chart(self) -> Panel:
        """Create loss visualization panel.

        Returns
        -------
        Panel
            Rich panel with loss chart.
        """
        if len(self.recent_policy_losses) < 2:
            return Panel(
                Text("📉 Loss Chart: Not enough data yet", style="dim"),
                title="[bold cyan]Loss Visualization[/bold cyan]",
                border_style="cyan",
            )

        content_lines = []
        content_lines.append(Text("📉 Policy Loss Trend (last 50 updates):", style="bold cyan"))
        content_lines.append("")

        # Add sparkline
        sparkline = self._create_sparkline(self.recent_policy_losses[-50:], width=45, style="cyan")
        content_lines.append(sparkline)
        content_lines.append("")

        # Add statistics
        if len(self.recent_policy_losses) >= 5:
            min_loss = min(self.recent_policy_losses[-20:])
            max_loss = max(self.recent_policy_losses[-20:])
            avg_loss = sum(self.recent_policy_losses[-20:]) / len(self.recent_policy_losses[-20:])
            
            content_lines.append(Text(f"Min: {min_loss:.4f}  Max: {max_loss:.4f}  Avg: {avg_loss:.4f}", style="dim"))
            content_lines.append("")

            # Add trend indicator
            recent_avg = sum(self.recent_policy_losses[-5:]) / 5
            older_avg = sum(self.recent_policy_losses[-10:-5]) / 5 if len(self.recent_policy_losses) >= 10 else recent_avg
            if recent_avg < older_avg:
                trend_msg = "📉 Improving! (Loss decreasing)"
                style = "green"
            elif recent_avg > older_avg:
                trend_msg = "📈 Increasing (Loss rising)"
                style = "red"
            else:
                trend_msg = "➡️ Stable"
                style = "yellow"
            content_lines.append(Text(trend_msg, style=style))

        return Panel(
            Group(*content_lines),
            title="[bold cyan]Loss Visualization[/bold cyan]",
            border_style="cyan",
        )

    def _create_layout(self) -> Layout:
        """Create dashboard layout.

        Returns
        -------
        Layout
            Rich layout with all components.
        """
        layout = Layout()

        # Top section: Progress bars
        progress_layout = Layout()
        progress_layout.split_column(
            Layout(self.stage_progress, size=3),
            Layout(self.training_progress, size=3),
        )

        # Middle section: Metrics and stats
        metrics_layout = Layout()
        metrics_layout.split_row(
            Layout(self._create_training_metrics_table(), size=45),
            Layout(self._create_stage_stats_table(), size=45),
            Layout(self._create_performance_panel(), size=30),
        )

        # Bottom section: Loss chart
        bottom_layout = Layout(self._create_loss_chart())

        # Combine everything
        layout.split_column(
            Layout(progress_layout, size=6),
            Layout(metrics_layout, size=14),
            Layout(bottom_layout, size=10),
        )

        return layout

    def start_live_display(self, refresh_rate: float = 2.0) -> Live:
        """Start live updating display.

        Parameters
        ----------
        refresh_rate : float
            Refresh rate in updates per second.

        Returns
        -------
        Live
            Rich Live context manager.
        """
        return Live(self._create_layout(), console=self.console, refresh_per_second=refresh_rate)

    def update_live_display(self, live_display: Live) -> None:
        """Update live display.

        Parameters
        ----------
        live_display : Live
            Live display context manager.
        """
        live_display.update(self._create_layout())

    def print_summary(self) -> None:
        """Print final training summary."""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        summary = Table(title="Training Summary", show_header=True, header_style="bold green")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")

        summary.add_row("Stage", self.current_stage)
        if self.current_stage == "trick_only":
            summary.add_row("Tricks Trained", str(self.trick_count))
        else:
            summary.add_row("Games Trained", str(self.game_count))
        summary.add_row("Training Time", f"{hours}h {minutes}m {seconds}s")

        if self.current_policy_loss is not None:
            summary.add_row("Final Policy Loss", f"{self.current_policy_loss:.4f}")
        if self.current_value_loss is not None:
            summary.add_row("Final Value Loss", f"{self.current_value_loss:.4f}")

        if self.current_stage == "trick_only" and (self.tricks_won + self.tricks_lost) > 0:
            win_rate = (self.tricks_won / (self.tricks_won + self.tricks_lost)) * 100
            summary.add_row("Trick Win Rate", f"{win_rate:.1f}%")
        elif (self.games_won + self.games_lost) > 0:
            win_rate = (self.games_won / (self.games_won + self.games_lost)) * 100
            summary.add_row("Game Win Rate", f"{win_rate:.1f}%")

        self.console.print()
        self.console.print(Panel(summary, border_style="green"))



