"""Rich-based dashboard for PyTorch self-play training."""

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


class SelfPlayDashboard:
    """Rich-based dashboard for self-play training with comprehensive metrics.

    Parameters
    ----------
    games_per_training : int
        Number of games per training cycle.
    player_config : List[tuple]
        Player configuration.
    """

    def __init__(
        self,
        games_per_training: int = 50,
        player_config: Optional[List] = None,
    ) -> None:
        """Initialize dashboard.

        Parameters
        ----------
        games_per_training : int
            Games per training cycle.
        player_config : Optional[List]
            Player configuration.
        """
        self.console = Console()
        self.games_per_training = games_per_training
        self.player_config = player_config or []
        self.start_time = time.time()

        # Training state
        self.current_cycle = 0
        self.total_games_played = 0
        self.current_game = 0
        self.is_training = False

        # Metrics
        self.loss_history: List[float] = []
        self.learning_rate_history: List[float] = []
        self.win_stats: Dict[str, int] = defaultdict(int)  # Player/team -> wins
        self.trick_stats: Dict[str, List[int]] = defaultdict(list)  # Player -> tricks won per game
        self.score_stats: Dict[str, List[int]] = defaultdict(list)  # Team -> scores per game
        self.recent_losses: List[float] = []  # Last 50 losses for display
        self.win_rate_history: List[float] = []  # Win rate over time (last 20 cycles)
        self.current_loss: Optional[float] = None
        self.current_lr: Optional[float] = None
        self.optimal_batch_size: Optional[int] = None
        self.training_time_per_cycle: List[float] = []  # Time per training cycle
        self.game_time_per_cycle: List[float] = []  # Time per game cycle

        # Game collection stats
        self.samples_collected = {
            "order_up": 0,
            "trump_selection": 0,
            "play_card": 0,
            "discard": 0,
        }

        # Create progress bars
        self.cycle_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console,
        )
        self.cycle_task_id = self.cycle_progress.add_task(
            "Training Cycle", total=None
        )

        self.game_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.game_task_id = self.game_progress.add_task(
            "Playing Games", total=games_per_training
        )

    def update_cycle(self, cycle: int, total_games: int) -> None:
        """Update cycle information.

        Parameters
        ----------
        cycle : int
            Current cycle number.
        total_games : int
            Total games played so far.
        """
        self.current_cycle = cycle
        self.total_games_played = total_games
        self.cycle_progress.update(
            self.cycle_task_id,
            description=f"Training Cycle {cycle}",
            completed=cycle,
        )

    def update_game_progress(self, current: int, total: Optional[int] = None) -> None:
        """Update game progress.

        Parameters
        ----------
        current : int
            Current game number.
        total : Optional[int]
            Total games (if None, uses games_per_training).
        """
        self.current_game = current
        if total is None:
            total = self.games_per_training
        self.game_progress.update(
            self.game_task_id,
            completed=current,
            total=total,
            description=f"Playing Games (Cycle {self.current_cycle + 1})",
        )

    def update_training_status(self, is_training: bool) -> None:
        """Update training status.

        Parameters
        ----------
        is_training : bool
            Whether currently training.
        """
        self.is_training = is_training

    def record_loss(self, loss: float, learning_rate: Optional[float] = None) -> None:
        """Record training loss.

        Parameters
        ----------
        loss : float
            Loss value.
        learning_rate : Optional[float]
            Current learning rate.
        """
        self.current_loss = loss
        self.loss_history.append(loss)
        self.recent_losses.append(loss)
        if len(self.recent_losses) > 50:
            self.recent_losses.pop(0)
        if learning_rate is not None:
            self.current_lr = learning_rate
            self.learning_rate_history.append(learning_rate)

    def record_game_outcome(
        self, winner: Optional[int], scores: tuple, tricks_won: List[List[int]]
    ) -> None:
        """Record game outcome for statistics.

        Parameters
        ----------
        winner : Optional[int]
            Winning team (0 or 1).
        scores : tuple
            Final scores (team0, team1).
        tricks_won : List[List[int]]
            Tricks won per hand for each team.
        """
        if winner is not None:
            self.win_stats[f"Team {winner}"] += 1
        self.score_stats["Team 0"].append(scores[0])
        self.score_stats["Team 1"].append(scores[1])

        # Update win rate history (calculate win rate for last N games)
        total_wins = sum(self.win_stats.values())
        if total_wins > 0:
            team0_wins = self.win_stats.get("Team 0", 0)
            win_rate = (team0_wins / total_wins) * 100
            self.win_rate_history.append(win_rate)
            if len(self.win_rate_history) > 20:
                self.win_rate_history.pop(0)

    def update_samples_collected(
        self, order_up: int = 0, trump_selection: int = 0, play_card: int = 0, discard: int = 0
    ) -> None:
        """Update collected sample counts.

        Parameters
        ----------
        order_up : int
            Order up samples.
        trump_selection : int
            Trump selection samples.
        play_card : int
            Play card samples.
        discard : int
            Discard samples.
        """
        self.samples_collected["order_up"] += order_up
        self.samples_collected["trump_selection"] += trump_selection
        self.samples_collected["play_card"] += play_card
        self.samples_collected["discard"] += discard

    def set_batch_size(self, batch_size: Optional[int]) -> None:
        """Set optimal batch size.

        Parameters
        ----------
        batch_size : Optional[int]
            Optimal batch size.
        """
        self.optimal_batch_size = batch_size

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

        # Current loss with sparkline
        if self.current_loss is not None:
            loss_str = f"{self.current_loss:.4f}"
            if len(self.recent_losses) >= 2:
                if self.recent_losses[-1] < self.recent_losses[-2]:
                    trend = "📉"
                    style = "green"
                else:
                    trend = "📈"
                    style = "red"
                loss_str = Text(f"{loss_str} {trend}", style=style)
            
            # Add sparkline if we have enough data
            table.add_row("🎯 Current Loss", loss_str)
            if len(self.recent_losses) >= 3:
                sparkline = self._create_sparkline(self.recent_losses[-20:], width=25, style="cyan")
                table.add_row("", sparkline)
        else:
            table.add_row("🎯 Current Loss", "[dim]N/A[/dim]")

        # Learning rate with history sparkline
        if self.current_lr is not None:
            lr_str = f"{self.current_lr:.2e}"
            table.add_row("📈 Learning Rate", lr_str)
            if len(self.learning_rate_history) >= 3:
                sparkline = self._create_sparkline(self.learning_rate_history[-15:], width=25, style="yellow")
                table.add_row("", sparkline)
        else:
            table.add_row("📈 Learning Rate", "[dim]N/A[/dim]")

        # Average loss (last 10)
        if len(self.recent_losses) >= 1:
            avg_loss = sum(self.recent_losses[-10:]) / min(len(self.recent_losses), 10)
            min_loss = min(self.recent_losses[-10:])
            max_loss = max(self.recent_losses[-10:])
            table.add_row("📊 Avg Loss (10)", f"{avg_loss:.4f}")
            table.add_row("   Min/Max", f"{min_loss:.4f} / {max_loss:.4f}")
        else:
            table.add_row("📊 Avg Loss (10)", "[dim]N/A[/dim]")

        # Loss trend
        if len(self.recent_losses) >= 5:
            recent_avg = sum(self.recent_losses[-5:]) / 5
            older_avg = sum(self.recent_losses[-10:-5]) / 5 if len(self.recent_losses) >= 10 else recent_avg
            improvement = ((older_avg - recent_avg) / older_avg * 100) if older_avg > 0 else 0
            trend_str = f"{improvement:+.1f}%"
            style = "green" if improvement > 0 else "red" if improvement < 0 else "yellow"
            table.add_row("📉 Trend (5 vs 5)", Text(trend_str, style=style))
        else:
            table.add_row("📉 Trend (5 vs 5)", "[dim]N/A[/dim]")

        # Batch size
        if self.optimal_batch_size is not None:
            table.add_row("⚙️  Batch Size", f"[bold]{self.optimal_batch_size}[/bold]")
        else:
            table.add_row("⚙️  Batch Size", "[dim]Auto[/dim]")

        # Training status
        status_emoji = "🔄" if self.is_training else "⏸️"
        status_text = "Training" if self.is_training else "Playing Games"
        table.add_row("", "")  # Separator
        table.add_row(f"{status_emoji} Status", status_text)

        return table

    def _create_game_stats_table(self) -> Table:
        """Create game statistics table.

        Returns
        -------
        Table
            Rich table with game statistics.
        """
        table = Table(title="[bold yellow]Game Statistics[/bold yellow]", show_header=True, header_style="bold yellow")
        table.add_column("Stat", style="yellow", no_wrap=True, width=22)
        table.add_column("Value", style="magenta")

        # Total games and cycle
        table.add_row("🎮 Total Games", f"[bold]{self.total_games_played}[/bold]")
        table.add_row("🔄 Current Cycle", f"[bold]{self.current_cycle}[/bold]")

        # Win rates with sparklines
        total_wins = sum(self.win_stats.values())
        if total_wins > 0:
            table.add_row("", "")  # Separator
            for team, wins in sorted(self.win_stats.items()):
                win_rate = (wins / total_wins) * 100
                # Emoji based on win rate
                if win_rate >= 60:
                    emoji = "🏆"
                elif win_rate >= 50:
                    emoji = "⭐"
                elif win_rate >= 40:
                    emoji = "🎯"
                else:
                    emoji = "💪"
                
                # Create visual bar
                bar_length = int(win_rate / 2)  # Scale to 30 chars max
                bar = "█" * bar_length + "░" * (30 - bar_length)
                
                table.add_row(
                    f"{emoji} {team}",
                    f"{win_rate:.1f}% ({wins}/{total_wins})"
                )
                table.add_row("", bar)
            
            # Win rate trend sparkline
            if len(self.win_rate_history) >= 3:
                sparkline = self._create_sparkline(self.win_rate_history[-15:], width=25, style="green")
                table.add_row("📈 Win Rate Trend", sparkline)
        else:
            table.add_row("🏆 Win Rates", "[dim]No games completed yet[/dim]")

        # Average scores
        if self.score_stats["Team 0"]:
            table.add_row("", "")  # Separator
            avg_score_0 = sum(self.score_stats["Team 0"]) / len(self.score_stats["Team 0"])
            avg_score_1 = sum(self.score_stats["Team 1"]) / len(self.score_stats["Team 1"])
            table.add_row("📊 Avg Score T0", f"{avg_score_0:.1f}")
            table.add_row("📊 Avg Score T1", f"{avg_score_1:.1f}")

        # Games per hour
        elapsed = time.time() - self.start_time
        if elapsed > 0 and self.total_games_played > 0:
            games_per_hour = (self.total_games_played / elapsed) * 3600
            table.add_row("⚡ Games/Hour", f"[bold]{games_per_hour:.1f}[/bold]")

        # Samples collected
        total_samples = sum(self.samples_collected.values())
        if total_samples > 0:
            table.add_row("", "")  # Separator
            table.add_row("📚 Total Samples", f"[bold]{total_samples}[/bold]")
            for action_type, count in self.samples_collected.items():
                if count > 0:
                    percentage = (count / total_samples) * 100
                    bar_length = int(percentage / 2)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    table.add_row(
                        f"  • {action_type.replace('_', ' ').title()}",
                        f"{count} ({percentage:.1f}%) {bar}"
                    )

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
        
        if self.total_games_played > 0 and elapsed > 0:
            games_per_hour = (self.total_games_played / elapsed) * 3600
            games_per_min = (self.total_games_played / elapsed) * 60
            content_lines.append(Text(f"⚡ Speed: {games_per_hour:.1f} games/hour", style="green"))
            content_lines.append(Text(f"   ({games_per_min:.2f} games/min)", style="dim"))
            avg_game_time = elapsed / self.total_games_played
            content_lines.append(Text(f"🎮 Avg Game: {avg_game_time:.1f}s", style="yellow"))
        
        content_lines.append("")

        # Cycle information
        if self.current_cycle > 0:
            avg_cycle_time = elapsed / self.current_cycle
            content_lines.append(Text(f"🔄 Avg Cycle: {avg_cycle_time:.1f}s", style="magenta"))
        
        # Estimated completion (if max_cycles set)
        content_lines.append("")
        content_lines.append(Text("📊 Current Status", style="bold"))
        content_lines.append(Text(f"   Cycle: {self.current_cycle}", style="cyan"))
        content_lines.append(Text(f"   Games: {self.total_games_played}", style="cyan"))
        
        if self.is_training:
            content_lines.append(Text("   Status: 🔄 Training", style="green"))
        else:
            content_lines.append(Text("   Status: ⏸️  Playing", style="yellow"))

        return Panel(
            Group(*content_lines),
            title="[bold yellow]Performance[/bold yellow]",
            border_style="yellow",
        )

    def _create_player_info_table(self) -> Table:
        """Create player information table.

        Returns
        -------
        Table
            Rich table with player info.
        """
        table = Table(title="Players", show_header=True, header_style="bold green")
        table.add_column("Position", style="green")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="yellow")

        for i, (name, profile_type) in enumerate(self.player_config):
            table.add_row(
                f"Player {i + 1}",
                name,
                profile_type.replace("_", " ").title(),
            )

        return table

    def _create_loss_chart(self) -> Panel:
        """Create loss visualization panel with sparkline.

        Returns
        -------
        Panel
            Rich panel with loss chart.
        """
        if len(self.recent_losses) < 2:
            return Panel(
                Text("📉 Loss Chart: Not enough data yet (need 2+ cycles)", style="dim"),
                title="[bold cyan]Loss Visualization[/bold cyan]",
                border_style="cyan",
            )

        # Create content with sparkline
        content_lines = []
        content_lines.append(Text("📉 Loss Trend (last 50 cycles):", style="bold cyan"))
        content_lines.append("")

        # Add sparkline
        sparkline = self._create_sparkline(self.recent_losses[-50:], width=45, style="cyan")
        content_lines.append(sparkline)
        content_lines.append("")

        # Add statistics
        min_loss = min(self.recent_losses[-20:])
        max_loss = max(self.recent_losses[-20:])
        avg_loss = sum(self.recent_losses[-20:]) / len(self.recent_losses[-20:])
        
        content_lines.append(Text(f"Min: {min_loss:.4f}  Max: {max_loss:.4f}  Avg: {avg_loss:.4f}", style="dim"))
        content_lines.append("")

        # Add trend indicator
        if len(self.recent_losses) >= 5:
            recent_avg = sum(self.recent_losses[-5:]) / 5
            older_avg = sum(self.recent_losses[-10:-5]) / 5 if len(self.recent_losses) >= 10 else recent_avg
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

        # Combine all content
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
            Layout(self.cycle_progress, size=3),
            Layout(self.game_progress, size=3),
        )

        # Middle section: Metrics and stats
        metrics_layout = Layout()
        metrics_layout.split_row(
            Layout(self._create_training_metrics_table(), size=40),
            Layout(self._create_game_stats_table(), size=40),
            Layout(self._create_player_info_table(), size=30),
        )

        # Bottom section: Loss chart and additional info
        bottom_layout = Layout()
        bottom_layout.split_row(
            Layout(self._create_loss_chart(), size=60),
            Layout(self._create_performance_panel(), size=40),
        )

        # Combine everything
        layout.split_column(
            Layout(progress_layout, size=6),
            Layout(metrics_layout, size=14),
            Layout(bottom_layout, size=12),
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

        summary.add_row("Total Cycles", str(self.current_cycle))
        summary.add_row("Total Games", str(self.total_games_played))
        summary.add_row("Training Time", f"{hours}h {minutes}m {seconds}s")
        summary.add_row("Games/Hour", f"{self.total_games_played / (elapsed / 3600):.1f}" if elapsed > 0 else "0")

        if self.current_loss is not None:
            summary.add_row("Final Loss", f"{self.current_loss:.4f}")

        if self.win_stats:
            total_wins = sum(self.win_stats.values())
            for team, wins in sorted(self.win_stats.items()):
                win_rate = (wins / total_wins) * 100
                summary.add_row(f"{team} Win Rate", f"{win_rate:.1f}%")

        self.console.print()
        self.console.print(Panel(summary, border_style="green"))

