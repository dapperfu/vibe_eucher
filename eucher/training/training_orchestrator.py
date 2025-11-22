"""Main training coordinator for ML models."""

import time
from pathlib import Path
from typing import Callable, Optional

from eucher.training.convergence_tracker import ConvergenceTracker
from eucher.training.training_scheduler import TrainingScheduler


class TrainingOrchestrator:
    """Coordinates training with time-based or convergence-based stopping."""

    def __init__(
        self,
        scheduler: TrainingScheduler,
        convergence_tracker: Optional[ConvergenceTracker] = None,
        checkpoint_interval: int = 100,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the training orchestrator.

        Parameters
        ----------
        scheduler : TrainingScheduler
            Training scheduler for time/convergence management.
        convergence_tracker : Optional[ConvergenceTracker]
            Convergence tracker. If None and using convergence-based training, creates one.
        checkpoint_interval : int
            Save checkpoint every N games (default 100).
        checkpoint_dir : Optional[Path]
            Directory for checkpoints. If None, uses default.
        """
        self.scheduler = scheduler
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize convergence tracker if needed
        if convergence_tracker is None and scheduler.until_converged:
            convergence_tracker = ConvergenceTracker()
        self.convergence_tracker = convergence_tracker

        # Training state
        self.current_game: int = 0
        self.last_checkpoint_game: int = 0

    def start_training(self) -> None:
        """Start the training session."""
        self.scheduler.start()
        if self.convergence_tracker:
            self.convergence_tracker.reset()
        self.current_game = 0
        self.last_checkpoint_game = 0

    def should_continue(self) -> bool:
        """
        Check if training should continue.

        Returns
        -------
        bool
            True if training should continue.
        """
        converged = False
        if self.convergence_tracker:
            converged = self.convergence_tracker.has_converged()

        return self.scheduler.should_continue(converged=converged)

    def record_game_result(self, won: bool) -> None:
        """
        Record the result of a training game.

        Parameters
        ----------
        won : bool
            True if the model/player won.
        """
        self.current_game += 1
        if self.convergence_tracker:
            self.convergence_tracker.record_game(won)

    def should_checkpoint(self) -> bool:
        """
        Check if it's time to save a checkpoint.

        Returns
        -------
        bool
            True if checkpoint should be saved.
        """
        return (self.current_game - self.last_checkpoint_game) >= self.checkpoint_interval

    def save_checkpoint(self, checkpoint_callback: Callable[[Path], None]) -> None:
        """
        Save a training checkpoint.

        Parameters
        ----------
        checkpoint_callback : Callable[[Path], None]
            Callback function that saves the checkpoint.
            Takes checkpoint path as argument.
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_game_{self.current_game}.pth"
        checkpoint_callback(checkpoint_path)
        self.last_checkpoint_game = self.current_game

    def get_status(self) -> dict:
        """
        Get current training status.

        Returns
        -------
        dict
            Dictionary with training status information.
        """
        status = {
            "current_game": self.current_game,
            "scheduler_progress": self.scheduler.get_progress(),
        }

        if self.convergence_tracker:
            status["convergence_stats"] = self.convergence_tracker.get_stats()

        return status

    def print_status(self) -> None:
        """Print current training status."""
        status = self.get_status()
        print(f"\n=== Training Status (Game {status['current_game']}) ===")

        scheduler_progress = status["scheduler_progress"]
        if scheduler_progress["mode"] == "time_based":
            elapsed = scheduler_progress["elapsed_seconds"]
            remaining = scheduler_progress["remaining_seconds"]
            progress = scheduler_progress["progress_percent"]
            print(f"Mode: Time-based")
            print(f"Elapsed: {elapsed:.1f}s")
            if remaining is not None:
                print(f"Remaining: {remaining:.1f}s")
            if progress is not None:
                print(f"Progress: {progress:.1f}%")
        else:
            print(f"Mode: Convergence-based")

        if self.convergence_tracker:
            stats = status["convergence_stats"]
            print(f"Overall Win Rate: {stats['overall_win_rate']:.2%}")
            print(f"Window Win Rate: {stats['window_win_rate']:.2%} (last {stats['window_size']} games)")
            print(f"Target Win Rate: {stats['target_win_rate']:.2%}")
            print(f"Converged: {stats['converged']}")

        print("=" * 50)

