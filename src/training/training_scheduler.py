"""Manage training schedules (time-based or convergence-based)."""

import time
from datetime import datetime, timedelta
from typing import Optional


class TrainingScheduler:
    """Manages training schedules for time-based or convergence-based training."""

    def __init__(
        self,
        duration_minutes: Optional[int] = None,
        duration_hours: Optional[int] = None,
        duration_days: Optional[int] = None,
        until_converged: bool = False,
    ) -> None:
        """
        Initialize the training scheduler.

        Parameters
        ----------
        duration_minutes : Optional[int]
            Training duration in minutes.
        duration_hours : Optional[int]
            Training duration in hours.
        duration_days : Optional[int]
            Training duration in days.
        until_converged : bool
            If True, train until convergence (ignores duration).
        """
        self.until_converged = until_converged
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Calculate end time from duration
        if not until_converged:
            total_seconds = 0
            if duration_days:
                total_seconds += duration_days * 24 * 60 * 60
            if duration_hours:
                total_seconds += duration_hours * 60 * 60
            if duration_minutes:
                total_seconds += duration_minutes * 60

            if total_seconds > 0:
                self.duration_seconds = total_seconds
            else:
                # Default: 1 hour if no duration specified
                self.duration_seconds = 3600
        else:
            self.duration_seconds = None

    def start(self) -> None:
        """Start the training schedule."""
        self.start_time = datetime.now()
        if self.duration_seconds is not None:
            self.end_time = self.start_time + timedelta(seconds=self.duration_seconds)

    def should_continue(self, converged: bool = False) -> bool:
        """
        Check if training should continue.

        Parameters
        ----------
        converged : bool
            Whether the model has converged (for convergence-based training).

        Returns
        -------
        bool
            True if training should continue, False otherwise.
        """
        if self.start_time is None:
            return True

        if self.until_converged:
            # Continue until converged
            return not converged

        # Time-based: check if we've exceeded duration
        if self.end_time is not None:
            return datetime.now() < self.end_time

        return True

    def get_elapsed_time(self) -> float:
        """
        Get elapsed time in seconds.

        Returns
        -------
        float
            Elapsed time in seconds.
        """
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()

    def get_remaining_time(self) -> Optional[float]:
        """
        Get remaining time in seconds (for time-based training).

        Returns
        -------
        Optional[float]
            Remaining time in seconds, or None if convergence-based or no end time.
        """
        if self.end_time is None:
            return None
        remaining = (self.end_time - datetime.now()).total_seconds()
        return max(0.0, remaining)

    def get_progress(self) -> dict:
        """
        Get training progress information.

        Returns
        -------
        dict
            Dictionary with progress information:
            - elapsed_seconds: Elapsed time in seconds
            - remaining_seconds: Remaining time (None if convergence-based)
            - progress_percent: Progress percentage (None if convergence-based)
            - mode: "time_based" or "convergence_based"
        """
        elapsed = self.get_elapsed_time()
        remaining = self.get_remaining_time()

        progress_percent = None
        if remaining is not None and self.duration_seconds is not None:
            progress_percent = (elapsed / self.duration_seconds) * 100.0

        return {
            "elapsed_seconds": elapsed,
            "remaining_seconds": remaining,
            "progress_percent": progress_percent,
            "mode": "convergence_based" if self.until_converged else "time_based",
        }

