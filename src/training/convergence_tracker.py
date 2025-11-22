"""Track win rates and detect convergence during training."""

from collections import deque
from typing import Deque, Optional


class ConvergenceTracker:
    """Tracks win rates and detects when model converges on target win rate."""

    def __init__(
        self,
        target_win_rate: float = 0.90,
        window_size: int = 100,
        min_games: int = 50,
    ) -> None:
        """
        Initialize the convergence tracker.

        Parameters
        ----------
        target_win_rate : float
            Target win rate for convergence (default 0.90 = 90%).
        window_size : int
            Number of recent games to consider for convergence check (default 100).
        min_games : int
            Minimum number of games before checking convergence (default 50).
        """
        self.target_win_rate = target_win_rate
        self.window_size = window_size
        self.min_games = min_games

        # Track win/loss results (True = win, False = loss)
        self.results: Deque[bool] = deque(maxlen=window_size)

        # Statistics
        self.total_games: int = 0
        self.total_wins: int = 0

    def record_game(self, won: bool) -> None:
        """
        Record the result of a game.

        Parameters
        ----------
        won : bool
            True if the model/player won, False otherwise.
        """
        self.results.append(won)
        self.total_games += 1
        if won:
            self.total_wins += 1

    def get_current_win_rate(self) -> float:
        """
        Get the current win rate over the tracking window.

        Returns
        -------
        float
            Win rate (0.0-1.0) over the last window_size games.
        """
        if len(self.results) == 0:
            return 0.0
        return sum(self.results) / len(self.results)

    def get_overall_win_rate(self) -> float:
        """
        Get the overall win rate across all games.

        Returns
        -------
        float
            Overall win rate (0.0-1.0) across all recorded games.
        """
        if self.total_games == 0:
            return 0.0
        return self.total_wins / self.total_games

    def has_converged(self) -> bool:
        """
        Check if the model has converged on the target win rate.

        Returns
        -------
        bool
            True if converged (current win rate >= target and minimum games played).
        """
        if self.total_games < self.min_games:
            return False

        current_win_rate = self.get_current_win_rate()
        return current_win_rate >= self.target_win_rate

    def get_stats(self) -> dict:
        """
        Get current statistics.

        Returns
        -------
        dict
            Dictionary with statistics:
            - total_games: Total number of games recorded
            - total_wins: Total number of wins
            - overall_win_rate: Overall win rate
            - window_win_rate: Win rate over tracking window
            - window_size: Current window size
            - target_win_rate: Target win rate for convergence
            - converged: Whether model has converged
        """
        return {
            "total_games": self.total_games,
            "total_wins": self.total_wins,
            "overall_win_rate": self.get_overall_win_rate(),
            "window_win_rate": self.get_current_win_rate(),
            "window_size": len(self.results),
            "target_win_rate": self.target_win_rate,
            "converged": self.has_converged(),
        }

    def reset(self) -> None:
        """Reset the tracker (clear all results)."""
        self.results.clear()
        self.total_games = 0
        self.total_wins = 0

