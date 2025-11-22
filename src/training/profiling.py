"""Profiling utilities for training performance analysis."""

import cProfile
import functools
import pstats
import time
from io import StringIO
from pathlib import Path
from typing import Callable, Optional


class Profiler:
    """Context manager for profiling code blocks."""

    def __init__(self, output_file: Optional[Path] = None, sort_by: str = "cumulative") -> None:
        """
        Initialize profiler.

        Parameters
        ----------
        output_file : Optional[Path]
            File to save profile results. If None, prints to stdout.
        sort_by : str
            Sort key for stats (default: "cumulative").
        """
        self.profiler = cProfile.Profile()
        self.output_file = output_file
        self.sort_by = sort_by

    def __enter__(self) -> cProfile.Profile:
        """Enter profiling context."""
        self.profiler.enable()
        return self.profiler

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit profiling context and print/save results."""
        self.profiler.disable()

        # Create stats
        stats_stream = StringIO()
        stats = pstats.Stats(self.profiler, stream=stats_stream)
        stats.sort_stats(self.sort_by)

        # Print top functions
        stats.print_stats(30)  # Top 30 functions

        results = stats_stream.getvalue()

        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "w") as f:
                f.write(results)
            print(f"\nProfile saved to {self.output_file}")
        else:
            print("\n" + "=" * 80)
            print("PROFILING RESULTS")
            print("=" * 80)
            print(results)


def time_function(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Parameters
    ----------
    func : Callable
        Function to time.

    Returns
    -------
    Callable
        Wrapped function with timing.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"[TIMING] {func.__module__}.{func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper


class TimingStats:
    """Collects timing statistics for multiple function calls."""

    def __init__(self) -> None:
        """Initialize timing stats."""
        self.timings: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}

    def record(self, name: str, elapsed: float) -> None:
        """
        Record a timing.

        Parameters
        ----------
        name : str
            Name of the operation.
        elapsed : float
            Elapsed time in seconds.
        """
        if name not in self.timings:
            self.timings[name] = []
            self.counts[name] = 0
        self.timings[name].append(elapsed)
        self.counts[name] += 1

    def get_stats(self) -> dict[str, dict[str, float]]:
        """
        Get statistics for all recorded timings.

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary mapping operation names to stats (total, mean, min, max, count).
        """
        stats = {}
        for name, times in self.timings.items():
            stats[name] = {
                "total": sum(times),
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "count": self.counts[name],
            }
        return stats

    def print_summary(self) -> None:
        """Print a summary of timing statistics."""
        stats = self.get_stats()
        if not stats:
            return

        print("\n" + "=" * 80)
        print("TIMING STATISTICS SUMMARY")
        print("=" * 80)
        header = (
            f"{'Operation':<40} {'Count':<10} {'Total (s)':<12} "
            f"{'Mean (s)':<12} {'Min (s)':<12} {'Max (s)':<12}"
        )
        print(header)
        print("-" * 80)

        # Sort by total time
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True)

        for name, stat in sorted_stats:
            print(
                f"{name:<40} {stat['count']:<10} {stat['total']:<12.4f} "
                f"{stat['mean']:<12.4f} {stat['min']:<12.4f} {stat['max']:<12.4f}"
            )

    def save_summary(self, filepath: Path) -> None:
        """
        Save timing summary to file.

        Parameters
        ----------
        filepath : Path
            Path to save file.
        """
        stats = self.get_stats()
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            f.write("TIMING STATISTICS SUMMARY\n")
            f.write("=" * 80 + "\n")
            header = (
                f"{'Operation':<40} {'Count':<10} {'Total (s)':<12} "
                f"{'Mean (s)':<12} {'Min (s)':<12} {'Max (s)':<12}\n"
            )
            f.write(header)
            f.write("-" * 80 + "\n")

            sorted_stats = sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True)

            for name, stat in sorted_stats:
                f.write(
                    f"{name:<40} {stat['count']:<10} {stat['total']:<12.4f} "
                    f"{stat['mean']:<12.4f} {stat['min']:<12.4f} {stat['max']:<12.4f}\n"
                )


# Global timing stats instance
_global_timing_stats = TimingStats()


def get_timing_stats() -> TimingStats:
    """
    Get the global timing stats instance.

    Returns
    -------
    TimingStats
        Global timing stats.
    """
    return _global_timing_stats


def timed_operation(name: Optional[str] = None) -> Callable:
    """
    Decorator to time operations and record in global stats.

    Parameters
    ----------
    name : Optional[str]
        Custom name for the operation. If None, uses function name.

    Returns
    -------
    Callable
        Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        operation_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            _global_timing_stats.record(operation_name, elapsed)
            return result
        return wrapper
    return decorator

