"""Benchmark manager for Monte Carlo simulation performance.

This module handles benchmarking simulation performance on different machines
and stores the results for configuring thinking time.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from eucher.cards import Card, Deck, Suit
from eucher.rules import RulesEngine


class BenchmarkManager:
    """Manages benchmarking of Monte Carlo simulation performance."""

    def __init__(self, benchmark_file: Optional[Path] = None) -> None:
        """
        Initialize the benchmark manager.

        Parameters
        ----------
        benchmark_file : Optional[Path]
            Path to benchmark JSON file. Defaults to `benchmarks/performance.json`.
        """
        if benchmark_file is None:
            benchmark_file = Path(__file__).parent.parent.parent / "benchmarks" / "performance.json"
        self.benchmark_file = Path(benchmark_file)
        self.benchmark_file.parent.mkdir(parents=True, exist_ok=True)
        self.rules = RulesEngine()

    def load_benchmarks(self) -> Dict:
        """
        Load benchmark data from file.

        Returns
        -------
        Dict
            Benchmark data with keys: 'simulations_per_second', 'device', 'timestamp'.
        """
        if not self.benchmark_file.exists():
            return {}
        with open(self.benchmark_file, "r") as f:
            return json.load(f)

    def save_benchmarks(self, data: Dict) -> None:
        """
        Save benchmark data to file.

        Parameters
        ----------
        data : Dict
            Benchmark data to save.
        """
        with open(self.benchmark_file, "w") as f:
            json.dump(data, f, indent=2)

    def run_benchmark(self, num_simulations: int = 1000, warmup: int = 100) -> float:
        """
        Run benchmark to determine simulations per second.

        Parameters
        ----------
        num_simulations : int
            Number of simulations to run for benchmark.
        warmup : int
            Number of warmup simulations to run before timing.

        Returns
        -------
        float
            Simulations per second.
        """
        # Warmup
        for _ in range(warmup):
            self._run_single_simulation()

        # Benchmark
        start_time = time.time()
        for _ in range(num_simulations):
            self._run_single_simulation()
        end_time = time.time()

        elapsed = end_time - start_time
        simulations_per_second = num_simulations / elapsed if elapsed > 0 else 0.0

        return simulations_per_second

    def _run_single_simulation(self) -> None:
        """
        Run a single simulation for benchmarking.

        This simulates a simple card play decision to measure performance.
        """
        deck = Deck()
        deck.shuffle()
        hand = deck.deal(5)
        turned_card = deck.draw_one()
        trump_suit = turned_card.suit

        # Simulate a simple decision: evaluate valid cards
        valid_cards = self.rules.get_valid_plays(hand, None, trump_suit)

        # Simulate trick play
        if valid_cards:
            card_to_play = valid_cards[0]
            remaining_hand = [c for c in hand if c != card_to_play]
            trick_cards = [card_to_play]

            # Simulate remaining players
            remaining_cards = [c for c in deck.cards if c not in hand and c != turned_card]
            for _ in range(3):
                if remaining_cards:
                    trick_cards.append(remaining_cards.pop())

            # Determine winner
            player_ids = list(range(4))
            led_suit = card_to_play.suit
            self.rules.determine_trick_winner(trick_cards, player_ids, led_suit, trump_suit)

    def benchmark_and_save(self, num_simulations: int = 1000) -> Dict:
        """
        Run benchmark and save results.

        Parameters
        ----------
        num_simulations : int
            Number of simulations to run.

        Returns
        -------
        Dict
            Benchmark data including simulations_per_second, device, timestamp.
        """
        # Detect device
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.get_device_name(0)}"
            device_type = "gpu"
        else:
            device = "cpu"
            device_type = "cpu"

        # Run benchmark
        simulations_per_second = self.run_benchmark(num_simulations)

        # Prepare data
        benchmark_data = {
            "simulations_per_second": simulations_per_second,
            "device": device,
            "device_type": device_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_simulations": num_simulations,
        }

        # Save
        self.save_benchmarks(benchmark_data)

        return benchmark_data

    def get_simulations_for_thinking_time(self, thinking_time_seconds: float) -> int:
        """
        Get number of simulations to run for a given thinking time.

        Parameters
        ----------
        thinking_time_seconds : float
            Desired thinking time in seconds (e.g., 0.5, 1.0, 10.0, 60.0, 120.0).

        Returns
        -------
        int
            Number of simulations to run.
        """
        benchmarks = self.load_benchmarks()
        simulations_per_second = benchmarks.get("simulations_per_second", 100.0)

        num_simulations = int(simulations_per_second * thinking_time_seconds)
        return max(1, num_simulations)  # At least 1 simulation

    def get_thinking_times_config(self) -> Dict[str, int]:
        """
        Get configuration for standard thinking times.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping thinking time names to number of simulations.
            Keys: 'fast' (0.5s), 'quick' (1s), 'normal' (10s), 'thorough' (60s), 'deep' (120s).
        """
        thinking_times = {
            "fast": 0.5,
            "quick": 1.0,
            "normal": 10.0,
            "thorough": 60.0,
            "deep": 120.0,
        }

        config = {}
        for name, seconds in thinking_times.items():
            config[name] = self.get_simulations_for_thinking_time(seconds)

        return config

