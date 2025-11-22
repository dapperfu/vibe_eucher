"""Collect training metrics during ML model training."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricsCollector:
    """Collects and tracks training metrics."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """
        Initialize the metrics collector.

        Parameters
        ----------
        output_dir : Optional[Path]
            Directory for saving metrics. If None, uses default.
        """
        self.output_dir = output_dir or Path("training_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.metrics: Dict[str, List[Any]] = defaultdict(list)
        self.decision_weights: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Game-level metrics
        self.game_metrics: List[Dict[str, Any]] = []

    def record_metric(self, name: str, value: Any) -> None:
        """
        Record a metric value.

        Parameters
        ----------
        name : str
            Metric name.
        value : Any
            Metric value.
        """
        self.metrics[name].append(value)

    def record_decision_weight(
        self,
        decision_type: str,
        action: str,
        weight: float,
        temperature: float,
        selected: bool,
    ) -> None:
        """
        Record a decision weight for analysis.

        Parameters
        ----------
        decision_type : str
            Type of decision (e.g., "order_up", "call_trump", "play_card", "discard").
        action : str
            Action taken (e.g., "order_up", "pass", "hearts", "card_index").
        weight : float
            Decision weight (confidence score).
        temperature : float
            Temperature threshold used.
        selected : bool
            Whether this action was selected.
        """
        self.decision_weights[decision_type].append(
            {
                "action": action,
                "weight": weight,
                "temperature": temperature,
                "selected": selected,
            }
        )

    def record_game_metrics(
        self,
        game_id: int,
        won: bool,
        game_length: int,
        tricks_won: int,
        score: int,
    ) -> None:
        """
        Record game-level metrics.

        Parameters
        ----------
        game_id : int
            Game identifier.
        won : bool
            Whether the model/player won.
        game_length : int
            Number of hands played.
        tricks_won : int
            Number of tricks won.
        score : int
            Final score.
        """
        self.game_metrics.append(
            {
                "game_id": game_id,
                "won": won,
                "game_length": game_length,
                "tricks_won": tricks_won,
                "score": score,
            }
        )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected metrics.

        Returns
        -------
        Dict[str, Any]
            Dictionary with metric summaries.
        """
        summary: Dict[str, Any] = {}

        # Time series metrics
        for name, values in self.metrics.items():
            if values:
                if isinstance(values[0], (int, float)):
                    summary[name] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "latest": values[-1],
                        "count": len(values),
                    }
                else:
                    summary[name] = {"count": len(values), "latest": values[-1]}

        # Decision weight statistics
        for decision_type, weights in self.decision_weights.items():
            if weights:
                selected_weights = [w["weight"] for w in weights if w["selected"]]
                all_weights = [w["weight"] for w in weights]
                summary[f"{decision_type}_weights"] = {
                    "mean_selected": sum(selected_weights) / len(selected_weights) if selected_weights else 0.0,
                    "mean_all": sum(all_weights) / len(all_weights) if all_weights else 0.0,
                    "count": len(weights),
                }

        # Game metrics summary
        if self.game_metrics:
            wins = sum(1 for g in self.game_metrics if g["won"])
            summary["games"] = {
                "total": len(self.game_metrics),
                "wins": wins,
                "win_rate": wins / len(self.game_metrics),
                "avg_game_length": sum(g["game_length"] for g in self.game_metrics) / len(self.game_metrics),
                "avg_tricks_won": sum(g["tricks_won"] for g in self.game_metrics) / len(self.game_metrics),
            }

        return summary

    def save_metrics(self, filename: Optional[str] = None) -> Path:
        """
        Save collected metrics to JSON file.

        Parameters
        ----------
        filename : Optional[str]
            Filename for metrics. If None, uses timestamp.

        Returns
        -------
        Path
            Path to saved metrics file.
        """
        import datetime

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"

        filepath = self.output_dir / filename

        data = {
            "metrics": dict(self.metrics),
            "decision_weights": dict(self.decision_weights),
            "game_metrics": self.game_metrics,
            "summary": self.get_metrics_summary(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return filepath

    def reset(self) -> None:
        """Reset all collected metrics."""
        self.metrics.clear()
        self.decision_weights.clear()
        self.game_metrics.clear()

