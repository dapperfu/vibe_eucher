#!/usr/bin/env python3
"""Analyze training data to identify patterns for improving HeuristicPlayer.

This script loads training data and analyzes decision patterns to identify
what strategies lead to wins vs losses. The insights can be used to improve
the rule-based logic in HeuristicPlayer.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eucher.training.data_collector import GameDataCollector


def load_training_data(data_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load training data from npz files.

    Parameters
    ----------
    data_dir : Path
        Directory containing training data files.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping decision types to their data (X, y).
    """
    collector = GameDataCollector(output_dir=data_dir)
    return collector.load_data(use_uuid_naming=True)


def analyze_order_up_decisions(
    data: Dict[str, np.ndarray], game_outcomes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze order up decisions and their outcomes.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Training data for order up decisions.
    game_outcomes : List[Dict[str, Any]]
        Game outcome information.

    Returns
    -------
    Dict[str, Any]
        Analysis results.
    """
    if "order_up" not in data:
        return {"error": "No order_up data available"}

    X = data["order_up"]["X"]
    y = data["order_up"]["y"]

    # Basic statistics
    total_decisions = len(y)
    order_up_count = np.sum(y == 1)
    pass_count = np.sum(y == 0)
    order_up_rate = order_up_count / total_decisions if total_decisions > 0 else 0.0

    # Analyze feature patterns (simplified - would need to decode features properly)
    # For now, just return basic stats
    return {
        "total_decisions": int(total_decisions),
        "order_up_count": int(order_up_count),
        "pass_count": int(pass_count),
        "order_up_rate": float(order_up_rate),
        "insights": [
            f"Players ordered up {order_up_rate:.1%} of the time",
            "Consider analyzing hand strength features to determine optimal thresholds",
        ],
    }


def analyze_call_trump_decisions(
    data: Dict[str, np.ndarray], game_outcomes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze call trump decisions and their outcomes.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Training data for call trump decisions.
    game_outcomes : List[Dict[str, Any]]
        Game outcome information.

    Returns
    -------
    Dict[str, Any]
        Analysis results.
    """
    if "call_trump" not in data:
        return {"error": "No call_trump data available"}

    X = data["call_trump"]["X"]
    y = data["call_trump"]["y"]

    # Count decisions by suit (y contains suit indices or None)
    suit_counts = Counter()
    for decision in y:
        if decision is not None and not np.isnan(decision):
            suit_counts[int(decision)] += 1

    total_decisions = len(y)
    called_trump_count = sum(suit_counts.values())
    passed_count = total_decisions - called_trump_count

    return {
        "total_decisions": int(total_decisions),
        "called_trump_count": int(called_trump_count),
        "passed_count": int(passed_count),
        "suit_distribution": dict(suit_counts),
        "insights": [
            f"Players called trump {called_trump_count/total_decisions:.1%} of the time",
            "Consider evaluating hand strength across all suits before deciding",
        ],
    }


def analyze_play_card_decisions(
    data: Dict[str, np.ndarray], game_outcomes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze play card decisions and their outcomes.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Training data for play card decisions.
    game_outcomes : List[Dict[str, Any]]
        Game outcome information.

    Returns
    -------
    Dict[str, Any]
        Analysis results.
    """
    if "play_card" not in data:
        return {"error": "No play_card data available"}

    X = data["play_card"]["X"]
    y = data["play_card"]["y"]

    total_decisions = len(y)

    return {
        "total_decisions": int(total_decisions),
        "insights": [
            "Consider analyzing trick win rates by position (leading vs following)",
            "Evaluate card power values in different contexts",
            "Track teammate vs opponent winning patterns",
        ],
    }


def analyze_discard_decisions(
    data: Dict[str, np.ndarray], game_outcomes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze discard decisions and their outcomes.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Training data for discard decisions.
    game_outcomes : List[Dict[str, Any]]
        Game outcome information.

    Returns
    -------
    Dict[str, Any]
        Analysis results.
    """
    if "discard" not in data:
        return {"error": "No discard data available"}

    X = data["discard"]["X"]
    y = data["discard"]["y"]

    total_decisions = len(y)

    return {
        "total_decisions": int(total_decisions),
        "insights": [
            "Consider trump suit when discarding",
            "Keep strong off-suit cards when appropriate",
            "Discard based on hand composition, not just rank",
        ],
    }


def generate_insights_report(
    analysis_results: Dict[str, Dict[str, Any]], output_file: Optional[Path] = None
) -> str:
    """
    Generate a human-readable insights report.

    Parameters
    ----------
    analysis_results : Dict[str, Dict[str, Any]]
        Results from all analysis functions.
    output_file : Optional[Path]
        Optional file to write report to.

    Returns
    -------
    str
        The report as a string.
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("HeuristicPlayer Improvement Analysis Report")
    report_lines.append("=" * 80)
    report_lines.append("")

    for decision_type, results in analysis_results.items():
        report_lines.append(f"## {decision_type.upper().replace('_', ' ')}")
        report_lines.append("")

        if "error" in results:
            report_lines.append(f"  Error: {results['error']}")
            report_lines.append("")
            continue

        # Print statistics
        for key, value in results.items():
            if key != "insights":
                report_lines.append(f"  {key}: {value}")

        # Print insights
        if "insights" in results:
            report_lines.append("")
            report_lines.append("  Recommendations:")
            for insight in results["insights"]:
                report_lines.append(f"    - {insight}")

        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("General Recommendations")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("1. Add hand strength evaluation for trump selection")
    report_lines.append("2. Consider position (dealer's partner vs opponent)")
    report_lines.append("3. Factor in off-suit strength (Aces, Kings)")
    report_lines.append("4. Account for score situation (aggressive when behind)")
    report_lines.append("5. Improve leading strategy (trump vs off-suit)")
    report_lines.append("6. Better teammate/opponent detection in card play")
    report_lines.append("7. Smarter discard logic based on trump suit")
    report_lines.append("")

    report = "\n".join(report_lines)

    if output_file:
        output_file.write_text(report)
        report_lines.append(f"Report saved to: {output_file}")

    return report


def main() -> None:
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze training data to improve HeuristicPlayer"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="training_data",
        help="Directory containing training data (default: training_data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for insights report (default: print to stdout)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)

    print(f"Loading training data from {data_dir}...")
    try:
        data = load_training_data(data_dir)
    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)

    print(f"Found data for: {list(data.keys())}")

    # Analyze each decision type
    game_outcomes = []  # Would need to load from game outcome data
    analysis_results = {}

    if "order_up" in data:
        print("Analyzing order_up decisions...")
        analysis_results["order_up"] = analyze_order_up_decisions(data, game_outcomes)

    if "call_trump" in data:
        print("Analyzing call_trump decisions...")
        analysis_results["call_trump"] = analyze_call_trump_decisions(data, game_outcomes)

    if "play_card" in data:
        print("Analyzing play_card decisions...")
        analysis_results["play_card"] = analyze_play_card_decisions(data, game_outcomes)

    if "discard" in data:
        print("Analyzing discard decisions...")
        analysis_results["discard"] = analyze_discard_decisions(data, game_outcomes)

    # Generate report
    output_file = Path(args.output) if args.output else None
    report = generate_insights_report(analysis_results, output_file)
    print("\n" + report)


if __name__ == "__main__":
    main()


