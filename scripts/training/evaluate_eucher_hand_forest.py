#!/usr/bin/env python3
"""Score a trained EucherHandForest checkpoint on held-out hands.

Examples
--------
::

    python scripts/training/evaluate_eucher_hand_forest.py
        --checkpoint checkpoints/EucherHandForest/checkpoint_best.pth
        --data-path data/EucherHandForest/v1
        --split test
        --output results/EucherHandForest/evaluation_test.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eucher.training.hand_forest import MODEL_NAME, evaluate_checkpoint  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation command line.

    Returns
    -------
    argparse.ArgumentParser
        Parser with help text for every flag.
    """
    parser = argparse.ArgumentParser(
        description=f"Evaluate an {MODEL_NAME} checkpoint on hands from the hand generator."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint to score")
    parser.add_argument(
        "--data-path", type=Path, required=True, help="Directory containing the hands"
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Dataset split to score",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the metrics")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Score one split and write the metrics.

    Parameters
    ----------
    argv : list of str, optional
        Arguments, excluding the program name. Defaults to the process arguments.
    """
    args = build_parser().parse_args(argv)
    report = evaluate_checkpoint(args.checkpoint, args.data_path, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, scores in report["decisions"].items():
        print(f"{name}: accuracy {scores['accuracy']:.3f} f1 {scores['f1']:.3f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
