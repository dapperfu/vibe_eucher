#!/usr/bin/env python3
"""Score a trained EucherHeuristicTiebreak checkpoint on held-out gray areas.

Examples
--------
::

    python scripts/training/evaluate_eucher_heuristic_tiebreak.py
        --checkpoint checkpoints/EucherHeuristicTiebreak/checkpoint_best.pth
        --data-path data/EucherHeuristicTiebreak/v1
        --split test
        --output results/EucherHeuristicTiebreak/evaluation_test.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from sklearn.metrics import (  # type: ignore[import-untyped]  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from euchre.ai.eucher_heuristic_tiebreak import MODEL_NAME, load_module  # noqa: E402
from euchre.training.gray_area_samples import load_gray_dataset  # noqa: E402

SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation command line.

    Returns
    -------
    argparse.ArgumentParser
        Parser with help text for every flag.
    """
    parser = argparse.ArgumentParser(
        description=f"Evaluate a {MODEL_NAME} checkpoint on gray-area trump decisions."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint to score")
    parser.add_argument(
        "--data-path", type=Path, required=True, help="Directory containing gray_samples.pt"
    )
    parser.add_argument(
        "--split", choices=sorted(SPLIT_IDS), default="test", help="Dataset split to score"
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the metrics")
    return parser


def evaluate(checkpoint: Path, data_path: Path, split_name: str) -> Dict[str, Any]:
    """Score one split.

    Parameters
    ----------
    checkpoint : Path
        Checkpoint file.
    data_path : Path
        Dataset directory.
    split_name : str
        ``train``, ``validation``, or ``test``.

    Returns
    -------
    Dict[str, Any]
        Accuracy, precision, recall, F1, and the confusion matrix.

    Raises
    ------
    ValueError
        If the requested split has no rows.
    """
    payload = load_gray_dataset(data_path)
    mask = payload["split"] == SPLIT_IDS[split_name]
    features = payload["features"][mask]
    labels = payload["labels"][mask]
    if len(labels) == 0:
        raise ValueError(f"The {split_name} split is empty")
    module = load_module(checkpoint)
    module.eval()
    with torch.no_grad():
        predicted = (torch.sigmoid(module(features)) >= 0.5).numpy().astype(int)
    actual = labels.numpy().astype(int)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    return {
        "model_name": MODEL_NAME,
        "checkpoint": str(checkpoint),
        "data_path": str(data_path),
        "split": split_name,
        "samples": int(len(actual)),
        "metrics": {
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
            "confusion_matrix": matrix.tolist(),
        },
    }


def main(argv: Optional[list] = None) -> None:
    """Evaluate a checkpoint and write the JSON report.

    Parameters
    ----------
    argv : list, optional
        Arguments, excluding the program name.
    """
    args = build_parser().parse_args(argv)
    report = evaluate(args.checkpoint, args.data_path, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    metrics = report["metrics"]
    print(
        f"{args.split}: accuracy {metrics['accuracy']:.3f} "
        f"precision {metrics['precision']:.3f} "
        f"recall {metrics['recall']:.3f} "
        f"f1 {metrics['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
