#!/usr/bin/env python3
"""Train EucherHeuristicTiebreak on simulated gray-area trump decisions.

Examples
--------
Train with the checked-in config::

    python scripts/training/train_eucher_heuristic_tiebreak.py

Collect the dataset and stop::

    python scripts/training/train_eucher_heuristic_tiebreak.py --collect-only

Start over and play fewer deals::

    python scripts/training/train_eucher_heuristic_tiebreak.py --no-resume --hands 50
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from euchre.ai.eucher_heuristic_tiebreak import MODEL_NAME, fit_tiebreak  # noqa: E402
from euchre.training.gray_area_samples import (  # noqa: E402
    collect_gray_samples,
    load_gray_dataset,
    save_gray_dataset,
)


def _load_config(path: Path) -> Dict[str, Any]:
    """Load the JSON config.

    Parameters
    ----------
    path : Path
        Config file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration.
    """
    with path.open() as handle:
        return cast(Dict[str, Any], json.load(handle))


def _git_hash() -> str:
    """Return the current git commit, or ``unknown`` when git is unavailable.

    Returns
    -------
    str
        Commit hash.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _system_snapshot() -> Dict[str, Any]:
    """Capture the machine state at one moment of a training run.

    Returns
    -------
    Dict[str, Any]
        Clock time, Python and torch versions, CPU count, load average,
        whether CUDA is present, and memory from ``/proc/meminfo``.
    """
    if hasattr(os, "getloadavg"):
        one, five, fifteen = os.getloadavg()
        load = {"1m": one, "5m": five, "15m": fifteen}
    else:
        load = {"1m": 0.0, "5m": 0.0, "15m": 0.0}
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "load_average": load,
        "cuda_available": torch.cuda.is_available(),
        "memory": _memory_mb(),
    }


def _memory_mb() -> Dict[str, float]:
    """Read available and total memory from ``/proc/meminfo``.

    Returns
    -------
    Dict[str, float]
        Megabytes available and total. Missing values are 0.
    """
    found = {"available_mb": 0.0, "total_mb": 0.0}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                found["available_mb"] = int(line.split()[1]) / 1024.0
            elif line.startswith("MemTotal:"):
                found["total_mb"] = int(line.split()[1]) / 1024.0
    except OSError:
        return found
    return found


def _apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Copy command-line values over the config file.

    Parameters
    ----------
    config : Dict[str, Any]
        Config loaded from disk. Updated in place.
    args : argparse.Namespace
        Parsed arguments. ``None`` leaves the config value alone.
    """
    pairs = [
        ("hands", "data", "hands"),
        ("seed", "data", "seed"),
        ("data_version", "data", "version"),
        ("epochs", "training", "epochs"),
        ("learning_rate", "training", "learning_rate"),
        ("batch_size", "training", "batch_size"),
        ("hidden_size", "model", "hidden_size"),
        ("trick_band", "model", "trick_band"),
        ("advantage_margin", "model", "advantage_margin"),
        ("checkpoint_interval", "checkpoint", "interval"),
    ]
    for arg_name, section, key in pairs:
        value = getattr(args, arg_name)
        if value is not None:
            config[section][key] = value
    if args.data_path is not None:
        config["data"]["train_path"] = args.data_path
    if args.checkpoint_dir is not None:
        config["checkpoint"]["directory"] = args.checkpoint_dir


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write one JSON file, creating parent directories.

    Parameters
    ----------
    path : Path
        Destination.
    payload : Dict[str, Any]
        JSON-serializable mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the training command line.

    Returns
    -------
    argparse.ArgumentParser
        Parser with help text for every flag.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train EucherHeuristicTiebreak, a micro model that breaks ties only "
            "when the euchre heuristic is close. Clear hands, such as all hearts "
            "with a spade turned up, are not in the training set."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / f"{MODEL_NAME}.json",
        help="JSON config file",
    )
    parser.add_argument("--data-path", default=None, help="Directory for the gray-area dataset")
    parser.add_argument(
        "--data-version", default=None, help="Dataset version identifier stored in the manifest"
    )
    parser.add_argument(
        "--hands", type=int, default=None, help="Simulated deals used to find gray-area decisions"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for deals and weight initialization"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=None, help="Adam learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument(
        "--hidden-size", type=int, default=None, help="Width of the single hidden layer"
    )
    parser.add_argument(
        "--trick-band",
        type=float,
        default=None,
        help="How close to 3 expected tricks counts as gray",
    )
    parser.add_argument(
        "--advantage-margin",
        type=float,
        default=None,
        help="Win-probability edge that still counts as a tie",
    )
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for checkpoints")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Write a numbered checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint to resume. Default is the latest file.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch instead of the latest checkpoint",
    )
    parser.add_argument(
        "--collect-only", action="store_true", help="Simulate gray-area deals and do not train"
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Simulate again even if the dataset already exists",
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    """Collect gray-area deals and train the tie-break model.

    Parameters
    ----------
    argv : list, optional
        Arguments, excluding the program name. Defaults to the process arguments.
    """
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    _apply_overrides(config, args)
    data_dir = Path(config["data"]["train_path"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    checkpoint_dir = Path(config["checkpoint"]["directory"])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT / checkpoint_dir
    samples_path = data_dir / "gray_samples.pt"
    if args.rebuild_data or args.collect_only or not samples_path.is_file():
        started = time.time()
        dataset = collect_gray_samples(
            hands=int(config["data"]["hands"]),
            seed=int(config["data"]["seed"]),
            version=str(config["data"]["version"]),
            trick_band=float(config["model"]["trick_band"]),
            advantage_margin=float(config["model"]["advantage_margin"]),
        )
        manifest = save_gray_dataset(
            dataset,
            data_dir,
            val_fraction=float(config["data"]["val_fraction"]),
            test_fraction=float(config["data"]["test_fraction"]),
        )
        manifest["git_hash"] = _git_hash()
        manifest["collected_seconds"] = round(time.time() - started, 3)
        (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"Kept {manifest['samples']} gray decisions "
            f"and dropped {manifest['clear_decisions']} clear decisions "
            f"from {manifest['hands_played']} deals."
        )
    if args.collect_only:
        return

    payload = load_gray_dataset(data_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    log_dir = Path(config["logging"]["directory"])
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    run_dir = log_dir / run_id
    started = time.time()
    system_start = _system_snapshot()
    result = fit_tiebreak(
        features=payload["features"],
        labels=payload["labels"],
        split=payload["split"],
        config=config,
        checkpoint_dir=checkpoint_dir,
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        batch_size=int(config["training"]["batch_size"]),
        hidden_size=int(config["model"]["hidden_size"]),
        checkpoint_interval=int(config["checkpoint"]["interval"]),
        seed=int(config["data"]["seed"]),
        resume=args.resume,
        no_resume=args.no_resume,
    )
    for record in result["history"]:
        print(
            f"epoch {int(record['epoch'])}: "
            f"train loss {record['train_loss']:.4f} "
            f"val loss {record['val_loss']:.4f} "
            f"val acc {record['val_accuracy']:.3f}"
        )
    manifest = json.loads((data_dir / "manifest.json").read_text())
    split = payload["split"]
    _write_json(
        run_dir / "config.json",
        {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "git_hash": _git_hash(),
            "data_version": config["data"]["version"],
            "data_checksum": manifest.get("checksum"),
            "data_path": str(data_dir),
            "dataset": {
                "samples": int(len(payload["labels"])),
                "train": int((split == 0).sum()),
                "validation": int((split == 1).sum()),
                "test": int((split == 2).sum()),
            },
            "config": config,
            "resumed_from": str(args.resume) if args.resume else None,
        },
    )
    _write_json(
        run_dir / "metrics.json",
        {"epochs": result["history"], "best_val_loss": result["best_val_loss"]},
    )
    _write_json(
        run_dir / "system_metrics.json",
        {
            "start": system_start,
            "end": dict(_system_snapshot(), elapsed_seconds=round(time.time() - started, 3)),
        },
    )
    print(f"Best checkpoint: {result['best_path']}")


if __name__ == "__main__":
    main()
