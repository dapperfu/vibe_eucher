#!/usr/bin/env python3
"""Train EucherHandForest on hands from the hand generator.

Examples
--------
Train with the checked-in config::

    python scripts/training/train_eucher_hand_forest.py

Deal the hands and stop::

    python scripts/training/train_eucher_hand_forest.py --collect-only

Deal fewer hands and start over::

    python scripts/training/train_eucher_hand_forest.py --no-resume --hand-seeds 2 --replays 4
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eucher.training.hand_forest import MODEL_NAME, collect_hands, fit_hand_forest  # noqa: E402
from eucher.training.system_info import get_system_info  # noqa: E402


def _git_hash() -> str:
    """Return the current commit, or ``unknown`` when git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_config(path: Path) -> dict[str, Any]:
    """Load a JSON config file."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], payload)


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Copy command-line values over the config. Omitted flags stay as configured."""
    data = config["data"]
    model = config["model"]
    training = config["training"]
    if args.data_path is not None:
        data["train_path"] = args.data_path
    if args.data_version is not None:
        data["version"] = args.data_version
    if args.game_seed is not None:
        data["game_seed"] = args.game_seed
    if args.hand_seeds is not None:
        data["hand_seeds"] = args.hand_seeds
    if args.replays is not None:
        data["replays"] = args.replays
    if args.learner is not None:
        data["learner"] = args.learner
    if args.seed is not None:
        data["seed"] = args.seed
        training["random_state"] = args.seed
    if args.n_estimators is not None:
        model["n_estimators"] = args.n_estimators
        training["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        model["max_depth"] = args.max_depth
        training["max_depth"] = args.max_depth
    if args.checkpoint_dir is not None:
        config["checkpoint"]["directory"] = args.checkpoint_dir


def _resolve(path_text: str) -> Path:
    """Return an absolute path, relative to the repository when needed."""
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build the training command line.

    Returns
    -------
    argparse.ArgumentParser
        Parser with help text for every flag.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train EucherHandForest. The hand generator deals hands of one "
            "game seed, the dealer rotates, and a random forest learns each "
            "recorded decision."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / f"{MODEL_NAME}.json",
        help="JSON config file",
    )
    parser.add_argument("--data-path", default=None, help="Directory for the generated hands")
    parser.add_argument(
        "--data-version", default=None, help="Dataset version stored in the manifest"
    )
    parser.add_argument("--game-seed", default=None, help="Game seed whose hands are dealt")
    parser.add_argument(
        "--hand-seeds", type=int, default=None, help="How many hands of the game to deal"
    )
    parser.add_argument(
        "--replays", type=int, default=None, help="Replays of each hand; the dealer rotates"
    )
    parser.add_argument("--learner", default=None, help="Personality recorded in seat 0")
    parser.add_argument("--seed", type=int, default=None, help="Split seed and forest random_state")
    parser.add_argument("--n-estimators", type=int, default=None, help="Trees in each forest")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for checkpoints")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Fit again instead of keeping the latest checkpoint",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Deal hands with the hand generator and do not fit",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Deal again even if the dataset already exists",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Deal hands and fit EucherHandForest.

    Parameters
    ----------
    argv : list of str, optional
        Arguments, excluding the program name. Defaults to the process arguments.
    """
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    _apply_overrides(config, args)
    data_dir = _resolve(str(config["data"]["train_path"]))
    checkpoint_dir = _resolve(str(config["checkpoint"]["directory"]))
    manifest_path = data_dir / "manifest.json"
    if args.rebuild_data or args.collect_only or not manifest_path.is_file():
        started = time.time()
        manifest = collect_hands(
            output_dir=data_dir,
            game_seed=str(config["data"]["game_seed"]),
            hand_seeds=int(config["data"]["hand_seeds"]),
            replays=int(config["data"]["replays"]),
            learner=str(config["data"]["learner"]),
            version=str(config["data"]["version"]),
            val_fraction=float(config["data"]["val_fraction"]),
            test_fraction=float(config["data"]["test_fraction"]),
            seed=int(config["data"]["seed"]),
        )
        manifest["git_hash"] = _git_hash()
        manifest["collected_seconds"] = round(time.time() - started, 3)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"Dealt {manifest['hands']} hands. "
            f"Decisions: {manifest['counts']}."
        )
    if args.collect_only:
        return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    log_dir = _resolve(str(config["logging"]["directory"])) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = fit_hand_forest(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        n_estimators=int(config["model"]["n_estimators"]),
        max_depth=int(config["model"]["max_depth"]),
        seed=int(config["training"]["random_state"]),
        config=config,
        no_resume=args.no_resume,
    )
    manifest = json.loads(manifest_path.read_text())
    metrics = {
        "resumed": result["resumed"],
        "checkpoint": str(result["checkpoint"]),
        "history": result["history"],
        "checksum": manifest.get("checksum"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _write_json(
        log_dir / "config.json",
        {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "git_hash": _git_hash(),
            "data_version": config["data"]["version"],
            "data_checksum": manifest.get("checksum"),
            "data_path": str(data_dir),
            "config": config,
        },
    )
    _write_json(log_dir / "metrics.json", metrics)
    _write_json(log_dir / "system_metrics.json", get_system_info())
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Experiment log: {log_dir}")


def _write_json(path: Path, payload: object) -> None:
    """Write JSON that can contain numpy values already converted to Python."""
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
