"""Train EucherHandForest on hands from the hand generator.

``play_training_hand`` deals one hand of a game seed and records the learner.
This module replays several hands of that seed, with the dealer rotating, and
fits one random forest for each recorded decision: order up, call trump, play
a card, and discard.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from eucher.training.data_collector import DECISION_TYPES
from eucher.training.hand_study import dealer_counts, dealers_are_balanced
from eucher.training.self_play import SelfPlayTrainer

MODEL_NAME = "EucherHandForest"
PREFIX = "hand"


def collect_hands(
    output_dir: Path,
    game_seed: str,
    hand_seeds: int,
    replays: int,
    learner: str,
    version: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    progress: bool = True,
) -> dict[str, Any]:
    """Deal hands with the hand generator and store the learner's decisions.

    Parameters
    ----------
    output_dir : Path
        Directory for the ``.npz`` files and ``manifest.json``.
    game_seed : str
        Game whose deal stream supplies the cards.
    hand_seeds : int
        How many hands of that game to deal. Hand 0 is the first hand.
    replays : int
        How many times to replay each hand. The dealer rotates across replays.
    learner : str
        Personality in seat 0. Its decisions are the labels.
    version : str
        Dataset version stored in the manifest.
    val_fraction : float
        Share of each decision's rows reserved for validation.
    test_fraction : float
        Share of each decision's rows reserved for test.
    seed : int
        Seed for the train, validation, and test split.
    progress : bool
        Whether the hand generator shows a progress bar.

    Returns
    -------
    dict[str, Any]
        The manifest written next to the arrays.

    Raises
    ------
    ValueError
        If the hand count or replay count is less than 1, or if no decision
        was recorded.
    """
    if hand_seeds < 1:
        raise ValueError(f"hand_seeds must be >= 1, got {hand_seeds}")
    if replays < 1:
        raise ValueError(f"replays must be >= 1, got {replays}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = SelfPlayTrainer(output_dir=output_dir)
    for hand_seed in range(hand_seeds):
        trainer.run_seeded_random_round(
            seed=game_seed,
            hand_seed=hand_seed,
            num_games=replays,
            learner_type=learner,
            save_data=False,
            show_progress=progress,
            workers=1,
        )
    saved = trainer.data_collector.save_data(
        prefix=PREFIX, use_uuid_naming=False, append=False
    )
    if not saved:
        raise ValueError("The hand generator recorded no decisions")
    studies = trainer.data_collector.hand_studies
    _write_studies(output_dir, studies)
    arrays = load_arrays(output_dir)
    splits = {
        name: assign_split(len(arrays[name]["y"]), val_fraction, test_fraction, seed).tolist()
        for name in arrays
    }
    (output_dir / "splits.json").write_text(json.dumps(splits) + "\n")
    counts = {
        name: int(len(arrays[name]["y"])) if name in arrays else 0 for name in DECISION_TYPES
    }
    manifest = {
        "model_name": MODEL_NAME,
        "version": version,
        "game_seed": str(game_seed),
        "hand_seeds": hand_seeds,
        "replays": replays,
        "learner": learner,
        "hands": len(studies),
        "dealer_counts": dealer_counts(studies),
        "dealer_balanced": dealers_are_balanced(dealer_counts(studies)),
        "counts": counts,
        "checksum": dataset_checksum(arrays),
        "feature_size": int(next(iter(arrays.values()))["X"].shape[1]),
        "source": "eucher.training.hand_study.play_training_hand",
        "preprocessing": (
            "Each hand seed of the game seed is dealt by the hand generator. "
            "The dealer rotates through seats 0, 1, 2, and 3. "
            "Seat 0 decisions are the labels."
        ),
        "studies": "hand_studies.jsonl",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_arrays(data_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load the decision arrays written by :func:`collect_hands`.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``hand_<decision>.npz`` files.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        ``X`` and ``y`` for each decision that was recorded.
    """
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for name in DECISION_TYPES:
        path = data_dir / f"{PREFIX}_{name}.npz"
        if not path.is_file():
            continue
        with np.load(path) as payload:
            loaded[name] = {
                "X": np.asarray(payload["X"], dtype=np.float32),
                "y": np.asarray(payload["y"], dtype=np.int32),
            }
    return loaded


def load_dataset(data_dir: Path) -> dict[str, Any]:
    """Load arrays, splits, and the manifest, checking the checksum.

    Parameters
    ----------
    data_dir : Path
        Dataset directory.

    Returns
    -------
    dict[str, Any]
        ``arrays``, ``splits``, and ``manifest``.

    Raises
    ------
    FileNotFoundError
        If the manifest or the split file is missing.
    ValueError
        If the checksum does not match the arrays.
    """
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No hand dataset at {data_dir}")
    manifest = json.loads(manifest_path.read_text())
    arrays = load_arrays(data_dir)
    checksum = dataset_checksum(arrays)
    if checksum != manifest.get("checksum"):
        raise ValueError(f"{data_dir} checksum does not match the saved arrays")
    splits_path = data_dir / "splits.json"
    if not splits_path.is_file():
        raise FileNotFoundError(f"No split file at {splits_path}")
    splits = {
        name: np.asarray(rows, dtype=np.int8)
        for name, rows in json.loads(splits_path.read_text()).items()
    }
    return {"arrays": arrays, "splits": splits, "manifest": manifest}


def dataset_checksum(arrays: Mapping[str, Mapping[str, np.ndarray]]) -> str:
    """Hash the feature and label bytes in decision order.

    Parameters
    ----------
    arrays : Mapping[str, Mapping[str, np.ndarray]]
        ``X`` and ``y`` for each recorded decision.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    for name in DECISION_TYPES:
        if name not in arrays:
            continue
        features = np.ascontiguousarray(arrays[name]["X"])
        labels = np.ascontiguousarray(arrays[name]["y"])
        digest.update(name.encode())
        digest.update(features.tobytes())
        digest.update(labels.tobytes())
    return digest.hexdigest()


def assign_split(n: int, val_fraction: float, test_fraction: float, seed: int) -> np.ndarray:
    """Assign train, validation, and test codes.

    Parameters
    ----------
    n : int
        Number of rows.
    val_fraction : float
        Share coded ``1``.
    test_fraction : float
        Share coded ``2``. The rest are train, coded ``0``.
    seed : int
        Permutation seed.

    Returns
    -------
    np.ndarray
        Integer codes of shape ``(n,)``. Fewer than five rows are all train.
    """
    if n < 5:
        return np.zeros(n, dtype=np.int8)
    order = np.random.RandomState(seed).permutation(n)
    n_val = int(round(n * val_fraction))
    n_test = int(round(n * test_fraction))
    if n_val + n_test >= n:
        n_val = n // 5
        n_test = n // 5
    split = np.zeros(n, dtype=np.int8)
    split[order[:n_val]] = 1
    split[order[n_val : n_val + n_test]] = 2
    return split


def fit_hand_forest(
    data_dir: Path,
    checkpoint_dir: Path,
    n_estimators: int,
    max_depth: int,
    seed: int,
    config: Mapping[str, Any],
    no_resume: bool = False,
) -> dict[str, Any]:
    """Fit one forest per decision and write the checkpoint.

    Parameters
    ----------
    data_dir : Path
        Directory produced by :func:`collect_hands`.
    checkpoint_dir : Path
        Directory for ``checkpoint_latest.pth`` and the ``.pkl`` heads.
    n_estimators : int
        Trees in each forest.
    max_depth : int
        Maximum tree depth.
    seed : int
        Forest ``random_state`` and the process random seed.
    config : Mapping[str, Any]
        Config stored inside the checkpoint.
    no_resume : bool
        Fit again even when ``checkpoint_latest.pth`` already exists.

    Returns
    -------
    dict[str, Any]
        ``history``, ``checkpoint``, and ``resumed``.

    Raises
    ------
    ValueError
        If no decision has two classes in the training rows.
    """
    latest = checkpoint_dir / "checkpoint_latest.pth"
    if latest.is_file() and not no_resume:
        payload = load_checkpoint(latest)
        return {"history": payload.get("history", []), "checkpoint": latest, "resumed": True}

    dataset = load_dataset(data_dir)
    random.seed(seed)
    np.random.seed(seed)
    history: list[dict[str, Any]] = []
    forests: dict[str, RandomForestClassifier] = {}
    for name, arrays in dataset["arrays"].items():
        split = dataset["splits"][name]
        trained = _fit_one(name, arrays["X"], arrays["y"], split, n_estimators, max_depth, seed)
        if trained is None:
            continue
        forest, record = trained
        forests[name] = forest
        history.append(record)
    if not forests:
        raise ValueError("No decision had two classes in the training rows")

    scores = [row["test_accuracy"] for row in history if row["test_rows"] > 0]
    if not scores:
        scores = [row["train_accuracy"] for row in history]
    payload = {
        "model_name": MODEL_NAME,
        "model_state": forests,
        "optimizer_state": {
            "algorithm": "sklearn.ensemble.RandomForestClassifier.fit",
            "random_state": seed,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        },
        "epoch": 1,
        "best_val_accuracy": float(np.mean(scores)),
        "history": history,
        "decisions": sorted(forests),
        "config": dict(config),
        "rng_state": {"python": random.getstate(), "numpy": np.random.get_state()},
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoint_latest.pth", "checkpoint_best.pth", "checkpoint_epoch_1.pth"):
        _dump(checkpoint_dir / name, payload)
    for name, forest in forests.items():
        _dump(checkpoint_dir / f"{name}.pkl", forest)
    return {
        "history": history,
        "checkpoint": checkpoint_dir / "checkpoint_best.pth",
        "resumed": False,
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a forest checkpoint.

    Parameters
    ----------
    path : Path
        Checkpoint file.

    Returns
    -------
    dict[str, Any]
        The stored payload. ``model_state`` maps a decision to its forest.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is a different model.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No hand-forest checkpoint at {path}")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, AttributeError, TypeError) as exc:
        raise ValueError(f"{path} is not an {MODEL_NAME} checkpoint") from exc
    if not isinstance(payload, dict) or payload.get("model_name") != MODEL_NAME:
        raise ValueError(f"{path} is not an {MODEL_NAME} checkpoint")
    return payload


def evaluate_checkpoint(checkpoint: Path, data_dir: Path, split_name: str) -> dict[str, Any]:
    """Score one split of a fitted forest.

    Parameters
    ----------
    checkpoint : Path
        Checkpoint file.
    data_dir : Path
        Dataset directory.
    split_name : str
        ``train``, ``validation``, or ``test``.

    Returns
    -------
    dict[str, Any]
        Per-decision accuracy, precision, recall, F1, and the confusion matrix.

    Raises
    ------
    ValueError
        If ``split_name`` is unknown or that split has no scored rows.
    """
    split_id = {"train": 0, "validation": 1, "test": 2}.get(split_name)
    if split_id is None:
        raise ValueError(f"Unknown split {split_name!r}")
    payload = load_checkpoint(checkpoint)
    dataset = load_dataset(data_dir)
    decisions: dict[str, Any] = {}
    for name, forest in payload["model_state"].items():
        if name not in dataset["arrays"]:
            continue
        chosen = dataset["splits"][name] == split_id
        if not np.any(chosen):
            continue
        features = dataset["arrays"][name]["X"][chosen]
        labels = dataset["arrays"][name]["y"][chosen]
        predicted = forest.predict(features)
        labels_present = sorted(set(labels.tolist()) | set(predicted.tolist()))
        decisions[name] = {
            "rows": int(len(labels)),
            **_scores(labels, predicted),
            "confusion_matrix": confusion_matrix(labels, predicted, labels=labels_present).tolist(),
        }
    if not decisions:
        raise ValueError(f"Split {split_name!r} has no rows for the fitted decisions")
    return {
        "model_name": MODEL_NAME,
        "checkpoint": str(checkpoint),
        "data_path": str(data_dir),
        "split": split_name,
        "decisions": decisions,
    }


def _fit_one(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    n_estimators: int,
    max_depth: int,
    seed: int,
) -> tuple[RandomForestClassifier, dict[str, Any]] | None:
    """Fit one decision when the training rows contain two classes."""
    train = split == 0
    if int(train.sum()) < 2 or len(np.unique(labels[train])) < 2:
        return None
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=1,
    )
    forest.fit(features[train], labels[train])
    record: dict[str, Any] = {"epoch": 1, "decision": name, "train_rows": int(train.sum())}
    record.update(_named_scores("train", labels[train], forest.predict(features[train])))
    for split_name, code in (("validation", 1), ("test", 2)):
        chosen = split == code
        record[f"{split_name}_rows"] = int(chosen.sum())
        if not np.any(chosen):
            record[f"{split_name}_accuracy"] = None
            record[f"{split_name}_f1"] = None
            continue
        record.update(
            _named_scores(split_name, labels[chosen], forest.predict(features[chosen]))
        )
    return forest, record


def _named_scores(prefix: str, labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return accuracy and F1 under a split prefix."""
    scores = _scores(labels, predicted)
    return {f"{prefix}_accuracy": scores["accuracy"], f"{prefix}_f1": scores["f1"]}


def _scores(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return accuracy, precision, recall, and F1."""
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(
            precision_score(labels, predicted, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(labels, predicted, average="weighted", zero_division=0)),
        "f1": float(f1_score(labels, predicted, average="weighted", zero_division=0)),
    }


def _write_studies(directory: Path, studies: list[dict[str, Any]]) -> None:
    """Write one JSON line per generated hand."""
    lines = [json.dumps(study, sort_keys=True) for study in studies]
    text = ("\n".join(lines) + "\n") if lines else ""
    (directory / "hand_studies.jsonl").write_text(text)


def _dump(path: Path, payload: object) -> None:
    """Pickle ``payload`` to ``path``."""
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
