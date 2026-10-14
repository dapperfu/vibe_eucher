"""Tests for forests trained on hands from the hand generator."""

import json
from pathlib import Path

import numpy as np
import pytest

from eucher.training.hand_forest import (
    MODEL_NAME,
    collect_hands,
    evaluate_checkpoint,
    fit_hand_forest,
    load_checkpoint,
    load_dataset,
)


def _collect(tmp_path: Path) -> dict:
    """Deal two hands, twice each, into ``tmp_path``."""
    return collect_hands(
        output_dir=tmp_path,
        game_seed="7",
        hand_seeds=2,
        replays=2,
        learner="random",
        version="v1",
        val_fraction=0.0,
        test_fraction=0.0,
        seed=3,
        progress=False,
    )


def test_hand_generator_writes_a_checksummed_dataset(tmp_path: Path) -> None:
    """The generator stores seat-0 decisions and a matching checksum."""
    manifest = _collect(tmp_path)
    loaded = load_dataset(tmp_path)
    assert manifest["source"] == "eucher.training.hand_study.play_training_hand"
    assert "dealer" in manifest["preprocessing"]
    assert manifest["hands"] == 4
    assert sum(manifest["counts"].values()) > 0
    assert loaded["manifest"]["checksum"] == manifest["checksum"]
    name = next(iter(loaded["arrays"]))
    arrays = loaded["arrays"][name]
    changed = dict(arrays)
    changed["y"] = arrays["y"].copy()
    changed["y"][0] = int(changed["y"][0]) + 1
    np.savez_compressed(tmp_path / f"hand_{name}.npz", X=changed["X"], y=changed["y"])
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(tmp_path)


def test_fit_writes_a_checkpoint_that_scores(tmp_path: Path) -> None:
    """A tiny forest loads, rejects another model name, and scores its train rows."""
    data_dir = tmp_path / "data"
    collect_hands(
        output_dir=data_dir,
        game_seed="7",
        hand_seeds=2,
        replays=4,
        learner="random",
        version="v1",
        val_fraction=0.0,
        test_fraction=0.25,
        seed=3,
        progress=False,
    )
    checkpoint_dir = tmp_path / "checkpoints"
    result = fit_hand_forest(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        n_estimators=2,
        max_depth=3,
        seed=3,
        config={"model": {"name": MODEL_NAME}},
        no_resume=True,
    )
    assert result["resumed"] is False
    payload = load_checkpoint(result["checkpoint"])
    assert payload["model_name"] == MODEL_NAME
    assert payload["optimizer_state"]["n_estimators"] == 2
    assert payload["rng_state"]["python"]
    heads = list(checkpoint_dir.glob("*.pkl"))
    assert heads
    report = evaluate_checkpoint(result["checkpoint"], data_dir, "train")
    assert report["decisions"]
    kept = checkpoint_dir / "checkpoint_latest.pth"
    kept.write_bytes(b"other-model")
    with pytest.raises(ValueError, match=MODEL_NAME):
        load_checkpoint(kept)


def test_resume_keeps_the_existing_forest(tmp_path: Path) -> None:
    """A second fit keeps the checkpoint unless resume is turned off."""
    data_dir = tmp_path / "data"
    collect_hands(
        output_dir=data_dir,
        game_seed="9",
        hand_seeds=1,
        replays=4,
        learner="random",
        version="v1",
        val_fraction=0.0,
        test_fraction=0.0,
        seed=1,
        progress=False,
    )
    checkpoint_dir = tmp_path / "checkpoints"
    first = fit_hand_forest(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        n_estimators=2,
        max_depth=2,
        seed=1,
        config={"model": {"name": MODEL_NAME}},
        no_resume=True,
    )
    second = fit_hand_forest(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        n_estimators=2,
        max_depth=2,
        seed=1,
        config={"model": {"name": MODEL_NAME}},
        no_resume=False,
    )
    assert first["resumed"] is False
    assert second["resumed"] is True
    assert json.loads((data_dir / "manifest.json").read_text())["version"] == "v1"
