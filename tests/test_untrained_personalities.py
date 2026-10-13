"""Tests that machine-learning personalities refuse to play before training."""

from pathlib import Path

import pytest

from eucher.game import Game
from eucher.personality_training import (
    UntrainedPersonalityError,
    assert_checkpoint_adequate,
    find_latest_checkpoint,
)
from plugins.ml.player import MLPlayer


def test_zero_progress_checkpoint_is_rejected(tmp_path: Path) -> None:
    """A checkpoint that records no completed training is not adequate."""
    source = tmp_path / "checkpoint.pt"
    with pytest.raises(UntrainedPersonalityError, match="not been adequately trained"):
        assert_checkpoint_adequate(
            "reinforcement_eucher",
            {"game_count": 0, "trick_count": 0},
            source,
        )


def test_completed_training_checkpoint_is_accepted(tmp_path: Path) -> None:
    """A checkpoint with completed games is adequate."""
    assert_checkpoint_adequate(
        "eucher_zero",
        {"metadata": {"game_count": 4, "iteration": 1}},
        tmp_path / "checkpoint.pt",
    )


def test_find_latest_checkpoint_prefers_newest_file(tmp_path: Path) -> None:
    """The newest checkpoint file in a directory is selected."""
    older = tmp_path / "a.pt"
    newer = tmp_path / "b.pt"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    older.touch()
    # Ensure newer mtime is strictly later.
    newer_mtime = older.stat().st_mtime + 10
    import os

    os.utime(newer, (newer_mtime, newer_mtime))

    assert find_latest_checkpoint(tmp_path) == newer


def test_supervised_player_without_models_raises(tmp_path: Path) -> None:
    """ml_sklearn raises when none of its classifiers have been fitted."""
    with pytest.raises(UntrainedPersonalityError, match="ml_sklearn"):
        MLPlayer(backend="supervised", model_dir=tmp_path, require_trained=True)


def test_supervised_player_can_be_built_for_training(tmp_path: Path) -> None:
    """Training code can still construct an unfitted supervised player."""
    player = MLPlayer(backend="supervised", model_dir=tmp_path, require_trained=False)
    assert player.backend == "supervised"


def test_eucher_zero_without_checkpoint_raises(tmp_path: Path) -> None:
    """EucherZero refuses to play when its checkpoint directory is empty."""
    from plugins.eucher_zero.config import EucherZeroConfig
    from plugins.eucher_zero.player import EucherZeroPlayer

    config = EucherZeroConfig(checkpoint_dir=str(tmp_path / "empty"))
    with pytest.raises(UntrainedPersonalityError, match="eucher_zero"):
        EucherZeroPlayer(config=config, require_trained=True)


def test_game_refuses_untrained_sklearn() -> None:
    """A normal game refuses an untrained supervised personality."""
    with pytest.raises(UntrainedPersonalityError, match="ml_sklearn"):
        Game(
            [
                ("ML1", "ml_sklearn"),
                ("ML2", "heuristic"),
                ("ML3", "heuristic"),
                ("ML4", "heuristic"),
            ]
        )


def test_game_allow_untrained_builds_sklearn(tmp_path: Path) -> None:
    """Trainers can opt in to an untrained supervised seat."""
    # The default models directory is empty in this workspace, so construction
    # succeeds only because allow_untrained skips the checkpoint requirement.
    del tmp_path
    game = Game(
        [
            ("ML1", "ml_sklearn"),
            ("ML2", "heuristic"),
            ("ML3", "heuristic"),
            ("ML4", "heuristic"),
        ],
        allow_untrained=True,
    )
    assert len(game.players) == 4
