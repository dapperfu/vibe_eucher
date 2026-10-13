"""Tests that machine-learning personalities refuse to play before training."""

import json
import os
import pickle
from pathlib import Path

import pytest
import torch

from eucher.game import Game
from eucher.personality_training import (
    UntrainedPersonalityError,
    assert_checkpoint_adequate,
    find_latest_checkpoint,
    personality_is_trained,
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


def test_missing_checkpoint_is_not_trained(tmp_path: Path) -> None:
    """A personality with no checkpoint is left out of the random draw."""
    assert personality_is_trained("eucher_zero", tmp_path) is False
    assert personality_is_trained("ml_pytorch", tmp_path) is False
    assert personality_is_trained("heuristic", tmp_path) is False


def test_zero_progress_checkpoint_is_not_trained(tmp_path: Path) -> None:
    """A checkpoint that records no completed training is not selectable."""
    directory = tmp_path / "checkpoints" / "perceiver_muzero"
    directory.mkdir(parents=True)
    torch.save({"metadata": {"game_count": 0, "iteration": 0}}, directory / "checkpoint_final.pt")

    assert personality_is_trained("perceiver_muzero", tmp_path) is False


def test_completed_checkpoint_is_trained(tmp_path: Path) -> None:
    """A checkpoint with completed games can be chosen at random."""
    directory = tmp_path / "checkpoints" / "eucher_zero"
    directory.mkdir(parents=True)
    torch.save({"metadata": {"game_count": 12}}, directory / "checkpoint_final.pt")

    assert personality_is_trained("eucher_zero", tmp_path) is True


def test_newer_zero_progress_checkpoint_blocks_an_older_trained_file(tmp_path: Path) -> None:
    """Play loads the newest checkpoint, so a later zero-progress file is untrained."""
    directory = tmp_path / "checkpoints" / "euchrego"
    directory.mkdir(parents=True)
    trained = directory / "checkpoint_1.pt"
    reset = directory / "checkpoint_2.pt"
    torch.save({"metadata": {"game_count": 4}}, trained)
    torch.save({"game_count": 0}, reset)
    later = trained.stat().st_mtime + 10
    os.utime(reset, (later, later))

    assert personality_is_trained("euchergo", tmp_path) is False


def test_pytorch_weight_file_counts_as_trained(tmp_path: Path) -> None:
    """PyTorch aliases share one weight file."""
    torch.save({"layer": 1}, tmp_path / "pytorch_ai_player.pth")

    assert personality_is_trained("ml_pytorch", tmp_path) is True
    assert personality_is_trained("pytorch_ai", tmp_path) is True


def test_pytorch_metadata_with_zero_games_is_not_trained(tmp_path: Path) -> None:
    """A PyTorch file whose metadata records no training is not selectable."""
    weight = tmp_path / "pytorch_ai_player.pth"
    torch.save({"layer": 1}, weight)
    metadata = {"models": [{"model_path": str(weight), "game_count": 0}]}
    (tmp_path / "model_metadata.json").write_text(json.dumps(metadata))

    assert personality_is_trained("ml_pytorch", tmp_path) is False


def test_rl_zero_games_are_not_trained(tmp_path: Path) -> None:
    """An RL checkpoint that records zero games is not selectable."""
    torch.save({"training_stats": {"games_played": 0}}, tmp_path / "rl_model.pth")

    assert personality_is_trained("ml_rl", tmp_path) is False


def test_rl_completed_games_are_trained(tmp_path: Path) -> None:
    """An RL checkpoint with completed games can be chosen at random."""
    torch.save({"training_stats": {"games_played": 3}}, tmp_path / "rl_model.pth")

    assert personality_is_trained("ml_rl", tmp_path) is True


def test_gan_weight_file_is_trained(tmp_path: Path) -> None:
    """A saved GAN weight file can be chosen at random."""
    torch.save({"generator": 1}, tmp_path / "gan_model.pth")

    assert personality_is_trained("ml_gan", tmp_path) is True


def test_sklearn_requires_all_four_fitted_models(tmp_path: Path) -> None:
    """Supervised play needs every decision model to be fitted."""
    from types import SimpleNamespace

    fitted = SimpleNamespace(classes_=(0, 1))
    bare = SimpleNamespace()
    for filename in ("order_up.pkl", "call_trump.pkl", "play_card.pkl"):
        with (tmp_path / filename).open("wb") as handle:
            pickle.dump(fitted, handle)
    with (tmp_path / "discard.pkl").open("wb") as handle:
        pickle.dump(bare, handle)

    assert personality_is_trained("ml_sklearn", tmp_path) is False

    with (tmp_path / "discard.pkl").open("wb") as handle:
        pickle.dump(fitted, handle)

    assert personality_is_trained("ml_sklearn", tmp_path) is True
