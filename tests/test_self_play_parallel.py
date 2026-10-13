"""Tests for parallel self-play game collection."""

from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

from eucher.training.self_play import SelfPlayTrainer, resolve_worker_count


def test_resolve_worker_count_uses_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero workers means one process per core, capped by the game count."""
    monkeypatch.setattr("eucher.training.self_play.os.cpu_count", lambda: 8)
    assert resolve_worker_count(0, 100) == 8
    assert resolve_worker_count(0, 3) == 3
    assert resolve_worker_count(4, 100) == 4
    assert resolve_worker_count(1, 100) == 1
    assert resolve_worker_count(8, 0) == 1


def test_resolve_worker_count_rejects_negative() -> None:
    """A negative process count is rejected."""
    with pytest.raises(ValueError, match="workers must be >= 0"):
        resolve_worker_count(-1, 10)


def test_serial_round_does_not_start_a_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default one-process path does not spawn workers."""
    monkeypatch.setattr(
        "eucher.training.self_play.ProcessPoolExecutor",
        MagicMock(side_effect=AssertionError("pool should not start")),
    )
    trainer = SelfPlayTrainer(output_dir=tmp_path)
    trainer.run_game_with_collection = MagicMock()  # type: ignore[method-assign]
    trainer.data_collector.save_data = MagicMock(return_value=[])  # type: ignore[method-assign]
    combinations = [[("A", "random"), ("B", "random"), ("C", "random"), ("D", "random")]]

    trainer.run_training_round(
        num_games=2,
        player_combinations=combinations,
        save_data=False,
        show_progress=False,
        workers=1,
    )

    assert trainer.run_game_with_collection.call_count == 2


def test_parallel_workers_write_datasets(tmp_path: Path) -> None:
    """Each worker writes UUID datasets and the parent records every game."""
    trainer = SelfPlayTrainer(output_dir=tmp_path)
    combinations: List[List[Tuple[str, str]]] = [
        [("A", "random"), ("B", "random"), ("C", "random"), ("D", "random")]
    ]

    trainer.run_training_round(
        num_games=2,
        player_combinations=combinations,
        save_data=True,
        show_progress=False,
        workers=2,
    )

    loaded = trainer.data_collector.load_data()
    assert "play_card" in loaded
    assert loaded["play_card"]["X"].shape[0] > 0
    assert len(trainer.data_collector.game_outcomes) == 2
    assert len(list(tmp_path.glob("*.gz"))) == 2
    assert trainer.data_collector.play_card_data == []


def test_parallel_workers_return_rows_for_named_prefix(tmp_path: Path) -> None:
    """Named dataset files are written once by the parent after workers finish."""
    trainer = SelfPlayTrainer(output_dir=tmp_path)
    combinations: List[List[Tuple[str, str]]] = [
        [("A", "random"), ("B", "random"), ("C", "random"), ("D", "random")]
    ]

    trainer.run_training_round(
        num_games=2,
        player_combinations=combinations,
        save_data=True,
        show_progress=False,
        data_prefix="batch",
        workers=2,
    )

    assert len(trainer.data_collector.game_outcomes) == 2
    assert trainer.data_collector.play_card_data
    assert (tmp_path / "batch_play_card.npz").exists()
    assert list(tmp_path.glob("*.txt")) == []
