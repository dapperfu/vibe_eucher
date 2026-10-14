"""Tests for replaying one deal seed against random opponents."""

import random
import subprocess
import sys
from pathlib import Path
from typing import List, Set, Tuple

from eucher.game import Game
from eucher.training.self_play import (
    SelfPlayTrainer,
    apply_decision_seed,
    random_opponent_lineup,
)


def _cards(game: Game) -> Tuple[tuple, tuple, tuple]:
    """Return hands, the turned card, and the kitty already dealt."""
    hands = tuple(
        tuple((card.suit.value, card.rank.value) for card in player.hand) for player in game.players
    )
    assert game.turned_card is not None
    turned = (game.turned_card.suit.value, game.turned_card.rank.value)
    kitty = tuple((card.suit.value, card.rank.value) for card in game.kitty)
    return hands, turned, kitty


def test_lineup_puts_the_learner_in_the_human_seat() -> None:
    """Seat 0 matches the human in ``eucher play``. The other seats are random."""
    lineup = random_opponent_lineup("ml_sklearn")
    assert lineup[0] == ("Learner", "ml_sklearn")
    assert [profile for _, profile in lineup[1:]] == ["random", "random", "random"]


def test_decision_seed_repeats_for_one_episode_and_changes_for_the_next() -> None:
    """The same episode replays the same choices. The next episode does not."""
    apply_decision_seed(42, 0)
    first = random.random()
    apply_decision_seed(42, 0)
    assert random.random() == first
    apply_decision_seed(42, 1)
    assert random.random() != first


def test_seeded_replays_share_the_opening_deal_and_record_the_learner(tmp_path: Path) -> None:
    """Two replays of one seed start with the same cards and record seat 0 only."""
    openings: List[Tuple[tuple, tuple, tuple]] = []
    seen: Set[int] = set()
    real_deal = Game._deal_cards

    def spy(self: Game) -> None:
        real_deal(self)
        if id(self) not in seen:
            seen.add(id(self))
            openings.append(_cards(self))

    Game._deal_cards = spy  # type: ignore[method-assign]
    try:
        trainer = SelfPlayTrainer(output_dir=tmp_path)
        trainer.run_seeded_random_round(
            seed=42,
            num_games=2,
            learner_type="random",
            save_data=False,
            show_progress=False,
            workers=1,
        )
    finally:
        Game._deal_cards = real_deal  # type: ignore[method-assign]

    assert len(openings) == 2
    assert openings[0] == openings[1]
    assert len(trainer.data_collector.game_outcomes) == 2
    recorded = (
        trainer.data_collector.order_up_data
        + trainer.data_collector.call_trump_data
        + trainer.data_collector.play_card_data
        + trainer.data_collector.discard_data
    )
    assert recorded
    assert {row["player_id"] for row in recorded} == {0}


def test_training_help_documents_seeded_random() -> None:
    """The training script explains the seeded random-opponent mode."""
    result = subprocess.run(
        [sys.executable, "scripts/training/train_models.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "seeded_random" in result.stdout
    assert "--seed" in result.stdout


def test_seeded_random_wrapper_requires_a_seed() -> None:
    """The shell wrapper tells you how to pass the seed."""
    result = subprocess.run(
        ["bash", "scripts/training/train_seeded_random.sh"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "SEED" in result.stderr
