"""Seat assignment for ``eucher play`` opponents and partner."""

import random

import pytest
from click.testing import CliRunner

from eucher.cli import (
    assign_computer_types,
    cli,
    ml_personality_names,
    trained_ml_personality_names,
)


@pytest.fixture(autouse=True)
def _treat_registered_models_as_trained(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep seating tests independent of checkpoints on disk.

    Tests that care about the trained filter override this patch.
    """
    monkeypatch.setattr("eucher.cli.personality_is_trained", lambda name: True)


def test_machine_learning_names_drop_aliases() -> None:
    """PyTorch aliases count as one personality, and heuristics are excluded."""
    names = ml_personality_names()

    assert "ml_pytorch" in names
    assert "pytorch_ai" not in names
    assert "pytorch_strategic" not in names
    assert "eucher_zero" in names
    assert "euchergo" in names
    assert "perceiver_muzero" in names
    assert "reinforcement_eucher" in names
    assert "heuristic" not in names
    assert "random" not in names
    assert "weighted_heuristic" not in names
    assert "ai" not in names


def test_one_opponent_type_fills_both_opponents() -> None:
    """A single opponent type is used for both opponents, not the partner."""
    random.seed(123)
    expected_partner = random.choice(trained_ml_personality_names())

    random.seed(123)
    seats = assign_computer_types("heuristic", None)

    assert seats == ["heuristic", expected_partner, "heuristic"]
    assert expected_partner != "heuristic"


def test_two_opponent_types_are_the_opponents() -> None:
    """Two listed types are the left and right opponents."""
    random.seed(7)
    expected_partner = random.choice(trained_ml_personality_names())

    random.seed(7)
    seats = assign_computer_types("euchergo,eucher_zero", None)

    assert seats == ["euchergo", expected_partner, "eucher_zero"]
    assert seats[0] == "euchergo"
    assert seats[2] == "eucher_zero"


def test_partner_type_overrides_the_random_partner() -> None:
    """An explicit partner is seated between the two opponents."""
    seats = assign_computer_types("EucherGo, Eucher_Zero", "heuristic2")

    assert seats == ["euchergo", "heuristic2", "eucher_zero"]


def test_omitted_types_draw_trained_machine_learning_seats() -> None:
    """Unspecified opponents and the partner come from the trained draw."""
    pool = trained_ml_personality_names()
    random.seed(1)
    left = random.choice(pool)
    right = random.choice(pool)
    partner = random.choice(pool)

    random.seed(1)
    assert assign_computer_types(None, None) == [left, partner, right]
    assert partner in pool


def test_explicit_partner_does_not_consume_the_partner_draw() -> None:
    """A chosen partner leaves the random draw for the two opponents only."""
    pool = trained_ml_personality_names()
    random.seed(3)
    left = random.choice(pool)
    right = random.choice(pool)

    random.seed(3)
    assert assign_computer_types(None, "heuristic") == [left, "heuristic", right]


def test_random_seats_skip_untrained_personalities(monkeypatch: pytest.MonkeyPatch) -> None:
    """An untrained model is not a random partner or opponent."""
    monkeypatch.setattr(
        "eucher.cli.personality_is_trained",
        lambda name: name != "ml_pytorch",
    )

    for seed in range(20):
        random.seed(seed)
        seats = assign_computer_types(None, None)
        assert "ml_pytorch" not in seats
        assert all(seat in trained_ml_personality_names() for seat in seats)


def test_explicit_request_can_still_name_an_untrained_personality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing a type seats it even when that model has not been trained."""
    monkeypatch.setattr("eucher.cli.personality_is_trained", lambda name: False)

    assert assign_computer_types("heuristic", "ml_pytorch") == [
        "heuristic",
        "ml_pytorch",
        "heuristic",
    ]


def test_random_seat_requires_a_trained_personality(monkeypatch: pytest.MonkeyPatch) -> None:
    """A random partner or opponent is an error when nothing has been trained."""
    monkeypatch.setattr("eucher.cli.personality_is_trained", lambda name: False)

    with pytest.raises(ValueError, match="No trained machine-learning"):
        assign_computer_types("heuristic", None)
    with pytest.raises(ValueError, match="No trained machine-learning"):
        assign_computer_types(None, "heuristic")


def test_more_than_two_opponent_types_are_rejected() -> None:
    """Three names are not a way to set the partner."""
    with pytest.raises(ValueError, match="--partner-type"):
        assign_computer_types("heuristic,heuristic2,heuristic", None)


def test_unknown_opponent_type_is_rejected() -> None:
    """An unregistered opponent name is an error."""
    with pytest.raises(ValueError, match="not_a_bot"):
        assign_computer_types("not_a_bot", None)


def test_partner_type_must_be_a_single_personality() -> None:
    """The partner flag accepts one personality."""
    with pytest.raises(ValueError, match="one personality"):
        assign_computer_types("heuristic", "heuristic,heuristic2")


def test_play_help_documents_separate_partner_type() -> None:
    """The play command documents that opponent types do not set the partner."""
    result = CliRunner().invoke(cli, ["play", "--help"])
    help_text = " ".join(result.output.split())

    assert result.exit_code == 0
    assert "--partner-type" in help_text
    assert "Does not set your partner" in help_text
    assert "random machine-learning personality" in help_text
    assert "that has been trained" in help_text


def test_play_command_rejects_three_opponent_types() -> None:
    """Listing three personalities stops before a game starts."""
    result = CliRunner().invoke(
        cli,
        ["play", "--opponent-type", "heuristic,heuristic2,heuristic"],
    )

    assert result.exit_code != 0
    assert "--partner-type" in result.output
