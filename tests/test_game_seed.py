"""Tests for deal seeds that stay fixed when decisions change."""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from click.testing import CliRunner

from eucher.cards import Suit
from eucher.cli import cli, prompt_for_seed
from eucher.game import Game, parse_seed_text, seeded_random_training_command
from eucher.game_file import load_game, save_game

_PLAYERS: List[Tuple[str, str]] = [
    ("A", "random"),
    ("B", "random"),
    ("C", "random"),
    ("D", "random"),
]


def _order_up(
    player: object,
    turned_card: object,
    dealer_id: int,
    trump_suit: object,
) -> bool:
    """Order up so the hand is played instead of passed."""
    return True


def _stay_with_partner(player: object, trump_suit: Suit) -> bool:
    """Decline going alone."""
    return False


def _dealt(game: Game) -> Tuple[tuple, tuple, tuple]:
    """Deal once and return hands, the turned card, and the kitty."""
    game._deal_cards()
    hands = tuple(
        tuple((card.suit.value, card.rank.value) for card in player.hand) for player in game.players
    )
    assert game.turned_card is not None
    turned = (game.turned_card.suit.value, game.turned_card.rank.value)
    kitty = tuple((card.suit.value, card.rank.value) for card in game.kitty)
    return hands, turned, kitty


def _current_cards(game: Game) -> Tuple[tuple, tuple, tuple]:
    """Return the cards already on the table, without dealing again."""
    hands = tuple(
        tuple((card.suit.value, card.rank.value) for card in player.hand) for player in game.players
    )
    assert game.turned_card is not None
    turned = (game.turned_card.suit.value, game.turned_card.rank.value)
    kitty = tuple((card.suit.value, card.rank.value) for card in game.kitty)
    return hands, turned, kitty


def test_prompt_for_seed_skips_a_non_interactive_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Piped play does not wait for a seed. A passed --seed is kept."""

    class SilentInput:
        """Stand-in for a pipe, which is not a terminal."""

        def isatty(self) -> bool:
            """Report that nobody is sitting at a keyboard."""
            return False

    monkeypatch.setattr("eucher.cli.sys.stdin", SilentInput())
    assert prompt_for_seed(None) is None
    assert prompt_for_seed("42") == "42"


def test_play_help_describes_the_seed() -> None:
    """The play command documents the seed that fixes the dealt hands."""
    result = CliRunner().invoke(cli, ["play", "--help"])
    assert result.exit_code == 0
    assert "--seed" in result.output
    assert "same hands" in result.output


def test_parse_seed_text() -> None:
    """Blank text starts a new game, and integers stay integers."""
    assert parse_seed_text(None) is None
    assert parse_seed_text("   ") is None
    assert parse_seed_text("42") == 42
    assert parse_seed_text(" 42 ") == 42
    assert parse_seed_text("not-a-number") == "not-a-number"


def test_integer_seed_label_is_the_integer() -> None:
    """Replaying an integer seed must pass that integer, not a derived UUID."""
    game = Game(_PLAYERS, seed=42)
    assert game.seed_label == "42"
    assert game.numeric_seed == 42
    assert "42" in seeded_random_training_command(game.seed_label)


def test_same_seed_deals_the_same_cards_after_different_random_draws() -> None:
    """Decision randomness does not move the shuffle for a seed."""
    random.seed(1)
    for _ in range(10):
        random.random()
    first = Game(_PLAYERS, seed=42, seed_decisions=False)
    first_deal = _dealt(first)

    random.seed(99)
    for _ in range(200):
        random.random()
    second = Game(_PLAYERS, seed=42, seed_decisions=False)
    assert _dealt(second) == first_deal


def test_different_seeds_deal_different_cards() -> None:
    """Two seeds are not the same deck."""
    first = _dealt(Game(_PLAYERS, seed=1, seed_decisions=False))
    second = _dealt(Game(_PLAYERS, seed=2, seed_decisions=False))
    assert first != second


def test_seed_decisions_false_does_not_reseed_numpy() -> None:
    """Training replays can keep their own decision seed."""
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    Game(_PLAYERS, seed=5, seed_decisions=False)
    assert np.random.rand() == expected


def test_next_hand_matches_when_the_first_hand_is_played_differently() -> None:
    """Hand 2 is the same deal even if hand 1 was played differently."""

    def play_first_hand(decision_seed: int) -> Tuple[tuple, tuple, tuple]:
        game = Game(_PLAYERS, seed=11, seed_decisions=False)
        random.seed(decision_seed)
        for player in game.players:
            player.profile.decide_order_up = _order_up  # type: ignore[method-assign]
            player.profile.decide_going_alone = _stay_with_partner  # type: ignore[method-assign]
        assert game.play_hand() is True
        assert game._hand_index == 1
        return _dealt(game)

    assert play_first_hand(1) == play_first_hand(2)


def test_passed_hand_redeal_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A redeal of the same hand number does not depend on decision draws."""

    def no_trump(self: object, turned_card: object, dealer_id: int) -> None:
        return None

    monkeypatch.setattr("eucher.game.TrumpSelector.select_trump", no_trump)

    def redeal(decision_seed: int) -> Tuple[tuple, tuple, tuple]:
        game = Game(_PLAYERS, seed=3, seed_decisions=False)
        random.seed(decision_seed)
        assert game.play_hand() is True
        assert game._hand_index == 0
        assert game._redeal_index == 1
        return _dealt(game)

    assert redeal(1) == redeal(8)


def test_saved_integer_seed_reloads_the_same_deal(tmp_path: Path) -> None:
    """A saved game remembers an integer seed instead of only its file UUID."""
    game = Game(_PLAYERS, seed=42, seed_decisions=False)
    opening = _dealt(game)
    path = save_game(game, tmp_path)
    loaded = load_game(path)
    assert loaded.seed_label == "42"
    assert loaded.numeric_seed == 42
    assert loaded.game_uuid == game.game_uuid
    assert _dealt(loaded) == opening
