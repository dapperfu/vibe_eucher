"""Tests for the heuristic player that asks a micro model only on close calls."""

import json
from pathlib import Path
from typing import Sequence

import pytest

from euchre.ai.ai_factory import AIFactory
from euchre.ai.eucher_heuristic_tiebreak import EucherHeuristicTiebreak, fit_tiebreak
from euchre.ai.gray_area import (
    FEATURE_COUNT,
    WEIGHT_MARGIN,
    feature_index,
    is_gray_call,
    trump_count,
)
from euchre.ai.heuristic_player import HeuristicPlayer
from euchre.ai.heuristic_tiebreak_player import HeuristicTiebreakPlayer
from euchre.game import EuchreGame
from euchre.models import Card, Rank, Suit
from euchre.training.gray_area_samples import (
    collect_gray_samples,
    load_gray_dataset,
    save_gray_dataset,
)


def card(rank: Rank, suit: Suit) -> Card:
    """Build one card."""
    return Card(rank, suit)


class FixedModel:
    """Tie-break stand-in that always returns one probability."""

    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.calls = 0

    def call_probability(self, features: Sequence[float]) -> float:
        """Count the call and return the fixed probability."""
        self.calls += 1
        assert len(features) == FEATURE_COUNT
        return self.probability


def seated(
    name: str, seat: int, dealer: int, hand: list, model: FixedModel
) -> HeuristicTiebreakPlayer:
    """A tie-break player that knows the seats and will not load a checkpoint."""
    player = HeuristicTiebreakPlayer(name, model=model, load_default=False)
    player.begin_round(seat, dealer, 0, 0)
    player.hand = list(hand)
    return player


def three_hearts() -> list:
    """Three hearts near the make line, plus two side cards."""
    return [
        card(Rank.NINE, Suit.HEARTS),
        card(Rank.JACK, Suit.HEARTS),
        card(Rank.QUEEN, Suit.HEARTS),
        card(Rank.NINE, Suit.CLUBS),
        card(Rank.TEN, Suit.SPADES),
    ]


def all_hearts() -> list:
    """Five hearts. Nothing here is a spade."""
    return [
        card(Rank.NINE, Suit.HEARTS),
        card(Rank.TEN, Suit.HEARTS),
        card(Rank.QUEEN, Suit.HEARTS),
        card(Rank.KING, Suit.HEARTS),
        card(Rank.ACE, Suit.HEARTS),
    ]


class TestGrayGate:
    """Obvious hands stay with the heuristic. Close three-trump hands do not."""

    def test_all_hearts_with_a_spade_upcard_does_not_ask_the_model(self) -> None:
        """Zero trump in the turned suit is a clear pass."""
        model = FixedModel(0.99)
        player = seated("Ada", 1, 0, all_hearts(), model)
        upcard = card(Rank.ACE, Suit.SPADES)
        assert trump_count(player.hand, upcard.suit) == 0
        assert player.should_order_up(upcard) is False
        assert model.calls == 0
        assert player.last_band == "clear"

    def test_both_bowers_stay_with_the_heuristic(self) -> None:
        """A hand well above three expected tricks does not ask the model."""
        model = FixedModel(0.01)
        hand = [
            card(Rank.JACK, Suit.HEARTS),
            card(Rank.JACK, Suit.DIAMONDS),
            card(Rank.ACE, Suit.CLUBS),
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.NINE, Suit.SPADES),
        ]
        player = seated("Ada", 1, 0, hand, model)
        upcard = card(Rank.QUEEN, Suit.HEARTS)
        plain = HeuristicPlayer("Bea")
        plain.begin_round(1, 0, 0, 0)
        plain.hand = list(hand)
        assert player.should_order_up(upcard) is True
        assert player.should_order_up(upcard) == plain.should_order_up(upcard)
        assert model.calls == 0

    def test_three_hearts_near_the_make_line_asks_the_model(self) -> None:
        """Three hearts with hearts up is a tie the model is allowed to break."""
        upcard = card(Rank.KING, Suit.HEARTS)
        calling = seated("Ada", 1, 0, three_hearts(), FixedModel(0.99))
        passing = seated("Bea", 1, 0, three_hearts(), FixedModel(0.01))
        evaluation = calling.evaluate_call(Suit.HEARTS, upcard, 1, calling._upcard_location(False))
        assert is_gray_call(3, evaluation.expected_tricks, evaluation.advantage)
        assert calling.should_order_up(upcard) is True
        assert passing.should_order_up(upcard) is False
        assert calling.model_consulted is True
        assert passing.model_consulted is True

    def test_dealer_partner_and_opponent_are_different_features(self) -> None:
        """The same three hearts encode who would be handed the upcard."""
        upcard = card(Rank.KING, Suit.HEARTS)
        partner_model = FixedModel(0.5)
        opponent_model = FixedModel(0.5)
        partner = seated("Ada", 0, 2, three_hearts(), partner_model)
        opponent = seated("Bea", 1, 0, three_hearts(), opponent_model)
        partner.should_order_up(upcard)
        opponent.should_order_up(upcard)
        partner_row = partner_model.calls
        assert partner_row == 1
        assert opponent_model.calls == 1
        partner_features = _last_features(partner, upcard)
        opponent_features = _last_features(opponent, upcard)
        assert partner_features[feature_index("dealer_partner")] == 1.0
        assert opponent_features[feature_index("dealer_opponent")] == 1.0
        assert partner_features != opponent_features

    def test_second_round_refuses_the_turned_suit(self) -> None:
        """A gray second-round call still cannot name the suit that was turned down."""
        player = seated("Ada", 1, 0, three_hearts(), FixedModel(0.99))
        named = player.choose_trump_suit(card(Rank.ACE, Suit.SPADES))
        assert named != Suit.SPADES
        if player.last_band == "gray":
            assert player.model_consulted is True
            assert named == Suit.HEARTS

    def test_factory_builds_the_tiebreak_player(self) -> None:
        """The AI factory knows both the plain heuristic and the tie-break."""
        plain = AIFactory.create_ai_player("Ada", "heuristic")
        hybrid = AIFactory.create_ai_player("Bea", "heuristic_tiebreak")
        assert isinstance(plain, HeuristicPlayer)
        assert not isinstance(plain, HeuristicTiebreakPlayer)
        assert isinstance(hybrid, HeuristicTiebreakPlayer)
        assert "heuristic_tiebreak" in AIFactory.get_available_ai_types()


class TestCardTieBreak:
    """Card play and the discard ask the model only when two weights are close."""

    def test_one_legal_card_does_not_ask_the_model(self) -> None:
        """Following suit with a single spade is not a tie."""
        model = FixedModel(0.99)
        player = seated(
            "Ada", 0, 1, [card(Rank.ACE, Suit.SPADES), card(Rank.ACE, Suit.HEARTS)], model
        )
        player.note_trump_call(1, Suit.HEARTS)
        played = player.choose_card_to_play(Suit.SPADES, Suit.HEARTS)
        assert played.suit == Suit.SPADES
        assert model.calls == 0
        assert player.last_band == "clear"

    def test_taking_with_the_only_winner_does_not_ask_the_model(self) -> None:
        """The ace beats the king and the nine does not, so the ace is clear."""
        model = FixedModel(0.01)
        player = seated(
            "Ada", 0, 1, [card(Rank.ACE, Suit.SPADES), card(Rank.NINE, Suit.SPADES)], model
        )
        player.note_trump_call(1, Suit.HEARTS)
        player.note_card_played(1, card(Rank.KING, Suit.SPADES))
        played = player.choose_card_to_play(Suit.SPADES, Suit.HEARTS)
        assert played.rank == Rank.ACE
        assert model.calls == 0
        assert player.last_band == "clear"

    def test_either_bower_on_lead_asks_the_model(self) -> None:
        """The left and right bower are one point apart, so the model may swap them."""
        hand = [
            card(Rank.JACK, Suit.HEARTS),
            card(Rank.JACK, Suit.DIAMONDS),
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.TEN, Suit.SPADES),
            card(Rank.NINE, Suit.SPADES),
        ]
        keep = seated("Ada", 0, 1, hand, FixedModel(0.01))
        keep.note_trump_call(0, Suit.HEARTS)
        kept = keep.choose_card_to_play(None, Suit.HEARTS)
        assert keep.model_consulted is True
        assert kept.suit == Suit.DIAMONDS

        swap = seated("Bea", 0, 1, hand, FixedModel(0.99))
        swap.note_trump_call(0, Suit.HEARTS)
        swapped = swap.choose_card_to_play(None, Suit.HEARTS)
        assert swap.model_consulted is True
        assert swapped.suit == Suit.HEARTS

    def test_a_clear_discard_ignores_the_model(self) -> None:
        """The bare diamond nine is well ahead of every other throw."""
        model = FixedModel(0.99)
        player = seated(
            "Ada",
            0,
            0,
            [
                card(Rank.JACK, Suit.HEARTS),
                card(Rank.ACE, Suit.HEARTS),
                card(Rank.NINE, Suit.CLUBS),
                card(Rank.TEN, Suit.CLUBS),
                card(Rank.NINE, Suit.DIAMONDS),
                card(Rank.ACE, Suit.SPADES),
            ],
            model,
        )
        thrown = player.choose_card_to_discard(Suit.HEARTS)
        assert thrown.suit == Suit.DIAMONDS
        assert thrown.rank == Rank.NINE
        assert model.calls == 0
        assert player.last_band == "clear"

    def test_two_singleton_discards_ask_the_model(self) -> None:
        """Two bare nines have the same throw weight, so the model can pick either."""
        hand = [
            card(Rank.JACK, Suit.HEARTS),
            card(Rank.ACE, Suit.HEARTS),
            card(Rank.NINE, Suit.DIAMONDS),
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.TEN, Suit.SPADES),
            card(Rank.QUEEN, Suit.SPADES),
        ]
        keep = seated("Ada", 0, 0, hand, FixedModel(0.01))
        kept = keep.choose_card_to_discard(Suit.HEARTS)
        assert keep.model_consulted is True
        assert kept == card(Rank.NINE, Suit.DIAMONDS)

        swap = seated("Bea", 0, 0, hand, FixedModel(0.99))
        swapped = swap.choose_card_to_discard(Suit.HEARTS)
        assert swap.model_consulted is True
        assert swapped == card(Rank.NINE, Suit.CLUBS)


def _last_features(player: HeuristicTiebreakPlayer, upcard: Card) -> list:
    """Recompute the feature row for the order-up decision this player just faced."""
    from euchre.ai.gray_area import features_for

    location = player._upcard_location(False)
    evaluation = player.evaluate_call(upcard.suit, upcard, 1, location)
    return features_for(player, evaluation, upcard, 1, location)


class TestGrayAreaTraining:
    """Simulated deals label only the gray spots, and a tiny net can fit them."""

    def test_collection_drops_clear_decisions(self, tmp_path: Path) -> None:
        """Every saved row is gray, and the walk also saw clear decisions."""
        dataset = collect_gray_samples(hands=40, seed=7, progress=False)
        assert dataset.clear_decisions > 0
        assert len(dataset) > 0
        assert any(kind in ("play", "discard") for kind in dataset.kinds)
        for kind, count, expected, advantage in zip(
            dataset.kinds, dataset.trump_counts, dataset.expected_tricks, dataset.advantages
        ):
            if kind == "call":
                assert is_gray_call(count, expected, advantage)
                assert 1 <= count <= 3
            else:
                assert kind in ("play", "discard")
                assert advantage <= WEIGHT_MARGIN
        manifest = save_gray_dataset(dataset, tmp_path)
        loaded = load_gray_dataset(tmp_path)
        assert manifest["checksum"] == loaded["checksum"]
        assert int(loaded["features"].shape[1]) == FEATURE_COUNT
        assert set(loaded["labels"].tolist()) <= {0.0, 1.0}

    def test_fit_writes_a_checkpoint_the_player_can_load(self, tmp_path: Path) -> None:
        """A short fit on collected rows resumes from the checkpoint file."""
        dataset = collect_gray_samples(hands=40, seed=11, progress=False)
        if len(dataset) < 5:
            pytest.skip("this seed did not produce enough gray decisions")
        save_gray_dataset(dataset, tmp_path / "data")
        loaded = load_gray_dataset(tmp_path / "data")
        result = fit_tiebreak(
            features=loaded["features"],
            labels=loaded["labels"],
            split=loaded["split"],
            config={"model": {"name": "EucherHeuristicTiebreak"}},
            checkpoint_dir=tmp_path / "checkpoints",
            epochs=2,
            learning_rate=0.05,
            weight_decay=0.0,
            batch_size=16,
            hidden_size=8,
            checkpoint_interval=1,
            seed=11,
            no_resume=True,
        )
        assert result["best_path"].is_file()
        player = HeuristicTiebreakPlayer(
            "Ada", checkpoint_path=str(result["best_path"]), load_default=False
        )
        player.begin_round(1, 0, 0, 0)
        player.hand = three_hearts()
        decision = player.should_order_up(card(Rank.KING, Suit.HEARTS))
        assert decision in (True, False)
        assert player.model_consulted is True
        assert isinstance(player.last_model_probability, float)
        module = EucherHeuristicTiebreak(hidden_size=8)
        assert module.hidden_size == 8

    def test_training_help_mentions_gray_areas(self) -> None:
        """The trainer documents the gray-area collection flags."""
        from scripts.training.train_eucher_heuristic_tiebreak import build_parser

        help_text = build_parser().format_help()
        assert "--hands" in help_text
        assert "--collect-only" in help_text
        assert "gray" in help_text.lower()


def test_four_tiebreak_players_finish_a_game() -> None:
    """Four untrained tie-break players, falling back to the heuristic, reach 10."""
    game = EuchreGame(quiet_mode=True)
    for name in ["Ada", "Bea", "Cia", "Dee"]:
        player = HeuristicTiebreakPlayer(name, load_default=False)
        game.players.append(player)
        game.tricks_won[player.name] = 0
    if len(game.players) == 4 and game.dealer_selection is None:
        from euchre.core.dealer_selection import DealerSelection

        game.dealer_selection = DealerSelection(game.players)
    game.run_full_game()
    assert game.is_game_over()
    scores = game.game_state_manager.game_scores
    assert scores["Team 1"] >= 10 or scores["Team 2"] >= 10


def test_manifest_records_only_gray_rows(tmp_path: Path) -> None:
    """The manifest says the rows came from simulated gray decisions."""
    dataset = collect_gray_samples(hands=15, seed=2, progress=False)
    if len(dataset) == 0:
        pytest.skip("no gray decisions in this small sample")
    save_gray_dataset(dataset, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["samples"] == len(dataset)
    assert manifest["clear_decisions"] >= 0
    assert "1 to 3 trump" in manifest["preprocessing"]
    assert "not stored" in manifest["preprocessing"]
