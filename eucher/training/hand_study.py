"""Study one euchre hand from a seed, without playing a game to 10.

The seed fixes the four hands, the upcard, and the kitty. Repeating the seed
deals those same cards again, so a learner can try a different bid or a
different card and compare the hand that follows. The dealer is not part of
that shuffle. Callers that sample many hands rotate the dealer through the
four seats, which is the only balance this step keeps.

Each decision records every legal option. The chosen option is one of those
options unless a profile reneges. A ``choices`` list or a ``choose`` callback
forces the learner's next decision, which is how a model experiments on the
same deal.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from eucher.cards import Card, Suit
from eucher.game import Game
from eucher.training.data_collector import GameDataCollector

PASS = "pass"
ORDER = "order"
ALONE = "alone"
PARTNER = "partner"
SEATS = 4

Chooser = Callable[["DecisionView"], str]


@dataclass(frozen=True)
class DecisionView:
    """One decision, including every legal option the player can take."""

    kind: str
    seat: int
    dealer: int
    options: tuple[str, ...]
    hand: tuple[str, ...]
    turned_card: str
    trump: Optional[str]
    led_suit: Optional[str]
    trick_cards: tuple[str, ...]
    must_choose: bool


@dataclass
class HandRecord:
    """The deal, every decision, and the score from a single hand."""

    seed: str
    episode: int
    dealer: int
    learner: str
    opening_hands: list[list[str]]
    upcard: str
    kitty: list[str]
    passed: bool
    trump: Optional[str]
    maker_seat: Optional[int]
    going_alone: bool
    tricks: list[int]
    scores: list[int]
    decisions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready copy of this hand.

        Returns
        -------
        dict[str, Any]
            The deal, the decisions, and the hand score.
        """
        return asdict(self)


def card_key(card: Card) -> str:
    """Return a stable label for a card.

    Parameters
    ----------
    card : Card
        Card to label.

    Returns
    -------
    str
        ``Suit_rank``, such as ``Hearts_14``.
    """
    return f"{card.suit.value}_{card.rank.value}"


def dealer_for_episode(episode: int) -> int:
    """Return the dealer seat for one sample so the four seats stay even.

    Parameters
    ----------
    episode : int
        Zero-based sample index.

    Returns
    -------
    int
        Dealer seat from 0 through 3. Episode 0 deals from seat 0.

    Raises
    ------
    ValueError
        If ``episode`` is negative.
    """
    if episode < 0:
        raise ValueError(f"episode must be >= 0, got {episode}")
    return episode % SEATS


def dealer_counts(studies: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count how often each seat dealt.

    Parameters
    ----------
    studies : Sequence[Mapping[str, Any]]
        Hand records. Each one has a ``dealer`` seat.

    Returns
    -------
    dict[str, int]
        Counts for seats ``0`` through ``3``.
    """
    counts = {str(seat): 0 for seat in range(SEATS)}
    for study in studies:
        seat = int(study["dealer"])
        counts[str(seat)] = counts.get(str(seat), 0) + 1
    return counts


def dealers_are_balanced(counts: Mapping[str, int]) -> bool:
    """Return whether dealer counts differ by at most one.

    Parameters
    ----------
    counts : Mapping[str, int]
        Counts from :func:`dealer_counts`.

    Returns
    -------
    bool
        True when the busiest seat dealt at most one more hand than the quietest.
    """
    if not counts:
        return True
    values = [int(value) for value in counts.values()]
    return max(values) - min(values) <= 1


def call_trump_options(turned: Card, must_choose: bool) -> list[str]:
    """Return the legal second-round calls.

    Parameters
    ----------
    turned : Card
        The upcard. Its suit cannot be named.
    must_choose : bool
        When True, the dealer must name a suit.

    Returns
    -------
    list[str]
        Suit names, with ``pass`` first when passing is legal.
    """
    suits = [suit.value for suit in Suit if suit != turned.suit]
    if must_choose:
        return suits
    return [PASS, *suits]


def card_from_key(label: str, hand: Sequence[Card]) -> Card:
    """Find the card a label names.

    Parameters
    ----------
    label : str
        Label from :func:`card_key`.
    hand : Sequence[Card]
        Cards that may be chosen.

    Returns
    -------
    Card
        The matching card.

    Raises
    ------
    ValueError
        If the label is not in ``hand``.
    """
    for card in hand:
        if card_key(card) == label:
            return card
    raise ValueError(f"{label!r} is not in the hand")


def write_hand_studies(
    directory: Path,
    studies: Sequence[Mapping[str, Any]],
    version: str,
    seed: str,
) -> Path:
    """Write the hand studies and a summary with a checksum.

    Parameters
    ----------
    directory : Path
        Directory for ``hand_studies.jsonl`` and the summary.
    studies : Sequence[Mapping[str, Any]]
        One record per hand.
    version : str
        Dataset version stored in the summary.
    seed : str
        Deal seed shared by the hands.

    Returns
    -------
    Path
        The summary file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(study), sort_keys=True) for study in studies]
    text = ("\n".join(lines) + "\n") if lines else ""
    (directory / "hand_studies.jsonl").write_text(text)
    counts = dealer_counts(studies)
    summary = {
        "version": version,
        "seed": seed,
        "hands": len(studies),
        "unit": "hand",
        "dealer_counts": counts,
        "dealer_balanced": dealers_are_balanced(counts),
        "checksum": hashlib.sha256(text.encode()).hexdigest(),
        "studies": "hand_studies.jsonl",
        "source": "one seeded hand per episode",
        "preprocessing": "Cards come from the seed. Dealer rotates through seats 0, 1, 2, and 3.",
    }
    path = directory / "hand_studies_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path


class _HandProbe:
    """Record every legal option while one hand is played."""

    def __init__(
        self,
        game: Game,
        dealer: int,
        learner_seat: int,
        collector: Optional[GameDataCollector],
        script: list[str],
        choose: Optional[Chooser],
    ) -> None:
        self.game = game
        self.dealer = dealer
        self.learner_seat = learner_seat
        self.collector = collector
        self.script = script
        self.choose = choose
        self.decisions: list[dict[str, Any]] = []
        self.opening_hands: Optional[list[list[str]]] = None
        self.upcard = ""
        self.kitty: list[str] = []

    def install(self) -> None:
        """Wrap every seat so the hand log sees the options before the choice."""
        for seat, player in enumerate(self.game.players):
            profile = player.profile
            profile.decide_order_up = self._order_up(seat, profile.decide_order_up)  # type: ignore[method-assign]
            profile.decide_call_trump = self._call_trump(seat, profile.decide_call_trump)  # type: ignore[method-assign]
            profile.choose_card_to_discard = self._discard(seat, profile.choose_card_to_discard)  # type: ignore[method-assign]
            profile.play_card = self._play(seat, profile.play_card)  # type: ignore[method-assign]
            profile.decide_going_alone = self._alone(seat, profile.decide_going_alone)  # type: ignore[method-assign]

    def finish_script(self) -> None:
        """Raise when the experiment named decisions the hand never reached.

        Raises
        ------
        ValueError
            If scripted choices remain.
        """
        if self.script:
            leftover = ", ".join(self.script)
            raise ValueError(f"Choices were left over after the hand: {leftover}")

    def _seen_opening(self) -> None:
        if self.opening_hands is not None:
            return
        turned = self.game.turned_card
        if turned is None:
            raise RuntimeError("The hand dealt no upcard")
        self.opening_hands = [[card_key(card) for card in player.hand] for player in self.game.players]
        self.upcard = card_key(turned)
        self.kitty = [card_key(card) for card in self.game.kitty]

    def _view(
        self,
        kind: str,
        seat: int,
        options: list[str],
        turned: Card,
        led: Optional[Suit],
        trick_cards: Sequence[Card],
        must_choose: bool,
    ) -> DecisionView:
        trump = self.game.trump_suit
        return DecisionView(
            kind=kind,
            seat=seat,
            dealer=self.dealer,
            options=tuple(options),
            hand=tuple(card_key(card) for card in self.game.players[seat].hand),
            turned_card=card_key(turned),
            trump=None if trump is None else trump.value,
            led_suit=None if led is None else led.value,
            trick_cards=tuple(card_key(card) for card in trick_cards),
            must_choose=must_choose,
        )

    def _pick(self, seat: int, options: list[str], view: DecisionView, fallback: Callable[[], str]) -> str:
        if seat == self.learner_seat and self.script:
            choice = self.script.pop(0)
            self._require_option(choice, options, view.kind)
            return choice
        if seat == self.learner_seat and self.choose is not None:
            choice = self.choose(view)
            self._require_option(choice, options, view.kind)
            return choice
        return fallback()

    def _require_option(self, choice: str, options: list[str], kind: str) -> None:
        if choice not in options:
            legal = ", ".join(options)
            raise ValueError(f"{choice!r} is not legal for {kind}. Legal options: {legal}")

    def _log(self, view: DecisionView, chosen: str) -> None:
        self.decisions.append(
            {
                "index": len(self.decisions),
                "kind": view.kind,
                "seat": view.seat,
                "options": list(view.options),
                "chosen": chosen,
                "hand": list(view.hand),
                "trump": view.trump,
                "led_suit": view.led_suit,
                "trick_cards": list(view.trick_cards),
                "must_choose": view.must_choose,
            }
        )

    def _state(self, **extra: Any) -> dict[str, Any]:
        tricks = getattr(self.game, "_current_tricks_won", [0, 0])
        state = {
            "trick_number": getattr(self.game, "_current_trick_number", 0),
            "tricks_won_team0": tricks[0],
            "tricks_won_team1": tricks[1],
        }
        state.update(extra)
        return state

    def _order_up(self, seat: int, original: Callable[..., bool]) -> Callable[..., bool]:
        def wrapper(player: Any, turned: Card, dealer_id: int, trump: Optional[Suit]) -> bool:
            self._seen_opening()
            options = [PASS, ORDER]
            view = self._view("order_up", seat, options, turned, None, [], False)

            def fallback() -> str:
                return ORDER if original(player, turned, dealer_id, trump) else PASS

            chosen = self._pick(seat, options, view, fallback)
            self._log(view, chosen)
            decision = chosen == ORDER
            if self.collector is not None and seat == self.learner_seat:
                self.collector.record_order_up_decision(
                    seat, player.hand, turned, dealer_id, decision, self._state()
                )
            return decision

        return wrapper

    def _call_trump(self, seat: int, original: Callable[..., Optional[Suit]]) -> Callable[..., Optional[Suit]]:
        def wrapper(
            player: Any,
            turned: Card,
            trump: Optional[Suit],
            must_choose: bool = False,
        ) -> Optional[Suit]:
            self._seen_opening()
            options = call_trump_options(turned, must_choose)
            view = self._view("call_trump", seat, options, turned, None, [], must_choose)

            def fallback() -> str:
                named = original(player, turned, trump, must_choose)
                return PASS if named is None else named.value

            chosen = self._pick(seat, options, view, fallback)
            self._log(view, chosen)
            decision = None if chosen == PASS else Suit(chosen)
            if self.collector is not None and seat == self.learner_seat:
                self.collector.record_call_trump_decision(seat, player.hand, turned, decision, self._state())
            return decision

        return wrapper

    def _discard(self, seat: int, original: Callable[..., Card]) -> Callable[..., Card]:
        def wrapper(player: Any, turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None) -> Card:
            self._seen_opening()
            turned = turned_card if turned_card is not None else self.game.turned_card
            if turned is None:
                raise RuntimeError("Discard with no upcard")
            options = [card_key(card) for card in player.hand]
            view = self._view("discard", seat, options, turned, None, [], False)

            def fallback() -> str:
                return card_key(original(player, turned_card, ordered_up_by))

            chosen = self._pick(seat, options, view, fallback)
            self._log(view, chosen)
            card = card_from_key(chosen, player.hand)
            if self.collector is not None and seat == self.learner_seat:
                self.collector.record_discard_decision(seat, player.hand, card, self._state())
            return card

        return wrapper

    def _play(self, seat: int, original: Callable[..., Card]) -> Callable[..., Card]:
        def wrapper(
            player: Any,
            led: Optional[Suit],
            trump: Optional[Suit],
            trick_cards: list[Card],
            trick_player_ids: list[int],
        ) -> Card:
            self._seen_opening()
            turned = self.game.turned_card
            if turned is None:
                raise RuntimeError("Card play with no upcard")
            legal = self.game.rules.get_valid_plays(player.hand, led, trump)
            options = [card_key(card) for card in legal] or [card_key(card) for card in player.hand]
            view = self._view("play_card", seat, options, turned, led, trick_cards, False)

            def fallback() -> str:
                return card_key(original(player, led, trump, trick_cards, trick_player_ids))

            chosen = self._pick(seat, options, view, fallback)
            self._log(view, chosen)
            card = card_from_key(chosen, player.hand)
            if self.collector is not None and seat == self.learner_seat:
                is_renege = not self.game.rules.can_play_card(card, player.hand, led, trump)
                self.collector.record_play_card_decision(
                    seat,
                    player.hand,
                    led,
                    trump,
                    trick_cards,
                    card,
                    self._state(is_renege=is_renege),
                )
            return card

        return wrapper

    def _alone(self, seat: int, original: Callable[..., bool]) -> Callable[..., bool]:
        def wrapper(player: Any, trump: Suit) -> bool:
            self._seen_opening()
            turned = self.game.turned_card
            if turned is None:
                raise RuntimeError("Going alone with no upcard")
            options = [PARTNER, ALONE]
            view = self._view("going_alone", seat, options, turned, None, [], False)

            def fallback() -> str:
                return ALONE if original(player, trump) else PARTNER

            chosen = self._pick(seat, options, view, fallback)
            self._log(view, chosen)
            return chosen == ALONE

        return wrapper


def play_training_hand(
    seed: int | str,
    dealer: int,
    episode: int = 0,
    learner_type: str = "random",
    choices: Optional[Sequence[str]] = None,
    choose: Optional[Chooser] = None,
    collector: Optional[GameDataCollector] = None,
    learner_seat: int = 0,
    player_config: Optional[Sequence[tuple[str, str]]] = None,
) -> HandRecord:
    """Play one hand from a seed and record every legal option.

    Parameters
    ----------
    seed : int or str
        Deal seed. The same seed deals the same cards for every dealer.
    dealer : int
        Dealer seat, from 0 through 3. This does not change the shuffle.
    episode : int
        Replay index. Opponents draw a different decision sequence per episode.
    learner_type : str
        Personality in the learner seat when ``player_config`` is omitted.
    choices : sequence of str, optional
        Labels the learner must take, in decision order. Each label has to be
        one of that decision's legal options.
    choose : callable, optional
        Called with a :class:`DecisionView` when ``choices`` has run out.
        The return value must be one of ``view.options``.
    collector : GameDataCollector, optional
        When set, the learner's bids and card plays are stored for training
        and this hand is appended to ``collector.hand_studies``.
    learner_seat : int
        Seat whose decisions can be forced, and whose rows are collected.
    player_config : sequence of tuple, optional
        Four ``(name, profile)`` seats. The default is the learner plus three
        random opponents.

    Returns
    -------
    HandRecord
        The opening deal, every decision with its options, and the hand score.

    Raises
    ------
    ValueError
        If the dealer, episode, or a forced choice is illegal, or if scripted
        choices remain after the hand.
    """
    if dealer not in range(SEATS):
        raise ValueError(f"dealer must be 0, 1, 2, or 3, got {dealer}")
    if learner_seat not in range(SEATS):
        raise ValueError(f"learner_seat must be 0, 1, 2, or 3, got {learner_seat}")
    if episode < 0:
        raise ValueError(f"episode must be >= 0, got {episode}")

    from eucher.training.self_play import apply_decision_seed, random_opponent_lineup

    apply_decision_seed(seed, episode)
    if player_config is None:
        lineup = random_opponent_lineup(learner_type)
    else:
        lineup = list(player_config)
    game_id = str(uuid.uuid4())
    if collector is not None:
        collector.start_game(game_id)
    game = Game(lineup, seed=seed, allow_untrained=True, seed_decisions=False)
    game.dealer_id = dealer
    if collector is not None:
        game.game_uuid = game_id

    probe = _HandProbe(
        game,
        dealer,
        learner_seat,
        collector,
        list(choices or []),
        choose,
    )
    probe.install()
    game.play_hand()
    probe.finish_script()
    if probe.opening_hands is None:
        raise RuntimeError("The hand ended before anyone was asked to decide")

    trump = game.trump_suit
    maker = game.trump_selector.trump_maker_id if game.trump_selector is not None else None
    tricks = list(getattr(game, "_current_tricks_won", [0, 0]))
    scores = list(game.get_scores())
    record = HandRecord(
        seed=str(seed),
        episode=episode,
        dealer=dealer,
        learner=learner_type,
        opening_hands=probe.opening_hands,
        upcard=probe.upcard,
        kitty=probe.kitty,
        passed=trump is None,
        trump=None if trump is None else trump.value,
        maker_seat=maker,
        going_alone=bool(getattr(game, "going_alone", False)),
        tricks=tricks,
        scores=scores,
        decisions=probe.decisions,
    )
    if collector is not None:
        collector.record_game_outcome(
            game_id,
            game.get_scores(),
            [tricks],
            game.get_winner(),
            bool(getattr(game, "_renege_occurred", False)),
            getattr(game, "_renege_player_id", None),
        )
        collector.game_outcomes[-1]["unit"] = "hand"
        collector.game_outcomes[-1]["dealer"] = dealer
        collector.game_outcomes[-1]["seed"] = str(seed)
        collector.game_outcomes[-1]["episode"] = episode
        collector.hand_studies.append(record.to_dict())
    return record
