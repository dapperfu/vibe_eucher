"""Simulated deals that label only the heuristic's gray decisions.

Each gray bid is played twice: once calling, once passing. Each gray card
play or discard is played twice as well: the heuristic card, and one close
alternative. The label is which branch leaves that team more likely to reach
10 first. Clear decisions are counted and then dropped.
"""

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import torch

from ..ai.gray_area import (
    ADVANTAGE_MARGIN,
    FEATURE_NAMES,
    TRICK_BAND,
    WEIGHT_MARGIN,
    close_alternatives,
    features_for,
    features_for_card,
    is_gray_call,
    trump_count,
)
from ..ai.heuristic_player import CallEvaluation, HeuristicPlayer, win_probability
from ..models import Card, Rank, Suit

RANKS: Tuple[Rank, ...] = (Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE)
Hands = List[List[Card]]


@dataclass
class GrayDataset:
    """Gray-area rows and the deal settings that produced them."""

    features: List[List[float]]
    labels: List[int]
    trump_counts: List[int]
    expected_tricks: List[float]
    advantages: List[float]
    rounds: List[int]
    kinds: List[str]
    version: str
    seed: int
    hands_played: int
    trick_band: float
    advantage_margin: float
    weight_margin: float = WEIGHT_MARGIN
    clear_decisions: int = 0
    gray_decisions: int = 0
    split: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of gray-area rows.

        Returns
        -------
        int
            How many labeled decisions were kept.
        """
        return len(self.labels)


def sample_scores(rng: random.Random) -> Tuple[int, int]:
    """Draw a score that spends extra time at 0-0 and near the end.

    Parameters
    ----------
    rng : random.Random
        The deal random source.

    Returns
    -------
    Tuple[int, int]
        Team 0 and team 1 scores, each from 0 through 9.
    """
    roll = rng.random()
    if roll < 0.4:
        return 0, 0
    if roll < 0.7:
        return rng.randrange(8, 10), rng.randrange(8, 10)
    return rng.randrange(10), rng.randrange(10)


def seat_scores(seat: int, team0: int, team1: int) -> Tuple[int, int]:
    """Return this seat's score and the opponents' score.

    Parameters
    ----------
    seat : int
        Seat index. Even seats are team 0.
    team0 : int
        Score of the even seats.
    team1 : int
        Score of the odd seats.

    Returns
    -------
    Tuple[int, int]
        ``(our score, opponent score)``.
    """
    if seat % 2 == 0:
        return team0, team1
    return team1, team0


def deal_hands(rng: random.Random) -> Tuple[Hands, Card, int]:
    """Deal four hands, an upcard, and a dealer seat.

    Parameters
    ----------
    rng : random.Random
        The deal random source.

    Returns
    -------
    Tuple[Hands, Card, int]
        Hands by seat, the upcard, and the dealer seat.
    """
    cards = [Card(rank, suit) for suit in Suit for rank in RANKS]
    rng.shuffle(cards)
    hands = [cards[index * 5 : (index + 1) * 5] for index in range(4)]
    return hands, cards[20], rng.randrange(4)


def copy_hands(hands: Sequence[Sequence[Card]]) -> Hands:
    """Copy hands so one branch cannot change another.

    Parameters
    ----------
    hands : Sequence[Sequence[Card]]
        Hands by seat.

    Returns
    -------
    Hands
        New card objects in the same seats.
    """
    return [[Card(card.rank, card.suit) for card in hand] for hand in hands]


def make_players(hands: Hands, dealer: int, team0: int, team1: int) -> List[HeuristicPlayer]:
    """Seat four heuristic players for one branch of a deal.

    Parameters
    ----------
    hands : Hands
        Hands by seat. The lists are used as-is and will be mutated.
    dealer : int
        Dealer seat.
    team0 : int
        Score of the even seats.
    team1 : int
        Score of the odd seats.

    Returns
    -------
    List[HeuristicPlayer]
        Players who know the score and the dealer.
    """
    players = []
    for seat, hand in enumerate(hands):
        mine, theirs = seat_scores(seat, team0, team1)
        player = HeuristicPlayer(f"Seat{seat}")
        player.hand = hand
        player.begin_round(seat, dealer, mine, theirs)
        players.append(player)
    return players


def speak_order(dealer: int) -> List[int]:
    """Return seats in bidding order, starting left of the dealer.

    Parameters
    ----------
    dealer : int
        Dealer seat.

    Returns
    -------
    List[int]
        Four seats, dealer last.
    """
    return [(dealer + 1 + step) % 4 for step in range(4)]


def seats_after(decider: int, dealer: int) -> List[int]:
    """Return seats that speak after ``decider`` in this round.

    Parameters
    ----------
    decider : int
        The seat that already passed.
    dealer : int
        Dealer seat.

    Returns
    -------
    List[int]
        The remaining seats. Empty when the dealer was the decider.
    """
    order = speak_order(dealer)
    return order[order.index(decider) + 1 :]


def _play_tricks(players: Sequence[HeuristicPlayer], dealer: int, caller: int, trump: Suit) -> int:
    """Play five tricks and return how many the makers won.

    Parameters
    ----------
    players : Sequence[HeuristicPlayer]
        Seated players holding their cards.
    dealer : int
        Dealer seat. The left of the dealer leads the first trick.
    caller : int
        Seat that named trump. That seat's team is the makers.
    trump : Suit
        The trump suit.

    Returns
    -------
    int
        Tricks won by the calling team.
    """
    for player in players:
        player.note_trump_call(caller, trump)
    leader = (dealer + 1) % 4
    maker_tricks = 0
    for _ in range(5):
        lead_suit: Optional[Suit] = None
        plays: List[Tuple[int, Card]] = []
        for step in range(4):
            seat = (leader + step) % 4
            player = players[seat]
            card = player.choose_card_to_play(lead_suit, trump)
            if card not in player.hand:
                raise RuntimeError(f"{player.name} chose {card}, which is not in hand")
            player.hand.remove(card)
            if lead_suit is None:
                lead_suit = card.suit
            for other in players:
                other.note_card_played(seat, card)
            plays.append((seat, card))
        winner_seat, winner_card = plays[0]
        for seat, card in plays[1:]:
            if card.beats(winner_card, trump, lead_suit):
                winner_seat, winner_card = seat, card
        for player in players:
            player.note_trick_complete(winner_seat)
        if winner_seat % 2 == caller % 2:
            maker_tricks += 1
        leader = winner_seat
    return maker_tricks


def _pickup(players: Sequence[HeuristicPlayer], dealer: int, upcard: Card, trump: Suit) -> None:
    """Give the dealer the upcard and let the heuristic discard.

    Parameters
    ----------
    players : Sequence[HeuristicPlayer]
        Seated players.
    dealer : int
        Dealer seat.
    upcard : Card
        The card ordered up.
    trump : Suit
        The trump suit, which is the upcard's suit.
    """
    hand = players[dealer].hand
    hand.append(Card(upcard.rank, upcard.suit))
    discarded = players[dealer].choose_card_to_discard(trump)
    hand.remove(discarded)


def finish_contract(
    players: Sequence[HeuristicPlayer],
    dealer: int,
    caller: int,
    trump: Suit,
    round_number: int,
    upcard: Card,
    decider: int,
    team0: int,
    team1: int,
) -> float:
    """Play a named contract and return the decider's resulting win probability.

    Parameters
    ----------
    players : Sequence[HeuristicPlayer]
        Seated players for this branch.
    dealer : int
        Dealer seat.
    caller : int
        Seat that named trump.
    trump : Suit
        The trump suit.
    round_number : int
        1 when the upcard was ordered up.
    upcard : Card
        The turned card.
    decider : int
        Seat whose team value is returned.
    team0 : int
        Score of the even seats before the hand.
    team1 : int
        Score of the odd seats before the hand.

    Returns
    -------
    float
        Probability that the decider's team reaches 10 first after this hand.
    """
    if round_number == 1 and trump == upcard.suit:
        _pickup(players, dealer, upcard, trump)
    maker_tricks = _play_tricks(players, dealer, caller, trump)
    if maker_tricks >= 5:
        maker_points, defender_points = 2, 0
    elif maker_tricks >= 3:
        maker_points, defender_points = 1, 0
    else:
        maker_points, defender_points = 0, 2
    if decider % 2 == caller % 2:
        gained, against = maker_points, defender_points
    else:
        gained, against = defender_points, maker_points
    mine, theirs = seat_scores(decider, team0, team1)
    return win_probability(mine + gained, theirs + against)


def _continue_auction(
    players: Sequence[HeuristicPlayer],
    dealer: int,
    decider: int,
    upcard: Card,
    round_number: int,
) -> Optional[Tuple[Suit, int, int]]:
    """Finish the auction after the decider passes.

    Parameters
    ----------
    players : Sequence[HeuristicPlayer]
        Seated heuristic players.
    dealer : int
        Dealer seat.
    decider : int
        The seat that passed the gray decision.
    upcard : Card
        The turned card.
    round_number : int
        The round the decider passed in.

    Returns
    -------
    Optional[Tuple[Suit, int, int]]
        ``(suit, caller seat, round)``, or None if nobody names trump.
    """
    if round_number == 1:
        for seat in seats_after(decider, dealer):
            if players[seat].should_order_up(upcard):
                return upcard.suit, seat, 1
        for seat in speak_order(dealer):
            named = players[seat].choose_trump_suit(upcard)
            if named is not None:
                return named, seat, 2
        return None
    for seat in seats_after(decider, dealer):
        named = players[seat].choose_trump_suit(upcard)
        if named is not None:
            return named, seat, 2
    return None


def call_value(
    hands: Hands,
    dealer: int,
    decider: int,
    suit: Suit,
    round_number: int,
    upcard: Card,
    team0: int,
    team1: int,
) -> float:
    """Value of the decider naming ``suit`` on this deal.

    Parameters
    ----------
    hands : Hands
        The dealt hands. They are copied.
    dealer : int
        Dealer seat.
    decider : int
        Seat that calls.
    suit : Suit
        Suit being named.
    round_number : int
        1 to order up, 2 to name a suit.
    upcard : Card
        The turned card.
    team0 : int
        Score of the even seats.
    team1 : int
        Score of the odd seats.

    Returns
    -------
    float
        Win probability for the decider's team after the hand.
    """
    players = make_players(copy_hands(hands), dealer, team0, team1)
    return finish_contract(
        players, dealer, decider, suit, round_number, upcard, decider, team0, team1
    )


def pass_value(
    hands: Hands,
    dealer: int,
    decider: int,
    round_number: int,
    upcard: Card,
    team0: int,
    team1: int,
) -> float:
    """Value of the decider passing, with the heuristic finishing the auction.

    Parameters
    ----------
    hands : Hands
        The dealt hands. They are copied.
    dealer : int
        Dealer seat.
    decider : int
        Seat that passes.
    round_number : int
        The round being passed.
    upcard : Card
        The turned card.
    team0 : int
        Score of the even seats.
    team1 : int
        Score of the odd seats.

    Returns
    -------
    float
        Win probability for the decider's team after the hand.
    """
    players = make_players(copy_hands(hands), dealer, team0, team1)
    contract = _continue_auction(players, dealer, decider, upcard, round_number)
    if contract is None:
        mine, theirs = seat_scores(decider, team0, team1)
        return win_probability(mine, theirs)
    suit, caller, named_round = contract
    return finish_contract(
        players, dealer, caller, suit, named_round, upcard, decider, team0, team1
    )


def _record(
    dataset: GrayDataset,
    player: HeuristicPlayer,
    evaluation: CallEvaluation,
    top_card: Card,
    round_number: int,
    location: str,
    label: int,
    count: int,
    kind: str = "call",
) -> None:
    """Append one gray-area row from an evaluation the heuristic already priced."""
    dataset.features.append(features_for(player, evaluation, top_card, round_number, location))
    dataset.labels.append(label)
    dataset.trump_counts.append(count)
    dataset.expected_tricks.append(evaluation.expected_tricks)
    dataset.advantages.append(evaluation.advantage)
    dataset.rounds.append(round_number)
    dataset.kinds.append(kind)


def collect_gray_samples(
    hands: int,
    seed: int,
    version: str = "v1",
    trick_band: float = TRICK_BAND,
    advantage_margin: float = ADVANTAGE_MARGIN,
    weight_margin: float = WEIGHT_MARGIN,
    progress: bool = True,
) -> GrayDataset:
    """Play deals and keep only the gray-area trump decisions.

    Parameters
    ----------
    hands : int
        How many deals to play.
    seed : int
        Seed for the deals and the score sample.
    version : str
        Dataset version stored with the rows.
    trick_band : float
        Gray band around three expected tricks.
    advantage_margin : float
        Gray band around a zero win-probability edge.
    weight_margin : float
        How close two card weights must be before that play or discard is kept.
    progress : bool
        Show a progress bar while the deals run.

    Returns
    -------
    GrayDataset
        Labeled gray decisions. Clear decisions are counted and omitted.
    """
    rng = random.Random(seed)
    dataset = GrayDataset(
        features=[],
        labels=[],
        trump_counts=[],
        expected_tricks=[],
        advantages=[],
        rounds=[],
        kinds=[],
        version=version,
        seed=seed,
        hands_played=hands,
        trick_band=trick_band,
        advantage_margin=advantage_margin,
        weight_margin=weight_margin,
    )
    iterator: Iterable[int] = range(hands)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="Gray-area deals", unit="deal")
    for _ in iterator:
        dealt, upcard, dealer = deal_hands(rng)
        team0, team1 = sample_scores(rng)
        players = make_players(copy_hands(dealt), dealer, team0, team1)
        _walk_auction(
            dataset, players, dealt, upcard, dealer, team0, team1, trick_band, advantage_margin
        )
        _record_hand_actions(dataset, dealt, upcard, dealer, team0, team1, weight_margin)
    return dataset


def _walk_auction(
    dataset: GrayDataset,
    players: Sequence[HeuristicPlayer],
    hands: Hands,
    upcard: Card,
    dealer: int,
    team0: int,
    team1: int,
    trick_band: float,
    advantage_margin: float,
) -> None:
    """Record gray bids, then stop once the heuristic names a contract."""
    for seat in speak_order(dealer):
        player = players[seat]
        location = player._upcard_location(False)
        evaluation = player.evaluate_call(upcard.suit, upcard, 1, location)
        count = trump_count(player.hand, upcard.suit)
        gray = is_gray_call(
            count, evaluation.expected_tricks, evaluation.advantage, trick_band, advantage_margin
        )
        if gray:
            dataset.gray_decisions += 1
            called = call_value(hands, dealer, seat, upcard.suit, 1, upcard, team0, team1)
            passed = pass_value(hands, dealer, seat, 1, upcard, team0, team1)
            _record(
                dataset,
                player,
                evaluation,
                upcard,
                1,
                location,
                1 if called > passed else 0,
                count,
            )
        else:
            dataset.clear_decisions += 1
        if evaluation.should_call:
            return

    for seat in speak_order(dealer):
        player = players[seat]
        if player._dealer_must_name_suit():
            return
        evaluations = []
        for suit in Suit:
            if suit == upcard.suit:
                continue
            evaluations.append(player.evaluate_call(suit, upcard, 2, "buried"))
        gray_evals = []
        clear_call = False
        for evaluation in evaluations:
            count = trump_count(player.hand, evaluation.suit)
            gray = is_gray_call(
                count,
                evaluation.expected_tricks,
                evaluation.advantage,
                trick_band,
                advantage_margin,
            )
            if gray:
                gray_evals.append((evaluation, count))
            elif evaluation.should_call:
                clear_call = True
        if clear_call or not gray_evals:
            dataset.clear_decisions += 1
        else:
            dataset.gray_decisions += 1
            passed = pass_value(hands, dealer, seat, 2, upcard, team0, team1)
            location = player._upcard_location(False)
            for evaluation, count in gray_evals:
                called = call_value(hands, dealer, seat, evaluation.suit, 2, upcard, team0, team1)
                _record(
                    dataset,
                    player,
                    evaluation,
                    upcard,
                    2,
                    location,
                    1 if called > passed else 0,
                    count,
                )
        named = player.choose_trump_suit(upcard)
        if named is not None:
            return


def _record_hand_actions(
    dataset: GrayDataset,
    hands: Hands,
    upcard: Card,
    dealer: int,
    team0: int,
    team1: int,
    weight_margin: float,
) -> None:
    """Label gray discards and gray card plays on the heuristic's chosen contract."""
    players = make_players(copy_hands(hands), dealer, team0, team1)
    found = _heuristic_contract(players, upcard, dealer)
    if found is None:
        return
    trump, caller, round_number = found
    for player in players:
        player.note_trump_call(caller, trump)
    if round_number == 1 and trump == upcard.suit:
        players[dealer].hand.append(Card(upcard.rank, upcard.suit))
        _consider_discard(dataset, players, dealer, caller, trump, team0, team1, weight_margin)
        _drop(players[dealer].hand, players[dealer].choose_card_to_discard(trump))
    _consider_plays(dataset, players, dealer, caller, trump, team0, team1, weight_margin)


def _heuristic_contract(
    players: Sequence[HeuristicPlayer], upcard: Card, dealer: int
) -> Optional[Tuple[Suit, int, int]]:
    """Return the suit, caller, and round the heuristic table actually names."""
    for seat in speak_order(dealer):
        if players[seat].should_order_up(upcard):
            return upcard.suit, seat, 1
    for seat in speak_order(dealer):
        named = players[seat].choose_trump_suit(upcard)
        if named is not None:
            return named, seat, 2
    return None


def _consider_discard(
    dataset: GrayDataset,
    players: Sequence[HeuristicPlayer],
    dealer: int,
    caller: int,
    trump: Suit,
    team0: int,
    team1: int,
    weight_margin: float,
) -> None:
    """Record a gray discard, then leave the hand for the caller to throw."""
    player = players[dealer]
    scored = player.discard_weights(trump)
    chosen = player.choose_card_to_discard(trump)
    alternatives = close_alternatives(scored, chosen, weight_margin)
    if not alternatives:
        dataset.clear_decisions += 1
        return
    dataset.gray_decisions += 1
    baseline = _discard_value(players, dealer, chosen, caller, trump, team0, team1)
    chosen_weight = _matching_weight(scored, chosen)
    for card, weight in alternatives:
        value = _discard_value(players, dealer, card, caller, trump, team0, team1)
        _record_card(
            dataset,
            player,
            card,
            trump,
            weight,
            chosen_weight - weight,
            1 + len(alternatives),
            "discard",
            value > baseline,
            None,
            False,
            "want",
        )


def _consider_plays(
    dataset: GrayDataset,
    players: List[HeuristicPlayer],
    dealer: int,
    caller: int,
    trump: Suit,
    team0: int,
    team1: int,
    weight_margin: float,
) -> None:
    """Play the hand and label every gray card choice along the way."""
    leader = (dealer + 1) % 4
    for _trick in range(5):
        lead_suit: Optional[Suit] = None
        plays: List[Tuple[int, Card]] = []
        for step in range(4):
            seat = (leader + step) % 4
            player = players[seat]
            scored = player.play_weights(lead_suit, trump)
            chosen = player.choose_card_to_play(lead_suit, trump)
            alternatives = close_alternatives(scored, chosen, weight_margin)
            actual_lead, partner_winning, urgency = _seat_context(player, trump)
            if not alternatives:
                dataset.clear_decisions += 1
            else:
                dataset.gray_decisions += 1
                baseline = _play_value(
                    players, seat, chosen, leader, trump, caller, team0, team1
                )
                chosen_weight = _matching_weight(scored, chosen)
                for card, weight in alternatives:
                    value = _play_value(
                        players, seat, card, leader, trump, caller, team0, team1
                    )
                    _record_card(
                        dataset,
                        player,
                        card,
                        trump,
                        weight,
                        chosen_weight - weight,
                        1 + len(alternatives),
                        "play",
                        value > baseline,
                        actual_lead,
                        partner_winning,
                        urgency,
                    )
            _drop(player.hand, chosen)
            if lead_suit is None:
                lead_suit = chosen.suit
            for other in players:
                other.note_card_played(seat, chosen)
            plays.append((seat, chosen))
        leader = _award(players, plays, trump, lead_suit)


def _discard_value(
    players: Sequence[HeuristicPlayer],
    dealer: int,
    discard: Card,
    caller: int,
    trump: Suit,
    team0: int,
    team1: int,
) -> float:
    """Win probability for the dealer after throwing ``discard``."""
    clones = _clone(players)
    _drop(clones[dealer].hand, discard)
    _play_out(clones, (dealer + 1) % 4, trump)
    return _hand_value(clones[caller].team_tricks, dealer, caller, team0, team1)


def _play_value(
    players: Sequence[HeuristicPlayer],
    seat: int,
    card: Card,
    leader: int,
    trump: Suit,
    caller: int,
    team0: int,
    team1: int,
) -> float:
    """Win probability for ``seat`` after that seat plays ``card`` now."""
    clones = _clone(players)
    _play_forced(clones, seat, card, leader, trump)
    return _hand_value(clones[caller].team_tricks, seat, caller, team0, team1)


def _play_forced(
    players: List[HeuristicPlayer], seat: int, card: Card, leader: int, trump: Suit
) -> None:
    """Play ``card`` for ``seat`` and let the heuristic finish the hand."""
    plays = list(players[seat].trick_plays)
    lead_suit = plays[0][1].suit if plays else None
    _drop(players[seat].hand, card)
    if lead_suit is None:
        lead_suit = card.suit
    for other in players:
        other.note_card_played(seat, card)
    plays.append((seat, card))
    step = len(plays)
    for offset in range(step, 4):
        next_seat = (leader + offset) % 4
        played = players[next_seat].choose_card_to_play(lead_suit, trump)
        _drop(players[next_seat].hand, played)
        for other in players:
            other.note_card_played(next_seat, played)
        plays.append((next_seat, played))
    leader = _award(players, plays, trump, lead_suit)
    if players[leader].hand:
        _play_out(players, leader, trump)


def _play_out(players: Sequence[HeuristicPlayer], leader: int, trump: Suit) -> None:
    """Play every remaining trick, starting with ``leader``."""
    while players[leader].hand:
        plays: List[Tuple[int, Card]] = []
        lead_suit: Optional[Suit] = None
        for step in range(4):
            seat = (leader + step) % 4
            card = players[seat].choose_card_to_play(lead_suit, trump)
            _drop(players[seat].hand, card)
            if lead_suit is None:
                lead_suit = card.suit
            for other in players:
                other.note_card_played(seat, card)
            plays.append((seat, card))
        leader = _award(players, plays, trump, lead_suit)


def _award(
    players: Sequence[HeuristicPlayer],
    plays: Sequence[Tuple[int, Card]],
    trump: Suit,
    lead_suit: Optional[Suit],
) -> int:
    """Give the trick to its winner and return that seat."""
    winner_seat, winner = plays[0]
    for seat, card in plays[1:]:
        if card.beats(winner, trump, lead_suit):
            winner_seat, winner = seat, card
    for player in players:
        player.note_trick_complete(winner_seat)
    return winner_seat


def _hand_value(
    maker_tricks: int, decider: int, caller: int, team0: int, team1: int
) -> float:
    """Win probability for ``decider`` after the makers take ``maker_tricks``."""
    if maker_tricks >= 5:
        maker_points, defender_points = 2, 0
    elif maker_tricks >= 3:
        maker_points, defender_points = 1, 0
    else:
        maker_points, defender_points = 0, 2
    if decider % 2 == caller % 2:
        gained, against = maker_points, defender_points
    else:
        gained, against = defender_points, maker_points
    mine, theirs = seat_scores(decider, team0, team1)
    return win_probability(mine + gained, theirs + against)


def _clone(players: Sequence[HeuristicPlayer]) -> List[HeuristicPlayer]:
    """Copy a table so a counterfactual play cannot change the real line."""
    clones = []
    for player in players:
        clone = HeuristicPlayer(player.name)
        clone.hand = [Card(card.rank, card.suit) for card in player.hand]
        seat = 0 if player.seat is None else player.seat
        dealer_seat = 0 if player.dealer_seat is None else player.dealer_seat
        clone.begin_round(seat, dealer_seat, player.team_score, player.opponent_score)
        if player.known_trump is not None:
            clone.note_trump_call(player.trump_caller_seat, player.known_trump)
        clone.team_tricks = player.team_tricks
        clone.opponent_tricks = player.opponent_tricks
        clone.played_cards = [
            (played_seat, Card(card.rank, card.suit)) for played_seat, card in player.played_cards
        ]
        clone.trick_plays = [
            (played_seat, Card(card.rank, card.suit)) for played_seat, card in player.trick_plays
        ]
        clones.append(clone)
    return clones


def _drop(hand: List[Card], card: Card) -> None:
    """Remove the card that matches ``card`` from ``hand``."""
    for index, held in enumerate(hand):
        if held.rank == card.rank and held.suit == card.suit:
            del hand[index]
            return
    raise RuntimeError(f"{card} is not in the hand")


def _matching_weight(scored: Sequence[Tuple[Card, float]], chosen: Card) -> float:
    """Return the weight stored for ``chosen``."""
    for card, weight in scored:
        if card.rank == chosen.rank and card.suit == chosen.suit:
            return weight
    return 0.0


def _seat_context(player: HeuristicPlayer, trump: Suit) -> Tuple[Optional[Suit], bool, str]:
    """Lead suit, whether partner is winning, and how badly this trick is needed."""
    plays = player._plays_before_us()
    if not plays:
        return None, False, player._urgency()
    lead_suit = plays[0][1].suit
    winner_seat, _winner = player._trick_winner(plays, trump, lead_suit)
    return lead_suit, player._is_partner(winner_seat), player._urgency()


def _record_card(
    dataset: GrayDataset,
    player: HeuristicPlayer,
    card: Card,
    trump: Suit,
    weight: float,
    gap: float,
    close_count: int,
    kind: str,
    better: bool,
    lead_suit: Optional[Suit],
    partner_winning: bool,
    urgency: str,
) -> None:
    """Append one gray play or discard row."""
    dataset.features.append(
        features_for_card(
            player,
            card,
            trump,
            weight,
            gap,
            close_count,
            kind,
            lead_suit=lead_suit,
            partner_winning=partner_winning,
            urgency=urgency,
        )
    )
    dataset.labels.append(1 if better else 0)
    dataset.trump_counts.append(trump_count(player.hand, trump))
    dataset.expected_tricks.append(weight)
    dataset.advantages.append(gap)
    dataset.rounds.append(0)
    dataset.kinds.append(kind)


def assign_split(count: int, seed: int, val_fraction: float, test_fraction: float) -> List[int]:
    """Assign train, validation, and test ids.

    Parameters
    ----------
    count : int
        Number of rows.
    seed : int
        Shuffle seed. The same seed reproduces the split.
    val_fraction : float
        Share of rows for validation.
    test_fraction : float
        Share of rows for test.

    Returns
    -------
    List[int]
        0 train, 1 validation, 2 test. Fewer than five rows all stay in train.
    """
    if count < 5:
        return [0] * count
    order = list(range(count))
    random.Random(seed).shuffle(order)
    n_test = max(1, int(round(count * test_fraction)))
    n_val = max(1, int(round(count * val_fraction)))
    if n_test + n_val >= count:
        n_test = max(1, count // 5)
        n_val = max(1, count // 5)
    split = [0] * count
    for index in order[:n_test]:
        split[index] = 2
    for index in order[n_test : n_test + n_val]:
        split[index] = 1
    return split


def tensor_checksum(features: torch.Tensor, labels: torch.Tensor) -> str:
    """Hash the feature and label bytes.

    Parameters
    ----------
    features : torch.Tensor
        Feature matrix.
    labels : torch.Tensor
        Labels.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    digest.update(features.numpy().tobytes())
    digest.update(labels.numpy().tobytes())
    return digest.hexdigest()


def save_gray_dataset(
    dataset: GrayDataset,
    directory: Path,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Dict[str, object]:
    """Write the rows, the split, and a manifest.

    Parameters
    ----------
    dataset : GrayDataset
        Rows to store. The split field is filled in.
    directory : Path
        Directory for ``gray_samples.pt`` and ``manifest.json``.
    val_fraction : float
        Validation share.
    test_fraction : float
        Test share.

    Returns
    -------
    Dict[str, object]
        The manifest.

    Raises
    ------
    ValueError
        If the dataset is empty.
    """
    if len(dataset) == 0:
        raise ValueError("No gray-area decisions were collected")
    directory.mkdir(parents=True, exist_ok=True)
    dataset.split = assign_split(len(dataset), dataset.seed, val_fraction, test_fraction)
    features = torch.tensor(dataset.features, dtype=torch.float32)
    labels = torch.tensor(dataset.labels, dtype=torch.float32)
    checksum = tensor_checksum(features, labels)
    payload = {
        "version": dataset.version,
        "features": features,
        "labels": labels,
        "split": torch.tensor(dataset.split, dtype=torch.long),
        "trump_counts": torch.tensor(dataset.trump_counts, dtype=torch.long),
        "expected_tricks": torch.tensor(dataset.expected_tricks, dtype=torch.float32),
        "advantages": torch.tensor(dataset.advantages, dtype=torch.float32),
        "rounds": torch.tensor(dataset.rounds, dtype=torch.long),
        "kinds": list(dataset.kinds),
        "feature_names": list(FEATURE_NAMES),
        "seed": dataset.seed,
        "hands_played": dataset.hands_played,
        "trick_band": dataset.trick_band,
        "advantage_margin": dataset.advantage_margin,
        "weight_margin": dataset.weight_margin,
        "clear_decisions": dataset.clear_decisions,
        "gray_decisions": dataset.gray_decisions,
        "checksum": checksum,
    }
    torch.save(payload, directory / "gray_samples.pt")
    manifest = {
        "version": dataset.version,
        "samples": len(dataset),
        "checksum": checksum,
        "seed": dataset.seed,
        "hands_played": dataset.hands_played,
        "trick_band": dataset.trick_band,
        "advantage_margin": dataset.advantage_margin,
        "weight_margin": dataset.weight_margin,
        "clear_decisions": dataset.clear_decisions,
        "gray_decisions": dataset.gray_decisions,
        "feature_names": list(FEATURE_NAMES),
        "source": "simulated euchre deals played by the rules-and-payoff heuristic",
        "preprocessing": (
            "Keep a trump decision only when the hand has 1 to 3 trump and either "
            "the expected tricks are near 3 or the win-probability edge is inside "
            "the advantage margin. Keep a card play or discard only when another "
            "legal card is within the weight margin of the heuristic's choice. "
            "Zero trump, four or more trump, lopsided trick estimates, and card "
            "choices with a wide weight gap are clear and are not stored."
        ),
        "train_fraction": 1.0 - val_fraction - test_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_gray_dataset(directory: Path) -> Dict[str, Any]:
    """Load gray-area rows and check them against the manifest.

    Parameters
    ----------
    directory : Path
        Directory written by :func:`save_gray_dataset`.

    Returns
    -------
    Dict[str, Any]
        The saved payload, including features, labels, and split.

    Raises
    ------
    ValueError
        If the checksum or feature names do not match.
    """
    payload = cast(
        Dict[str, Any],
        torch.load(directory / "gray_samples.pt", map_location="cpu", weights_only=False),
    )
    manifest_path = directory / "manifest.json"
    checksum = tensor_checksum(payload["features"], payload["labels"])
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("checksum") != checksum:
            raise ValueError(f"Checksum mismatch for {directory}")
    if payload.get("checksum") != checksum:
        raise ValueError(f"Checksum mismatch inside {directory / 'gray_samples.pt'}")
    if list(payload.get("feature_names", [])) != list(FEATURE_NAMES):
        raise ValueError(f"{directory} uses a different feature layout")
    return payload
