"""Gray-area gate for the heuristic tie-break model.

A decision stays with the heuristic when one option is clearly ahead. A hand
of hearts does not need a model to pass a spade turned up. A single legal
card does not need one either. The model is asked when a call sits near the
three-trick line, or when the top card weights are within
:data:`WEIGHT_MARGIN` of each other. Training rows are those gray spots only.
"""

from typing import List, Optional, Sequence, Tuple

from ..models import Card, Rank, Suit
from .heuristic_player import SAME_COLOR, CallEvaluation, HeuristicPlayer

# Expected tricks within this distance of 3 are close enough to ask the model.
TRICK_BAND = 0.55

# A win-probability edge smaller than this is a tie even if the trick estimate
# sits a little outside the band. Both bowers at the start of a game sit above it.
ADVANTAGE_MARGIN = 0.012

# Card and discard weights within this distance of the heuristic's choice are
# close enough to ask the model. One rank is one point. A change of role is larger.
WEIGHT_MARGIN = 3.0

FEATURE_NAMES: Tuple[str, ...] = (
    "round_is_second",
    "trump_count",
    "has_right",
    "has_left",
    "has_trump_ace",
    "low_trump_count",
    "off_ace_count",
    "missing_suits",
    "dealer_self",
    "dealer_partner",
    "dealer_opponent",
    "trump_times_partner",
    "trump_times_opponent",
    "trump_times_self",
    "seat_from_dealer",
    "team_score",
    "opponent_score",
    "expected_tricks",
    "advantage_x10",
    "heuristic_call",
    "is_next",
    "opponent_passes",
    "upcard_trump_value",
    "can_win_on_make",
    "euchre_loses_game",
    "is_play",
    "is_discard",
    "candidate_is_trump",
    "candidate_trump_value",
    "candidate_rank",
    "candidate_is_ace",
    "candidate_is_bower",
    "candidate_follows_lead",
    "candidate_weight",
    "weight_gap",
    "close_count",
    "is_leading",
    "partner_winning",
    "urgency_must",
    "urgency_safe",
    "we_are_makers",
    "tricks_played",
    "hand_size",
    "team_tricks",
    "opponent_tricks",
)

ACTION_FEATURE_COUNT = 20

FEATURE_COUNT = len(FEATURE_NAMES)


def feature_index(name: str) -> int:
    """Return the column of one tie-break feature.

    Parameters
    ----------
    name : str
        A name from :data:`FEATURE_NAMES`.

    Returns
    -------
    int
        The column index.
    """
    return FEATURE_NAMES.index(name)


def trump_count(hand: Sequence[Card], suit: Suit) -> int:
    """Count cards in ``hand`` that would be trump if ``suit`` is named.

    Parameters
    ----------
    hand : Sequence[Card]
        Cards the player currently holds. The upcard is not included.
    suit : Suit
        The suit being considered as trump.

    Returns
    -------
    int
        Trump cards, counting the left bower.
    """
    return sum(1 for card in hand if card.get_trump_value(suit) > 0)


def is_gray_call(
    count: int,
    expected_tricks: float,
    advantage: float,
    trick_band: float = TRICK_BAND,
    advantage_margin: float = ADVANTAGE_MARGIN,
) -> bool:
    """Return True when a trump call should be broken by the micro model.

    Parameters
    ----------
    count : int
        Trump cards in hand for the suit being considered.
    expected_tricks : float
        Tricks the heuristic expects the calling team to win.
    advantage : float
        Heuristic win probability of calling, minus the probability of passing.
    trick_band : float
        How close to three expected tricks still counts as gray.
    advantage_margin : float
        Win-probability edge that still counts as a tie.

    Returns
    -------
    bool
        True when the heuristic should ask the micro model.
    """
    if count <= 0 or count >= 4:
        return False
    if abs(expected_tricks - 3.0) <= trick_band:
        return True
    return abs(advantage) < advantage_margin


def close_alternatives(
    scored: Sequence[Tuple[Card, float]],
    chosen: Card,
    margin: float = WEIGHT_MARGIN,
) -> List[Tuple[Card, float]]:
    """Return the other cards whose weight sits near the heuristic's choice.

    Parameters
    ----------
    scored : Sequence[Tuple[Card, float]]
        Weights for the legal options.
    chosen : Card
        The card the heuristic would play or discard.
    margin : float
        How far behind the chosen weight still counts as a tie.

    Returns
    -------
    List[Tuple[Card, float]]
        Alternatives, highest weight first. Empty when the choice is clear.
    """
    chosen_weight = None
    for card, weight in scored:
        if card.rank == chosen.rank and card.suit == chosen.suit:
            chosen_weight = weight
            break
    if chosen_weight is None:
        return []
    near = [
        (card, weight)
        for card, weight in scored
        if not (card.rank == chosen.rank and card.suit == chosen.suit)
        and chosen_weight - weight <= margin
    ]
    near.sort(key=lambda item: item[1], reverse=True)
    return near


def features_for(
    player: HeuristicPlayer,
    evaluation: CallEvaluation,
    top_card: Card,
    round_number: int,
    dealer_location: str,
) -> List[float]:
    """Encode one gray-area trump decision for the micro model.

    Parameters
    ----------
    player : HeuristicPlayer
        The player about to bid. The hand, score, and seats are read from it.
    evaluation : CallEvaluation
        The heuristic's evaluation of the suit being considered.
    top_card : Card
        The turned card.
    round_number : int
        1 when the upcard can be ordered up, 2 when a suit is named.
    dealer_location : str
        ``self``, ``partner``, or ``opponent``: who would receive the upcard.

    Returns
    -------
    List[float]
        One value per :data:`FEATURE_NAMES`.
    """
    suit = evaluation.suit
    count = trump_count(player.hand, suit)
    has_right = 0.0
    has_left = 0.0
    has_ace = 0.0
    low_trump = 0
    off_aces = 0
    suits_seen = set()
    for card in player.hand:
        suits_seen.add(card.suit)
        value = card.get_trump_value(suit)
        if value == 7:
            has_right = 1.0
        elif value == 6:
            has_left = 1.0
        elif value == 5:
            has_ace = 1.0
        elif value in (1, 2, 3):
            low_trump += 1
        elif card.rank == Rank.ACE:
            off_aces += 1
    dealer_self = 1.0 if dealer_location == "self" else 0.0
    dealer_partner = 1.0 if dealer_location == "partner" else 0.0
    dealer_opponent = 1.0 if dealer_location == "opponent" else 0.0
    scaled_count = count / 5.0
    if player.seat is None or player.dealer_seat is None:
        seat_from_dealer = 0.0
        passes = 0
    else:
        seat_from_dealer = ((player.seat - player.dealer_seat) % 4) / 3.0
        passes = player._opponent_passes(round_number)
    encoded = [
        1.0 if round_number == 2 else 0.0,
        scaled_count,
        has_right,
        has_left,
        has_ace,
        low_trump / 5.0,
        off_aces / 3.0,
        (4 - len(suits_seen)) / 3.0,
        dealer_self,
        dealer_partner,
        dealer_opponent,
        scaled_count * dealer_partner,
        scaled_count * dealer_opponent,
        scaled_count * dealer_self,
        seat_from_dealer,
        player.team_score / 10.0,
        player.opponent_score / 10.0,
        evaluation.expected_tricks / 5.0,
        evaluation.advantage * 10.0,
        1.0 if evaluation.should_call else 0.0,
        1.0 if suit == SAME_COLOR[top_card.suit] else 0.0,
        passes / 2.0,
        top_card.get_trump_value(suit) / 7.0,
        1.0 if player.team_score >= 9 else 0.0,
        1.0 if player.opponent_score >= 8 else 0.0,
    ]
    encoded.extend(_action_features())
    if len(encoded) != FEATURE_COUNT:
        raise RuntimeError("tie-break feature vector does not match FEATURE_NAMES")
    return encoded


def features_for_card(
    player: HeuristicPlayer,
    card: Card,
    trump: Optional[Suit],
    weight: float,
    gap: float,
    close_count: int,
    kind: str,
    lead_suit: Optional[Suit] = None,
    partner_winning: bool = False,
    urgency: str = "want",
) -> List[float]:
    """Encode one gray card play or discard.

    Parameters
    ----------
    player : HeuristicPlayer
        The player about to choose a card.
    card : Card
        The alternative the model is scoring.
    trump : Suit, optional
        Trump for this hand.
    weight : float
        Heuristic weight of ``card``.
    gap : float
        How far ``weight`` sits behind the heuristic's choice.
    close_count : int
        How many cards, including the heuristic's choice, are inside the margin.
    kind : str
        ``play`` or ``discard``.
    lead_suit : Suit, optional
        The suit led, when this is a play on an existing trick.
    partner_winning : bool
        True when partner currently holds the trick.
    urgency : str
        ``must``, ``safe``, or ``want``.

    Returns
    -------
    List[float]
        One value per :data:`FEATURE_NAMES`.
    """
    suit = trump if trump is not None else card.suit
    count = trump_count(player.hand, suit) if trump is not None else 0
    has_right, has_left, has_ace, low_trump, off_aces, missing = _trump_shape(player.hand, suit)
    if trump is None:
        has_right = has_left = has_ace = 0.0
        low_trump = 0
        off_aces = 0
    location = player._upcard_location(False)
    dealer_self = 1.0 if location == "self" else 0.0
    dealer_partner = 1.0 if location == "partner" else 0.0
    dealer_opponent = 1.0 if location == "opponent" else 0.0
    scaled_count = count / 5.0
    if player.seat is None or player.dealer_seat is None:
        seat_from_dealer = 0.0
    else:
        seat_from_dealer = ((player.seat - player.dealer_seat) % 4) / 3.0
    trump_value = card.get_trump_value(trump) if trump is not None else 0
    follows = 1.0 if lead_suit is not None and card.suit == lead_suit else 0.0
    tricks_played = player.team_tricks + player.opponent_tricks
    encoded = [
        0.0,
        scaled_count,
        has_right,
        has_left,
        has_ace,
        low_trump / 5.0,
        off_aces / 3.0,
        missing,
        dealer_self,
        dealer_partner,
        dealer_opponent,
        scaled_count * dealer_partner,
        scaled_count * dealer_opponent,
        scaled_count * dealer_self,
        seat_from_dealer,
        player.team_score / 10.0,
        player.opponent_score / 10.0,
        0.0,
        gap,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 if player.team_score >= 9 else 0.0,
        1.0 if player.opponent_score >= 8 else 0.0,
    ]
    encoded.extend(
        _action_features(
            is_play=kind == "play",
            is_discard=kind == "discard",
            candidate_is_trump=trump_value > 0,
            candidate_trump_value=trump_value / 7.0,
            candidate_rank=card.rank.value / 14.0,
            candidate_is_ace=card.rank == Rank.ACE,
            candidate_is_bower=trump_value >= 6,
            candidate_follows_lead=follows == 1.0,
            candidate_weight=weight / 100.0,
            weight_gap=gap / 10.0,
            close_count=close_count / 5.0,
            is_leading=lead_suit is None and kind == "play",
            partner_winning=partner_winning,
            urgency_must=urgency == "must",
            urgency_safe=urgency == "safe",
            we_are_makers=player._we_are_makers(),
            tricks_played=tricks_played / 5.0,
            hand_size=len(player.hand) / 6.0,
            team_tricks=player.team_tricks / 5.0,
            opponent_tricks=player.opponent_tricks / 5.0,
        )
    )
    if len(encoded) != FEATURE_COUNT:
        raise RuntimeError("tie-break feature vector does not match FEATURE_NAMES")
    return encoded


def _trump_shape(hand: Sequence[Card], suit: Suit) -> Tuple[float, float, float, int, int, float]:
    """Right, left, ace, low trump, off aces, and missing suits for one suit."""
    has_right = 0.0
    has_left = 0.0
    has_ace = 0.0
    low_trump = 0
    off_aces = 0
    suits_seen = set()
    for card in hand:
        suits_seen.add(card.suit)
        value = card.get_trump_value(suit)
        if value == 7:
            has_right = 1.0
        elif value == 6:
            has_left = 1.0
        elif value == 5:
            has_ace = 1.0
        elif value in (1, 2, 3):
            low_trump += 1
        elif card.rank == Rank.ACE:
            off_aces += 1
    missing = (4 - len(suits_seen)) / 3.0
    return has_right, has_left, has_ace, low_trump, off_aces, missing


def _action_features(
    is_play: bool = False,
    is_discard: bool = False,
    candidate_is_trump: bool = False,
    candidate_trump_value: float = 0.0,
    candidate_rank: float = 0.0,
    candidate_is_ace: bool = False,
    candidate_is_bower: bool = False,
    candidate_follows_lead: bool = False,
    candidate_weight: float = 0.0,
    weight_gap: float = 0.0,
    close_count: float = 0.0,
    is_leading: bool = False,
    partner_winning: bool = False,
    urgency_must: bool = False,
    urgency_safe: bool = False,
    we_are_makers: bool = False,
    tricks_played: float = 0.0,
    hand_size: float = 0.0,
    team_tricks: float = 0.0,
    opponent_tricks: float = 0.0,
) -> List[float]:
    """Build the card-decision tail of a feature vector.

    Call rows leave every value at zero.
    """
    encoded = [
        1.0 if is_play else 0.0,
        1.0 if is_discard else 0.0,
        1.0 if candidate_is_trump else 0.0,
        candidate_trump_value,
        candidate_rank,
        1.0 if candidate_is_ace else 0.0,
        1.0 if candidate_is_bower else 0.0,
        1.0 if candidate_follows_lead else 0.0,
        candidate_weight,
        weight_gap,
        close_count,
        1.0 if is_leading else 0.0,
        1.0 if partner_winning else 0.0,
        1.0 if urgency_must else 0.0,
        1.0 if urgency_safe else 0.0,
        1.0 if we_are_makers else 0.0,
        tricks_played,
        hand_size,
        team_tricks,
        opponent_tricks,
    ]
    if len(encoded) != ACTION_FEATURE_COUNT:
        raise RuntimeError("action features do not match ACTION_FEATURE_COUNT")
    return encoded
