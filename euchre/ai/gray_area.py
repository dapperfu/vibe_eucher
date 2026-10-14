"""Gray-area gate for the heuristic tie-break model.

A trump decision stays with the heuristic when the hand is obvious. A hand of
hearts does not need a model to pass a spade turned up, and four or more
trump do not need one to call. One, two, or three trump is a gray area when
the expected tricks sit near the three-trick make line, which is where the
dealer being a partner or an opponent can flip the call. The micro model is
trained only on those decisions.
"""

from typing import List, Sequence, Tuple

from ..models import Card, Rank, Suit
from .heuristic_player import SAME_COLOR, CallEvaluation, HeuristicPlayer

# Expected tricks within this distance of 3 are close enough to ask the model.
TRICK_BAND = 0.55

# A win-probability edge smaller than this is a tie even if the trick estimate
# sits a little outside the band. Both bowers at the start of a game sit above it.
ADVANTAGE_MARGIN = 0.012

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
)

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
    if len(encoded) != FEATURE_COUNT:
        raise RuntimeError("tie-break feature vector does not match FEATURE_NAMES")
    return encoded
