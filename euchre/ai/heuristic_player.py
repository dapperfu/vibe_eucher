"""Heuristic euchre player driven by the rules and the score payoff.

The player extends :class:`euchre.models.Player` and decides from public
information only: its hand, the upcard, the cards already played, the score,
and which seat is the dealer. There is no learned weight.

Two payoffs come straight from the rules. The makers score 1 point for three
or four tricks and 2 for all five. If they take fewer than three, the
defenders score 2 (a euchre). The game is a race to 10, so a call is worth
making only when it raises the chance of reaching 10 before the opponents.
That chance, while the race is still open, is the gambler's-ruin value
``(10 - opponent) / ((10 - us) + (10 - opponent))``.

Unseen cards are treated as a random deal of the remaining deck. A trump card
wins a lead when every higher trump sits with partner or in the kitty, which
is a hypergeometric count over those safe slots. Lower trump still take
tricks once a boss has pulled the outstanding higher ones. The same ranking
the rules give the bowers — right bower, left bower, ace, king, queen, ten,
nine — is the ranking used for every comparison.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..models import Card, Player, PlayerType, Rank, Suit

TARGET_SCORE = 10
TRICKS_IN_ROUND = 5

# Spread of team tricks around the expected value. Boss-heavy hands are a
# little more predictable; 0.8 matches a round that can still break either way.
TRICK_SIGMA = 0.8

# A non-boss trump takes a trick this often once we can draw with a boss,
# and less often when every boss is still out.
LENGTH_WITH_CONTROL = 0.62
LENGTH_WITHOUT_CONTROL = 0.28

# Partner's other cards, before the known upcard is added. The upcard
# replaces a discard, so a boss given to the dealer is counted on top.
PARTNER_BASELINE = 0.75
PARTNER_BASELINE_PER_TRUMP = 0.05

OFF_ACE = 0.58
GUARDED_KING = 0.22
BARE_KING = 0.08

# Holding the top trump lets the maker draw. Both bowers leave the other
# side with no boss at all.
RIGHT_BOWER_CONTROL = 0.10
TEAM_BOWERS_CONTROL = 0.35
THREE_TRUMP_WITH_BOWER = 0.25

# In the second round the same color as the turned-down suit is "next":
# every opponent has already refused that color, and one card of it is buried.
NEXT_SUIT_INFORMATION = 0.15

SAME_COLOR = {
    Suit.HEARTS: Suit.DIAMONDS,
    Suit.DIAMONDS: Suit.HEARTS,
    Suit.CLUBS: Suit.SPADES,
    Suit.SPADES: Suit.CLUBS,
}


@dataclass(frozen=True)
class CallEvaluation:
    """Value of naming one suit trump, in win probability."""

    suit: Suit
    expected_tricks: float
    win_probability: float
    pass_win_probability: float
    should_call: bool
    reasoning: str

    @property
    def advantage(self) -> float:
        """How much naming this suit improves the chance of winning the game."""
        return self.win_probability - self.pass_win_probability


def win_probability(my_score: int, opponent_score: int, target: int = TARGET_SCORE) -> float:
    """Probability of reaching ``target`` first if each future point is a coin flip.

    This is the gambler's-ruin probability for two teams that are equally
    likely to score the next point. A finished race is 1 or 0.
    """
    if my_score >= target:
        return 1.0
    if opponent_score >= target:
        return 0.0
    my_needed = target - my_score
    opponent_needed = target - opponent_score
    return opponent_needed / (my_needed + opponent_needed)


def trick_distribution(mean: float, sigma: float = TRICK_SIGMA) -> List[float]:
    """Discrete distribution over 0..5 tricks, centered on ``mean``."""
    mean = min(float(TRICKS_IN_ROUND), max(0.0, mean))
    weights = [
        math.exp(-0.5 * ((tricks - mean) / sigma) ** 2) for tricks in range(TRICKS_IN_ROUND + 1)
    ]
    total = sum(weights)
    return [weight / total for weight in weights]


def probability_all_safe(count: int, safe_slots: int, total_slots: int) -> float:
    """Probability that ``count`` specific cards all lie in ``safe_slots``.

    The unseen cards are dealt at random. Partner and the kitty are safe:
    a higher trump there cannot be played against the calling team. Opponents'
    hands are not safe. The ratio of combinations is
    ``C(safe, count) / C(total, count)``.
    """
    if count <= 0:
        return 1.0
    if safe_slots < count or total_slots < count or total_slots <= 0:
        return 0.0
    probability = 1.0
    for index in range(count):
        probability *= (safe_slots - index) / (total_slots - index)
    return probability


def maker_outcome_score(my_score: int, opponent_score: int, tricks: int) -> Tuple[int, int]:
    """Scores after the makers take ``tricks`` tricks."""
    if tricks >= 5:
        return my_score + 2, opponent_score
    if tricks >= 3:
        return my_score + 1, opponent_score
    return my_score, opponent_score + 2


class HeuristicPlayer(Player):
    """Rule-and-payoff player. Subclass of the base :class:`Player`."""

    def __init__(self, name: str) -> None:
        super().__init__(name, PlayerType.AI)
        self.seat: Optional[int] = None
        self.dealer_seat: Optional[int] = None
        self.team_score = 0
        self.opponent_score = 0
        self.played_cards: List[Tuple[int, Card]] = []
        self.trick_plays: List[Tuple[int, Card]] = []
        self.team_tricks = 0
        self.opponent_tricks = 0
        self.trump_caller_seat: Optional[int] = None
        self.known_trump: Optional[Suit] = None
        self.last_reasoning = ""

    def begin_round(
        self,
        seat: int,
        dealer_seat: int,
        team_score: int = 0,
        opponent_score: int = 0,
    ) -> None:
        """Forget the previous deal and record the new score and seats."""
        self.team_score = max(0, int(team_score))
        self.opponent_score = max(0, int(opponent_score))
        self.played_cards = []
        self.trick_plays = []
        self.team_tricks = 0
        self.opponent_tricks = 0
        self.trump_caller_seat = None
        self.known_trump = None
        self.last_reasoning = ""
        self.note_table(seat, dealer_seat)

    def note_table(self, seat: int, dealer_seat: int) -> None:
        """Record this player's seat and the dealer's seat.

        Seats are indexes around the table. Partners sit opposite each other,
        so a player's partner is two seats away and the two teams are the
        even and odd seats.
        """
        self.seat = seat
        self.dealer_seat = dealer_seat
        self.is_dealer = seat == dealer_seat
        self.set_team(seat % 2)

    def note_trump_call(self, caller_seat: Optional[int], trump_suit: Suit) -> None:
        """Record who named trump. Their team is the makers."""
        self.trump_caller_seat = caller_seat
        self.known_trump = trump_suit

    def note_card_played(self, seat: int, card: Card) -> None:
        """Remember a card that hit the table. Higher trump that have been seen are dead."""
        self.played_cards.append((seat, card))
        self.trick_plays.append((seat, card))

    def note_trick_complete(self, winner_seat: int) -> None:
        """Count the trick for the winner's team and clear the table."""
        if self.team is not None and winner_seat % 2 == self.team:
            self.team_tricks += 1
        else:
            self.opponent_tricks += 1
        self.trick_plays = []

    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Order the upcard suit up when doing so raises our chance of winning.

        Parameters
        ----------
        top_card : Card
            The card turned up. Ordering it up makes its suit trump and gives
            that card to the dealer.
        is_partner_dealing : bool
            Used only when this player has not been told the seats. When the
            seats are known, the dealer is read from them.
        """
        if top_card is None or not self.hand:
            self.last_reasoning = "Pass: no upcard or no cards."
            return False
        evaluation = self.evaluate_call(
            top_card.suit,
            top_card,
            round_number=1,
            upcard_location=self._upcard_location(is_partner_dealing),
        )
        self.last_reasoning = evaluation.reasoning
        return evaluation.should_call

    def choose_trump_suit(self, top_card: Card, must_call: bool = False) -> Optional[Suit]:
        """Name a suit in the second round, or pass.

        The turned-down suit is illegal. The dealer has to name some suit if
        the choice reaches them; every other player passes when the best
        legal suit does not improve their chance of winning.
        """
        if top_card is None:
            return None
        forced = must_call or self._dealer_must_name_suit()
        best_suit: Optional[Suit] = None
        best_evaluation: Optional[CallEvaluation] = None
        for suit in Suit:
            if suit == top_card.suit:
                continue
            evaluation = self.evaluate_call(
                suit,
                top_card,
                round_number=2,
                upcard_location="buried",
            )
            if best_evaluation is None or self._prefer(evaluation, best_evaluation, top_card.suit):
                best_suit = suit
                best_evaluation = evaluation
        if best_evaluation is None or best_suit is None:
            self.last_reasoning = "No legal suit to name."
            return None
        if forced or best_evaluation.should_call:
            if forced and not best_evaluation.should_call:
                self.last_reasoning = (
                    f"Dealer must name {best_suit.name}. {best_evaluation.reasoning}"
                )
            else:
                self.last_reasoning = best_evaluation.reasoning
            return best_suit
        self.last_reasoning = best_evaluation.reasoning
        return None

    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Play one legal card.

        Follow the led printed suit when this hand has one. Otherwise lead a
        boss trump if we are the makers, cash an off-ace, or save trump when
        defending. In a trick already started, take the trick with the
        cheapest winner when an opponent is winning, and duck when partner
        is winning unless this trick is required to make or euchre.
        """
        if not self.hand:
            raise ValueError(f"{self.name} has no cards to play")

        plays = self._plays_before_us()
        if plays:
            lead_suit = plays[0][1].suit
        legal = self._legal_cards(lead_suit)
        if len(legal) == 1:
            chosen = legal[0]
            self.last_reasoning = f"Only legal card is {chosen}."
            return chosen
        if not plays:
            chosen = self._lead_card(legal, trump_suit)
            self.last_reasoning = f"Lead {chosen}."
            return chosen
        chosen = self._answer_trick(legal, plays, lead_suit, trump_suit)
        self.last_reasoning = f"Play {chosen} on the current trick."
        return chosen

    def choose_card_to_discard(self, trump_suit: Suit) -> Card:
        """Discard after the dealer picks up the upcard.

        Throw a low side singleton so the hand gains a void, keep off-aces
        over low side cards from a longer suit, and keep trump unless every
        card is trump.
        """
        if not self.hand:
            raise ValueError(f"{self.name} has no card to discard")
        return self._choose_discard(self.hand, trump_suit)

    def play_weights(
        self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]
    ) -> List[Tuple[Card, float]]:
        """Score every legal card. Higher means the heuristic would rather play it.

        The gaps match the play rules. A boss trump and a side nine land far
        apart. Two cards the rules treat as neighbors, such as the left and
        right bower on lead, land one point apart.

        Parameters
        ----------
        lead_suit : Suit, optional
            The suit led to this trick. Ignored once cards are already on the table.
        trump_suit : Suit, optional
            Trump for this hand.

        Returns
        -------
        List[Tuple[Card, float]]
            One weight per legal card.
        """
        plays = self._plays_before_us()
        if plays:
            lead_suit = plays[0][1].suit
        legal = self._legal_cards(lead_suit)
        if len(legal) <= 1:
            return [(card, 100.0) for card in legal]
        if not plays:
            return self._lead_weights(legal, trump_suit)
        return self._answer_weights(legal, plays, lead_suit, trump_suit)

    def discard_weights(self, trump_suit: Suit) -> List[Tuple[Card, float]]:
        """Score every card as a discard. Higher means the heuristic would rather throw it.

        Parameters
        ----------
        trump_suit : Suit
            The trump suit after the dealer picks up the upcard.

        Returns
        -------
        List[Tuple[Card, float]]
            One weight per card in hand.
        """
        weights = []
        for held in self.hand:
            trump_value = held.get_trump_value(trump_suit)
            if trump_value:
                weights.append((held, -float(trump_value)))
                continue
            if held.rank == Rank.ACE:
                weights.append((held, 15.0))
                continue
            mates = self._side_mates(held, trump_suit)
            base = 50.0 if not mates else 30.0
            weights.append((held, base - float(held.rank.value)))
        return weights

    def evaluate_call(
        self,
        suit: Suit,
        top_card: Optional[Card],
        round_number: int,
        upcard_location: str,
        must_call: bool = False,
    ) -> CallEvaluation:
        """Expected tricks and win probability for naming ``suit`` trump."""
        cards, partner_cards, opponent_cards, dead_cards = self._call_holdings(
            suit, top_card, upcard_location
        )
        removed_from = None
        if top_card is not None and top_card.get_trump_value(suit) > 0:
            if upcard_location == "partner":
                removed_from = "partner"
            elif upcard_location == "opponent":
                removed_from = "opponent"
        safe_slots, total_slots = self._unseen_slots(round_number, removed_from)
        expected = self._expected_team_tricks(
            cards,
            suit,
            partner_cards,
            opponent_cards,
            dead_cards,
            safe_slots,
            total_slots,
        )
        if (
            round_number == 2
            and top_card is not None
            and suit == SAME_COLOR[top_card.suit]
        ):
            expected += NEXT_SUIT_INFORMATION
        expected = min(float(TRICKS_IN_ROUND), max(0.0, expected))

        distribution = trick_distribution(expected)
        win_prob = 0.0
        for tricks, probability in enumerate(distribution):
            my_score, opponent_score = maker_outcome_score(
                self.team_score, self.opponent_score, tricks
            )
            win_prob += probability * win_probability(my_score, opponent_score)
        pass_prob = win_probability(self.team_score, self.opponent_score)
        should_call = must_call or win_prob > pass_prob
        decision = "Call" if should_call else "Pass"
        reasoning = (
            f"{decision} {suit.name}: {expected:.2f} expected team tricks, "
            f"win probability {win_prob:.3f} against passing at {pass_prob:.3f} "
            f"({upcard_location} holds the upcard)."
        )
        return CallEvaluation(
            suit=suit,
            expected_tricks=expected,
            win_probability=win_prob,
            pass_win_probability=pass_prob,
            should_call=should_call,
            reasoning=reasoning,
        )

    def _call_holdings(
        self,
        suit: Suit,
        top_card: Optional[Card],
        upcard_location: str,
    ) -> Tuple[List[Card], List[Card], List[Card], List[Card]]:
        """Place the upcard, and drop the dealer's discard when we receive it."""
        cards = list(self.hand)
        partner_cards: List[Card] = []
        opponent_cards: List[Card] = []
        dead_cards: List[Card] = []
        if top_card is None:
            return cards, partner_cards, opponent_cards, dead_cards
        if upcard_location == "self":
            trial = cards + [top_card]
            discarded = self._choose_discard(trial, suit)
            cards = [card for card in trial if card is not discarded]
            # The throw is out of play. A discarded trump is buried, not
            # available for an opponent to win a trick with.
            if discarded.get_trump_value(suit) > 0:
                dead_cards.append(discarded)
        elif upcard_location == "partner":
            if top_card.get_trump_value(suit) > 0:
                partner_cards.append(top_card)
        elif upcard_location == "opponent":
            if top_card.get_trump_value(suit) > 0:
                opponent_cards.append(top_card)
        else:
            if top_card.get_trump_value(suit) > 0:
                dead_cards.append(top_card)
        return cards, partner_cards, opponent_cards, dead_cards

    def _expected_team_tricks(
        self,
        cards: Sequence[Card],
        trump: Suit,
        partner_cards: Sequence[Card],
        opponent_cards: Sequence[Card],
        dead_cards: Sequence[Card],
        safe_slots: int,
        total_slots: int,
    ) -> float:
        """Tricks the calling team can expect from this hand and the upcard."""
        our_values = {
            card.get_trump_value(trump) for card in cards if card.get_trump_value(trump) > 0
        }
        partner_values = {
            card.get_trump_value(trump) for card in partner_cards if card.get_trump_value(trump) > 0
        }
        opponent_values = {
            card.get_trump_value(trump)
            for card in opponent_cards
            if card.get_trump_value(trump) > 0
        }
        dead_values = {
            card.get_trump_value(trump) for card in dead_cards if card.get_trump_value(trump) > 0
        }

        we_control = self._callers_control_trump(our_values, opponent_values)
        tricks = 0.0
        for value in sorted(our_values, reverse=True):
            boss_probability = self._boss_probability(
                value,
                our_values,
                partner_values,
                opponent_values,
                dead_values,
                safe_slots,
                total_slots,
            )
            length = LENGTH_WITH_CONTROL if we_control else LENGTH_WITHOUT_CONTROL
            tricks += boss_probability + (1.0 - boss_probability) * length

        for value in partner_values:
            boss_probability = self._boss_probability(
                value,
                our_values,
                partner_values,
                opponent_values,
                dead_values,
                safe_slots,
                total_slots,
            )
            tricks += boss_probability

        tricks += self._side_suit_tricks(cards, trump)
        tricks += max(0.30, PARTNER_BASELINE - PARTNER_BASELINE_PER_TRUMP * len(our_values))

        if 7 in our_values:
            tricks += RIGHT_BOWER_CONTROL
        if (
            6 in our_values
            and 7 not in opponent_values
            and (7 in our_values or 7 in partner_values or 7 in dead_values)
        ):
            tricks += TEAM_BOWERS_CONTROL
        if len(our_values) >= 3 and (7 in our_values or 6 in our_values):
            tricks += THREE_TRUMP_WITH_BOWER
        return tricks

    def _boss_probability(
        self,
        value: int,
        our_values: set,
        partner_values: set,
        opponent_values: set,
        dead_values: set,
        safe_slots: int,
        total_slots: int,
    ) -> float:
        """Probability this trump rank wins the trick it is led into."""
        unknown_higher = 0
        for higher in range(value + 1, 8):
            if higher in opponent_values:
                return 0.0
            if higher in our_values or higher in partner_values or higher in dead_values:
                continue
            unknown_higher += 1
        return probability_all_safe(unknown_higher, safe_slots, total_slots)

    def _callers_control_trump(self, our_values: set, opponent_values: set) -> bool:
        """True when a boss trump is in this hand and not known to sit with an opponent."""
        if 7 in our_values:
            return True
        return 6 in our_values and 7 not in opponent_values

    def _side_suit_tricks(self, cards: Sequence[Card], trump: Suit) -> float:
        """Off-aces and guarded kings. The left bower is trump, not a side jack."""
        by_suit = {}
        for card in cards:
            if card.get_trump_value(trump) > 0:
                continue
            by_suit.setdefault(card.suit, []).append(card)
        tricks = 0.0
        for suit_cards in by_suit.values():
            ranks = {card.rank for card in suit_cards}
            if Rank.ACE in ranks:
                tricks += OFF_ACE
                if Rank.KING in ranks:
                    tricks += GUARDED_KING
            elif Rank.KING in ranks:
                tricks += BARE_KING
        return tricks

    def _unseen_slots(self, round_number: int, removed_from: Optional[str]) -> Tuple[int, int]:
        """Safe slots (partner and kitty) and total unseen slots.

        An opponent who has already passed is less likely to be sitting on
        the boss trump, so one of their slots is treated as buried instead.
        A placed upcard is removed from the hand that received it.
        """
        partner, opponents, kitty = 5, 10, 3
        shift = min(self._opponent_passes(round_number), 3)
        opponents -= shift
        kitty += shift
        if removed_from == "partner":
            partner -= 1
        elif removed_from == "opponent":
            opponents -= 1
        total = partner + opponents + kitty
        safe = partner + kitty
        return max(safe, 0), max(total, 1)

    def _opponent_passes(self, round_number: int) -> int:
        """How many opponents have already had a chance to name trump."""
        del round_number  # Speaking order is the same in both rounds.
        if self.seat is None or self.dealer_seat is None:
            return 0
        passes = 0
        for step in range(1, 4):
            seat = (self.dealer_seat + step) % 4
            if seat == self.seat:
                break
            if (seat - self.seat) % 4 != 2:
                passes += 1
        return passes

    def _upcard_location(self, is_partner_dealing: bool) -> str:
        """Who receives the upcard if it is ordered up."""
        if self.seat is not None and self.dealer_seat is not None:
            if self.seat == self.dealer_seat:
                return "self"
            if (self.dealer_seat - self.seat) % 4 == 2:
                return "partner"
            return "opponent"
        if self.is_dealer:
            return "self"
        if is_partner_dealing:
            return "partner"
        return "opponent"

    def _dealer_must_name_suit(self) -> bool:
        """The dealer must name a suit once the second round reaches them."""
        if self.seat is not None and self.dealer_seat is not None:
            return self.seat == self.dealer_seat
        return self.is_dealer

    def _prefer(
        self, candidate: CallEvaluation, current: CallEvaluation, turned_down: Suit
    ) -> bool:
        """Prefer the higher win probability, then the same color as the card turned down."""
        if candidate.win_probability > current.win_probability + 1e-12:
            return True
        if abs(candidate.win_probability - current.win_probability) > 1e-12:
            return False
        candidate_is_next = candidate.suit == SAME_COLOR[turned_down]
        current_is_next = current.suit == SAME_COLOR[turned_down]
        return candidate_is_next and not current_is_next

    def _choose_discard(self, hand: Sequence[Card], trump: Suit) -> Card:
        """Card the dealer should throw from a hand that includes the upcard."""

        def key(card: Card) -> Tuple[int, int, int]:
            trump_value = card.get_trump_value(trump)
            if trump_value:
                return (2, trump_value, 0)
            if card.rank == Rank.ACE:
                return (1, card.rank.value, 0)
            mates = [
                other
                for other in hand
                if other is not card
                and other.suit == card.suit
                and other.get_trump_value(trump) == 0
            ]
            singleton = 0 if not mates else 1
            return (0, singleton, card.rank.value)

        return min(hand, key=key)

    def _side_mates(self, held: Card, trump: Optional[Suit]) -> List[Card]:
        """Other side cards of the same printed suit."""
        return [
            other
            for other in self.hand
            if other is not held
            and other.suit == held.suit
            and (not trump or other.get_trump_value(trump) == 0)
        ]

    def _lead_weights(
        self, legal: Sequence[Card], trump: Optional[Suit]
    ) -> List[Tuple[Card, float]]:
        """Weights for leading a trick. Mirrors :meth:`_lead_card`."""
        if not trump:
            return [(card, float(card.rank.value)) for card in legal]
        trump_cards = [card for card in legal if card.get_trump_value(trump) > 0]
        masters = [card for card in trump_cards if self._is_boss_trump(card, trump)]
        off_aces = [
            card for card in legal if card.get_trump_value(trump) == 0 and card.rank == Rank.ACE
        ]
        if self._we_are_makers():
            if masters:
                return [
                    (card, 100.0 - float(card.get_trump_value(trump)))
                    if card in masters
                    else (card, 0.0)
                    for card in legal
                ]
            if off_aces:
                return [(card, 80.0 if card in off_aces else 0.0) for card in legal]
            if trump_cards:
                return [
                    (card, 60.0 - float(card.get_trump_value(trump)))
                    if card in trump_cards
                    else (card, 0.0)
                    for card in legal
                ]
            return [(card, float(card.rank.value)) for card in legal]
        if off_aces:
            return [(card, 80.0 if card in off_aces else 0.0) for card in legal]
        if len(trump_cards) >= 3 and masters:
            return [
                (card, 100.0 - float(card.get_trump_value(trump)))
                if card in masters
                else (card, 0.0)
                for card in legal
            ]
        return self._side_lead_weights(legal, trump)

    def _side_lead_weights(
        self, legal: Sequence[Card], trump: Optional[Suit]
    ) -> List[Tuple[Card, float]]:
        """Low card from the longest side suit scores highest."""
        side = [card for card in legal if not trump or card.get_trump_value(trump) == 0]
        if not side:
            return [(card, 20.0 - float(card.get_trump_value(trump))) for card in legal]
        groups: Dict[Suit, List[Card]] = {}
        for card in side:
            groups.setdefault(card.suit, []).append(card)
        weights = []
        for card in legal:
            if card not in side:
                weights.append((card, 0.0))
                continue
            suit_cards = groups[card.suit]
            high_card = max(other.rank.value for other in suit_cards)
            suit_priority = len(suit_cards) * 100 + (20 - high_card)
            weights.append((card, float(suit_priority - card.rank.value)))
        return weights

    def _answer_weights(
        self,
        legal: Sequence[Card],
        plays: Sequence[Tuple[int, Card]],
        lead_suit: Optional[Suit],
        trump: Optional[Suit],
    ) -> List[Tuple[Card, float]]:
        """Weights for a trick already started. Mirrors :meth:`_answer_trick`."""
        winner_seat, winner = self._trick_winner(plays, trump, lead_suit)
        partner_winning = self._is_partner(winner_seat)
        urgency = self._urgency()
        beaters = [card for card in legal if card.beats(winner, trump, lead_suit)]
        duck = (partner_winning and urgency != "must") or not beaters
        if not duck:
            cheapest = min(beaters, key=lambda card: self._spend_key(card, trump))
            if self._should_save_bower(cheapest, legal, trump, urgency, plays):
                duck = True
        if duck:
            return self._loser_weights(legal, trump, winner, lead_suit)
        weights = []
        for card in legal:
            if card not in beaters:
                weights.append((card, 0.0))
                continue
            trump_value = card.get_trump_value(trump) if trump else 0
            if trump_value:
                weights.append((card, 70.0 - float(trump_value)))
            else:
                weights.append((card, 90.0 - float(card.rank.value)))
        return weights

    def _loser_weights(
        self,
        legal: Sequence[Card],
        trump: Optional[Suit],
        winner: Card,
        lead_suit: Optional[Suit],
    ) -> List[Tuple[Card, float]]:
        """Weights for ducking. A low side singleton outranks trump."""
        ducks = [card for card in legal if not card.beats(winner, trump, lead_suit)]
        pool = ducks or list(legal)
        weights = []
        for card in legal:
            if card not in pool:
                weights.append((card, 0.0))
                continue
            trump_value = card.get_trump_value(trump) if trump else 0
            if trump_value:
                weights.append((card, 10.0 - float(trump_value)))
                continue
            base = 40.0 if not self._side_mates(card, trump) else 20.0
            weights.append((card, base - float(card.rank.value)))
        return weights

    def _legal_cards(self, lead_suit: Optional[Suit]) -> List[Card]:
        """Cards this table accepts.

        The rules rank the left bower as trump. Following suit on this table
        uses the printed suit, which is also how the game builds the legal
        set, so a left bower follows its printed suit when that suit is led.
        """
        if lead_suit is None:
            return list(self.hand)
        followers = [card for card in self.hand if card.suit == lead_suit]
        if followers:
            return followers
        return list(self.hand)

    def _plays_before_us(self) -> List[Tuple[int, Card]]:
        if self.seat is None:
            return list(self.trick_plays)
        return [(seat, card) for seat, card in self.trick_plays if seat != self.seat]

    def _lead_card(self, legal: Sequence[Card], trump: Optional[Suit]) -> Card:
        if not trump:
            return max(legal, key=lambda card: card.rank.value)
        trump_cards = [card for card in legal if card.get_trump_value(trump) > 0]
        masters = [card for card in trump_cards if self._is_boss_trump(card, trump)]
        off_aces = [
            card for card in legal if card.get_trump_value(trump) == 0 and card.rank == Rank.ACE
        ]
        if self._we_are_makers():
            if masters:
                return min(masters, key=lambda card: card.get_trump_value(trump))
            if off_aces:
                return off_aces[0]
            if trump_cards:
                return min(trump_cards, key=lambda card: card.get_trump_value(trump))
            return max(legal, key=lambda card: card.rank.value)
        if off_aces:
            return off_aces[0]
        if len(trump_cards) >= 3 and masters:
            return min(masters, key=lambda card: card.get_trump_value(trump))
        return self._low_from_longest_side_suit(legal, trump)

    def _low_from_longest_side_suit(self, legal: Sequence[Card], trump: Optional[Suit]) -> Card:
        side = [card for card in legal if not trump or card.get_trump_value(trump) == 0]
        if not side:
            return min(legal, key=lambda card: card.get_trump_value(trump))
        groups = {}
        for card in side:
            groups.setdefault(card.suit, []).append(card)

        def suit_key(suit: Suit) -> Tuple[int, int]:
            cards = groups[suit]
            return (len(cards), -max(card.rank.value for card in cards))

        suit = max(groups, key=suit_key)
        return min(groups[suit], key=lambda card: card.rank.value)

    def _answer_trick(
        self,
        legal: Sequence[Card],
        plays: Sequence[Tuple[int, Card]],
        lead_suit: Optional[Suit],
        trump: Optional[Suit],
    ) -> Card:
        winner_seat, winner = self._trick_winner(plays, trump, lead_suit)
        partner_winning = self._is_partner(winner_seat)
        urgency = self._urgency()
        beaters = [card for card in legal if card.beats(winner, trump, lead_suit)]

        if partner_winning and urgency != "must":
            return self._cheapest_loser(legal, trump, winner, lead_suit)
        if not beaters:
            return self._cheapest_loser(legal, trump, winner, lead_suit)

        cheapest = min(beaters, key=lambda card: self._spend_key(card, trump))
        if self._should_save_bower(cheapest, legal, trump, urgency, plays):
            return self._cheapest_loser(legal, trump, winner, lead_suit)
        return cheapest

    def _should_save_bower(
        self,
        card: Card,
        legal: Sequence[Card],
        trump: Optional[Suit],
        urgency: str,
        plays: Sequence[Tuple[int, Card]],
    ) -> bool:
        """Once the contract is safe, keep a bower if an opponent still has to play."""
        if urgency != "safe" or not trump or not self._opponents_remain(plays):
            return False
        if card.get_trump_value(trump) < 6:
            return False
        return any(other.get_trump_value(trump) == 0 for other in legal)

    def _cheapest_loser(
        self,
        legal: Sequence[Card],
        trump: Optional[Suit],
        winner: Card,
        lead_suit: Optional[Suit],
    ) -> Card:
        ducks = [card for card in legal if not card.beats(winner, trump, lead_suit)]
        pool = ducks or list(legal)
        return min(pool, key=self._loser_key(trump))

    def _loser_key(self, trump: Optional[Suit]):
        def key(card: Card) -> Tuple[int, int, int]:
            trump_value = card.get_trump_value(trump) if trump else 0
            if trump_value:
                return (2, trump_value, 0)
            mates = [
                other
                for other in self.hand
                if other is not card
                and other.suit == card.suit
                and (not trump or other.get_trump_value(trump) == 0)
            ]
            singleton = 0 if not mates else 1
            return (0, singleton, card.rank.value)

        return key

    def _spend_key(self, card: Card, trump: Optional[Suit]) -> Tuple[int, int]:
        trump_value = card.get_trump_value(trump) if trump else 0
        if trump_value:
            return (1, trump_value)
        return (0, card.rank.value)

    def _trick_winner(
        self,
        plays: Sequence[Tuple[int, Card]],
        trump: Optional[Suit],
        lead_suit: Optional[Suit],
    ) -> Tuple[int, Card]:
        seat, card = plays[0]
        for other_seat, other in plays[1:]:
            if other.beats(card, trump, lead_suit):
                seat, card = other_seat, other
        return seat, card

    def _is_partner(self, seat: int) -> bool:
        if self.seat is None:
            return False
        return (seat - self.seat) % 4 == 2

    def _opponents_remain(self, plays: Sequence[Tuple[int, Card]]) -> bool:
        if self.seat is None:
            return len(plays) < 3
        played = {seat for seat, _ in plays}
        played.add(self.seat)
        for seat in range(4):
            if seat in played:
                continue
            if not self._is_partner(seat):
                return True
        return False

    def _urgency(self) -> str:
        """Whether this trick is required.

        Makers need three tricks to score. Defenders need three to euchre
        them. ``must`` means every remaining trick is required, ``safe``
        means the three are already in hand.
        """
        remaining = TRICKS_IN_ROUND - (self.team_tricks + self.opponent_tricks)
        if remaining < 1:
            remaining = 1
        if self.trump_caller_seat is None or self.team is None:
            return "want"
        need = 3 - self.team_tricks
        if need <= 0:
            return "safe"
        if remaining <= need:
            return "must"
        return "want"

    def _we_are_makers(self) -> bool:
        if self.trump_caller_seat is None or self.team is None:
            return False
        return self.trump_caller_seat % 2 == self.team

    def _is_boss_trump(self, card: Card, trump: Suit) -> bool:
        value = card.get_trump_value(trump)
        if value <= 0:
            return False
        for higher in range(value + 1, 8):
            if self._trump_rank_still_out(higher, trump):
                return False
        return True

    def _trump_rank_still_out(self, value: int, trump: Suit) -> bool:
        for card in self.hand:
            if card.get_trump_value(trump) == value:
                return False
        for _, card in self.played_cards:
            if card.get_trump_value(trump) == value:
                return False
        return True
