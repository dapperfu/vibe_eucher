"""Heuristic player that asks a micro model only when a decision is close.

Ordering up, naming a suit, leading, following, and discarding stay with the
rules-and-payoff heuristic when one option is clearly ahead. A gray spot,
such as three hearts with hearts turned up, or a lead of either bower, is
passed to :class:`EucherHeuristicTiebreak`. With no checkpoint, a gray spot
falls back to the heuristic too.
"""

from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

from ..models import Card, Suit
from .eucher_heuristic_tiebreak import default_checkpoint_path, load_tiebreak_predictor
from .gray_area import (
    ADVANTAGE_MARGIN,
    TRICK_BAND,
    WEIGHT_MARGIN,
    close_alternatives,
    features_for,
    features_for_card,
    is_gray_call,
    trump_count,
)
from .heuristic_player import CallEvaluation, HeuristicPlayer


class TiebreakModel(Protocol):
    """Something that scores one gray-area decision."""

    def call_probability(self, features: Sequence[float]) -> float:
        """Return the probability that calling is better than passing.

        Parameters
        ----------
        features : Sequence[float]
            One gray-area feature vector.

        Returns
        -------
        float
            A probability in ``[0, 1]``.
        """


class HeuristicTiebreakPlayer(HeuristicPlayer):
    """Heuristic bidder that uses a micro model as a tie-break."""

    def __init__(
        self,
        name: str,
        checkpoint_path: Optional[str] = None,
        model: Optional[TiebreakModel] = None,
        load_default: bool = True,
        trick_band: float = TRICK_BAND,
        advantage_margin: float = ADVANTAGE_MARGIN,
        weight_margin: float = WEIGHT_MARGIN,
        call_threshold: float = 0.5,
    ) -> None:
        """Build a heuristic player with an optional tie-break model.

        Parameters
        ----------
        name : str
            The player's name.
        checkpoint_path : str, optional
            Checkpoint to load. A missing file is an error.
        model : TiebreakModel, optional
            A predictor to use instead of loading a file.
        load_default : bool
            When no model or path is given, load the default checkpoint if
            that file exists.
        trick_band : float
            How close to three expected tricks still counts as gray.
        advantage_margin : float
            Win-probability edge that still counts as a tie.
        weight_margin : float
            How close two card weights must be before the model is asked.
        call_threshold : float
            Take the model's action when its probability is at least this value.
        """
        super().__init__(name)
        self.trick_band = trick_band
        self.advantage_margin = advantage_margin
        self.weight_margin = weight_margin
        self.call_threshold = call_threshold
        self.model_consulted = False
        self.last_band = "clear"
        self.last_model_probability: Optional[float] = None
        if model is not None:
            self.model: Optional[TiebreakModel] = model
        elif checkpoint_path:
            self.model = load_tiebreak_predictor(Path(checkpoint_path))
        elif load_default and default_checkpoint_path().is_file():
            try:
                self.model = load_tiebreak_predictor(default_checkpoint_path())
            except ValueError:
                # A leftover checkpoint from an older feature layout is not usable.
                self.model = None
        else:
            self.model = None

    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Order the upcard up, asking the model only when the call is gray.

        Parameters
        ----------
        top_card : Card
            The card turned up.
        is_partner_dealing : bool
            Used only when this player has not been told the seats.

        Returns
        -------
        bool
            True when this player orders the suit up.
        """
        self._reset_decision()
        if top_card is None or not self.hand:
            self.last_band = "clear"
            return super().should_order_up(top_card, is_partner_dealing)
        location = self._upcard_location(is_partner_dealing)
        evaluation = self.evaluate_call(top_card.suit, top_card, 1, location)
        if not self._is_gray(evaluation):
            self.last_band = "clear"
            self.last_reasoning = evaluation.reasoning
            return evaluation.should_call
        self.last_band = "gray"
        return self._break_order_tie(evaluation, top_card, location)

    def choose_trump_suit(self, top_card: Card, must_call: bool = False) -> Optional[Suit]:
        """Name a suit in the second round, or pass.

        A clear call is taken from the heuristic. Gray suits are given to the
        micro model. The dealer still has to name a suit.

        Parameters
        ----------
        top_card : Card
            The card that was turned down.
        must_call : bool
            Force a suit even when every evaluation says to pass.

        Returns
        -------
        Optional[Suit]
            The named suit, or None when this player passes.
        """
        self._reset_decision()
        if top_card is None:
            self.last_band = "clear"
            self.last_reasoning = "No upcard."
            return None
        forced = must_call or self._dealer_must_name_suit()
        evaluations = self._second_round_evaluations(top_card)
        if not evaluations:
            self.last_band = "clear"
            self.last_reasoning = "No legal suit to name."
            return None
        gray = [evaluation for evaluation in evaluations if self._is_gray(evaluation)]
        gray_ids = {id(evaluation) for evaluation in gray}
        clear_calls = [
            evaluation
            for evaluation in evaluations
            if id(evaluation) not in gray_ids and evaluation.should_call
        ]
        if clear_calls:
            best = self._best_evaluation(clear_calls, top_card.suit)
            self.last_band = "clear"
            self.last_reasoning = best.reasoning
            return best.suit
        if not gray:
            return self._heuristic_suit(evaluations, top_card.suit, forced)
        self.last_band = "gray"
        if self.model is None:
            return self._heuristic_suit(gray, top_card.suit, forced, gray_fallback=True)
        return self._model_choose(gray, top_card, forced)

    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Play a card, asking the model when the top weights are close.

        Parameters
        ----------
        lead_suit : Suit, optional
            The suit led to this trick.
        trump_suit : Suit, optional
            Trump for this hand.

        Returns
        -------
        Card
            The card to play. It stays in the hand until the table removes it.
        """
        self._reset_decision()
        heuristic = super().choose_card_to_play(lead_suit, trump_suit)
        scored = self.play_weights(lead_suit, trump_suit)
        alternatives = close_alternatives(scored, heuristic, self.weight_margin)
        if not alternatives:
            self.last_band = "clear"
            self.last_reasoning = f"Play {heuristic}."
            return heuristic
        self.last_band = "gray"
        if self.model is None:
            self.last_reasoning = f"Gray area, heuristic decides. Play {heuristic}."
            return heuristic
        context = self._play_context(lead_suit, trump_suit)
        return self._break_card_tie(heuristic, alternatives, scored, trump_suit, "play", context)

    def choose_card_to_discard(self, trump_suit: Suit) -> Card:
        """Discard, asking the model when two throws have nearly the same weight.

        Parameters
        ----------
        trump_suit : Suit
            The trump suit the dealer just accepted.

        Returns
        -------
        Card
            The card to throw.
        """
        self._reset_decision()
        heuristic = super().choose_card_to_discard(trump_suit)
        scored = self.discard_weights(trump_suit)
        alternatives = close_alternatives(scored, heuristic, self.weight_margin)
        if not alternatives:
            self.last_band = "clear"
            self.last_reasoning = f"Discard {heuristic}."
            return heuristic
        self.last_band = "gray"
        if self.model is None:
            self.last_reasoning = f"Gray area, heuristic decides. Discard {heuristic}."
            return heuristic
        return self._break_card_tie(
            heuristic,
            alternatives,
            scored,
            trump_suit,
            "discard",
            {"lead_suit": None, "partner_winning": False, "urgency": "want"},
        )

    def _reset_decision(self) -> None:
        """Clear the record of the decision about to be made."""
        self.model_consulted = False
        self.last_model_probability = None

    def _is_gray(self, evaluation: CallEvaluation) -> bool:
        """Return True when this evaluation is a tie-break spot.

        Parameters
        ----------
        evaluation : CallEvaluation
            One suit the heuristic has already priced.

        Returns
        -------
        bool
            True when the micro model should be asked.
        """
        count = trump_count(self.hand, evaluation.suit)
        return is_gray_call(
            count,
            evaluation.expected_tricks,
            evaluation.advantage,
            trick_band=self.trick_band,
            advantage_margin=self.advantage_margin,
        )

    def _second_round_evaluations(self, top_card: Card) -> List[CallEvaluation]:
        """Price every suit other than the one that was turned down.

        Parameters
        ----------
        top_card : Card
            The card that was turned down.

        Returns
        -------
        List[CallEvaluation]
            One evaluation per legal suit.
        """
        evaluations = []
        for suit in Suit:
            if suit == top_card.suit:
                continue
            evaluations.append(self.evaluate_call(suit, top_card, 2, "buried"))
        return evaluations

    def _best_evaluation(
        self, evaluations: Sequence[CallEvaluation], turned_down: Suit
    ) -> CallEvaluation:
        """Return the evaluation the heuristic prefers.

        Parameters
        ----------
        evaluations : Sequence[CallEvaluation]
            Candidates. Must be non-empty.
        turned_down : Suit
            The suit that cannot be named, used to break a probability tie.

        Returns
        -------
        CallEvaluation
            The preferred evaluation.
        """
        best = evaluations[0]
        for candidate in evaluations[1:]:
            if self._prefer(candidate, best, turned_down):
                best = candidate
        return best

    def _heuristic_suit(
        self,
        evaluations: Sequence[CallEvaluation],
        turned_down: Suit,
        forced: bool,
        gray_fallback: bool = False,
    ) -> Optional[Suit]:
        """Follow the heuristic for these evaluations.

        Parameters
        ----------
        evaluations : Sequence[CallEvaluation]
            Suits still in consideration.
        turned_down : Suit
            The suit that cannot be named.
        forced : bool
            Return the best suit even when the heuristic would pass.
        gray_fallback : bool
            Note that the model was unavailable for a gray decision.

        Returns
        -------
        Optional[Suit]
            The named suit, or None when passing.
        """
        best = self._best_evaluation(evaluations, turned_down)
        prefix = "Gray area, heuristic decides. " if gray_fallback else ""
        if forced or best.should_call:
            self.last_reasoning = f"{prefix}{best.reasoning}"
            return best.suit
        self.last_reasoning = f"{prefix}{best.reasoning}"
        return None

    def _break_order_tie(self, evaluation: CallEvaluation, top_card: Card, location: str) -> bool:
        """Ask the model whether to order up a gray suit.

        Parameters
        ----------
        evaluation : CallEvaluation
            The heuristic's price for ordering up.
        top_card : Card
            The upcard.
        location : str
            Who would receive the upcard.

        Returns
        -------
        bool
            True when the suit is ordered up.
        """
        if self.model is None:
            self.last_reasoning = f"Gray area, heuristic decides. {evaluation.reasoning}"
            return evaluation.should_call
        features = features_for(self, evaluation, top_card, 1, location)
        probability = float(self.model.call_probability(features))
        self.model_consulted = True
        self.last_model_probability = probability
        call = probability >= self.call_threshold
        decision = "Call" if call else "Pass"
        self.last_reasoning = (
            f"Tie-break {decision} {top_card.suit.name}: "
            f"micro model {probability:.3f}. {evaluation.reasoning}"
        )
        return call

    def _model_choose(
        self, gray: Sequence[CallEvaluation], top_card: Card, forced: bool
    ) -> Optional[Suit]:
        """Pick a gray suit with the micro model.

        Parameters
        ----------
        gray : Sequence[CallEvaluation]
            Suits the heuristic will not decide alone.
        top_card : Card
            The turned-down card.
        forced : bool
            Pick the best gray suit even below the call threshold.

        Returns
        -------
        Optional[Suit]
            The named suit, or None when the model passes.
        """
        model = self.model
        if model is None:
            return self._heuristic_suit(gray, top_card.suit, forced, gray_fallback=True)
        location = self._upcard_location(False)
        chosen: Optional[CallEvaluation] = None
        chosen_probability = -1.0
        for evaluation in gray:
            features = features_for(self, evaluation, top_card, 2, location)
            probability = float(model.call_probability(features))
            self.model_consulted = True
            if not forced and probability < self.call_threshold:
                continue
            if (
                chosen is None
                or probability > chosen_probability + 1e-12
                or (
                    abs(probability - chosen_probability) <= 1e-12
                    and self._prefer(evaluation, chosen, top_card.suit)
                )
            ):
                chosen = evaluation
                chosen_probability = probability
        if chosen is None:
            best = self._best_evaluation(gray, top_card.suit)
            self.last_reasoning = (
                f"Tie-break pass. Micro model stayed below {self.call_threshold:.2f}. "
                f"{best.reasoning}"
            )
            return None
        self.last_model_probability = chosen_probability
        self.last_reasoning = (
            f"Tie-break Call {chosen.suit.name}: "
            f"micro model {chosen_probability:.3f}. {chosen.reasoning}"
        )
        return chosen.suit

    def _play_context(
        self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]
    ) -> dict:
        """Lead, partner, and urgency for the card about to be played.

        Parameters
        ----------
        lead_suit : Suit, optional
            The suit the caller thinks was led.
        trump_suit : Suit, optional
            Trump. Unused except to keep the signature next to the play call.

        Returns
        -------
        dict
            ``lead_suit``, ``partner_winning``, and ``urgency``.
        """
        plays = self._plays_before_us()
        if not plays:
            return {"lead_suit": None, "partner_winning": False, "urgency": self._urgency()}
        actual_lead = plays[0][1].suit
        winner_seat, _winner = self._trick_winner(plays, trump_suit, actual_lead)
        return {
            "lead_suit": actual_lead,
            "partner_winning": self._is_partner(winner_seat),
            "urgency": self._urgency(),
        }

    def _break_card_tie(
        self,
        heuristic: Card,
        alternatives: Sequence[Tuple[Card, float]],
        scored: Sequence[Tuple[Card, float]],
        trump: Optional[Suit],
        kind: str,
        context: dict,
    ) -> Card:
        """Pick a close card when the model prefers it to the heuristic.

        Parameters
        ----------
        heuristic : Card
            The card the rules would play or discard.
        alternatives : Sequence[Tuple[Card, float]]
            Other cards inside the weight margin.
        scored : Sequence[Tuple[Card, float]]
            Every legal weight, used for the gap and the close count.
        trump : Suit, optional
            Trump for this hand.
        kind : str
            ``play`` or ``discard``.
        context : dict
            Lead, partner, and urgency fields.

        Returns
        -------
        Card
            The model's card, or the heuristic card when every alternative
            stays under the threshold.
        """
        model = self.model
        if model is None:
            return heuristic
        heuristic_weight = _weight_of(scored, heuristic)
        close_count = 1 + len(alternatives)
        chosen = heuristic
        chosen_probability = -1.0
        for card, weight in alternatives:
            features = features_for_card(
                self,
                card,
                trump,
                weight,
                heuristic_weight - weight,
                close_count,
                kind,
                lead_suit=context["lead_suit"],
                partner_winning=bool(context["partner_winning"]),
                urgency=str(context["urgency"]),
            )
            probability = float(model.call_probability(features))
            self.model_consulted = True
            if probability < self.call_threshold:
                continue
            if probability > chosen_probability + 1e-12:
                chosen = card
                chosen_probability = probability
        if chosen_probability < 0.0:
            self.last_reasoning = (
                f"Tie-break keep {heuristic}. "
                f"Micro model stayed below {self.call_threshold:.2f}."
            )
            return heuristic
        self.last_model_probability = chosen_probability
        verb = "Play" if kind == "play" else "Discard"
        self.last_reasoning = f"Tie-break {verb} {chosen}: micro model {chosen_probability:.3f}."
        return chosen


def _weight_of(scored: Sequence[Tuple[Card, float]], chosen: Card) -> float:
    """Return the weight recorded for ``chosen``."""
    for card, weight in scored:
        if card.rank == chosen.rank and card.suit == chosen.suit:
            return weight
    return 0.0
