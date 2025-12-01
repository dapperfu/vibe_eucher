"""Heuristic-based player with tunable risk factor."""

from typing import TYPE_CHECKING, List, Optional

from eucher.cards import Card, Rank, Suit
from eucher.players.computer.base import ComputerPlayer

if TYPE_CHECKING:
    from eucher.players.base import Player


class HeuristicPlayer2(ComputerPlayer):
    """
    Heuristic-based player with tunable risk factor.

    This player uses rule-based logic to make decisions, with a risk_factor
    parameter that controls how aggressively the player plays. Higher risk
    factor (closer to 1.0) means more aggressive play (lower thresholds for
    trump selection, more likely to go alone, more aggressive card play).
    Lower risk factor (closer to 0.0) means more conservative play.

    Parameters
    ----------
    risk_factor : float
        Risk factor controlling aggressiveness (0.0-1.0). Default is 0.5.
        0.0 = very conservative, 1.0 = very aggressive.
    """

    def __init__(self, risk_factor: float = 0.5) -> None:
        """
        Initialize a heuristic-based player with risk factor.

        Parameters
        ----------
        risk_factor : float
            Risk factor controlling aggressiveness (0.0-1.0). Default is 0.5.
        """
        super().__init__()
        if not 0.0 <= risk_factor <= 1.0:
            raise ValueError("risk_factor must be between 0.0 and 1.0")
        self.risk_factor: float = risk_factor

        # Base thresholds (will be adjusted by risk_factor)
        self.ORDER_UP_BASE_THRESHOLD: float = 150.0
        self.CALL_TRUMP_BASE_THRESHOLD: float = 140.0
        self.CALL_TRUMP_MUST_CHOOSE_THRESHOLD: float = 80.0
        self.GOING_ALONE_BASE_THRESHOLD: float = 250.0

        # Risk adjustment factors
        self.THRESHOLD_REDUCTION_MAX: float = 0.20  # Max 20% reduction at risk=1.0 (reduced from 0.30)
        self.GOING_ALONE_BONUS_MAX: float = 0.40  # Max 40% bonus at risk=1.0

        # Leading strategy weights (context-aware multipliers)
        self.LEAD_TRUMP_PREFERENCE: float = 1.2  # 20% bonus for trump cards when leading
        self.LEAD_OFFSUIT_ACE: float = 1.1  # 10% bonus for off-suit Aces when leading
        self.LEAD_OFFSUIT_KING: float = 1.05  # 5% bonus for off-suit Kings when leading

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Decide whether to order up the turned card.

        Higher risk factor lowers the threshold for ordering up, making the
        player more likely to order up with weaker hands.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up.
        dealer_id : int
            The ID of the dealer.
        trump_suit : Optional[Suit]
            Current trump suit if already determined.

        Returns
        -------
        bool
            True to order up, False to pass.
        """
        if trump_suit is not None:
            return False

        potential_trump = turned_card.suit

        # Evaluate hand strength for this potential trump
        hand_strength = self._evaluate_hand_strength(player.hand, potential_trump)
        trump_count = self._count_trump_cards(player.hand, potential_trump)

        # Base threshold
        threshold = self.ORDER_UP_BASE_THRESHOLD

        # Adjust threshold based on risk factor (higher risk = lower threshold)
        threshold_reduction = self.risk_factor * self.THRESHOLD_REDUCTION_MAX * threshold
        threshold = threshold - threshold_reduction

        # Adjust threshold based on position (being dealer's partner is better)
        is_dealer_partner = (player.player_id % 2) == (dealer_id % 2)
        if is_dealer_partner:
            threshold -= 20.0  # Lower threshold when dealer's partner

        # Order up if hand strength exceeds adjusted threshold
        if hand_strength >= threshold:
            return True

        # Order up if we have 2+ trump cards (conservative threshold)
        # With higher risk, might order up with just 1 trump
        min_trump_needed = 2 if self.risk_factor < 0.7 else 1
        if trump_count >= min_trump_needed:
            return True

        # Order up if we have a bower (very strong)
        if self._has_bower(player.hand, potential_trump):
            return True

        # Order up if we have strong trump (Ace or King) and at least one other trump
        # With higher risk, might order up with just strong trump
        if self._has_strong_trump(player.hand, potential_trump):
            if trump_count >= 1 or self.risk_factor > 0.6:
                return True

        return False

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Decide which suit to call as trump.

        Higher risk factor lowers the threshold for calling trump, making the
        player more likely to call with weaker hands.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up (cannot be chosen).
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, player must choose a suit (cannot pass).

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass (only if must_choose=False).
        """
        if trump_suit is not None:
            if must_choose:
                return trump_suit
            return None

        forbidden_suit = turned_card.suit
        suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]

        best_suit = None
        best_score = 0.0

        # Evaluate each potential trump suit
        for suit in suits:
            if suit == forbidden_suit:
                continue

            # Calculate hand strength for this suit
            hand_strength = self._evaluate_hand_strength(player.hand, suit)
            trump_count = self._count_trump_cards(player.hand, suit)

            # Score based on hand strength and trump count
            score = hand_strength

            # Bonus for having bowers
            if self._has_bower(player.hand, suit):
                score += 30.0

            # Bonus for multiple trump cards
            if trump_count >= 3:
                score += 20.0
            elif trump_count >= 2:
                score += 10.0

            if score > best_score:
                best_score = score
                best_suit = suit

        # Threshold for calling trump
        threshold = self.CALL_TRUMP_BASE_THRESHOLD

        # Adjust threshold based on risk factor
        threshold_reduction = self.risk_factor * self.THRESHOLD_REDUCTION_MAX * threshold
        threshold = threshold - threshold_reduction

        # If must choose, lower threshold significantly
        if must_choose:
            threshold = self.CALL_TRUMP_MUST_CHOOSE_THRESHOLD
            # Further reduce if high risk
            threshold = threshold - (self.risk_factor * 20.0)

        # Call trump if score exceeds threshold
        if best_suit is not None and best_score >= threshold:
            return best_suit

        # If must choose, return best suit even if weak
        if must_choose and best_suit is not None:
            return best_suit

        return None

    def choose_card_to_discard(
        self,
        player: "Player",
        turned_card: Optional[Card] = None,
        ordered_up_by: Optional[str] = None,
    ) -> Card:
        """
        Choose a card to discard (dealer only, after ordering up).

        Higher risk factor may lead to discarding slightly better cards
        if it improves the hand composition.

        Parameters
        ----------
        player : Player
            The dealer player.
        turned_card : Optional[Card]
            The card that was ordered up, if available.
        ordered_up_by : Optional[str]
            Name of the player who ordered up, if available.

        Returns
        -------
        Card
            The card to discard.
        """
        if not player.hand:
            raise ValueError("Player has no cards to discard")

        # If we know the trump suit (turned_card was ordered up), use it
        trump_suit: Optional[Suit] = None
        if turned_card is not None:
            trump_suit = turned_card.suit

        # Find the card with lowest power value
        # With higher risk, might be slightly more selective about what to discard
        return min(
            player.hand,
            key=lambda c: self._calculate_card_power(c, trump_suit, context="general"),
        )

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Choose a card to play in a trick.

        Higher risk factor makes the player more aggressive:
        - More likely to try to win tricks even when teammate is winning
        - More likely to lead with strong cards
        - Less likely to duck when teammate is winning

        Parameters
        ----------
        player : Player
            The player making the decision.
        led_suit : Optional[Suit]
            The suit that was led, if any.
        trump_suit : Optional[Suit]
            The current trump suit, if any.
        trick_cards : List[Card]
            Cards already played in the trick.
        trick_player_ids : List[int]
            Player IDs who played each card in trick_cards (same order).

        Returns
        -------
        Card
            The card to play.
        """
        valid_cards = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)
        if not valid_cards:
            return player.hand[0]

        # Scenario detection and routing
        if led_suit is None:
            # Leading: no cards played yet
            return self._decide_lead(player, valid_cards, trump_suit)
        elif len(trick_cards) == 3:
            # Last to play: 3 cards already played
            return self._decide_last_play(
                player, valid_cards, trick_cards, trick_player_ids, led_suit, trump_suit
            )
        else:
            # Following: 1-2 cards already played
            return self._decide_follow(
                player, valid_cards, trick_cards, trick_player_ids, led_suit, trump_suit
            )

    def decide_going_alone(self, player: "Player", trump_suit: Suit) -> bool:
        """
        Decide whether to go alone after making trump.

        Higher risk factor increases the likelihood of going alone by
        lowering the threshold for this decision.

        Parameters
        ----------
        player : Player
            The player making the decision (must be the trump maker).
        trump_suit : Suit
            The trump suit that was selected.

        Returns
        -------
        bool
            True to go alone, False to play with partner.
        """
        # Check for flush in trump (all 5 cards same suit as trump)
        if self._has_flush_in_trump(player.hand, trump_suit):
            return True

        # Evaluate hand strength
        hand_strength = self._evaluate_hand_strength(player.hand, trump_suit)
        trump_count = self._count_trump_cards(player.hand, trump_suit)

        # Base threshold for going alone
        threshold = self.GOING_ALONE_BASE_THRESHOLD

        # Adjust threshold based on risk factor (higher risk = lower threshold)
        threshold_reduction = self.risk_factor * self.GOING_ALONE_BONUS_MAX * threshold
        threshold = threshold - threshold_reduction

        # Check for top 3 trump cards (Right Bower, Left Bower, Ace)
        has_top_trump = self._has_top_trump_cards(player.hand, trump_suit)

        # Check for off-suit Aces
        off_suit_aces = self._count_off_suit_aces(player.hand, trump_suit)

        # Go alone if:
        # 1. Flush in trump (already checked above)
        # 2. Very high hand strength AND has top trump cards AND off-suit Aces
        if hand_strength >= threshold and has_top_trump and off_suit_aces >= 1:
            return True

        # Go alone if we have 4+ trump cards (very strong)
        # With higher risk, might go alone with 3 trump
        min_trump_for_alone = 4 if self.risk_factor < 0.7 else 3
        if trump_count >= min_trump_for_alone:
            return True

        # With very high risk, might go alone with strong hand even without all top trump
        if self.risk_factor > 0.8 and hand_strength >= threshold * 0.9 and trump_count >= 3:
            return True

        return False

    def decide_trade_in(self, player: "Player", eligible_cards: List[Card]) -> bool:
        """
        Decide whether to trade-in eligible cards for kitty cards.

        Higher risk factor makes the player more likely to trade in,
        as it's a riskier move that could improve the hand.

        Parameters
        ----------
        player : Player
            The player making the decision.
        eligible_cards : List[Card]
            The three cards that are eligible for trade-in (same suit, all 9s or 10s).

        Returns
        -------
        bool
            True to trade-in, False to pass.
        """
        # Calculate current hand strength (without knowing trump yet)
        # Use average rank value as a proxy
        current_hand_strength = sum(card.rank.value for card in player.hand)

        # Count high cards (Ace, King, Queen) in current hand
        high_card_count = sum(
            1 for card in player.hand if card.rank in (Rank.ACE, Rank.KING, Rank.QUEEN)
        )

        # Count Jacks (potential bowers)
        jack_count = sum(1 for card in player.hand if card.rank == Rank.JACK)

        # Trade-in if:
        # 1. Hand has very few high cards (2 or fewer, or 3 or fewer with high risk)
        high_card_threshold = 2 if self.risk_factor < 0.6 else 3
        if high_card_count <= high_card_threshold and jack_count == 0:
            return True

        # Trade-in if hand strength is very low (mostly 9s and 10s)
        # With higher risk, might trade in with slightly better hands
        strength_threshold = 50.0 - (self.risk_factor * 10.0)
        if current_hand_strength <= strength_threshold:
            return True

        # With very high risk, might trade in even with moderate hands
        if self.risk_factor > 0.8 and high_card_count <= 3:
            return True

        # Otherwise, keep the hand
        return False

    def _evaluate_hand_strength(self, hand: List[Card], trump_suit: Suit) -> float:
        """
        Evaluate the overall strength of a hand for a given trump suit.

        Parameters
        ----------
        hand : List[Card]
            The hand to evaluate.
        trump_suit : Suit
            The trump suit to evaluate against.

        Returns
        -------
        float
            Hand strength score (higher is better).
        """
        total_power = 0.0
        trump_count = 0
        has_bower = False

        for card in hand:
            power = self._calculate_card_power(card, trump_suit, context="general")
            total_power += power

            if card.rank == Rank.JACK:
                if card.suit == trump_suit:
                    has_bower = True
                    trump_count += 1
                else:
                    trump_card = Card(trump_suit, Rank.ACE)
                    if card.is_same_color(trump_card):
                        has_bower = True
                        trump_count += 1
            elif card.suit == trump_suit:
                trump_count += 1

        # Bonus for having bowers
        if has_bower:
            total_power += 30.0

        # Bonus for multiple trump cards
        if trump_count >= 3:
            total_power += 20.0
        elif trump_count >= 2:
            total_power += 10.0

        return total_power

    def _calculate_card_power(
        self, card: Card, trump_suit: Optional[Suit], context: str = "general"
    ) -> float:
        """
        Calculate the power/weight of a card.

        Parameters
        ----------
        card : Card
            The card to evaluate.
        trump_suit : Optional[Suit]
            The current trump suit, if any.
        context : str
            Context of evaluation: "general", "leading", "following".
            When "leading", applies multipliers for trump and strong off-suit cards.

        Returns
        -------
        float
            Card power value (higher is better).
        """
        if trump_suit is None:
            # No trump: use standard rank values
            return float(card.rank.value)

        # Check if card is trump
        is_right_bower = card.rank == Rank.JACK and card.suit == trump_suit
        is_left_bower = False
        if card.rank == Rank.JACK and not is_right_bower:
            trump_card = Card(trump_suit, Rank.ACE)
            is_left_bower = card.is_same_color(trump_card)

        if is_right_bower:
            power = 100.0
        elif is_left_bower:
            power = 90.0
        elif card.suit == trump_suit:
            # Regular trump suit card
            rank_power_map = {
                Rank.ACE: 80.0,
                Rank.KING: 70.0,
                Rank.QUEEN: 60.0,
                Rank.TEN: 50.0,
                Rank.NINE: 40.0,
            }
            power = rank_power_map.get(card.rank, 0.0)
        else:
            # Off-suit card
            rank_power_map = {
                Rank.ACE: 35.0,
                Rank.KING: 30.0,
                Rank.QUEEN: 25.0,
                Rank.JACK: 20.0,
                Rank.TEN: 15.0,
                Rank.NINE: 10.0,
            }
            power = rank_power_map.get(card.rank, 0.0)

        # Apply context-based adjustments for leading
        if context == "leading":
            if card.suit == trump_suit or is_right_bower or is_left_bower:
                power *= self.LEAD_TRUMP_PREFERENCE
            elif card.rank == Rank.ACE:
                power *= self.LEAD_OFFSUIT_ACE
            elif card.rank == Rank.KING:
                power *= self.LEAD_OFFSUIT_KING

        return power

    def _is_teammate(self, player: "Player", other_player_id: int) -> bool:
        """
        Check if another player is a teammate.

        Parameters
        ----------
        player : Player
            The current player.
        other_player_id : int
            The other player's ID.

        Returns
        -------
        bool
            True if other player is a teammate, False otherwise.
        """
        # Teams are determined by player_id % 2
        return (player.player_id % 2) == (other_player_id % 2)

    def _find_current_winner_player_id(
        self,
        trick_cards: List[Card],
        trick_player_ids: List[int],
        led_suit: Suit,
        trump_suit: Optional[Suit],
    ) -> Optional[int]:
        """
        Find the player ID of the current winner of the trick.

        Parameters
        ----------
        trick_cards : List[Card]
            Cards played so far.
        trick_player_ids : List[int]
            Player IDs who played each card.
        led_suit : Suit
            The led suit.
        trump_suit : Optional[Suit]
            The trump suit.

        Returns
        -------
        Optional[int]
            Player ID of winning player, or None if no cards.
        """
        if not trick_cards:
            return None

        winner_idx = 0
        for i in range(1, len(trick_cards)):
            comparison = trick_cards[i].compare_to(trick_cards[winner_idx], trump_suit, led_suit)
            if comparison > 0:
                winner_idx = i
        return trick_player_ids[winner_idx]

    def _decide_lead(
        self, player: "Player", valid_cards: List[Card], trump_suit: Optional[Suit]
    ) -> Card:
        """
        Decide which card to play when leading a trick.

        Higher risk factor leads to playing stronger cards more aggressively.

        Parameters
        ----------
        player : Player
            The player making the decision.
        valid_cards : List[Card]
            Valid cards that can be played.
        trump_suit : Optional[Suit]
            The current trump suit, if any.

        Returns
        -------
        Card
            The card to lead with.
        """
        # Calculate power for each card in leading context (applies multipliers)
        card_powers = [
            (card, self._calculate_card_power(card, trump_suit, context="leading"))
            for card in valid_cards
        ]

        # Count trump cards in hand
        trump_count = 0
        if trump_suit is not None:
            trump_count = self._count_trump_cards(player.hand, trump_suit)

        # Strategy: If we have multiple strong trump, lead with one
        # With higher risk, more likely to lead trump even with fewer trump cards
        min_trump_for_lead = 2 if self.risk_factor < 0.6 else 1
        if trump_suit is not None and trump_count >= min_trump_for_lead:
            # Prefer leading with strong trump
            trump_cards = [
                (card, power)
                for card, power in card_powers
                if self._is_trump_card(card, trump_suit)
            ]
            if trump_cards:
                # Lead with highest power trump
                return max(trump_cards, key=lambda x: x[1])[0]

        # Lead with highest power card
        # With higher risk, prefer stronger cards
        return max(card_powers, key=lambda x: x[1])[0]

    def _decide_follow(
        self,
        player: "Player",
        valid_cards: List[Card],
        trick_cards: List[Card],
        trick_player_ids: List[int],
        led_suit: Suit,
        trump_suit: Optional[Suit],
    ) -> Card:
        """
        Decide which card to play when following (not leading, not last).

        Higher risk factor makes the player more likely to try to win even
        when teammate is winning.

        Parameters
        ----------
        player : Player
            The player making the decision.
        valid_cards : List[Card]
            Valid cards that can be played.
        trick_cards : List[Card]
            Cards already played in the trick.
        trick_player_ids : List[int]
            Player IDs who played each card.
        led_suit : Suit
            The suit that was led.
        trump_suit : Optional[Suit]
            The current trump suit, if any.

        Returns
        -------
        Card
            The card to play.
        """
        # Find current winner
        current_winner_id = self._find_current_winner_player_id(
            trick_cards, trick_player_ids, led_suit, trump_suit
        )

        if current_winner_id is None:
            # No winner yet (shouldn't happen, but fallback)
            return max(
                valid_cards,
                key=lambda c: self._calculate_card_power(c, trump_suit, context="general"),
            )

        # Check if teammate is winning
        is_teammate_winning = self._is_teammate(player, current_winner_id)

        if is_teammate_winning:
            # Teammate is winning: normally duck (play lowest card)
            # But with higher risk, might still try to win if we have a very strong card
            if self.risk_factor > 0.7:
                # Check if we have a card that can win by a lot
                winning_cards = []
                for card in valid_cards:
                    if self._can_win_trick(card, trick_cards, led_suit, trump_suit):
                        power = self._calculate_card_power(card, trump_suit, context="general")
                        # Only try to win if card is very strong
                        if power > 70.0:  # Strong trump or off-suit Ace
                            winning_cards.append((card, power))
                if winning_cards:
                    # Win with lowest power winning card that's still strong
                    return min(winning_cards, key=lambda x: x[1])[0]

            # Default: duck (play lowest card)
            return min(
                valid_cards,
                key=lambda c: self._calculate_card_power(c, trump_suit, context="general"),
            )
        else:
            # Opponent is winning: try to win with lowest winning card
            winning_cards = []
            for card in valid_cards:
                if self._can_win_trick(card, trick_cards, led_suit, trump_suit):
                    winning_cards.append(card)

            if winning_cards:
                # Win with lowest power winning card
                return min(
                    winning_cards,
                    key=lambda c: self._calculate_card_power(c, trump_suit, context="general"),
                )
            else:
                # Can't win: play lowest card to preserve high cards
                return min(
                    valid_cards,
                    key=lambda c: self._calculate_card_power(c, trump_suit, context="general"),
                )

    def _decide_last_play(
        self,
        player: "Player",
        valid_cards: List[Card],
        trick_cards: List[Card],
        trick_player_ids: List[int],
        led_suit: Suit,
        trump_suit: Optional[Suit],
    ) -> Card:
        """
        Decide which card to play when playing last (3 cards already played).

        Higher risk factor makes the player more likely to try to win even
        when teammate is winning.

        Parameters
        ----------
        player : Player
            The player making the decision.
        valid_cards : List[Card]
            Valid cards that can be played.
        trick_cards : List[Card]
            Cards already played in the trick.
        trick_player_ids : List[int]
            Player IDs who played each card.
        led_suit : Suit
            The suit that was led.
        trump_suit : Optional[Suit]
            The current trump suit, if any.

        Returns
        -------
        Card
            The card to play.
        """
        # Find current winner
        current_winner_id = self._find_current_winner_player_id(
            trick_cards, trick_player_ids, led_suit, trump_suit
        )

        if current_winner_id is None:
            # No winner yet (shouldn't happen, but fallback)
            return max(
                valid_cards,
                key=lambda c: self._calculate_card_power(c, trump_suit, context="general"),
            )

        # Check if teammate is winning
        is_teammate_winning = self._is_teammate(player, current_winner_id)

        if is_teammate_winning:
            # Teammate is winning: normally duck (play lowest card)
            # But with higher risk, might still try to win if we have a very strong card
            if self.risk_factor > 0.7:
                # Check if we have a card that can win by a lot
                winning_cards = []
                for card in valid_cards:
                    if self._can_win_trick(card, trick_cards, led_suit, trump_suit):
                        power = self._calculate_card_power(card, trump_suit, context="general")
                        # Only try to win if card is very strong
                        if power > 70.0:  # Strong trump or off-suit Ace
                            winning_cards.append((card, power))
                if winning_cards:
                    # Win with lowest power winning card that's still strong
                    return min(winning_cards, key=lambda x: x[1])[0]

            # Default: duck (play lowest card)
            return min(
                valid_cards,
                key=lambda c: self._calculate_card_power(c, trump_suit, context="general"),
            )
        else:
            # Opponent is winning: win if possible with minimal card
            winning_cards = []
            for card in valid_cards:
                if self._can_win_trick(card, trick_cards, led_suit, trump_suit):
                    winning_cards.append(card)

            if winning_cards:
                # Win with lowest power winning card
                return min(
                    winning_cards,
                    key=lambda c: self._calculate_card_power(c, trump_suit, context="general"),
                )
            else:
                # Can't win: play lowest card
                return min(
                    valid_cards,
                    key=lambda c: self._calculate_card_power(c, trump_suit, context="general"),
                )

    def _count_off_suit_aces(self, hand: List[Card], trump_suit: Optional[Suit]) -> int:
        """
        Count strong off-suit cards (Aces and Kings) that are not trump.

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Optional[Suit]
            The trump suit, if any.

        Returns
        -------
        int
            Number of strong off-suit cards.
        """
        count = 0
        for card in hand:
            # Skip trump cards
            if trump_suit is not None and self._is_trump_card(card, trump_suit):
                continue
            # Count Aces and Kings
            if card.rank in [Rank.ACE, Rank.KING]:
                count += 1
        return count

    def _has_flush_in_trump(self, hand: List[Card], trump_suit: Suit) -> bool:
        """
        Check if hand has a flush in trump (all 5 cards are trump).

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        bool
            True if all cards are trump, False otherwise.
        """
        for card in hand:
            if not self._is_trump_card(card, trump_suit):
                return False
        return True

    def _has_top_trump_cards(self, hand: List[Card], trump_suit: Suit) -> bool:
        """
        Check if hand has top 3 trump cards (Right Bower, Left Bower, Ace).

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        bool
            True if has top 3 trump cards, False otherwise.
        """
        has_right_bower = False
        has_left_bower = False
        has_trump_ace = False

        for card in hand:
            if self._is_trump_card(card, trump_suit):
                # Right Bower
                if card.rank == Rank.JACK and card.suit == trump_suit:
                    has_right_bower = True
                # Left Bower
                elif card.rank == Rank.JACK:
                    trump_card = Card(trump_suit, Rank.ACE)
                    if card.is_same_color(trump_card):
                        has_left_bower = True
                # Trump Ace
                elif card.suit == trump_suit and card.rank == Rank.ACE:
                    has_trump_ace = True

        return has_right_bower and has_left_bower and has_trump_ace





