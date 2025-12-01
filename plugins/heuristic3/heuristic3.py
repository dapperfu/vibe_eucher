"""Advanced heuristic-based player with card counting, opponent analysis, and adaptive strategies."""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from eucher.cards import Card, Rank, Suit
from eucher.players.computer.base import ComputerPlayer

if TYPE_CHECKING:
    from eucher.players.base import Player


class AdvancedHeuristicWeights:
    """Configuration class for advanced heuristic weights in HeuristicPlayer3 decision making."""

    # Trump selection thresholds (more nuanced than basic heuristics)
    ORDER_UP_BASE_THRESHOLD = 150.0
    ORDER_UP_DEALER_PARTNER_BONUS = -25.0
    ORDER_UP_DEALER_BONUS = -15.0
    ORDER_UP_FIRST_SEAT_BONUS = -10.0
    CALL_TRUMP_BASE_THRESHOLD = 140.0
    CALL_TRUMP_MUST_CHOOSE_THRESHOLD = 75.0
    CALL_TRUMP_SECOND_ROUND_BONUS = -15.0

    # Hand strength bonuses (more granular)
    RIGHT_BOWER_BONUS = 50.0
    LEFT_BOWER_BONUS = 45.0
    BOTH_BOWERS_BONUS = 120.0  # Extra bonus for having both
    TRUMP_ACE_BONUS = 35.0
    TRUMP_KING_BONUS = 25.0
    MULTIPLE_TRUMP_BONUS_4PLUS = 35.0
    MULTIPLE_TRUMP_BONUS_3PLUS = 25.0
    MULTIPLE_TRUMP_BONUS_2PLUS = 12.0
    STRONG_TRUMP_COMBO_BONUS = 20.0

    # Off-suit strength bonuses
    OFFSUIT_ACE_BONUS = 15.0
    OFFSUIT_KING_BONUS = 10.0
    MULTIPLE_OFFSUIT_ACES_BONUS = 25.0

    # Card power base values (more detailed)
    RIGHT_BOWER_POWER = 110
    LEFT_BOWER_POWER = 100
    TRUMP_ACE_POWER = 85
    TRUMP_KING_POWER = 75
    TRUMP_QUEEN_POWER = 65
    TRUMP_TEN_POWER = 55
    TRUMP_NINE_POWER = 45

    # Off-suit power values
    OFFSUIT_ACE_POWER = 40
    OFFSUIT_KING_POWER = 35
    OFFSUIT_QUEEN_POWER = 30
    OFFSUIT_JACK_POWER = 25
    OFFSUIT_TEN_POWER = 18
    OFFSUIT_NINE_POWER = 12

    # Leading strategy weights
    LEAD_TRUMP_PREFERENCE = 1.3
    LEAD_OFFSUIT_ACE = 1.15
    LEAD_OFFSUIT_KING = 1.08
    LEAD_DEFENSIVE_TRUMP = 1.1

    # Score-based adjustments
    BEHIND_BY_2_PLUS_AGGRESSION = 0.15
    BEHIND_BY_4_PLUS_AGGRESSION = 0.30
    AHEAD_BY_2_PLUS_CONSERVATIVE = -0.10
    AHEAD_BY_4_PLUS_CONSERVATIVE = -0.20
    ENDGAME_AGGRESSION = 0.25  # When close to 10 points

    # Going alone thresholds
    GOING_ALONE_BASE_THRESHOLD = 280.0
    GOING_ALONE_WITH_BOTH_BOWERS = 240.0
    GOING_ALONE_FLUSH_BONUS = 50.0

    # Defensive play weights
    PREVENT_MARCH_BONUS = 30.0
    BREAK_TRICK_BONUS = 20.0
    FORCE_WASTE_BONUS = 15.0


class CardTracker:
    """Tracks cards played and estimates remaining card distributions."""

    def __init__(self) -> None:
        """Initialize an empty card tracker."""
        self.cards_seen: Set[Card] = set()
        self.cards_by_suit: Dict[Suit, Set[Card]] = defaultdict(set)
        self.cards_by_rank: Dict[Rank, Set[Card]] = defaultdict(set)
        self.tricks_played: List[List[Card]] = []
        self.trump_suit: Optional[Suit] = None

    def reset(self) -> None:
        """Reset the tracker for a new hand."""
        self.cards_seen.clear()
        self.cards_by_suit.clear()
        self.cards_by_rank.clear()
        self.tricks_played.clear()
        self.trump_suit = None

    def record_card(self, card: Card) -> None:
        """Record a card that has been played."""
        self.cards_seen.add(card)
        self.cards_by_suit[card.suit].add(card)
        self.cards_by_rank[card.rank].add(card)

    def record_trick(self, trick_cards: List[Card]) -> None:
        """Record a complete trick."""
        for card in trick_cards:
            self.record_card(card)
        self.tricks_played.append(trick_cards.copy())

    def set_trump(self, trump_suit: Suit) -> None:
        """Set the trump suit for this hand."""
        self.trump_suit = trump_suit

    def get_remaining_cards(self, known_hand: List[Card]) -> List[Card]:
        """Get list of cards that haven't been seen yet."""
        all_cards = self._generate_all_cards()
        seen_and_held = self.cards_seen | set(known_hand)
        return [card for card in all_cards if card not in seen_and_held]

    def estimate_trump_remaining(self, known_hand: List[Card], trump_suit: Suit) -> int:
        """Estimate how many trump cards remain unplayed."""
        all_trump = self._get_all_trump_cards(trump_suit)
        seen_trump = {c for c in self.cards_seen if self._is_trump(c, trump_suit)}
        held_trump = {c for c in known_hand if self._is_trump(c, trump_suit)}
        remaining = len(all_trump) - len(seen_trump | held_trump)
        return max(0, remaining)

    def estimate_high_cards_remaining(
        self, known_hand: List[Card], suit: Optional[Suit] = None
    ) -> Dict[Rank, int]:
        """Estimate how many high cards remain in a suit."""
        high_ranks = [Rank.ACE, Rank.KING, Rank.QUEEN]
        estimates: Dict[Rank, int] = {}
        
        for rank in high_ranks:
            all_cards_of_rank = self._get_cards_by_rank(rank, suit)
            seen_of_rank = {c for c in self.cards_seen if c.rank == rank and (suit is None or c.suit == suit)}
            held_of_rank = {c for c in known_hand if c.rank == rank and (suit is None or c.suit == suit)}
            remaining = len(all_cards_of_rank) - len(seen_of_rank | held_of_rank)
            estimates[rank] = max(0, remaining)
        
        return estimates

    def _generate_all_cards(self) -> List[Card]:
        """Generate all possible cards in a Euchre deck."""
        cards: List[Card] = []
        for suit in Suit:
            for rank in [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]:
                cards.append(Card(suit, rank))
        return cards

    def _get_all_trump_cards(self, trump_suit: Suit) -> List[Card]:
        """Get all trump cards for a given trump suit."""
        trump_cards: List[Card] = []
        # Right bower
        trump_cards.append(Card(trump_suit, Rank.JACK))
        # Left bower (jack of same color)
        for suit in Suit:
            if suit != trump_suit:
                trump_card = Card(trump_suit, Rank.ACE)
                test_card = Card(suit, Rank.ACE)
                if test_card.is_same_color(trump_card):
                    trump_cards.append(Card(suit, Rank.JACK))
                    break
        # Other trump suit cards
        for rank in [Rank.NINE, Rank.TEN, Rank.QUEEN, Rank.KING, Rank.ACE]:
            trump_cards.append(Card(trump_suit, rank))
        return trump_cards

    def _is_trump(self, card: Card, trump_suit: Suit) -> bool:
        """Check if a card is trump."""
        if card.rank == Rank.JACK and card.suit == trump_suit:
            return True
        if card.rank == Rank.JACK:
            trump_card = Card(trump_suit, Rank.ACE)
            if card.is_same_color(trump_card):
                return True
        return card.suit == trump_suit

    def _get_cards_by_rank(self, rank: Rank, suit: Optional[Suit] = None) -> List[Card]:
        """Get all cards of a given rank (and optionally suit)."""
        cards: List[Card] = []
        suits_to_check = [suit] if suit else list(Suit)
        for s in suits_to_check:
            cards.append(Card(s, rank))
        return cards


class OpponentPatternAnalyzer:
    """Analyzes opponent patterns in bidding and play."""

    def __init__(self) -> None:
        """Initialize pattern analyzer."""
        self.bidding_patterns: Dict[int, List[bool]] = defaultdict(list)  # player_id -> list of bids
        self.aggression_scores: Dict[int, float] = defaultdict(float)
        self.leading_preferences: Dict[int, Dict[Suit, int]] = defaultdict(lambda: defaultdict(int))
        self.trump_play_patterns: Dict[int, List[int]] = defaultdict(list)  # trick numbers when they played trump
        self.going_alone_frequency: Dict[int, int] = defaultdict(int)

    def reset(self) -> None:
        """Reset patterns for a new hand."""
        # Keep historical data but reset current hand tracking
        pass

    def record_bid(self, player_id: int, bid_made: bool) -> None:
        """Record a bidding decision."""
        self.bidding_patterns[player_id].append(bid_made)
        # Update aggression score (1.0 = always bids, 0.0 = never bids)
        if len(self.bidding_patterns[player_id]) > 0:
            bids_made = sum(self.bidding_patterns[player_id])
            total_bids = len(self.bidding_patterns[player_id])
            self.aggression_scores[player_id] = bids_made / total_bids

    def record_lead(self, player_id: int, suit: Suit) -> None:
        """Record a suit that was led."""
        self.leading_preferences[player_id][suit] += 1

    def record_trump_play(self, player_id: int, trick_number: int) -> None:
        """Record when a player played trump."""
        self.trump_play_patterns[player_id].append(trick_number)

    def record_going_alone(self, player_id: int) -> None:
        """Record a going alone decision."""
        self.going_alone_frequency[player_id] += 1

    def is_aggressive_bidder(self, player_id: int) -> bool:
        """Check if player tends to bid aggressively."""
        return self.aggression_scores.get(player_id, 0.5) > 0.6

    def is_conservative_bidder(self, player_id: int) -> bool:
        """Check if player tends to bid conservatively."""
        return self.aggression_scores.get(player_id, 0.5) < 0.4

    def get_preferred_lead_suit(self, player_id: int) -> Optional[Suit]:
        """Get the suit a player most often leads."""
        preferences = self.leading_preferences.get(player_id, {})
        if not preferences:
            return None
        return max(preferences.items(), key=lambda x: x[1])[0]

    def tends_to_play_trump_early(self, player_id: int) -> bool:
        """Check if player tends to play trump early in tricks."""
        trump_plays = self.trump_play_patterns.get(player_id, [])
        if len(trump_plays) < 2:
            return False
        # If they play trump in first 2 tricks often, they're aggressive
        early_plays = sum(1 for t in trump_plays if t <= 2)
        return early_plays / len(trump_plays) > 0.5


class HeuristicPlayer3(ComputerPlayer):
    """
    Advanced heuristic-based player with card counting, opponent analysis, and adaptive strategies.

    This player implements sophisticated Euchre strategies including:
    - Card counting and probability estimation
    - Opponent pattern recognition
    - Score-based strategy adaptation
    - Advanced trump evaluation
    - Defensive play detection
    - Complex decision trees for all scenarios

    The player adapts its strategy based on:
    - Current score differential
    - Position at the table
    - Opponent tendencies
    - Cards already played
    - Remaining card distributions
    """

    def __init__(self) -> None:
        """Initialize an advanced heuristic-based player."""
        super().__init__()
        self.card_tracker = CardTracker()
        self.pattern_analyzer = OpponentPatternAnalyzer()
        self.current_tricks_won: Dict[int, int] = defaultdict(int)  # team -> tricks won this hand
        self.current_scores: Optional[List[int]] = None  # [team0_score, team1_score]
        self.trump_maker_team: Optional[int] = None
        self.hand_number: int = 0

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Decide whether to order up using advanced heuristics with position awareness.

        Considers:
        - Hand strength evaluation
        - Position at table (dealer, dealer's partner, first seat, second seat)
        - Score situation
        - Opponent patterns
        - Risk/reward analysis

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
        self.card_tracker.set_trump(potential_trump)

        # Comprehensive hand strength evaluation
        hand_strength = self._evaluate_advanced_hand_strength(player.hand, potential_trump)
        trump_count = self._count_trump_cards(player.hand, potential_trump)
        
        # Check for specific strong combinations
        has_both_bowers = self._has_both_bowers(player.hand, potential_trump)
        has_right_bower = self._has_right_bower(player.hand, potential_trump)
        has_left_bower = self._has_left_bower(player.hand, potential_trump)
        has_trump_ace = self._has_trump_ace(player.hand, potential_trump)
        has_trump_king = self._has_trump_king(player.hand, potential_trump)

        # Base threshold
        threshold = AdvancedHeuristicWeights.ORDER_UP_BASE_THRESHOLD

        # Position-based adjustments
        player_position = self._get_player_position(player.player_id, dealer_id)
        if player_position == "dealer_partner":
            threshold += AdvancedHeuristicWeights.ORDER_UP_DEALER_PARTNER_BONUS
        elif player_position == "dealer":
            threshold += AdvancedHeuristicWeights.ORDER_UP_DEALER_BONUS
        elif player_position == "first_seat":
            threshold += AdvancedHeuristicWeights.ORDER_UP_FIRST_SEAT_BONUS

        # Score-based adjustments
        score_adjustment = self._calculate_score_based_adjustment(player.team)
        threshold += score_adjustment * threshold

        # Opponent pattern adjustments
        if self._should_be_more_aggressive_due_to_opponents(player.player_id):
            threshold -= 15.0

        # Strong hand combinations - order up immediately
        if has_both_bowers:
            return True
        if has_right_bower and trump_count >= 2:
            return True
        if has_left_bower and has_trump_ace and trump_count >= 2:
            return True
        if trump_count >= 3:
            return True
        if trump_count >= 2 and (has_trump_ace or has_trump_king):
            return True

        # Evaluate based on hand strength
        if hand_strength >= threshold:
            return True

        # Conservative check: at least 2 trump
        if trump_count >= 2 and hand_strength >= threshold * 0.85:
            return True

        # Very aggressive if behind significantly
        if self._is_significantly_behind(player.team) and trump_count >= 1 and hand_strength >= threshold * 0.75:
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
        Decide which suit to call as trump using sophisticated multi-factor analysis.

        Evaluates all possible suits considering:
        - Hand strength for each suit
        - Position and score situation
        - Opponent patterns
        - Risk assessment

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up (cannot be chosen).
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit.

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        if trump_suit is not None:
            if must_choose:
                return trump_suit
            return None

        forbidden_suit = turned_card.suit
        suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]

        best_suit: Optional[Suit] = None
        best_score = 0.0

        # Evaluate each potential trump suit
        for suit in suits:
            if suit == forbidden_suit:
                continue

            # Calculate comprehensive hand strength
            hand_strength = self._evaluate_advanced_hand_strength(player.hand, suit)
            trump_count = self._count_trump_cards(player.hand, suit)

            # Start with base hand strength
            score = hand_strength

            # Bonus for having bowers
            if self._has_both_bowers(player.hand, suit):
                score += AdvancedHeuristicWeights.BOTH_BOWERS_BONUS
            elif self._has_right_bower(player.hand, suit):
                score += AdvancedHeuristicWeights.RIGHT_BOWER_BONUS
            elif self._has_left_bower(player.hand, suit):
                score += AdvancedHeuristicWeights.LEFT_BOWER_BONUS

            # Bonus for multiple trump cards
            if trump_count >= 4:
                score += AdvancedHeuristicWeights.MULTIPLE_TRUMP_BONUS_4PLUS
            elif trump_count >= 3:
                score += AdvancedHeuristicWeights.MULTIPLE_TRUMP_BONUS_3PLUS
            elif trump_count >= 2:
                score += AdvancedHeuristicWeights.MULTIPLE_TRUMP_BONUS_2PLUS

            # Bonus for strong trump combinations
            if self._has_trump_ace(player.hand, suit) and self._has_trump_king(player.hand, suit):
                score += AdvancedHeuristicWeights.STRONG_TRUMP_COMBO_BONUS

            # Bonus for off-suit aces
            offsuit_aces = self._count_offsuit_aces(player.hand, suit)
            if offsuit_aces >= 2:
                score += AdvancedHeuristicWeights.MULTIPLE_OFFSUIT_ACES_BONUS
            elif offsuit_aces >= 1:
                score += AdvancedHeuristicWeights.OFFSUIT_ACE_BONUS

            # Score-based adjustment
            score_adjustment = self._calculate_score_based_adjustment(player.team)
            score *= (1.0 + score_adjustment)

            if score > best_score:
                best_score = score
                best_suit = suit

        # Determine threshold
        threshold = AdvancedHeuristicWeights.CALL_TRUMP_BASE_THRESHOLD
        
        # Second round bonus (more likely to call)
        if must_choose:
            threshold = AdvancedHeuristicWeights.CALL_TRUMP_MUST_CHOOSE_THRESHOLD
        else:
            threshold += AdvancedHeuristicWeights.CALL_TRUMP_SECOND_ROUND_BONUS

        # Score-based adjustment
        score_adjustment = self._calculate_score_based_adjustment(player.team)
        threshold += score_adjustment * threshold

        # Call trump if score exceeds threshold
        if best_suit is not None and best_score >= threshold:
            self.card_tracker.set_trump(best_suit)
            return best_suit

        # If must choose, return best suit even if weak
        if must_choose and best_suit is not None:
            self.card_tracker.set_trump(best_suit)
            return best_suit

        return None

    def choose_card_to_discard(
        self,
        player: "Player",
        turned_card: Optional[Card] = None,
        ordered_up_by: Optional[str] = None,
    ) -> Card:
        """
        Choose a card to discard using strategic evaluation.

        Considers:
        - Hand composition after picking up turned card
        - Trump suit strength
        - Off-suit card values
        - Defensive considerations

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

        trump_suit: Optional[Suit] = None
        if turned_card is not None:
            trump_suit = turned_card.suit
            self.card_tracker.set_trump(trump_suit)

        # If we know trump, discard the weakest card considering trump value
        if trump_suit is not None:
            # Evaluate each card's value in context
            card_values: List[Tuple[Card, float]] = []
            for card in player.hand:
                value = self._calculate_advanced_card_power(card, trump_suit, context="discard")
                card_values.append((card, value))
            
            # Discard the card with lowest value
            return min(card_values, key=lambda x: x[1])[0]

        # If trump unknown, discard lowest rank card
        return min(player.hand, key=lambda c: c.rank.value)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Choose a card to play using complex decision trees.

        Implements advanced strategies for:
        - Leading: Trump control, off-suit pressure, defensive leads
        - Following: Advanced ducking, overtrumping decisions
        - Last to play: Complex trick outcome analysis

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

        # Record cards played so far in this trick
        for card in trick_cards:
            self.card_tracker.record_card(card)

        # Record leads for pattern analysis
        if led_suit is not None and len(trick_cards) == 0:
            # This is a new trick, record the lead
            if len(trick_player_ids) > 0:
                leader_id = trick_player_ids[0]
                self.pattern_analyzer.record_lead(leader_id, led_suit)

        # Route to appropriate decision method
        if led_suit is None:
            # Leading: no cards played yet
            return self._decide_advanced_lead(player, valid_cards, trump_suit)
        elif len(trick_cards) == 3:
            # Last to play: 3 cards already played
            return self._decide_advanced_last_play(
                player, valid_cards, trick_cards, trick_player_ids, led_suit, trump_suit
            )
        else:
            # Following: 1-2 cards already played
            return self._decide_advanced_follow(
                player, valid_cards, trick_cards, trick_player_ids, led_suit, trump_suit
            )

    def decide_going_alone(self, player: "Player", trump_suit: Suit) -> bool:
        """
        Decide whether to go alone using advanced risk assessment.

        Considers:
        - Hand strength with both bowers
        - Off-suit aces
        - Score situation
        - Opponent patterns
        - Risk/reward analysis

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
        # Check for flush in trump (all 5 cards are trump)
        if self._has_flush_in_trump(player.hand, trump_suit):
            return True

        # Evaluate hand strength
        hand_strength = self._evaluate_advanced_hand_strength(player.hand, trump_suit)
        trump_count = self._count_trump_cards(player.hand, trump_suit)

        # Check for both bowers (very strong)
        has_both_bowers = self._has_both_bowers(player.hand, trump_suit)
        has_right_bower = self._has_right_bower(player.hand, trump_suit)
        has_left_bower = self._has_left_bower(player.hand, trump_suit)

        # Base threshold
        threshold = AdvancedHeuristicWeights.GOING_ALONE_BASE_THRESHOLD

        # Lower threshold if have both bowers
        if has_both_bowers:
            threshold = AdvancedHeuristicWeights.GOING_ALONE_WITH_BOTH_BOWERS

        # Score-based adjustment
        score_adjustment = self._calculate_score_based_adjustment(player.team)
        threshold += score_adjustment * threshold

        # Check for top 3 trump cards (Right Bower, Left Bower, Ace)
        has_top_trump = self._has_top_trump_cards(player.hand, trump_suit)

        # Check for off-suit Aces
        off_suit_aces = self._count_offsuit_aces(player.hand, trump_suit)

        # Go alone if:
        # 1. Flush in trump (already checked above)
        # 2. Very high hand strength AND has top trump cards AND off-suit Aces
        if hand_strength >= threshold and has_top_trump and off_suit_aces >= 1:
            return True

        # Go alone if we have both bowers and at least 3 trump total
        if has_both_bowers and trump_count >= 3:
            return True

        # Go alone if we have 4+ trump cards (very strong)
        if trump_count >= 4:
            return True

        # Go alone if we have right bower, left bower, and trump ace
        if has_right_bower and has_left_bower and self._has_trump_ace(player.hand, trump_suit):
            if off_suit_aces >= 1:
                return True

        # Aggressive going alone if significantly behind
        if self._is_significantly_behind(player.team):
            if hand_strength >= threshold * 0.85 and trump_count >= 3:
                return True

        return False

    def decide_trade_in(self, player: "Player", eligible_cards: List[Card]) -> bool:
        """
        Decide whether to trade-in eligible cards using strategic evaluation.

        Parameters
        ----------
        player : Player
            The player making the decision.
        eligible_cards : List[Card]
            The three cards eligible for trade-in.

        Returns
        -------
        bool
            True to trade-in, False to pass.
        """
        # Calculate current hand strength (without knowing trump yet)
        current_hand_strength = sum(card.rank.value for card in player.hand)

        # Count high cards (Ace, King, Queen) in current hand
        high_card_count = sum(
            1 for card in player.hand if card.rank in (Rank.ACE, Rank.KING, Rank.QUEEN)
        )

        # Count Jacks (potential bowers)
        jack_count = sum(1 for card in player.hand if card.rank == Rank.JACK)

        # Trade-in if:
        # 1. Hand has very few high cards (2 or fewer)
        # 2. Hand has no Jacks (no potential bowers)
        # 3. Overall hand strength is low

        if high_card_count <= 2 and jack_count == 0:
            return True

        # Trade-in if hand strength is very low (mostly 9s and 10s)
        if current_hand_strength <= 50:  # Average of 10 per card (all low cards)
            return True

        # Trade-in if significantly behind and hand is weak
        if self._is_significantly_behind(player.team) and high_card_count <= 3:
            return True

        # Otherwise, keep the hand
        return False

    # ========== Advanced Helper Methods ==========

    def _evaluate_advanced_hand_strength(self, hand: List[Card], trump_suit: Suit) -> float:
        """
        Evaluate hand strength using comprehensive multi-factor analysis.

        Parameters
        ----------
        hand : List[Card]
            The hand to evaluate.
        trump_suit : Suit
            The trump suit to evaluate against.

        Returns
        -------
        float
            Comprehensive hand strength score (higher is better).
        """
        total_power = 0.0
        trump_count = 0
        has_right_bower = False
        has_left_bower = False
        has_trump_ace = False
        has_trump_king = False

        for card in hand:
            power = self._calculate_advanced_card_power(card, trump_suit, context="general")
            total_power += power

            if self._is_trump_card(card, trump_suit):
                trump_count += 1
                if card.rank == Rank.JACK and card.suit == trump_suit:
                    has_right_bower = True
                elif card.rank == Rank.JACK:
                    trump_card = Card(trump_suit, Rank.ACE)
                    if card.is_same_color(trump_card):
                        has_left_bower = True
                elif card.suit == trump_suit and card.rank == Rank.ACE:
                    has_trump_ace = True
                elif card.suit == trump_suit and card.rank == Rank.KING:
                    has_trump_king = True

        # Bonuses for specific combinations
        if has_right_bower and has_left_bower:
            total_power += AdvancedHeuristicWeights.BOTH_BOWERS_BONUS
        elif has_right_bower:
            total_power += AdvancedHeuristicWeights.RIGHT_BOWER_BONUS
        elif has_left_bower:
            total_power += AdvancedHeuristicWeights.LEFT_BOWER_BONUS

        if has_trump_ace:
            total_power += AdvancedHeuristicWeights.TRUMP_ACE_BONUS
        if has_trump_king:
            total_power += AdvancedHeuristicWeights.TRUMP_KING_BONUS

        # Bonus for multiple trump cards
        if trump_count >= 4:
            total_power += AdvancedHeuristicWeights.MULTIPLE_TRUMP_BONUS_4PLUS
        elif trump_count >= 3:
            total_power += AdvancedHeuristicWeights.MULTIPLE_TRUMP_BONUS_3PLUS
        elif trump_count >= 2:
            total_power += AdvancedHeuristicWeights.MULTIPLE_TRUMP_BONUS_2PLUS

        # Bonus for strong trump combinations
        if has_trump_ace and has_trump_king:
            total_power += AdvancedHeuristicWeights.STRONG_TRUMP_COMBO_BONUS

        # Bonus for off-suit aces
        offsuit_aces = self._count_offsuit_aces(hand, trump_suit)
        if offsuit_aces >= 2:
            total_power += AdvancedHeuristicWeights.MULTIPLE_OFFSUIT_ACES_BONUS
        elif offsuit_aces >= 1:
            total_power += AdvancedHeuristicWeights.OFFSUIT_ACE_BONUS

        return total_power

    def _calculate_advanced_card_power(
        self, card: Card, trump_suit: Optional[Suit], context: str = "general"
    ) -> float:
        """
        Calculate card power using advanced heuristics with context awareness.

        Parameters
        ----------
        card : Card
            The card to evaluate.
        trump_suit : Optional[Suit]
            The current trump suit, if any.
        context : str
            Context of evaluation: "general", "leading", "following", "discard"

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
            power = float(AdvancedHeuristicWeights.RIGHT_BOWER_POWER)
        elif is_left_bower:
            power = float(AdvancedHeuristicWeights.LEFT_BOWER_POWER)
        elif card.suit == trump_suit:
            # Regular trump suit card
            rank_power_map = {
                Rank.ACE: AdvancedHeuristicWeights.TRUMP_ACE_POWER,
                Rank.KING: AdvancedHeuristicWeights.TRUMP_KING_POWER,
                Rank.QUEEN: AdvancedHeuristicWeights.TRUMP_QUEEN_POWER,
                Rank.TEN: AdvancedHeuristicWeights.TRUMP_TEN_POWER,
                Rank.NINE: AdvancedHeuristicWeights.TRUMP_NINE_POWER,
            }
            power = float(rank_power_map.get(card.rank, 0))
        else:
            # Off-suit card
            rank_power_map = {
                Rank.ACE: AdvancedHeuristicWeights.OFFSUIT_ACE_POWER,
                Rank.KING: AdvancedHeuristicWeights.OFFSUIT_KING_POWER,
                Rank.QUEEN: AdvancedHeuristicWeights.OFFSUIT_QUEEN_POWER,
                Rank.JACK: AdvancedHeuristicWeights.OFFSUIT_JACK_POWER,
                Rank.TEN: AdvancedHeuristicWeights.OFFSUIT_TEN_POWER,
                Rank.NINE: AdvancedHeuristicWeights.OFFSUIT_NINE_POWER,
            }
            power = float(rank_power_map.get(card.rank, 0))

        # Apply context-based adjustments
        if context == "leading":
            if card.suit == trump_suit or is_right_bower or is_left_bower:
                power *= AdvancedHeuristicWeights.LEAD_TRUMP_PREFERENCE
            elif card.rank == Rank.ACE:
                power *= AdvancedHeuristicWeights.LEAD_OFFSUIT_ACE
            elif card.rank == Rank.KING:
                power *= AdvancedHeuristicWeights.LEAD_OFFSUIT_KING
        elif context == "discard":
            # When discarding, we want to keep high-value cards
            # So lower the power (we discard low power cards)
            pass  # Already calculated correctly

        return power

    def _decide_advanced_lead(
        self, player: "Player", valid_cards: List[Card], trump_suit: Optional[Suit]
    ) -> Card:
        """
        Decide which card to lead using advanced strategies.

        Strategies include:
        - Trump control leads
        - Off-suit pressure leads
        - Defensive leads
        - Score-based leading

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
        if trump_suit is None:
            # No trump yet - lead highest card
            return max(valid_cards, key=lambda c: c.rank.value)

        # Calculate power for each card in leading context
        card_powers = [
            (card, self._calculate_advanced_card_power(card, trump_suit, context="leading"))
            for card in valid_cards
        ]

        # Count trump cards in hand
        trump_count = self._count_trump_cards(player.hand, trump_suit)

        # Strategy selection based on situation
        tricks_needed = self._calculate_tricks_needed(player.team)
        is_ahead = self._is_team_ahead(player.team)

        # If we need tricks and have strong trump, lead trump
        if tricks_needed > 0 and trump_count >= 2:
            trump_cards = [
                (card, power)
                for card, power in card_powers
                if self._is_trump_card(card, trump_suit)
            ]
            if trump_cards:
                # Lead with strong trump to establish control
                return max(trump_cards, key=lambda x: x[1])[0]

        # If ahead and have good off-suit, lead off-suit to draw out trump
        if is_ahead and trump_count >= 1:
            offsuit_cards = [
                (card, power)
                for card, power in card_powers
                if not self._is_trump_card(card, trump_suit)
            ]
            if offsuit_cards:
                # Lead with strong off-suit ace or king
                strong_offsuit = [
                    (card, power)
                    for card, power in offsuit_cards
                    if card.rank in [Rank.ACE, Rank.KING]
                ]
                if strong_offsuit:
                    return max(strong_offsuit, key=lambda x: x[1])[0]
                return max(offsuit_cards, key=lambda x: x[1])[0]

        # Default: lead with highest power card
        return max(card_powers, key=lambda x: x[1])[0]

    def _decide_advanced_follow(
        self,
        player: "Player",
        valid_cards: List[Card],
        trick_cards: List[Card],
        trick_player_ids: List[int],
        led_suit: Suit,
        trump_suit: Optional[Suit],
    ) -> Card:
        """
        Decide which card to play when following suit.

        Implements advanced strategies:
        - Smart ducking when teammate is winning
        - Overtrumping decisions
        - Defensive plays
        - Score-based decisions

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
        if not trick_cards:
            return valid_cards[0]

        # Find current winner
        current_winner_id = self._find_current_winner_player_id(
            trick_cards, trick_player_ids, led_suit, trump_suit
        )

        if current_winner_id is None:
            # Fallback
            return max(
                valid_cards,
                key=lambda c: self._calculate_advanced_card_power(c, trump_suit, context="general"),
            )

        # Check if teammate is winning
        is_teammate_winning = self._is_teammate(player, current_winner_id)

        if is_teammate_winning:
            # Teammate is winning: advanced ducking strategy
            tricks_needed = self._calculate_tricks_needed(player.team)
            
            # If we need more tricks, might still try to win with a very strong card
            if tricks_needed > 0 and self._is_significantly_behind(player.team):
                winning_cards = []
                for card in valid_cards:
                    if self._can_win_trick(card, trick_cards, led_suit, trump_suit):
                        power = self._calculate_advanced_card_power(card, trump_suit, context="general")
                        # Only try to win if card is very strong (top trump or off-suit ace)
                        if power > 75.0:
                            winning_cards.append((card, power))
                if winning_cards:
                    # Win with lowest power winning card that's still strong
                    return min(winning_cards, key=lambda x: x[1])[0]

            # Default: duck (play lowest card)
            return min(
                valid_cards,
                key=lambda c: self._calculate_advanced_card_power(c, trump_suit, context="general"),
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
                    key=lambda c: self._calculate_advanced_card_power(c, trump_suit, context="general"),
                )
            else:
                # Can't win: play lowest card to preserve high cards
                return min(
                    valid_cards,
                    key=lambda c: self._calculate_advanced_card_power(c, trump_suit, context="general"),
                )

    def _decide_advanced_last_play(
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

        Complex decision tree considering:
        - Trick outcome analysis
        - Score situation
        - Defensive needs
        - Trump management

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
        if not trick_cards or len(trick_cards) != 3:
            return valid_cards[0]

        # Find current winner
        current_winner_id = self._find_current_winner_player_id(
            trick_cards, trick_player_ids, led_suit, trump_suit
        )

        if current_winner_id is None:
            # Fallback
            return max(
                valid_cards,
                key=lambda c: self._calculate_advanced_card_power(c, trump_suit, context="general"),
            )

        # Check if teammate is winning
        is_teammate_winning = self._is_teammate(player, current_winner_id)

        tricks_needed = self._calculate_tricks_needed(player.team)
        is_behind = self._is_significantly_behind(player.team)

        if is_teammate_winning:
            # Teammate is winning: advanced ducking
            # If we're ahead and have enough tricks, definitely duck
            if not is_behind and tricks_needed <= 0:
                return min(
                    valid_cards,
                    key=lambda c: self._calculate_advanced_card_power(c, trump_suit, context="general"),
                )

            # If we need tricks and behind, might still try to win with very strong card
            if is_behind and tricks_needed > 0:
                winning_cards = []
                for card in valid_cards:
                    if self._can_win_trick(card, trick_cards, led_suit, trump_suit):
                        power = self._calculate_advanced_card_power(card, trump_suit, context="general")
                        if power > 80.0:  # Very strong card only
                            winning_cards.append((card, power))
                if winning_cards:
                    return min(winning_cards, key=lambda x: x[1])[0]

            # Default: duck
            return min(
                valid_cards,
                key=lambda c: self._calculate_advanced_card_power(c, trump_suit, context="general"),
            )
        else:
            # Opponent is winning: try to win if possible
            winning_cards = []
            for card in valid_cards:
                if self._can_win_trick(card, trick_cards, led_suit, trump_suit):
                    winning_cards.append(card)

            if winning_cards:
                # Win with lowest power winning card
                return min(
                    winning_cards,
                    key=lambda c: self._calculate_advanced_card_power(c, trump_suit, context="general"),
                )
            else:
                # Can't win: play lowest card
                return min(
                    valid_cards,
                    key=lambda c: self._calculate_advanced_card_power(c, trump_suit, context="general"),
                )

    def _get_player_position(self, player_id: int, dealer_id: int) -> str:
        """
        Get player's position relative to dealer.

        Parameters
        ----------
        player_id : int
            The player's ID.
        dealer_id : int
            The dealer's ID.

        Returns
        -------
        str
            Position: "dealer", "dealer_partner", "first_seat", "second_seat"
        """
        if player_id == dealer_id:
            return "dealer"
        
        # Check if dealer's partner
        if (player_id % 2) == (dealer_id % 2):
            return "dealer_partner"
        
        # Determine seat order (0,1,2,3 clockwise)
        # First seat is left of dealer
        first_seat = (dealer_id + 1) % 4
        if player_id == first_seat:
            return "first_seat"
        
        return "second_seat"

    def _calculate_score_based_adjustment(self, team: int) -> float:
        """
        Calculate strategy adjustment based on score situation.

        Returns positive for more aggressive, negative for more conservative.

        Parameters
        ----------
        team : int
            The player's team (0 or 1).

        Returns
        -------
        float
            Adjustment factor (-1.0 to 1.0).
        """
        if self.current_scores is None:
            return 0.0

        my_score = self.current_scores[team]
        opponent_score = self.current_scores[1 - team]
        score_diff = my_score - opponent_score

        # Behind significantly: be more aggressive
        if score_diff <= -4:
            return AdvancedHeuristicWeights.BEHIND_BY_4_PLUS_AGGRESSION
        elif score_diff <= -2:
            return AdvancedHeuristicWeights.BEHIND_BY_2_PLUS_AGGRESSION

        # Ahead significantly: be more conservative
        if score_diff >= 4:
            return AdvancedHeuristicWeights.AHEAD_BY_4_PLUS_CONSERVATIVE
        elif score_diff >= 2:
            return AdvancedHeuristicWeights.AHEAD_BY_2_PLUS_CONSERVATIVE

        # Endgame: close to 10 points
        if my_score >= 8 or opponent_score >= 8:
            if my_score < opponent_score:
                return AdvancedHeuristicWeights.ENDGAME_AGGRESSION
            else:
                return AdvancedHeuristicWeights.AHEAD_BY_2_PLUS_CONSERVATIVE

        return 0.0

    def _is_significantly_behind(self, team: int) -> bool:
        """Check if team is significantly behind in score."""
        if self.current_scores is None:
            return False
        my_score = self.current_scores[team]
        opponent_score = self.current_scores[1 - team]
        return opponent_score - my_score >= 2

    def _is_team_ahead(self, team: int) -> bool:
        """Check if team is ahead in score."""
        if self.current_scores is None:
            return False
        my_score = self.current_scores[team]
        opponent_score = self.current_scores[1 - team]
        return my_score > opponent_score

    def _calculate_tricks_needed(self, team: int) -> int:
        """
        Calculate how many tricks are needed this hand.

        Parameters
        ----------
        team : int
            The team to check.

        Returns
        -------
        int
            Number of tricks needed (negative if already have enough).
        """
        if self.trump_maker_team is None:
            return 0  # Can't determine yet

        tricks_won = self.current_tricks_won.get(team, 0)
        
        if self.trump_maker_team == team:
            # We're makers, need 3 tricks
            return max(0, 3 - tricks_won)
        else:
            # We're defenders, need to prevent makers from getting 3
            # This is defensive, so return 0 (we're trying to euchre them)
            return 0

    def _should_be_more_aggressive_due_to_opponents(self, player_id: int) -> bool:
        """Check if should be more aggressive based on opponent patterns."""
        # If opponents are conservative, we can be more aggressive
        # This is a simplified check - in full implementation would analyze all opponents
        return False  # Placeholder

    def _is_teammate(self, player: "Player", other_player_id: int) -> bool:
        """Check if another player is a teammate."""
        return (player.player_id % 2) == (other_player_id % 2)

    def _find_current_winner_player_id(
        self,
        trick_cards: List[Card],
        trick_player_ids: List[int],
        led_suit: Suit,
        trump_suit: Optional[Suit],
    ) -> Optional[int]:
        """Find the player ID of the current winner of the trick."""
        if not trick_cards:
            return None

        winner_idx = 0
        for i in range(1, len(trick_cards)):
            comparison = trick_cards[i].compare_to(trick_cards[winner_idx], trump_suit, led_suit)
            if comparison > 0:
                winner_idx = i

        if winner_idx < len(trick_player_ids):
            return trick_player_ids[winner_idx]
        return None

    def _has_both_bowers(self, hand: List[Card], trump_suit: Suit) -> bool:
        """Check if hand has both Right and Left Bower."""
        has_right = False
        has_left = False

        for card in hand:
            if card.rank == Rank.JACK:
                if card.suit == trump_suit:
                    has_right = True
                else:
                    trump_card = Card(trump_suit, Rank.ACE)
                    if card.is_same_color(trump_card):
                        has_left = True

        return has_right and has_left

    def _has_right_bower(self, hand: List[Card], trump_suit: Suit) -> bool:
        """Check if hand has Right Bower."""
        for card in hand:
            if card.rank == Rank.JACK and card.suit == trump_suit:
                return True
        return False

    def _has_left_bower(self, hand: List[Card], trump_suit: Suit) -> bool:
        """Check if hand has Left Bower."""
        for card in hand:
            if card.rank == Rank.JACK and card.suit != trump_suit:
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    return True
        return False

    def _has_trump_ace(self, hand: List[Card], trump_suit: Suit) -> bool:
        """Check if hand has trump Ace."""
        for card in hand:
            if card.suit == trump_suit and card.rank == Rank.ACE:
                return True
        return False

    def _has_trump_king(self, hand: List[Card], trump_suit: Suit) -> bool:
        """Check if hand has trump King."""
        for card in hand:
            if card.suit == trump_suit and card.rank == Rank.KING:
                return True
        return False

    def _count_offsuit_aces(self, hand: List[Card], trump_suit: Suit) -> int:
        """Count off-suit Aces (Aces that are not trump)."""
        count = 0
        for card in hand:
            if card.rank == Rank.ACE and not self._is_trump_card(card, trump_suit):
                count += 1
        return count

    def _has_flush_in_trump(self, hand: List[Card], trump_suit: Suit) -> bool:
        """Check if hand has a flush in trump (all 5 cards are trump)."""
        for card in hand:
            if not self._is_trump_card(card, trump_suit):
                return False
        return True

    def _has_top_trump_cards(self, hand: List[Card], trump_suit: Suit) -> bool:
        """Check if hand has top 3 trump cards (Right Bower, Left Bower, Ace)."""
        has_right_bower = self._has_right_bower(hand, trump_suit)
        has_left_bower = self._has_left_bower(hand, trump_suit)
        has_trump_ace = self._has_trump_ace(hand, trump_suit)
        return has_right_bower and has_left_bower and has_trump_ace

    def _estimate_opponent_hand_strength(
        self, player_id: int, trump_suit: Suit, known_cards: List[Card]
    ) -> float:
        """
        Estimate opponent's hand strength based on cards played and patterns.

        Uses card tracking and pattern analysis to estimate how strong
        an opponent's hand might be.

        Parameters
        ----------
        player_id : int
            The opponent's player ID.
        trump_suit : Suit
            The current trump suit.
        known_cards : List[Card]
            Cards we know the opponent has played or doesn't have.

        Returns
        -------
        float
            Estimated hand strength (0-500 range).
        """
        # Get remaining cards that opponent might have
        remaining_cards = self.card_tracker.get_remaining_cards(known_cards)
        
        # Estimate based on cards they've played
        # If they've played high cards, they might have more
        # If they've played low cards, they might have high cards remaining
        
        # Simple heuristic: if they've been conservative, assume stronger hand
        if self.pattern_analyzer.is_conservative_bidder(player_id):
            return 180.0  # Assume moderate-strong hand
        
        # If they've been aggressive, might have strong hand or might be bluffing
        if self.pattern_analyzer.is_aggressive_bidder(player_id):
            return 160.0  # Moderate estimate
        
        # Default moderate estimate
        return 140.0

    def _simulate_trick_outcome(
        self,
        card_to_play: Card,
        trick_cards: List[Card],
        led_suit: Suit,
        trump_suit: Optional[Suit],
        remaining_cards_estimate: List[Card],
    ) -> Dict[str, float]:
        """
        Simulate possible trick outcomes to evaluate card play.

        Parameters
        ----------
        card_to_play : Card
            The card we're considering playing.
        trick_cards : List[Card]
            Cards already played in the trick.
        led_suit : Suit
            The suit that was led.
        trump_suit : Optional[Suit]
            The current trump suit.
        remaining_cards_estimate : List[Card]
            Estimated remaining cards opponents might have.

        Returns
        -------
        Dict[str, float]
            Dictionary with win_probability, expected_value, etc.
        """
        # Simple simulation: check if card can win against current trick
        can_win = self._can_win_trick(card_to_play, trick_cards, led_suit, trump_suit)
        
        # Estimate probability of winning based on card strength
        card_power = self._calculate_advanced_card_power(card_to_play, trump_suit, context="general")
        
        # Higher power cards have better chance of winning
        win_probability = min(0.95, max(0.05, card_power / 100.0)) if can_win else 0.05
        
        return {
            "can_win": can_win,
            "win_probability": win_probability,
            "card_power": card_power,
            "expected_value": win_probability * card_power,
        }

    def _evaluate_defensive_play_needed(
        self, player: "Player", trump_suit: Optional[Suit], tricks_played: int
    ) -> bool:
        """
        Evaluate if defensive play is needed to prevent opponent march.

        Parameters
        ----------
        player : Player
            The player making the decision.
        trump_suit : Optional[Suit]
            The current trump suit.
        tricks_played : int
            Number of tricks already played this hand.

        Returns
        -------
        bool
            True if defensive play is needed.
        """
        if self.trump_maker_team is None or trump_suit is None:
            return False

        # If opponents are makers and have won 2+ tricks, need to defend
        if self.trump_maker_team != player.team:
            opponent_tricks = self.current_tricks_won.get(1 - player.team, 0)
            if opponent_tricks >= 2:
                return True

        return False

    def _find_best_defensive_card(
        self,
        valid_cards: List[Card],
        trick_cards: List[Card],
        led_suit: Suit,
        trump_suit: Optional[Suit],
    ) -> Optional[Card]:
        """
        Find the best card to play defensively to break up opponent's trick.

        Parameters
        ----------
        valid_cards : List[Card]
            Valid cards that can be played.
        trick_cards : List[Card]
            Cards already played in the trick.
        led_suit : Suit
            The suit that was led.
        trump_suit : Optional[Suit]
            The current trump suit.

        Returns
        -------
        Optional[Card]
            Best defensive card, or None if no good defensive play.
        """
        if not trick_cards:
            return None

        # Find cards that can win the trick (break it up)
        winning_cards = [
            card
            for card in valid_cards
            if self._can_win_trick(card, trick_cards, led_suit, trump_suit)
        ]

        if not winning_cards:
            return None

        # Prefer trump cards for defensive plays
        trump_winners = []
        if trump_suit:
            trump_winners = [
                card for card in winning_cards if self._is_trump_card(card, trump_suit)
            ]
        if trump_winners:
            # Use lowest trump that can win
            return min(
                trump_winners,
                key=lambda c: self._calculate_advanced_card_power(c, trump_suit, context="general"),
            )

        # Otherwise use lowest winning card
        return min(
            winning_cards,
            key=lambda c: self._calculate_advanced_card_power(c, trump_suit, context="general"),
        )

    def _analyze_suit_distribution(self, hand: List[Card], trump_suit: Optional[Suit]) -> Dict[Suit, int]:
        """
        Analyze suit distribution in hand.

        Parameters
        ----------
        hand : List[Card]
            The hand to analyze.
        trump_suit : Optional[Suit]
            The current trump suit.

        Returns
        -------
        Dict[Suit, int]
            Count of cards per suit.
        """
        distribution: Dict[Suit, int] = defaultdict(int)
        for card in hand:
            if trump_suit and self._is_trump_card(card, trump_suit):
                distribution[trump_suit] += 1
            else:
                distribution[card.suit] += 1
        return distribution

    def _identify_void_suits(self, hand: List[Card], trump_suit: Optional[Suit]) -> List[Suit]:
        """
        Identify suits the player is void in (has no cards).

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Optional[Suit]
            The current trump suit.

        Returns
        -------
        List[Suit]
            List of suits the player is void in.
        """
        suits_in_hand: Set[Suit] = set()
        for card in hand:
            if trump_suit and self._is_trump_card(card, trump_suit):
                suits_in_hand.add(trump_suit)
            else:
                suits_in_hand.add(card.suit)

        all_suits = set(Suit)
        void_suits = all_suits - suits_in_hand
        return list(void_suits)

    def _evaluate_trump_management_strategy(
        self, player: "Player", trump_suit: Suit, tricks_played: int
    ) -> str:
        """
        Evaluate trump management strategy for current situation.

        Returns strategy: "aggressive", "conservative", "balanced", "defensive"

        Parameters
        ----------
        player : Player
            The player making the decision.
        trump_suit : Suit
            The current trump suit.
        tricks_played : int
            Number of tricks already played.

        Returns
        -------
        str
            Strategy recommendation.
        """
        trump_count = self._count_trump_cards(player.hand, trump_suit)
        tricks_needed = self._calculate_tricks_needed(player.team)
        is_behind = self._is_significantly_behind(player.team)

        # If we need tricks and have many trump, be aggressive
        if tricks_needed > 0 and trump_count >= 3:
            return "aggressive"

        # If ahead and have few trump left, be conservative
        if not is_behind and trump_count <= 1 and tricks_played >= 2:
            return "conservative"

        # If defending and opponents are close to march, be defensive
        if self._evaluate_defensive_play_needed(player, trump_suit, tricks_played):
            return "defensive"

        return "balanced"

    def _calculate_hand_balance(self, hand: List[Card], trump_suit: Suit) -> float:
        """
        Calculate hand balance score (distribution of strength).

        A balanced hand has strength spread across suits, while an unbalanced
        hand has all strength in one suit (usually trump).

        Parameters
        ----------
        hand : List[Card]
            The hand to evaluate.
        trump_suit : Suit
            The current trump suit.

        Returns
        -------
        float
            Balance score (higher = more balanced).
        """
        suit_strengths: Dict[Suit, float] = defaultdict(float)
        
        for card in hand:
            power = self._calculate_advanced_card_power(card, trump_suit, context="general")
            if self._is_trump_card(card, trump_suit):
                suit_strengths[trump_suit] += power
            else:
                suit_strengths[card.suit] += power

        if not suit_strengths:
            return 0.0

        # Calculate variance - lower variance = more balanced
        strengths = list(suit_strengths.values())
        mean_strength = sum(strengths) / len(strengths)
        variance = sum((s - mean_strength) ** 2 for s in strengths) / len(strengths)
        
        # Return inverse of variance (higher = more balanced)
        return 1000.0 / (1.0 + variance)

    def _evaluate_offensive_potential(
        self, hand: List[Card], trump_suit: Suit
    ) -> Dict[str, float]:
        """
        Evaluate offensive potential of hand.

        Parameters
        ----------
        hand : List[Card]
            The hand to evaluate.
        trump_suit : Suit
            The current trump suit.

        Returns
        -------
        Dict[str, float]
            Dictionary with offensive metrics.
        """
        trump_count = self._count_trump_cards(hand, trump_suit)
        has_both_bowers = self._has_both_bowers(hand, trump_suit)
        has_right_bower = self._has_right_bower(hand, trump_suit)
        offsuit_aces = self._count_offsuit_aces(hand, trump_suit)

        # Calculate offensive score
        offensive_score = 0.0
        offensive_score += trump_count * 20.0
        if has_both_bowers:
            offensive_score += 50.0
        elif has_right_bower:
            offensive_score += 30.0
        offensive_score += offsuit_aces * 15.0

        # Estimate tricks we can likely win
        estimated_tricks = min(5.0, offensive_score / 40.0)

        return {
            "offensive_score": offensive_score,
            "estimated_tricks": estimated_tricks,
            "trump_count": trump_count,
            "has_both_bowers": has_both_bowers,
        }

    def _evaluate_defensive_potential(
        self, hand: List[Card], trump_suit: Suit
    ) -> Dict[str, float]:
        """
        Evaluate defensive potential of hand.

        Parameters
        ----------
        hand : List[Card]
            The hand to evaluate.
        trump_suit : Suit
            The current trump suit.

        Returns
        -------
        Dict[str, float]
            Dictionary with defensive metrics.
        """
        trump_count = self._count_trump_cards(hand, trump_suit)
        void_suits = self._identify_void_suits(hand, trump_suit)
        offsuit_high_cards = sum(
            1
            for card in hand
            if not self._is_trump_card(card, trump_suit) and card.rank in [Rank.ACE, Rank.KING]
        )

        # Calculate defensive score
        defensive_score = 0.0
        defensive_score += trump_count * 15.0  # Trump can break tricks
        defensive_score += len(void_suits) * 10.0  # Voids allow trump plays
        defensive_score += offsuit_high_cards * 5.0  # High cards can win

        # Estimate ability to euchre opponents
        euchre_potential = min(1.0, defensive_score / 50.0)

        return {
            "defensive_score": defensive_score,
            "euchre_potential": euchre_potential,
            "trump_count": trump_count,
            "void_suits_count": len(void_suits),
        }

    def _analyze_opponent_card_play_pattern(
        self, player_id: int, card_played: Card, trick_number: int, was_winning: bool
    ) -> None:
        """
        Analyze opponent's card play pattern for future predictions.

        Parameters
        ----------
        player_id : int
            The opponent's player ID.
        card_played : Card
            The card they played.
        trick_number : int
            Which trick this was (0-4).
        was_winning : bool
            Whether this card won the trick.
        """
        # Track if they play trump early
        if self._is_trump_card(card_played, self.card_tracker.trump_suit) if self.card_tracker.trump_suit else False:
            self.pattern_analyzer.record_trump_play(player_id, trick_number)

        # Track leading preferences
        if trick_number == 0:  # First trick
            self.pattern_analyzer.record_lead(player_id, card_played.suit)

    def _estimate_probability_opponent_has_card(
        self, player_id: int, card: Card, known_cards: List[Card]
    ) -> float:
        """
        Estimate probability that an opponent has a specific card.

        Parameters
        ----------
        player_id : int
            The opponent's player ID.
        card : Card
            The card to check for.
        known_cards : List[Card]
            Cards we know are not in opponent's hand.

        Returns
        -------
        float
            Probability (0.0 to 1.0).
        """
        remaining_cards = self.card_tracker.get_remaining_cards(known_cards)
        
        if card not in remaining_cards:
            return 0.0

        # Simple estimate: 1 / number of remaining cards
        # In reality, this would be more sophisticated
        if len(remaining_cards) == 0:
            return 0.0

        # Each opponent has roughly 5 cards, so probability is roughly 5 / remaining
        # But we don't know exactly how many cards remain, so use a heuristic
        return min(1.0, 5.0 / max(1, len(remaining_cards)))

    def _evaluate_risk_reward_ordering_up(
        self, player: "Player", turned_card: Card, dealer_id: int
    ) -> Dict[str, float]:
        """
        Evaluate risk/reward of ordering up.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up.
        dealer_id : int
            The dealer's ID.

        Returns
        -------
        Dict[str, float]
            Dictionary with risk, reward, and recommendation.
        """
        potential_trump = turned_card.suit
        hand_strength = self._evaluate_advanced_hand_strength(player.hand, potential_trump)
        trump_count = self._count_trump_cards(player.hand, potential_trump)

        # Calculate expected value
        # If we order up and make it, we get points
        # If we order up and get euchred, we lose points
        
        # Estimate probability of making it
        make_probability = min(0.95, max(0.05, hand_strength / 200.0))
        
        # Expected value calculation
        # Making it: +1 point (or +2 if march, +4 if alone march)
        # Getting euchred: -2 points for opponents
        
        # Simplified: positive EV if make_probability > 0.6
        expected_value = (make_probability * 1.0) - ((1.0 - make_probability) * 0.5)
        
        risk = 1.0 - make_probability
        reward = make_probability

        return {
            "risk": risk,
            "reward": reward,
            "expected_value": expected_value,
            "make_probability": make_probability,
            "recommendation": "order" if expected_value > 0.3 else "pass",
        }

    def _calculate_position_value(self, player_id: int, dealer_id: int) -> float:
        """
        Calculate the value of player's position at the table.

        Parameters
        ----------
        player_id : int
            The player's ID.
        dealer_id : int
            The dealer's ID.

        Returns
        -------
        float
            Position value (higher = better position).
        """
        position = self._get_player_position(player_id, dealer_id)
        
        position_values = {
            "dealer": 1.0,  # Best position - can see all bids
            "dealer_partner": 0.9,  # Good position - partner can help
            "first_seat": 0.7,  # Moderate - first to act
            "second_seat": 0.8,  # Good - can see first seat's decision
        }
        
        return position_values.get(position, 0.5)

    def _evaluate_endgame_strategy(self, team: int) -> str:
        """
        Evaluate endgame strategy based on score.

        Parameters
        ----------
        team : int
            The team to evaluate for.

        Returns
        -------
        str
            Strategy: "aggressive", "conservative", "balanced"
        """
        if self.current_scores is None:
            return "balanced"

        my_score = self.current_scores[team]
        opponent_score = self.current_scores[1 - team]

        # Endgame: close to 10 points
        if my_score >= 9:
            return "conservative"  # Don't risk losing when so close
        elif opponent_score >= 9:
            return "aggressive"  # Must take risks to catch up
        elif my_score >= 8:
            return "balanced"  # Be careful but still play
        elif opponent_score >= 8:
            return "aggressive"  # Need to prevent opponent from winning

        return "balanced"

    def _analyze_trick_requirements(
        self, team: int, tricks_won: int
    ) -> Dict[str, int]:
        """
        Analyze trick requirements for current hand.

        Parameters
        ----------
        team : int
            The team to analyze.
        tricks_won : int
            Number of tricks won so far.

        Returns
        -------
        Dict[str, int]
            Dictionary with requirements.
        """
        if self.trump_maker_team is None:
            return {"needed": 0, "remaining": 5, "status": "unknown"}

        if self.trump_maker_team == team:
            # We're makers, need 3 tricks
            needed = max(0, 3 - tricks_won)
            status = "maker"
        else:
            # We're defenders, need to prevent makers from getting 3
            maker_tricks = self.current_tricks_won.get(self.trump_maker_team, 0)
            needed = max(0, 3 - maker_tricks)  # Tricks to prevent
            status = "defender"

        remaining = 5 - sum(self.current_tricks_won.values())

        return {
            "needed": needed,
            "remaining": remaining,
            "status": status,
            "tricks_won": tricks_won,
        }

    def _evaluate_card_utility(
        self,
        card: Card,
        hand: List[Card],
        trump_suit: Optional[Suit],
        context: str,
        trick_cards: Optional[List[Card]] = None,
    ) -> float:
        """
        Evaluate utility of playing a specific card in a given context.

        Parameters
        ----------
        card : Card
            The card to evaluate.
        hand : List[Card]
            The player's hand.
        trump_suit : Optional[Suit]
            The current trump suit.
        context : str
            Context: "lead", "follow", "last", "defensive"
        trick_cards : Optional[List[Card]]
            Cards already played in trick (if following).

        Returns
        -------
        float
            Utility score (higher = better play).
        """
        base_power = self._calculate_advanced_card_power(card, trump_suit, context=context)
        utility = base_power

        if context == "lead":
            # Leading: prefer trump or strong off-suit
            if trump_suit and self._is_trump_card(card, trump_suit):
                utility *= 1.2
            elif card.rank in [Rank.ACE, Rank.KING]:
                utility *= 1.1

        elif context == "follow" and trick_cards:
            # Following: adjust based on whether we can win
            can_win = self._can_win_trick(card, trick_cards, trick_cards[0].suit if trick_cards else Suit.HEARTS, trump_suit)
            if can_win:
                utility *= 1.1
            else:
                utility *= 0.9  # Slightly lower if can't win

        elif context == "defensive":
            # Defensive: prefer trump to break tricks
            if trump_suit and self._is_trump_card(card, trump_suit):
                utility *= 1.3

        return utility

    def _calculate_hand_versatility(self, hand: List[Card], trump_suit: Suit) -> float:
        """
        Calculate how versatile the hand is (can play in multiple situations).

        Parameters
        ----------
        hand : List[Card]
            The hand to evaluate.
        trump_suit : Suit
            The current trump suit.

        Returns
        -------
        float
            Versatility score (higher = more versatile).
        """
        trump_count = self._count_trump_cards(hand, trump_suit)
        offsuit_aces = self._count_offsuit_aces(hand, trump_suit)
        void_suits = self._identify_void_suits(hand, trump_suit)

        # Versatile hand has:
        # - Some trump (but not all)
        # - Some off-suit strength
        # - Can play in multiple suits

        versatility = 0.0
        
        # Trump versatility (having 2-3 trump is versatile)
        if 2 <= trump_count <= 3:
            versatility += 30.0
        elif trump_count == 1:
            versatility += 15.0

        # Off-suit versatility
        versatility += offsuit_aces * 20.0

        # Suit versatility (not void in many suits)
        versatility += (4 - len(void_suits)) * 10.0

        return versatility

    def _estimate_remaining_trick_value(
        self, tricks_remaining: int, team: int
    ) -> float:
        """
        Estimate the value of remaining tricks.

        Parameters
        ----------
        tricks_remaining : int
            Number of tricks remaining.
        team : int
            The team to evaluate for.

        Returns
        -------
        float
            Estimated value of remaining tricks.
        """
        if self.trump_maker_team is None:
            return 0.0

        tricks_won = self.current_tricks_won.get(team, 0)
        
        if self.trump_maker_team == team:
            # We're makers
            if tricks_won >= 3:
                return 0.0  # Already have enough
            needed = 3 - tricks_won
            if tricks_remaining >= needed:
                return 1.0 * needed  # Value of making it
            else:
                return 0.0  # Can't make it
        else:
            # We're defenders
            maker_tricks = self.current_tricks_won.get(self.trump_maker_team, 0)
            if maker_tricks >= 3:
                return 0.0  # Already euchred
            needed_to_euchre = 3 - maker_tricks
            if tricks_remaining >= needed_to_euchre:
                return 2.0  # Value of euchring
            else:
                return 0.0

    def _analyze_opponent_bidding_tendency(self, player_id: int) -> Dict[str, float]:
        """
        Analyze opponent's bidding tendency.

        Parameters
        ----------
        player_id : int
            The opponent's player ID.

        Returns
        -------
        Dict[str, float]
            Dictionary with tendency metrics.
        """
        bids = self.pattern_analyzer.bidding_patterns.get(player_id, [])
        
        if not bids:
            return {
                "aggression": 0.5,
                "frequency": 0.0,
                "tendency": "unknown",
            }

        aggression = self.pattern_analyzer.aggression_scores.get(player_id, 0.5)
        frequency = len(bids)
        
        if aggression > 0.6:
            tendency = "aggressive"
        elif aggression < 0.4:
            tendency = "conservative"
        else:
            tendency = "balanced"

        return {
            "aggression": aggression,
            "frequency": frequency,
            "tendency": tendency,
        }

