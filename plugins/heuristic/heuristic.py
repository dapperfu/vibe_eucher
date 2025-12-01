"""Heuristic-based player using rule-based logic."""

from typing import TYPE_CHECKING, Dict, List, Optional

from eucher.cards import Card, Rank, Suit
from eucher.players.computer.base import ComputerPlayer

if TYPE_CHECKING:
    from eucher.players.base import Player


class HeuristicWeights:
    """Configuration class for heuristic weights in HeuristicPlayer decision making."""

    # Trump selection thresholds
    ORDER_UP_BASE_THRESHOLD = 150.0
    ORDER_UP_DEALER_PARTNER_BONUS = -20.0  # Lower threshold when dealer's partner
    CALL_TRUMP_BASE_THRESHOLD = 140.0
    CALL_TRUMP_MUST_CHOOSE_THRESHOLD = 80.0

    # Hand strength bonuses
    BOWER_BONUS = 30.0
    MULTIPLE_TRUMP_BONUS_3PLUS = 20.0
    MULTIPLE_TRUMP_BONUS_2PLUS = 10.0
    STRONG_TRUMP_BONUS = 15.0

    # Card power base values
    RIGHT_BOWER_POWER = 100
    LEFT_BOWER_POWER = 90
    TRUMP_ACE_POWER = 80
    TRUMP_KING_POWER = 70
    TRUMP_QUEEN_POWER = 60
    TRUMP_TEN_POWER = 50
    TRUMP_NINE_POWER = 40

    # Off-suit power values (when trump is known)
    OFFSUIT_ACE_POWER = 35
    OFFSUIT_KING_POWER = 30
    OFFSUIT_QUEEN_POWER = 25
    OFFSUIT_JACK_POWER = 20
    OFFSUIT_TEN_POWER = 15
    OFFSUIT_NINE_POWER = 10

    # Leading strategy weights
    LEAD_TRUMP_PREFERENCE = 1.2
    LEAD_OFFSUIT_ACE = 1.1
    LEAD_OFFSUIT_KING = 1.05


class HeuristicPlayer(ComputerPlayer):
    """Heuristic-based player using rule-based logic (formerly SimpleRuleBasedProfile)."""

    def __init__(self) -> None:
        """Initialize a heuristic-based player."""
        super().__init__()

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Decide whether to order up using improved heuristics.

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

        # Count trump cards
        trump_count = self._count_trump_cards(player.hand, potential_trump)

        # Base threshold for ordering up
        threshold = HeuristicWeights.ORDER_UP_BASE_THRESHOLD

        # Adjust threshold based on position (being dealer's partner is better)
        is_dealer_partner = (player.player_id % 2) == (dealer_id % 2)
        if is_dealer_partner:
            threshold += HeuristicWeights.ORDER_UP_DEALER_PARTNER_BONUS

        # Order up if hand strength exceeds threshold
        if hand_strength >= threshold:
            return True

        # Order up if we have 2+ trump cards (conservative threshold)
        if trump_count >= 2:
            return True

        # Order up if we have a bower (very strong)
        if self._has_bower(player.hand, potential_trump):
            return True

        # Order up if we have strong trump (Ace or King) and at least one other trump
        if self._has_strong_trump(player.hand, potential_trump) and trump_count >= 1:
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
        Decide which suit to call as trump using improved heuristics.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up.
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
                score += HeuristicWeights.BOWER_BONUS

            # Bonus for multiple trump cards
            if trump_count >= 3:
                score += HeuristicWeights.MULTIPLE_TRUMP_BONUS_3PLUS
            elif trump_count >= 2:
                score += HeuristicWeights.MULTIPLE_TRUMP_BONUS_2PLUS

            if score > best_score:
                best_score = score
                best_suit = suit

        # Threshold for calling trump
        threshold = HeuristicWeights.CALL_TRUMP_BASE_THRESHOLD

        # If must choose, lower threshold significantly
        if must_choose:
            threshold = HeuristicWeights.CALL_TRUMP_MUST_CHOOSE_THRESHOLD

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
        Choose a card to discard using improved heuristics.

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
        # This considers trump suit if known, otherwise uses rank
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
        Choose a card to play using improved heuristics.

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
            total_power += HeuristicWeights.BOWER_BONUS

        # Bonus for multiple trump cards
        if trump_count >= 3:
            total_power += HeuristicWeights.MULTIPLE_TRUMP_BONUS_3PLUS
        elif trump_count >= 2:
            total_power += HeuristicWeights.MULTIPLE_TRUMP_BONUS_2PLUS

        return total_power

    def _calculate_card_power(
        self, card: Card, trump_suit: Optional[Suit], context: str = "general"
    ) -> float:
        """
        Calculate the power/weight of a card using comprehensive heuristics.

        Parameters
        ----------
        card : Card
            The card to evaluate.
        trump_suit : Optional[Suit]
            The current trump suit, if any.
        context : str
            Context of evaluation: "general", "leading", "following"

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
            power = float(HeuristicWeights.RIGHT_BOWER_POWER)
        elif is_left_bower:
            power = float(HeuristicWeights.LEFT_BOWER_POWER)
        elif card.suit == trump_suit:
            # Regular trump suit card
            rank_power_map = {
                Rank.ACE: HeuristicWeights.TRUMP_ACE_POWER,
                Rank.KING: HeuristicWeights.TRUMP_KING_POWER,
                Rank.QUEEN: HeuristicWeights.TRUMP_QUEEN_POWER,
                Rank.TEN: HeuristicWeights.TRUMP_TEN_POWER,
                Rank.NINE: HeuristicWeights.TRUMP_NINE_POWER,
            }
            power = float(rank_power_map.get(card.rank, 0))
        else:
            # Off-suit card
            rank_power_map = {
                Rank.ACE: HeuristicWeights.OFFSUIT_ACE_POWER,
                Rank.KING: HeuristicWeights.OFFSUIT_KING_POWER,
                Rank.QUEEN: HeuristicWeights.OFFSUIT_QUEEN_POWER,
                Rank.JACK: HeuristicWeights.OFFSUIT_JACK_POWER,
                Rank.TEN: HeuristicWeights.OFFSUIT_TEN_POWER,
                Rank.NINE: HeuristicWeights.OFFSUIT_NINE_POWER,
            }
            power = float(rank_power_map.get(card.rank, 0))

        # Apply context-based adjustments
        if context == "leading":
            if card.suit == trump_suit or is_right_bower or is_left_bower:
                power *= HeuristicWeights.LEAD_TRUMP_PREFERENCE
            elif card.rank == Rank.ACE:
                power *= HeuristicWeights.LEAD_OFFSUIT_ACE
            elif card.rank == Rank.KING:
                power *= HeuristicWeights.LEAD_OFFSUIT_KING

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

    def _find_current_winner(
        self, trick_cards: List[Card], led_suit: Suit, trump_suit: Optional[Suit]
    ) -> Optional[int]:
        """
        Find the current winner of the trick.

        Parameters
        ----------
        trick_cards : List[Card]
            Cards played so far.
        led_suit : Suit
            The led suit.
        trump_suit : Optional[Suit]
            The trump suit.

        Returns
        -------
        Optional[int]
            Index of winning card, or None if no cards.
        """
        if not trick_cards:
            return None

        winner_idx = 0
        for i in range(1, len(trick_cards)):
            comparison = trick_cards[i].compare_to(trick_cards[winner_idx], trump_suit, led_suit)
            if comparison > 0:
                winner_idx = i
        return winner_idx

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
        winner_idx = self._find_current_winner(trick_cards, led_suit, trump_suit)
        if winner_idx is None:
            return None
        return trick_player_ids[winner_idx]

    def _decide_lead(
        self, player: "Player", valid_cards: List[Card], trump_suit: Optional[Suit]
    ) -> Card:
        """
        Decide which card to play when leading a trick.

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
        # Calculate power for each card in leading context
        card_powers = [
            (card, self._calculate_card_power(card, trump_suit, context="leading"))
            for card in valid_cards
        ]

        # Count trump cards in hand
        trump_count = 0
        if trump_suit is not None:
            trump_count = self._count_trump_cards(player.hand, trump_suit)

        # Strategy: If we have multiple strong trump, lead with one
        # Otherwise, lead with strong off-suit to draw out trump
        if trump_suit is not None and trump_count >= 2:
            # Prefer leading with strong trump
            trump_cards = [
                (card, power)
                for card, power in card_powers
                if self._is_trump_card(card, trump_suit)
            ]
            if trump_cards:
                # Lead with highest power trump
                return max(trump_cards, key=lambda x: x[1])[0]

        # Lead with highest power card (weighted for context)
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
            # Teammate is winning: duck (play lowest card)
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
            # Teammate is winning: duck (play lowest card)
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

    def _should_be_aggressive(
        self, player: "Player", team_scores: Optional[List[int]] = None
    ) -> bool:
        """
        Determine if aggressive play is needed based on score situation.

        Parameters
        ----------
        player : Player
            The player making the decision.
        team_scores : Optional[List[int]]
            Current team scores [team0, team1]. If None, assumes neutral.

        Returns
        -------
        bool
            True if aggressive play is recommended.
        """
        if team_scores is None:
            return False  # Neutral if scores unknown

        player_team_score = team_scores[player.team]
        opponent_team_score = team_scores[1 - player.team]

        # Be aggressive if behind by 2+ points
        if opponent_team_score - player_team_score >= 2:
            return True

        # Be aggressive if opponent is close to winning (8+ points)
        if opponent_team_score >= 8:
            return True

        return False

    def decide_going_alone(self, player: "Player", trump_suit: Suit) -> bool:
        """
        Decide whether to go alone using heuristic evaluation.

        Parameters
        ----------
        player : Player
            The player making the decision.
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

        # High threshold for going alone - need very strong hand
        # Typically need top 3 trump cards (Right Bower, Left Bower, Ace) + off-suit Ace
        alone_threshold = 250.0

        # Check for top 3 trump cards
        has_top_trump = self._has_top_trump_cards(player.hand, trump_suit)

        # Check for off-suit Aces
        off_suit_aces = self._count_off_suit_aces(player.hand, trump_suit)

        # Go alone if:
        # 1. Flush in trump (already checked above)
        # 2. Very high hand strength AND has top trump cards AND off-suit Aces
        if hand_strength >= alone_threshold and has_top_trump and off_suit_aces >= 1:
            return True

        # Go alone if we have 4+ trump cards (very strong)
        if trump_count >= 4:
            return True

        return False

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

    def decide_trade_in(self, player: "Player", eligible_cards: List[Card]) -> bool:
        """
        Decide whether to trade-in using heuristics.

        Trade-in if the current hand is weak (low overall strength)
        and trading in would likely improve it.

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
        # Use average rank value as a proxy
        current_hand_strength = sum(card.rank.value for card in player.hand)

        # The eligible cards are all 9s or 10s, so they're weak
        # Removing them and getting 3 random cards from kitty is likely beneficial
        # if current hand is weak

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

        # Otherwise, keep the hand
        return False
