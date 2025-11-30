"""AI decision making for Euchre game."""

import random
from typing import Dict, List, Optional

from typing import TYPE_CHECKING

from eucher.cards import Card, Rank, Suit
from eucher.rules import RulesEngine

if TYPE_CHECKING:
    from eucher.players import Player


class HeuristicWeights:
    """Configuration class for heuristic weights in AI decision making."""

    # Leading strategy weights
    LEAD_TRUMP_PREFERENCE = 1.2
    LEAD_OFFSUIT_ACE = 1.1
    LEAD_OFFSUIT_KING = 1.05

    # Following strategy weights
    FOLLOW_TEAMMATE_WINNING = 0.3  # Low value = duck (play low)
    FOLLOW_OPPONENT_WINNING = 1.5  # High value = try to win

    # Last play strategy weights
    LAST_PLAY_DUCK = 0.2
    LAST_PLAY_WIN = 2.0

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


class AIDecisionMaker:
    """Makes AI decisions for trump selection and card play."""

    def __init__(self) -> None:
        """Initialize the AI decision maker."""
        self.rules = RulesEngine()

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit] = None
    ) -> bool:
        """
        Decide whether to order up the turned card using weighted heuristics.

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
            return False  # Trump already determined

        potential_trump = turned_card.suit

        # Evaluate hand strength for this potential trump
        hand_strength = self._evaluate_hand_strength(player.hand, potential_trump)

        # Count trump cards with power weighting
        trump_count = self._count_trump_cards(player.hand, potential_trump)

        # Base threshold for ordering up
        base_threshold = 150.0

        # Adjust threshold based on position (being dealer's partner is better)
        is_dealer_partner = (player.player_id % 2) == (dealer_id % 2)
        if is_dealer_partner:
            base_threshold -= 20.0  # More willing to order up as dealer's partner

        # Order up if hand strength exceeds threshold
        if hand_strength >= base_threshold:
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
        trump_suit: Optional[Suit] = None,
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Decide which suit to call as trump using weighted scoring.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up (cannot be chosen).
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit (cannot pass).

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        if trump_suit is not None:
            if must_choose:
                return trump_suit
            return None  # Trump already determined

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
        threshold = 140.0

        # If must choose, lower threshold significantly
        if must_choose:
            threshold = 80.0

        # Call trump if score exceeds threshold
        if best_suit is not None and best_score >= threshold:
            return best_suit

        # If must choose, return best suit even if weak
        if must_choose and best_suit is not None:
            return best_suit

        return None

    def choose_card_to_discard(self, player: "Player") -> Card:
        """
        Choose a card to discard (dealer only) using power-based heuristics.

        Parameters
        ----------
        player : Player
            The dealer player.

        Returns
        -------
        Card
            The card to discard.
        """
        if not player.hand:
            raise ValueError("Player has no cards to discard")

        # Need to know trump suit to make good decision
        # For now, discard lowest power card
        # In a full implementation, we'd have access to trump_suit here
        # For simplicity, discard lowest rank card
        # (This will be improved when we have trump context)
        lowest_card = min(player.hand, key=lambda c: c.rank.value)
        return lowest_card

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Choose a card to play in a trick using heuristic-based decision making.

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
            # Should not happen, but fallback
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

    def _count_trump_cards(self, hand: List[Card], trump_suit: Suit) -> int:
        """
        Count how many trump cards are in a hand.

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        int
            Number of trump cards.
        """
        count = 0
        for card in hand:
            # Right Bower (Jack of trump suit)
            if card.rank == Rank.JACK and card.suit == trump_suit:
                count += 1
            # Left Bower (Jack of same color as trump)
            elif card.rank == Rank.JACK:
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    count += 1
            # Regular trump suit card
            elif card.suit == trump_suit:
                count += 1
        return count

    def _has_bower(self, hand: List[Card], trump_suit: Suit) -> bool:
        """
        Check if hand has a bower (Right or Left).

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        bool
            True if hand has a bower, False otherwise.
        """
        for card in hand:
            if card.rank == Rank.JACK:
                # Right Bower
                if card.suit == trump_suit:
                    return True
                # Left Bower
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    return True
        return False

    def _has_strong_trump(self, hand: List[Card], trump_suit: Suit) -> bool:
        """
        Check if hand has strong trump cards (Ace or King).

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        bool
            True if hand has strong trump, False otherwise.
        """
        for card in hand:
            if card.suit == trump_suit and card.rank in [Rank.ACE, Rank.KING]:
                return True
        return False

    def _card_value(self, card: Card, trump_suit: Optional[Suit], led_suit: Optional[Suit]) -> int:
        """
        Get the value of a card for comparison.

        Parameters
        ----------
        card : Card
            The card to value.
        trump_suit : Optional[Suit]
            The trump suit.
        led_suit : Optional[Suit]
            The led suit.

        Returns
        -------
        int
            Card value (higher is better).
        """
        if trump_suit is not None:
            # Right Bower (Jack of trump suit) is highest
            if card.rank == Rank.JACK and card.suit == trump_suit:
                return 7
            # Left Bower (Jack of same color as trump) is second
            if card.rank == Rank.JACK:
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    return 6
            # Regular trump suit cards
            if card.suit == trump_suit:
                rank_map = {
                    Rank.ACE: 5,
                    Rank.KING: 4,
                    Rank.QUEEN: 3,
                    Rank.TEN: 2,
                    Rank.NINE: 1,
                }
                return rank_map.get(card.rank, 0)
        return card.rank.value

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
                (card, power) for card, power in card_powers if self._is_trump_card(card, trump_suit)
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
            return max(valid_cards, key=lambda c: self._calculate_card_power(c, trump_suit))

        # Check if teammate is winning
        is_teammate_winning = self._is_teammate(player, current_winner_id)

        if is_teammate_winning:
            # Teammate is winning: duck (play lowest card)
            return min(valid_cards, key=lambda c: self._calculate_card_power(c, trump_suit))
        else:
            # Opponent is winning: try to win with lowest winning card
            winning_cards = []
            for card in valid_cards:
                if self._can_win_trick(card, trick_cards, led_suit, trump_suit):
                    winning_cards.append(card)

            if winning_cards:
                # Win with lowest power winning card
                return min(winning_cards, key=lambda c: self._calculate_card_power(c, trump_suit))
            else:
                # Can't win: play lowest card to preserve high cards
                return min(valid_cards, key=lambda c: self._calculate_card_power(c, trump_suit))

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
            return max(valid_cards, key=lambda c: self._calculate_card_power(c, trump_suit))

        # Check if teammate is winning
        is_teammate_winning = self._is_teammate(player, current_winner_id)

        if is_teammate_winning:
            # Teammate is winning: duck (play lowest card)
            return min(valid_cards, key=lambda c: self._calculate_card_power(c, trump_suit))
        else:
            # Opponent is winning: win if possible with minimal card
            winning_cards = []
            for card in valid_cards:
                if self._can_win_trick(card, trick_cards, led_suit, trump_suit):
                    winning_cards.append(card)

            if winning_cards:
                # Win with lowest power winning card
                return min(winning_cards, key=lambda c: self._calculate_card_power(c, trump_suit))
            else:
                # Can't win: play lowest card
                return min(valid_cards, key=lambda c: self._calculate_card_power(c, trump_suit))

    def _is_trump_card(self, card: Card, trump_suit: Suit) -> bool:
        """
        Check if a card is a trump card.

        Parameters
        ----------
        card : Card
            The card to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        bool
            True if card is trump, False otherwise.
        """
        # Right Bower
        if card.rank == Rank.JACK and card.suit == trump_suit:
            return True
        # Left Bower
        if card.rank == Rank.JACK:
            trump_card = Card(trump_suit, Rank.ACE)
            if card.is_same_color(trump_card):
                return True
        # Regular trump
        return card.suit == trump_suit

    def _can_win_trick(
        self, card: Card, trick_cards: List[Card], led_suit: Suit, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Check if a card can win the current trick.

        Parameters
        ----------
        card : Card
            The card to check.
        trick_cards : List[Card]
            Cards already played.
        led_suit : Suit
            The led suit.
        trump_suit : Optional[Suit]
            The trump suit.

        Returns
        -------
        bool
            True if card can win, False otherwise.
        """
        if not trick_cards:
            return True

        current_winner_idx = self._find_current_winner(trick_cards, led_suit, trump_suit)
        if current_winner_idx is None:
            return True

        current_winner = trick_cards[current_winner_idx]
        comparison = card.compare_to(current_winner, trump_suit, led_suit)
        return comparison > 0

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
            power = self._calculate_card_power(card, trump_suit)
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
            total_power += 20.0

        # Bonus for multiple trump cards
        if trump_count >= 3:
            total_power += 15.0
        elif trump_count >= 2:
            total_power += 10.0

        return total_power

    def decide_going_alone(self, player: "Player", trump_suit: Suit) -> bool:
        """
        Decide whether to go alone using AI decision making.

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

    def _count_off_suit_aces(self, hand: List[Card], trump_suit: Suit) -> int:
        """
        Count off-suit Aces (Aces that are not trump).

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        int
            Number of off-suit Aces.
        """
        count = 0
        for card in hand:
            # Skip trump cards
            if self._is_trump_card(card, trump_suit):
                continue
            # Count Aces
            if card.rank == Rank.ACE:
                count += 1
        return count

    def decide_trade_in(self, player: "Player", eligible_cards: List[Card]) -> bool:
        """
        Decide whether to trade-in using AI heuristics.

        Uses similar logic to heuristic player - trade-in if hand is weak.

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
        # Calculate current hand strength
        current_hand_strength = sum(card.rank.value for card in player.hand)

        # Count high cards (Ace, King, Queen) in current hand
        high_card_count = sum(
            1 for card in player.hand if card.rank in (Rank.ACE, Rank.KING, Rank.QUEEN)
        )

        # Count Jacks (potential bowers)
        jack_count = sum(1 for card in player.hand if card.rank == Rank.JACK)

        # Trade-in if hand is weak
        if high_card_count <= 2 and jack_count == 0:
            return True

        # Trade-in if hand strength is very low
        if current_hand_strength <= 50:
            return True

        return False



from typing import TYPE_CHECKING, Optional

from eucher.cards import Card, Suit
from eucher.players.computer.base import ComputerPlayer

if TYPE_CHECKING:
    from eucher.players.base import Player


class AIPlayer(ComputerPlayer):
    """AI-based player using the existing AIDecisionMaker (formerly AIBasedProfile)."""

    def __init__(self, ai_decision_maker) -> None:
        """
        Initialize an AI-based player.

        Parameters
        ----------
        ai_decision_maker
            The AIDecisionMaker instance to use.
        """
        super().__init__()
        self.ai = ai_decision_maker

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """Use AI to decide whether to order up."""
        return self.ai.decide_order_up(player, turned_card, dealer_id, trump_suit)

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """Use AI to decide which suit to call as trump."""
        return self.ai.decide_call_trump(player, turned_card, trump_suit, must_choose)

    def choose_card_to_discard(self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None) -> Card:
        """Use AI to choose a card to discard."""
        return self.ai.choose_card_to_discard(player)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: list[Card],
        trick_player_ids: list[int],
    ) -> Card:
        """Use AI to choose a card to play."""
        return self.ai.play_card(player, led_suit, trump_suit, trick_cards, trick_player_ids)

    def decide_trade_in(self, player: "Player", eligible_cards: List[Card]) -> bool:
        """Use AI to decide whether to trade-in."""
        return self.ai.decide_trade_in(player, eligible_cards)

    def get_order_up_probabilities(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> Dict[bool, float]:
        """
        Return probability distribution for order up decision based on hand strength.

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
        Dict[bool, float]
            Dictionary mapping decision (True=order up, False=pass) to probability.
        """
        if trump_suit is not None:
            return {False: 1.0, True: 0.0}

        potential_trump = turned_card.suit
        hand_strength = self.ai._evaluate_hand_strength(player.hand, potential_trump)

        # Base threshold for ordering up
        base_threshold = 150.0

        # Adjust threshold based on position (being dealer's partner is better)
        is_dealer_partner = (player.player_id % 2) == (dealer_id % 2)
        if is_dealer_partner:
            base_threshold -= 20.0

        # Convert hand strength to probability using sigmoid-like function
        # Higher hand strength = higher probability of ordering up
        diff = hand_strength - base_threshold
        # Use sigmoid to convert difference to probability
        import math
        prob_order_up = 1.0 / (1.0 + math.exp(-diff / 20.0))  # Scale factor of 20

        return {True: float(prob_order_up), False: float(1.0 - prob_order_up)}

    def get_call_trump_probabilities(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Dict[Optional[Suit], float]:
        """
        Return probability distribution for call trump decision based on hand strength scores.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up (cannot be chosen).
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit (cannot pass).

        Returns
        -------
        Dict[Optional[Suit], float]
            Dictionary mapping suit (or None for pass) to probability.
        """
        if trump_suit is not None:
            return {trump_suit: 1.0}

        forbidden_suit = turned_card.suit
        suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]

        # Calculate scores for each suit
        suit_scores: Dict[Suit, float] = {}
        for suit in suits:
            if suit == forbidden_suit:
                continue

            hand_strength = self.ai._evaluate_hand_strength(player.hand, suit)
            trump_count = self.ai._count_trump_cards(player.hand, suit)

            score = hand_strength

            # Bonus for having bowers
            if self.ai._has_bower(player.hand, suit):
                score += 30.0

            # Bonus for multiple trump cards
            if trump_count >= 3:
                score += 20.0
            elif trump_count >= 2:
                score += 10.0

            suit_scores[suit] = score

        # Calculate pass score (lower threshold means more likely to pass)
        threshold = 140.0 if not must_choose else 80.0
        pass_score = threshold  # Pass is equivalent to threshold score

        # Convert scores to probabilities using softmax
        all_scores = [pass_score] + [suit_scores[suit] for suit in suit_scores]
        # Normalize scores to positive values for softmax
        min_score = min(all_scores)
        normalized_scores = [s - min_score + 1.0 for s in all_scores]

        # Apply softmax
        import math
        exp_scores = [math.exp(s) for s in normalized_scores]
        total_exp = sum(exp_scores)
        probs = [s / total_exp for s in exp_scores]

        result: Dict[Optional[Suit], float] = {}
        if not must_choose:
            result[None] = float(probs[0])
        for i, suit in enumerate(suit_scores.keys()):
            result[suit] = float(probs[i + (0 if must_choose else 1)])

        return result

    def get_play_card_probabilities(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Dict[Card, float]:
        """
        Return probability distribution for play card decision based on card power scores.

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
            Player IDs who played each card in trick_cards.

        Returns
        -------
        Dict[Card, float]
            Dictionary mapping card to probability (only valid cards).
        """
        valid_cards = self.ai.rules.get_valid_plays(player.hand, led_suit, trump_suit)
        if not valid_cards:
            return {}

        # Determine context for card power calculation
        context = "general"
        if led_suit is None:
            context = "leading"
        elif len(trick_cards) == 3:
            context = "last_play"
        else:
            context = "following"

        # Calculate power scores for each valid card
        card_scores: Dict[Card, float] = {}
        for card in valid_cards:
            power = self.ai._calculate_card_power(card, trump_suit, context)
            card_scores[card] = power

        # Convert scores to probabilities using softmax
        scores = list(card_scores.values())
        if not scores:
            return {}

        import math
        # Normalize scores to positive values
        min_score = min(scores)
        normalized_scores = [s - min_score + 1.0 for s in scores]

        # Apply softmax
        exp_scores = [math.exp(s) for s in normalized_scores]
        total_exp = sum(exp_scores)
        probs = [s / total_exp for s in exp_scores]

        result: Dict[Card, float] = {}
        for i, card in enumerate(valid_cards):
            result[card] = float(probs[i])

        return result

    def get_discard_probabilities(
        self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
    ) -> Dict[Card, float]:
        """
        Return probability distribution for discard decision based on card values.

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
        Dict[Card, float]
            Dictionary mapping card to probability.
        """
        if not player.hand:
            return {}

        # Calculate discard value for each card (lower is better to discard)
        # Use rank value as proxy (lower rank = more likely to discard)
        card_values: Dict[Card, float] = {}
        for card in player.hand:
            # Higher rank value = less likely to discard
            # Invert so higher values become lower discard probabilities
            value = card.rank.value
            card_values[card] = value

        # Convert to discard probabilities (lower value = higher discard probability)
        # Invert the scores
        max_value = max(card_values.values())
        discard_scores = {card: max_value - value + 1.0 for card, value in card_values.items()}

        # Normalize to probabilities
        total_score = sum(discard_scores.values())
        if total_score > 0:
            result: Dict[Card, float] = {}
            for card in player.hand:
                result[card] = discard_scores[card] / total_score
            return result
        else:
            # Fallback: equal distribution
            prob = 1.0 / len(player.hand)
            return {card: prob for card in player.hand}

    def decide_going_alone(self, player: "Player", trump_suit: Suit) -> bool:
        """Use AI to decide whether to go alone."""
        return self.ai.decide_going_alone(player, trump_suit)
