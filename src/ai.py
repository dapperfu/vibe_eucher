"""AI decision making for Euchre game."""

import random
from typing import List, Optional

from src.cards import Card, Rank, Suit
from src.players import AIPlayer, Player
from src.rules import RulesEngine


class AIDecisionMaker:
    """Makes AI decisions for trump selection and card play."""

    def __init__(self) -> None:
        """Initialize the AI decision maker."""
        self.rules = RulesEngine()

    def decide_order_up(
        self, player: Player, turned_card: Card, dealer_id: int, trump_suit: Optional[Suit] = None
    ) -> bool:
        """
        Decide whether to order up the turned card.

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
        trump_count = self._count_trump_cards(player.hand, potential_trump)

        # Order up if we have 2+ trump cards
        if trump_count >= 2:
            return True

        # Order up if we have Right or Left Bower
        if self._has_bower(player.hand, potential_trump):
            return True

        # Order up if we have strong hand (Ace or King of trump)
        if self._has_strong_trump(player.hand, potential_trump):
            return True

        return False

    def decide_call_trump(
        self, player: Player, turned_card: Card, trump_suit: Optional[Suit] = None
    ) -> Optional[Suit]:
        """
        Decide which suit to call as trump (or pass).

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up (cannot be chosen).
        trump_suit : Optional[Suit]
            Current trump suit if already determined.

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        if trump_suit is not None:
            return None  # Trump already determined

        forbidden_suit = turned_card.suit
        suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]

        best_suit = None
        best_count = 0

        for suit in suits:
            if suit == forbidden_suit:
                continue
            count = self._count_trump_cards(player.hand, suit)
            if count > best_count:
                best_count = count
                best_suit = suit

        # Call trump if we have at least 2 cards of that suit
        if best_count >= 2:
            return best_suit

        # Call trump if we have a bower
        if best_suit is not None and self._has_bower(player.hand, best_suit):
            return best_suit

        return None

    def choose_card_to_discard(self, player: Player) -> Card:
        """
        Choose a card to discard (dealer only).

        Parameters
        ----------
        player : Player
            The dealer player.

        Returns
        -------
        Card
            The card to discard.
        """
        # Discard the lowest non-trump card, or lowest card if all are trump
        if not player.hand:
            raise ValueError("Player has no cards to discard")

        # Simple strategy: discard the lowest card
        lowest_card = min(player.hand, key=lambda c: c.rank.value)
        return lowest_card

    def play_card(
        self,
        player: Player,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
    ) -> Card:
        """
        Choose a card to play in a trick.

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

        Returns
        -------
        Card
            The card to play.
        """
        valid_cards = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)

        if not valid_cards:
            # Should not happen, but fallback
            return player.hand[0]

        # If leading, play highest card
        if led_suit is None:
            return max(valid_cards, key=lambda c: self._card_value(c, trump_suit, led_suit))

        # Determine if we need to win or can lose
        current_winner = self._find_current_winner(trick_cards, led_suit, trump_suit)
        is_teammate_winning = False
        if current_winner is not None and len(trick_cards) > 0:
            # Check if teammate is winning (simplified - would need player info)
            pass

        # Try to win if teammate is not winning, otherwise play low
        if is_teammate_winning:
            # Play lowest card
            return min(valid_cards, key=lambda c: self._card_value(c, trump_suit, led_suit))

        # Try to win with lowest winning card
        winning_cards = []
        for card in valid_cards:
            if self._can_win_trick(card, trick_cards, led_suit, trump_suit):
                winning_cards.append(card)

        if winning_cards:
            return min(winning_cards, key=lambda c: self._card_value(c, trump_suit, led_suit))

        # Can't win, play lowest
        return min(valid_cards, key=lambda c: self._card_value(c, trump_suit, led_suit))

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

