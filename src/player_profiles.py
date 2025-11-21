"""Player profile system for pluggable decision-making backends."""

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from src.cards import Card, Rank, Suit
from src.rules import RulesEngine

if TYPE_CHECKING:
    from src.players import Player


# Import TUI for HumanProfile
try:
    from src.tui import TextTUI
except ImportError:
    TextTUI = None  # type: ignore


class PlayerProfile(ABC):
    """Abstract base class for player decision-making profiles."""

    @abstractmethod
    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
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
        pass

    @abstractmethod
    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
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
        must_choose : bool
            If True, player must choose a suit (cannot pass).
            Used for "screw the dealer" rule.

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass (only if must_choose=False).

        Raises
        ------
        ValueError
            If must_choose=True and None is returned.
        """
        pass

    @abstractmethod
    def choose_card_to_discard(self, player: "Player") -> Card:
        """
        Choose a card to discard (dealer only, after ordering up).

        Parameters
        ----------
        player : Player
            The dealer player.

        Returns
        -------
        Card
            The card to discard.
        """
        pass

    @abstractmethod
    def play_card(
        self,
        player: "Player",
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
        pass


class HumanProfile(PlayerProfile):
    """Profile for human players using TUI (Terminal User Interface)."""

    def __init__(self, tui: Optional[TextTUI] = None) -> None:
        """
        Initialize a human profile.

        Parameters
        ----------
        tui : Optional[TextTUI]
            The TUI object. If None, will raise error when used.
        """
        self.tui = tui

    def set_tui(self, tui: TextTUI) -> None:
        """
        Set the TUI object.

        Parameters
        ----------
        tui : TextTUI
            The TUI object.
        """
        self.tui = tui

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Get user input for ordering up.

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
        if self.tui is None:
            raise RuntimeError("TUI not set for human profile")
        return self.tui.get_order_up_decision(player, turned_card, dealer_id)

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Get user input for calling trump.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up.
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit (cannot pass).

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        if self.tui is None:
            raise RuntimeError("TUI not set for human profile")
        result = self.tui.get_call_trump_decision(player, turned_card, must_choose)
        if must_choose and result is None:
            raise ValueError("Must choose a suit when must_choose=True")
        return result

    def choose_card_to_discard(self, player: "Player") -> Card:
        """
        Get user input for discarding a card.

        Parameters
        ----------
        player : Player
            The dealer player.

        Returns
        -------
        Card
            The card to discard.
        """
        if self.tui is None:
            raise RuntimeError("TUI not set for human profile")
        return self.tui.get_discard_decision(player)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
    ) -> Card:
        """
        Get user input for playing a card.

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
        if self.tui is None:
            raise RuntimeError("TUI not set for human profile")
        return self.tui.get_play_card_decision(player, led_suit, trump_suit, trick_cards)


class SimpleRuleBasedProfile(PlayerProfile):
    """Simple rule-based profile using if/else logic."""

    def __init__(self) -> None:
        """Initialize a simple rule-based profile."""
        self.rules = RulesEngine()

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Decide whether to order up using simple rules.

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
        trump_count = self._count_trump_cards(player.hand, potential_trump)

        # Order up if we have 2+ trump cards
        if trump_count >= 2:
            return True

        # Order up if we have Right or Left Bower
        if self._has_bower(player.hand, potential_trump):
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
        Decide which suit to call as trump using simple rules.

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
                # Must return something, but trump already set
                return trump_suit
            return None

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

        # If must choose, return best suit even if weak
        if must_choose and best_suit is not None:
            return best_suit

        return None

    def choose_card_to_discard(self, player: "Player") -> Card:
        """
        Choose a card to discard using simple rules.

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
        # Discard the lowest card
        return min(player.hand, key=lambda c: c.rank.value)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
    ) -> Card:
        """
        Choose a card to play using simple rules.

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
            return player.hand[0]

        # If leading, play highest card
        if led_suit is None:
            return max(valid_cards, key=lambda c: self._card_value(c, trump_suit))

        # Try to win with lowest winning card, or play lowest
        winning_cards = []
        for card in valid_cards:
            if self._can_win_trick(card, trick_cards, led_suit, trump_suit):
                winning_cards.append(card)

        if winning_cards:
            return min(winning_cards, key=lambda c: self._card_value(c, trump_suit))

        # Can't win, play lowest
        return min(valid_cards, key=lambda c: self._card_value(c, trump_suit))

    def _count_trump_cards(self, hand: List[Card], trump_suit: Suit) -> int:
        """Count trump cards in hand."""
        count = 0
        for card in hand:
            if card.rank == Rank.JACK and card.suit == trump_suit:
                count += 1
            elif card.rank == Rank.JACK:
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    count += 1
            elif card.suit == trump_suit:
                count += 1
        return count

    def _has_bower(self, hand: List[Card], trump_suit: Suit) -> bool:
        """Check if hand has a bower."""
        for card in hand:
            if card.rank == Rank.JACK:
                if card.suit == trump_suit:
                    return True
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    return True
        return False

    def _card_value(self, card: Card, trump_suit: Optional[Suit]) -> int:
        """Get card value for comparison."""
        if trump_suit is not None:
            if card.rank == Rank.JACK and card.suit == trump_suit:
                return 7
            if card.rank == Rank.JACK:
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    return 6
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

    def _can_win_trick(
        self, card: Card, trick_cards: List[Card], led_suit: Suit, trump_suit: Optional[Suit]
    ) -> bool:
        """Check if card can win the trick."""
        if not trick_cards:
            return True

        current_winner = trick_cards[0]
        for c in trick_cards[1:]:
            if c.compare_to(current_winner, trump_suit, led_suit) > 0:
                current_winner = c

        return card.compare_to(current_winner, trump_suit, led_suit) > 0


class AIBasedProfile(PlayerProfile):
    """Profile that uses the existing AIDecisionMaker."""

    def __init__(self, ai_decision_maker) -> None:
        """
        Initialize an AI-based profile.

        Parameters
        ----------
        ai_decision_maker
            The AIDecisionMaker instance to use.
        """
        self.ai = ai_decision_maker

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Use AI to decide whether to order up.

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
        return self.ai.decide_order_up(player, turned_card, dealer_id, trump_suit)

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Use AI to decide which suit to call as trump.

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
        return self.ai.decide_call_trump(player, turned_card, trump_suit, must_choose)

    def choose_card_to_discard(self, player: "Player") -> Card:
        """
        Use AI to choose a card to discard.

        Parameters
        ----------
        player : Player
            The dealer player.

        Returns
        -------
        Card
            The card to discard.
        """
        return self.ai.choose_card_to_discard(player)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
    ) -> Card:
        """
        Use AI to choose a card to play.

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
        return self.ai.play_card(player, led_suit, trump_suit, trick_cards)


class RandomProfile(PlayerProfile):
    """Profile that makes all decisions randomly."""

    def __init__(self) -> None:
        """Initialize a random profile."""
        self.rules = RulesEngine()

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Randomly decide whether to order up the turned card.

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
            Randomly True to order up, False to pass.
        """
        return random.choice([True, False])

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Randomly decide which suit to call as trump (or pass).

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
            Randomly chosen suit, or None to pass (only if must_choose=False).
        """
        if trump_suit is not None:
            return None

        forbidden_suit = turned_card.suit
        available_suits = [suit for suit in Suit if suit != forbidden_suit]

        if must_choose:
            # Must choose a suit
            return random.choice(available_suits)
        else:
            # Can pass or choose a suit
            choices: List[Optional[Suit]] = [None] + available_suits
            return random.choice(choices)

    def choose_card_to_discard(self, player: "Player") -> Card:
        """
        Randomly choose a card to discard.

        Parameters
        ----------
        player : Player
            The dealer player.

        Returns
        -------
        Card
            A randomly chosen card from the hand.
        """
        if not player.hand:
            raise ValueError("Player has no cards to discard")
        return random.choice(player.hand)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
    ) -> Card:
        """
        Randomly choose a card to play from valid plays.

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
            A randomly chosen valid card to play.
        """
        valid_cards = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)

        if not valid_cards:
            # Fallback if no valid cards (should not happen)
            return player.hand[0]

        return random.choice(valid_cards)
