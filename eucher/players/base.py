"""Base player classes for Euchre game."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from eucher.cards import Card, Suit

if TYPE_CHECKING:
    pass  # Forward reference for Player


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
    def choose_card_to_discard(
        self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
    ) -> Card:
        """
        Choose a card to discard (dealer only, after ordering up).

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
        pass

    @abstractmethod
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
        pass


class Player:
    """Unified player class that uses a PlayerProfile for decision-making."""

    def __init__(self, name: str, player_id: int, profile: PlayerProfile) -> None:
        """
        Initialize a player.

        Parameters
        ----------
        name : str
            The player's name.
        player_id : int
            The player's ID (0-3).
        profile : PlayerProfile
            The decision-making profile for this player.
        """
        self.name: str = name
        self.player_id: int = player_id
        self.hand: List[Card] = []
        self.team: int = player_id % 2  # Players 0,2 are team 0; 1,3 are team 1
        self.profile: PlayerProfile = profile

    def receive_card(self, card: Card) -> None:
        """
        Add a card to the player's hand.

        Parameters
        ----------
        card : Card
            The card to add.
        """
        self.hand.append(card)

    def receive_hand(self, cards: List[Card]) -> None:
        """
        Set the player's hand.

        Parameters
        ----------
        cards : List[Card]
            The cards to set as the hand.
        """
        self.hand = cards.copy()

    def remove_card(self, card: Card) -> None:
        """
        Remove a card from the player's hand.

        Parameters
        ----------
        card : Card
            The card to remove.

        Raises
        ------
        ValueError
            If the card is not in the hand.
        """
        if card not in self.hand:
            raise ValueError(f"Card {card} not in hand")
        self.hand.remove(card)

    def has_card(self, card: Card) -> bool:
        """
        Check if player has a specific card.

        Parameters
        ----------
        card : Card
            The card to check.

        Returns
        -------
        bool
            True if player has the card, False otherwise.
        """
        return card in self.hand

    def decide_order_up(
        self, turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Decide whether to order up the turned card.

        Parameters
        ----------
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
        return self.profile.decide_order_up(self, turned_card, dealer_id, trump_suit)

    def decide_call_trump(
        self, turned_card: Card, trump_suit: Optional[Suit], must_choose: bool = False
    ) -> Optional[Suit]:
        """
        Decide which suit to call as trump (or pass).

        Parameters
        ----------
        turned_card : Card
            The card that was turned up (cannot be chosen).
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit (cannot pass). Used for "screw the dealer" rule.

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass (only if must_choose=False).
        """
        return self.profile.decide_call_trump(self, turned_card, trump_suit, must_choose)

    def choose_card_to_discard(
        self, turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
    ) -> Card:
        """
        Choose a card to discard (dealer only, after ordering up).

        Parameters
        ----------
        turned_card : Optional[Card]
            The card that was ordered up, if available.
        ordered_up_by : Optional[str]
            Name of the player who ordered up, if available.

        Returns
        -------
        Card
            The card to discard.
        """
        return self.profile.choose_card_to_discard(self, turned_card, ordered_up_by)

    def play_card(
        self,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Choose a card to play in a trick.

        Parameters
        ----------
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
        return self.profile.play_card(self, led_suit, trump_suit, trick_cards, trick_player_ids)

    def __repr__(self) -> str:
        """
        Return string representation of the player.

        Returns
        -------
        str
            String representation.
        """
        profile_name = self.profile.__class__.__name__
        return f"Player({self.name}, id={self.player_id}, profile={profile_name})"

