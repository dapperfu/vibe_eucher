"""Player classes for Euchre game."""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.cards import Card, Suit


class Player(ABC):
    """Base class for all players."""

    def __init__(self, name: str, player_id: int) -> None:
        """
        Initialize a player.

        Parameters
        ----------
        name : str
            The player's name.
        player_id : int
            The player's ID (0-3).
        """
        self.name: str = name
        self.player_id: int = player_id
        self.hand: List[Card] = []
        self.team: int = player_id % 2  # Players 0,2 are team 0; 1,3 are team 1

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

    @abstractmethod
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
        pass

    @abstractmethod
    def decide_call_trump(
        self, turned_card: Card, trump_suit: Optional[Suit]
    ) -> Optional[Suit]:
        """
        Decide which suit to call as trump (or pass).

        Parameters
        ----------
        turned_card : Card
            The card that was turned up (cannot be chosen).
        trump_suit : Optional[Suit]
            Current trump suit if already determined.

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        pass

    @abstractmethod
    def choose_card_to_discard(self) -> Card:
        """
        Choose a card to discard (dealer only, after ordering up).

        Returns
        -------
        Card
            The card to discard.
        """
        pass

    @abstractmethod
    def play_card(
        self,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
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

        Returns
        -------
        Card
            The card to play.
        """
        pass

    def __repr__(self) -> str:
        """
        Return string representation of the player.

        Returns
        -------
        str
            String representation.
        """
        return f"{self.__class__.__name__}({self.name}, id={self.player_id})"


class HumanPlayer(Player):
    """Human player that gets input from the user."""

    def __init__(self, name: str, player_id: int) -> None:
        """
        Initialize a human player.

        Parameters
        ----------
        name : str
            The player's name.
        player_id : int
            The player's ID.
        """
        super().__init__(name, player_id)
        self._ui = None  # Will be set by game

    def set_ui(self, ui) -> None:
        """
        Set the UI object for user interaction.

        Parameters
        ----------
        ui
            The UI object.
        """
        self._ui = ui

    def decide_order_up(
        self, turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Get user input for ordering up.

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
        if self._ui is None:
            raise RuntimeError("UI not set for human player")
        return self._ui.get_order_up_decision(self, turned_card, dealer_id)

    def decide_call_trump(
        self, turned_card: Card, trump_suit: Optional[Suit]
    ) -> Optional[Suit]:
        """
        Get user input for calling trump.

        Parameters
        ----------
        turned_card : Card
            The card that was turned up.
        trump_suit : Optional[Suit]
            Current trump suit if already determined.

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        if self._ui is None:
            raise RuntimeError("UI not set for human player")
        return self._ui.get_call_trump_decision(self, turned_card)

    def choose_card_to_discard(self) -> Card:
        """
        Get user input for discarding a card.

        Returns
        -------
        Card
            The card to discard.
        """
        if self._ui is None:
            raise RuntimeError("UI not set for human player")
        return self._ui.get_discard_decision(self)

    def play_card(
        self,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
    ) -> Card:
        """
        Get user input for playing a card.

        Parameters
        ----------
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
        if self._ui is None:
            raise RuntimeError("UI not set for human player")
        return self._ui.get_play_card_decision(self, led_suit, trump_suit, trick_cards)


class AIPlayer(Player):
    """AI player that makes decisions automatically."""

    def __init__(self, name: str, player_id: int, ai_decision_maker) -> None:
        """
        Initialize an AI player.

        Parameters
        ----------
        name : str
            The player's name.
        player_id : int
            The player's ID.
        ai_decision_maker
            The AI decision maker object.
        """
        super().__init__(name, player_id)
        self.ai = ai_decision_maker

    def decide_order_up(
        self, turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Use AI to decide whether to order up.

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
        return self.ai.decide_order_up(self, turned_card, dealer_id)

    def decide_call_trump(
        self, turned_card: Card, trump_suit: Optional[Suit]
    ) -> Optional[Suit]:
        """
        Use AI to decide which suit to call as trump.

        Parameters
        ----------
        turned_card : Card
            The card that was turned up.
        trump_suit : Optional[Suit]
            Current trump suit if already determined.

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        return self.ai.decide_call_trump(self, turned_card)

    def choose_card_to_discard(self) -> Card:
        """
        Use AI to choose a card to discard.

        Returns
        -------
        Card
            The card to discard.
        """
        return self.ai.choose_card_to_discard(self)

    def play_card(
        self,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
    ) -> Card:
        """
        Use AI to choose a card to play.

        Parameters
        ----------
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
        return self.ai.play_card(self, led_suit, trump_suit, trick_cards)

