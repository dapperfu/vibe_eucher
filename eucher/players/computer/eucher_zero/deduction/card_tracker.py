"""Card tracker for perfect memory."""

from enum import Enum
from typing import Dict, List, Optional

from eucher.cards import Card


class CardLocation(Enum):
    """Card location types."""

    IN_HAND = "in_hand"
    PLAYED = "played"
    DISCARDED = "discarded"
    UNKNOWN = "unknown"


class CardTracker:
    """Track all 24 cards with perfect memory."""

    NUM_CARDS = 24

    def __init__(self) -> None:
        """Initialize card tracker."""
        # Map card -> location
        self.card_locations: Dict[Card, CardLocation] = {}
        # Map location -> list of cards
        self.location_cards: Dict[CardLocation, List[Card]] = {
            CardLocation.IN_HAND: [],
            CardLocation.PLAYED: [],
            CardLocation.DISCARDED: [],
            CardLocation.UNKNOWN: [],
        }

    def initialize(self, player_hand: List[Card], turned_card: Optional[Card] = None) -> None:
        """
        Initialize tracker with known cards.

        Parameters
        ----------
        player_hand : List[Card]
            Cards in player's hand.
        turned_card : Optional[Card]
            Turned card if available.
        """
        # All cards start as unknown
        from eucher.cards import Deck

        deck = Deck()
        all_cards = deck.cards.copy()

        for card in all_cards:
            self.card_locations[card] = CardLocation.UNKNOWN
            self.location_cards[CardLocation.UNKNOWN].append(card)

        # Mark player's hand
        for card in player_hand:
            self._move_card(card, CardLocation.IN_HAND)

        # Mark turned card as visible
        if turned_card:
            self._move_card(turned_card, CardLocation.PLAYED)

    def _move_card(self, card: Card, new_location: CardLocation) -> None:
        """
        Move card to new location.

        Parameters
        ----------
        card : Card
            Card to move.
        new_location : CardLocation
            New location.
        """
        old_location = self.card_locations.get(card, CardLocation.UNKNOWN)

        # Remove from old location
        if card in self.location_cards[old_location]:
            self.location_cards[old_location].remove(card)

        # Add to new location
        self.card_locations[card] = new_location
        if card not in self.location_cards[new_location]:
            self.location_cards[new_location].append(card)

    def mark_card_played(self, card: Card) -> None:
        """
        Mark card as played.

        Parameters
        ----------
        card : Card
            Card that was played.
        """
        self._move_card(card, CardLocation.PLAYED)

    def mark_card_discarded(self, card: Card) -> None:
        """
        Mark card as discarded.

        Parameters
        ----------
        card : Card
            Card that was discarded.
        """
        self._move_card(card, CardLocation.DISCARDED)

    def get_unknown_cards(self) -> List[Card]:
        """
        Get list of unknown cards.

        Returns
        -------
        List[Card]
            Unknown cards.
        """
        return self.location_cards[CardLocation.UNKNOWN].copy()

    def get_location(self, card: Card) -> CardLocation:
        """
        Get location of a card.

        Parameters
        ----------
        card : Card
            Card to check.

        Returns
        -------
        CardLocation
            Card location.
        """
        return self.card_locations.get(card, CardLocation.UNKNOWN)

