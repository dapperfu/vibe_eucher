"""Deck management and card operations for the euchre game."""

from typing import List
from ..models import Card, Suit, Rank


class Deck:
    """Manages a deck of cards for euchre."""
    
    def __init__(self) -> None:
        """Initialize a standard euchre deck (24 cards, 9-A of each suit)."""
        self.cards: List[Card] = []
        self._initialize_deck()
    
    def _initialize_deck(self) -> None:
        """Create the standard euchre deck."""
        for suit in Suit:
            for rank in [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]:
                self.cards.append(Card(suit=suit, rank=rank))
    
    def shuffle(self) -> None:
        """Shuffle the deck."""
        import random
        random.shuffle(self.cards)
    
    def deal_cards(self, num_players: int, cards_per_player: int = 5) -> List[List[Card]]:
        """Deal cards to players.
        
        Parameters
        ----------
        num_players : int
            Number of players to deal to
        cards_per_player : int
            Number of cards per player
            
        Returns
        -------
        List[List[Card]]
            List of hands for each player
        """
        hands = [[] for _ in range(num_players)]
        
        for i in range(cards_per_player):
            for j in range(num_players):
                if self.cards:
                    hands[j].append(self.cards.pop())
        
        return hands
    
    def draw_top_card(self) -> Card:
        """Draw the top card from the deck.
        
        Returns
        -------
        Card
            The top card
            
        Raises
        ------
        ValueError
            If deck is empty
        """
        if not self.cards:
            raise ValueError("Cannot draw from empty deck")
        return self.cards.pop()
    
    def get_remaining_cards(self) -> List[Card]:
        """Get all remaining cards in the deck.
        
        Returns
        -------
        List[Card]
            List of remaining cards
        """
        return self.cards.copy()
    
    def reset(self) -> None:
        """Reset the deck to its initial state."""
        self.cards.clear()
        self._initialize_deck()
    
    def __len__(self) -> int:
        """Return the number of cards in the deck."""
        return len(self.cards)
    
    def is_empty(self) -> bool:
        """Check if the deck is empty."""
        return len(self.cards) == 0 