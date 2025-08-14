"""Deck management and card operations for the euchre game."""

from typing import List, Optional, Dict
from ..models import Card, Suit, Rank


class Deck:
    """Manages a deck of cards for euchre."""
    
    def __init__(self) -> None:
        """Initialize a standard euchre deck (24 cards, 9-A of each suit)."""
        self._cards: List[Card] = []
        self._original_cards: List[Card] = []
        self._initialize_deck()
    
    def _initialize_deck(self) -> None:
        """Create the standard euchre deck."""
        self._cards.clear()
        for suit in Suit:
            for rank in [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]:
                card = Card(suit=suit, rank=rank)
                self._cards.append(card)
        self._original_cards = self._cards.copy()
    
    @property
    def cards(self) -> List[Card]:
        """Get a copy of the current cards in the deck."""
        return self._cards.copy()
    
    @property
    def size(self) -> int:
        """Get the current number of cards in the deck."""
        return len(self._cards)
    
    @property
    def is_empty(self) -> bool:
        """Check if the deck is empty."""
        return len(self._cards) == 0
    
    @property
    def is_full(self) -> bool:
        """Check if the deck has all 24 cards."""
        return len(self._cards) == 24
    
    def shuffle(self) -> None:
        """Shuffle the deck."""
        import random
        random.shuffle(self._cards)
    
    def shuffle_times(self, times: int) -> None:
        """Shuffle the deck multiple times.
        
        Parameters
        ----------
        times : int
            Number of times to shuffle
        """
        for _ in range(times):
            self.shuffle()
    
    def cut(self, position: Optional[int] = None) -> None:
        """Cut the deck at a specific position.
        
        Parameters
        ----------
        position : Optional[int]
            Position to cut at (defaults to random)
        """
        import random
        if position is None:
            position = random.randint(0, len(self._cards))
        
        if 0 <= position <= len(self._cards):
            self._cards = self._cards[position:] + self._cards[:position]
    
    def deal_cards(self, num_players: int, cards_per_player: int = 5) -> List[List[Card]]:
        """Deal cards to players using traditional Euchre pattern.
        
        Traditional Euchre dealing: Deal 2 cards to each player, then 3 cards to each player.
        This gives each player 5 cards and leaves 3 cards in the kitty.
        
        Parameters
        ----------
        num_players : int
            Number of players to deal to
        cards_per_player : int
            Number of cards per player (should be 5 for Euchre)
            
        Returns
        -------
        List[List[Card]]
            List of hands for each player
        """
        if len(self._cards) < num_players * cards_per_player:
            raise ValueError(f"Not enough cards to deal {cards_per_player} to {num_players} players")
        
        hands = [[] for _ in range(num_players)]
        
        # Traditional Euchre dealing: 2, 3, 2, 3, 3, 2, 3, 2 pattern
        if cards_per_player == 5:
            # Traditional Euchre dealing: Deal 2 cards to each player
            for i in range(2):
                for j in range(num_players):
                    if self._cards:
                        hands[j].append(self._cards.pop())
            
            # Second round: Deal 3 cards to each player
            for i in range(3):
                for j in range(num_players):
                    if self._cards:
                        hands[j].append(self._cards.pop())
        else:
            # Fallback to round-robin for non-standard card counts
            for i in range(cards_per_player):
                for j in range(num_players):
                    if self._cards:
                        hands[j].append(self._cards.pop())
        
        return hands
    
    def deal_specific_cards(self, num_players: int, cards_per_player: int = 5) -> List[List[Card]]:
        """Deal cards in a specific pattern (for testing or specific scenarios).
        
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
        
        # Deal in round-robin fashion
        for i in range(cards_per_player):
            for j in range(num_players):
                if self._cards:
                    hands[j].append(self._cards.pop())
        
        return hands
    
    def draw_top_card(self) -> Card:
        """Draw the top card from the deck.
        
        Returns
        -------
        Card
            The top card
            
        Raises
        -------
        ValueError
            If deck is empty
        """
        if not self._cards:
            raise ValueError("Cannot draw from empty deck")
        return self._cards.pop()
    
    def draw_bottom_card(self) -> Card:
        """Draw the bottom card from the deck.
        
        Returns
        -------
        Card
            The bottom card
            
        Raises
        -------
        ValueError
            If deck is empty
        """
        if not self._cards:
            raise ValueError("Cannot draw from empty deck")
        return self._cards.pop(0)
    
    def peek_top_card(self) -> Card:
        """Look at the top card without removing it.
        
        Returns
        -------
        Card
            The top card
            
        Raises
        -------
        ValueError
            If deck is empty
        """
        if not self._cards:
            raise ValueError("Cannot peek at empty deck")
        return self._cards[-1]
    
    def peek_bottom_card(self) -> Card:
        """Look at the bottom card without removing it.
        
        Returns
        -------
        Card
            The bottom card
            
        Raises
        -------
        ValueError
            If deck is empty
        """
        if not self._cards:
            raise ValueError("Cannot peek at empty deck")
        return self._cards[0]
    
    def add_card(self, card: Card, position: str = "top") -> None:
        """Add a card to the deck.
        
        Parameters
        ----------
        card : Card
            The card to add
        position : str
            Where to add the card ("top" or "bottom")
        """
        if position == "top":
            self._cards.append(card)
        elif position == "bottom":
            self._cards.insert(0, card)
        else:
            raise ValueError("Position must be 'top' or 'bottom'")
    
    def remove_card(self, card: Card) -> bool:
        """Remove a specific card from the deck.
        
        Parameters
        ----------
        card : Card
            The card to remove
            
        Returns
        -------
        bool
            True if card was removed, False if not found
        """
        try:
            self._cards.remove(card)
            return True
        except ValueError:
            return False
    
    def get_remaining_cards(self) -> List[Card]:
        """Get all remaining cards in the deck.
        
        Returns
        -------
        List[Card]
            List of remaining cards
        """
        return self._cards.copy()
    
    def reset(self) -> None:
        """Reset the deck to its initial state."""
        self._cards = self._original_cards.copy()
    
    def reset_and_shuffle(self) -> None:
        """Reset the deck and shuffle it."""
        self.reset()
        self.shuffle()
    
    def get_card_count_by_suit(self) -> Dict[Suit, int]:
        """Get the count of cards by suit.
        
        Returns
        -------
        Dict[Suit, int]
            Dictionary mapping suits to card counts
        """
        counts = {suit: 0 for suit in Suit}
        for card in self._cards:
            counts[card.suit] += 1
        return counts
    
    def get_card_count_by_rank(self) -> Dict[Rank, int]:
        """Get the count of cards by rank.
        
        Returns
        -------
        Dict[Rank, int]
            Dictionary mapping ranks to card counts
        """
        counts = {rank: 0 for rank in [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]}
        for card in self._cards:
            counts[card.rank] += 1
        return counts
    
    def __len__(self) -> int:
        """Return the number of cards in the deck."""
        return len(self._cards)
    
    def __contains__(self, card: Card) -> bool:
        """Check if a card is in the deck."""
        return card in self._cards
    
    def __iter__(self):
        """Iterate over the cards in the deck."""
        return iter(self._cards.copy()) 