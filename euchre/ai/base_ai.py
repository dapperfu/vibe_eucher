"""Base AI player class with common functionality."""

from typing import List, Optional
from ..models import Player, PlayerType, Card, Suit, Rank


class BaseAI(Player):
    """Base class for AI players with common functionality."""
    
    def __init__(self, name: str, risk_ratio: float = 0.5) -> None:
        """Initialize base AI player.
        
        Parameters
        ----------
        name : str
            The player's name
        risk_ratio : float
            Risk tolerance (0.0 = conservative, 1.0 = aggressive)
        """
        super().__init__(name, PlayerType.AI)
        self.risk_ratio = max(0.0, min(1.0, risk_ratio))  # Clamp to [0, 1]
    
    def _get_left_bower_suit(self, trump_suit: Suit) -> Suit:
        """Get the left bower suit for a given trump suit.
        
        Parameters
        ----------
        trump_suit : Suit
            The trump suit
            
        Returns
        -------
        Suit
            The left bower suit
        """
        if trump_suit == Suit.HEARTS:
            return Suit.DIAMONDS
        elif trump_suit == Suit.DIAMONDS:
            return Suit.HEARTS
        elif trump_suit == Suit.CLUBS:
            return Suit.SPADES
        else:  # SPADES
            return Suit.CLUBS
    
    def _evaluate_hand_for_trump(self, trump_suit: Suit, top_card: Optional[Card] = None, is_partner_dealing: bool = False) -> float:
        """Evaluate hand strength for a potential trump suit.
        
        Parameters
        ----------
        trump_suit : Suit
            The potential trump suit
        top_card : Optional[Card]
            The top card if ordering up
        is_partner_dealing : bool
            Whether the player's partner is dealing
            
        Returns
        -------
        float
            Hand strength score
        """
        # Get left bower suit
        left_bower_suit = self._get_left_bower_suit(trump_suit)
        
        # Count trump cards
        trump_cards = [card for card in self.hand if card.suit == trump_suit]
        left_bower_cards = [card for card in self.hand if card.suit == left_bower_suit and card.rank == Rank.JACK]
        
        # Count high cards (10, J, Q, K, A)
        high_cards = [card for card in trump_cards if card.rank.value >= 10]
        left_bower_high = [card for card in left_bower_cards if card.rank.value >= 10]
        
        # Calculate hand strength
        total_trump_potential = len(trump_cards) + len(left_bower_cards)
        high_card_bonus = len(high_cards) + len(left_bower_high)
        
        # Base score
        score = total_trump_potential * 3.0 + high_card_bonus * 2.0
        
        # Penalty for giving dealer left bower
        if top_card and is_partner_dealing:
            if any(card.suit == left_bower_suit and card.rank == Rank.JACK for card in self.hand):
                score -= 5.0
        
        return score
    
    def _card_value(self, card: Card, trump_suit: Optional[Suit]) -> float:
        """Calculate the value of a card.
        
        Parameters
        ----------
        card : Card
            The card to evaluate
        trump_suit : Optional[Suit]
            The trump suit for this round
            
        Returns
        -------
        float
            Card value (higher is better)
        """
        if not trump_suit:
            return card.rank.value
        
        # Trump cards are worth more
        if card.is_trump:
            return card.rank.value + 20.0
        
        # Left bower is second highest trump
        left_bower_suit = self._get_left_bower_suit(trump_suit)
        if card.suit == left_bower_suit and card.rank == Rank.JACK:
            return 19.0
        
        # Regular cards
        return card.rank.value
    
    def has_suit(self, suit: Suit) -> bool:
        """Check if player has cards of a specific suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to check
            
        Returns
        -------
        bool
            True if player has cards of the suit
        """
        return any(card.suit == suit for card in self.hand)
    
    def get_cards_of_suit(self, suit: Suit) -> List[Card]:
        """Get all cards of a specific suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to get cards for
            
        Returns
        -------
        List[Card]
            List of cards of the specified suit
        """
        return [card for card in self.hand if card.suit == suit]
    
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Decide whether to order up the top card.
        
        Parameters
        ----------
        top_card : Card
            The top card flipped up
        is_partner_dealing : bool
            Whether the player's partner is dealing
            
        Returns
        -------
        bool
            True if the AI should order up
        """
        # Base implementation - subclasses should override
        hand_strength = self._evaluate_hand_for_trump(top_card.suit, top_card, is_partner_dealing)
        return hand_strength >= 12.0
    
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play.
        
        Parameters
        ----------
        lead_suit : Optional[Suit]
            The lead suit of the trick
        trump_suit : Optional[Suit]
            The trump suit for this round
            
        Returns
        -------
        Card
            The card to play
        """
        if not self.hand:
            raise ValueError("AI player has no cards to play")
        
        # If leading (no lead suit), play highest card
        if not lead_suit:
            return max(self.hand, key=lambda c: self._card_value(c, trump_suit))
        
        # MUST follow suit if possible
        if self.has_suit(lead_suit):
            cards_of_suit = self.get_cards_of_suit(lead_suit)
            return max(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
        else:
            # Cannot follow suit - can play any card
            return max(self.hand, key=lambda c: self._card_value(c, trump_suit)) 