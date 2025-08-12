"""AI player profiles with different risk tolerances and playing styles."""

from typing import List, Optional, Dict, Any
from .models import Player, PlayerType, Card, Suit, Rank


class AggressiveAI(Player):
    """Aggressive AI player - high risk tolerance, leads with high cards."""
    
    def __init__(self, name: str, risk_ratio: float = 0.8) -> None:
        """Initialize aggressive AI player.
        
        Parameters
        ----------
        name : str
            The player's name
        risk_ratio : float
            Risk tolerance (0.0 = conservative, 1.0 = very aggressive)
        """
        super().__init__(name, PlayerType.AI)
        self.risk_ratio = max(0.0, min(1.0, risk_ratio))  # Clamp to [0, 1]
        
    def _get_left_bower_suit(self, trump_suit: Suit) -> Suit:
        """Get the left bower suit for a given trump suit."""
        if trump_suit == Suit.HEARTS:
            return Suit.DIAMONDS
        elif trump_suit == Suit.DIAMONDS:
            return Suit.HEARTS
        elif trump_suit == Suit.CLUBS:
            return Suit.SPADES
        else:  # SPADES
            return Suit.CLUBS
        
    def should_order_up(self, top_card: Card) -> bool:
        """Decide whether to order up the top card.
        
        Aggressive players are more likely to order up with fewer cards.
        """
        # Count cards of the potential trump suit
        cards_of_suit = self.get_cards_of_suit(top_card.suit)
        
        # Also count left bower (jack of same color)
        left_bower_suit = self._get_left_bower_suit(top_card.suit)
        left_bower_cards = self.get_cards_of_suit(left_bower_suit)
        
        total_trump_potential = len(cards_of_suit) + len(left_bower_cards)
        
        # Aggressive threshold: lower requirement based on risk ratio
        # Conservative: 3+ cards, Aggressive: 1+ card
        threshold = max(1, 3 - int(self.risk_ratio * 2))
        
        return total_trump_potential >= threshold
        
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> None:
        """Choose which card to play.
        
        Aggressive players lead with high cards and play aggressively.
        """
        if not self.hand:
            raise ValueError("AI player has no cards to play")
            
        # If leading (no lead suit), play highest card
        if not lead_suit:
            return max(self.hand, key=lambda c: self._card_value(c, trump_suit))
            
        # Must follow suit if possible
        if self.has_suit(lead_suit):
            cards_of_suit = self.get_cards_of_suit(lead_suit)
            
            # Aggressive: play highest card of lead suit
            if self.risk_ratio > 0.5:
                return max(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
            else:
                # Moderate aggressive: play high card but not necessarily highest
                sorted_cards = sorted(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
                # Choose from top 60% of cards
                top_cards = sorted_cards[int(len(sorted_cards) * 0.4):]
                return top_cards[0] if top_cards else sorted_cards[0]
        else:
            # Can play any card - aggressive players play high cards
            if self.risk_ratio > 0.7:
                # Very aggressive: play highest card
                return max(self.hand, key=lambda c: self._card_value(c, trump_suit))
            else:
                # Moderate: play high card but not necessarily highest
                sorted_cards = sorted(self.hand, key=lambda c: self._card_value(c, trump_suit))
                top_cards = sorted_cards[int(len(sorted_cards) * 0.6):]
                return top_cards[0] if top_cards else sorted_cards[0]
                
    def _card_value(self, card: Card, trump_suit: Optional[Suit]) -> int:
        """Calculate the value of a card for decision making."""
        base_value = card.rank.value
        
        # Trump cards are worth more
        if trump_suit and card.suit == trump_suit:
            base_value += 20
            
        # Right bower (jack of trump suit) is highest
        if trump_suit and card.suit == trump_suit and card.rank == Rank.JACK:
            base_value += 10
            
        # Left bower (jack of same color as trump) is second highest
        if trump_suit:
            left_bower_suit = self._get_left_bower_suit(trump_suit)
            if card.suit == left_bower_suit and card.rank == Rank.JACK:
                base_value += 15
                
        return base_value


class ConservativeAI(Player):
    """Conservative AI player - low risk tolerance, leads with low cards."""
    
    def __init__(self, name: str, risk_ratio: float = 0.2) -> None:
        """Initialize conservative AI player.
        
        Parameters
        ----------
        name : str
            The player's name
        risk_ratio : float
            Risk tolerance (0.0 = very conservative, 1.0 = aggressive)
        """
        super().__init__(name, PlayerType.AI)
        self.risk_ratio = max(0.0, min(1.0, risk_ratio))  # Clamp to [0, 1]
        
    def _get_left_bower_suit(self, trump_suit: Suit) -> Suit:
        """Get the left bower suit for a given trump suit."""
        if trump_suit == Suit.HEARTS:
            return Suit.DIAMONDS
        elif trump_suit == Suit.DIAMONDS:
            return Suit.HEARTS
        elif trump_suit == Suit.CLUBS:
            return Suit.SPADES
        else:  # SPADES
            return Suit.CLUBS
        
    def should_order_up(self, top_card: Card) -> bool:
        """Decide whether to order up the top card.
        
        Conservative players require more cards to order up.
        """
        # Count cards of the potential trump suit
        cards_of_suit = self.get_cards_of_suit(top_card.suit)
        
        # Also count left bower (jack of same color)
        left_bower_suit = self._get_left_bower_suit(top_card.suit)
        left_bower_cards = self.get_cards_of_suit(left_bower_suit)
        
        total_trump_potential = len(cards_of_suit) + len(left_bower_cards)
        
        # Conservative threshold: higher requirement based on risk ratio
        # Very conservative: 4+ cards, Aggressive: 2+ cards
        threshold = max(2, 4 - int(self.risk_ratio * 2))
        
        return total_trump_potential >= threshold
        
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> None:
        """Choose which card to play.
        
        Conservative players lead with low cards and play defensively.
        """
        if not self.hand:
            raise ValueError("AI player has no cards to play")
            
        # If leading (no lead suit), play lowest card
        if not lead_suit:
            return min(self.hand, key=lambda c: self._card_value(c, trump_suit))
            
        # Must follow suit if possible
        if self.has_suit(lead_suit):
            cards_of_suit = self.get_cards_of_suit(lead_suit)
            
            # Conservative: play lowest card of lead suit
            if self.risk_ratio < 0.3:
                return min(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
            else:
                # Moderate conservative: play low card but not necessarily lowest
                sorted_cards = sorted(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
                # Choose from bottom 60% of cards
                bottom_cards = sorted_cards[:int(len(sorted_cards) * 0.6)]
                return bottom_cards[-1] if bottom_cards else sorted_cards[0]
        else:
            # Can play any card - conservative players play low cards
            if self.risk_ratio < 0.2:
                # Very conservative: play lowest card
                return min(self.hand, key=lambda c: self._card_value(c, trump_suit))
            else:
                # Moderate: play low card but not necessarily lowest
                sorted_cards = sorted(self.hand, key=lambda c: self._card_value(c, trump_suit))
                bottom_cards = sorted_cards[:int(len(sorted_cards) * 0.4)]
                return bottom_cards[-1] if bottom_cards else sorted_cards[0]
                
    def _card_value(self, card: Card, trump_suit: Optional[Suit]) -> int:
        """Calculate the value of a card for decision making."""
        base_value = card.rank.value
        
        # Trump cards are worth more
        if trump_suit and card.suit == trump_suit:
            base_value += 20
            
        # Right bower (jack of trump suit) is highest
        if trump_suit and card.suit == trump_suit and card.rank == Rank.JACK:
            base_value += 10
            
        # Left bower (jack of same color as trump) is second highest
        if trump_suit:
            left_bower_suit = self._get_left_bower_suit(trump_suit)
            if card.suit == left_bower_suit and card.rank == Rank.JACK:
                base_value += 15
                
        return base_value


class BalancedAI(Player):
    """Balanced AI player - moderate risk tolerance, adaptive playing style."""
    
    def __init__(self, name: str, risk_ratio: float = 0.5) -> None:
        """Initialize balanced AI player.
        
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
        """Get the left bower suit for a given trump suit."""
        if trump_suit == Suit.HEARTS:
            return Suit.DIAMONDS
        elif trump_suit == Suit.DIAMONDS:
            return Suit.HEARTS
        elif trump_suit == Suit.CLUBS:
            return Suit.SPADES
        else:  # SPADES
            return Suit.CLUBS
        
    def should_order_up(self, top_card: Card) -> bool:
        """Decide whether to order up the top card.
        
        Balanced players use moderate thresholds.
        """
        # Count cards of the potential trump suit
        cards_of_suit = self.get_cards_of_suit(top_card.suit)
        
        # Also count left bower (jack of same color)
        left_bower_suit = self._get_left_bower_suit(top_card.suit)
        left_bower_cards = self.get_cards_of_suit(left_bower_suit)
        
        total_trump_potential = len(cards_of_suit) + len(left_bower_cards)
        
        # Balanced threshold: moderate requirement
        threshold = max(2, 3 - int(self.risk_ratio * 1))
        
        return total_trump_potential >= threshold
        
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> None:
        """Choose which card to play.
        
        Balanced players adapt their strategy based on the situation.
        """
        if not self.hand:
            raise ValueError("AI player has no cards to play")
            
        # If leading (no lead suit), choose based on risk ratio
        if not lead_suit:
            if self.risk_ratio > 0.6:
                # More aggressive: lead with high card
                return max(self.hand, key=lambda c: self._card_value(c, trump_suit))
            elif self.risk_ratio < 0.4:
                # More conservative: lead with low card
                return min(self.hand, key=lambda c: self._card_value(c, trump_suit))
            else:
                # Balanced: lead with middle card
                sorted_cards = sorted(self.hand, key=lambda c: self._card_value(c, trump_suit))
                return sorted_cards[len(sorted_cards) // 2]
            
        # Must follow suit if possible
        if self.has_suit(lead_suit):
            cards_of_suit = self.get_cards_of_suit(lead_suit)
            
            # Balanced: choose based on risk ratio
            if self.risk_ratio > 0.6:
                # More aggressive: play high card
                sorted_cards = sorted(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
                top_cards = sorted_cards[int(len(sorted_cards) * 0.7):]
                return top_cards[0] if top_cards else sorted_cards[0]
            elif self.risk_ratio < 0.4:
                # More conservative: play low card
                sorted_cards = sorted(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
                bottom_cards = sorted_cards[:int(len(sorted_cards) * 0.3)]
                return bottom_cards[-1] if bottom_cards else sorted_cards[0]
            else:
                # Balanced: play middle card
                sorted_cards = sorted(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
                return sorted_cards[len(sorted_cards) // 2]
        else:
            # Can play any card - balanced approach
            sorted_cards = sorted(self.hand, key=lambda c: self._card_value(c, trump_suit))
            
            if self.risk_ratio > 0.6:
                # More aggressive: play high card
                top_cards = sorted_cards[int(len(sorted_cards) * 0.7):]
                return top_cards[0] if top_cards else sorted_cards[0]
            elif self.risk_ratio < 0.4:
                # More conservative: play low card
                bottom_cards = sorted_cards[:int(len(sorted_cards) * 0.3)]
                return bottom_cards[-1] if bottom_cards else sorted_cards[0]
            else:
                # Balanced: play middle card
                return sorted_cards[len(sorted_cards) // 2]
                
    def _card_value(self, card: Card, trump_suit: Optional[Suit]) -> int:
        """Calculate the value of a card for decision making."""
        base_value = card.rank.value
        
        # Trump cards are worth more
        if trump_suit and card.suit == trump_suit:
            base_value += 20
            
        # Right bower (jack of trump suit) is highest
        if trump_suit and card.suit == trump_suit and card.rank == Rank.JACK:
            base_value += 10
            
        # Left bower (jack of same color as trump) is second highest
        if trump_suit:
            left_bower_suit = self._get_left_bower_suit(trump_suit)
            if card.suit == left_bower_suit and card.rank == Rank.JACK:
                base_value += 15
                
        return base_value


class OpportunisticAI(Player):
    """Opportunistic AI player - adapts strategy based on game state."""
    
    def __init__(self, name: str, risk_ratio: float = 0.6) -> None:
        """Initialize opportunistic AI player.
        
        Parameters
        ----------
        name : str
            The player's name
        risk_ratio : float
            Base risk tolerance (0.0 = conservative, 1.0 = aggressive)
        """
        super().__init__(name, PlayerType.AI)
        self.risk_ratio = max(0.0, min(1.0, risk_ratio))  # Clamp to [0, 1]
        self.tricks_won_this_round = 0
        
    def _get_left_bower_suit(self, trump_suit: Suit) -> Suit:
        """Get the left bower suit for a given trump suit."""
        if trump_suit == Suit.HEARTS:
            return Suit.DIAMONDS
        elif trump_suit == Suit.DIAMONDS:
            return Suit.HEARTS
        elif trump_suit == Suit.CLUBS:
            return Suit.SPADES
        else:  # SPADES
            return Suit.CLUBS
        
    def should_order_up(self, top_card: Card) -> bool:
        """Decide whether to order up the top card.
        
        Opportunistic players consider their position and current score.
        """
        # Count cards of the potential trump suit
        cards_of_suit = self.get_cards_of_suit(top_card.suit)
        
        # Also count left bower (jack of same color)
        left_bower_suit = self._get_left_bower_suit(top_card.suit)
        left_bower_cards = self.get_cards_of_suit(left_bower_suit)
        
        total_trump_potential = len(cards_of_suit) + len(left_bower_cards)
        
        # Base threshold
        base_threshold = 2
        
        # Adjust based on risk ratio
        adjusted_threshold = max(1, base_threshold - int(self.risk_ratio * 1))
        
        # Be more aggressive if we have strong trump potential
        if total_trump_potential >= 3:
            adjusted_threshold = max(1, adjusted_threshold - 1)
            
        return total_trump_potential >= adjusted_threshold
        
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> None:
        """Choose which card to play.
        
        Opportunistic players adapt based on game situation.
        """
        if not self.hand:
            raise ValueError("AI player has no cards to play")
            
        # If leading (no lead suit), consider game state
        if not lead_suit:
            # If we're ahead in tricks, be more conservative
            if self.tricks_won_this_round >= 2:
                return min(self.hand, key=lambda c: self._card_value(c, trump_suit))
            # If we're behind, be more aggressive
            elif self.tricks_won_this_round == 0:
                return max(self.hand, key=lambda c: self._card_value(c, trump_suit))
            else:
                # Balanced approach
                sorted_cards = sorted(self.hand, key=lambda c: self._card_value(c, trump_suit))
                return sorted_cards[len(sorted_cards) // 2]
            
        # Must follow suit if possible
        if self.has_suit(lead_suit):
            cards_of_suit = self.get_cards_of_suit(lead_suit)
            
            # If we're ahead in tricks, play conservatively
            if self.tricks_won_this_round >= 2:
                return min(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
            # If we're behind, play aggressively
            elif self.tricks_won_this_round == 0:
                return max(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
            else:
                # Balanced approach
                sorted_cards = sorted(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
                return sorted_cards[len(sorted_cards) // 2]
        else:
            # Can play any card - consider game state
            if self.tricks_won_this_round >= 2:
                # Ahead: play low card
                return min(self.hand, key=lambda c: self._card_value(c, trump_suit))
            elif self.tricks_won_this_round == 0:
                # Behind: play high card
                return max(self.hand, key=lambda c: self._card_value(c, trump_suit))
            else:
                # Balanced: play middle card
                sorted_cards = sorted(self.hand, key=lambda c: self._card_value(c, trump_suit))
                return sorted_cards[len(sorted_cards) // 2]
                
    def _card_value(self, card: Card, trump_suit: Optional[Suit]) -> int:
        """Calculate the value of a card for decision making."""
        base_value = card.rank.value
        
        # Trump cards are worth more
        if trump_suit and card.suit == trump_suit:
            base_value += 20
            
        # Right bower (jack of trump suit) is highest
        if trump_suit and card.suit == trump_suit and card.rank == Rank.JACK:
            base_value += 10
            
        # Left bower (jack of same color as trump) is second highest
        if trump_suit:
            left_bower_suit = self._get_left_bower_suit(trump_suit)
            if card.suit == left_bower_suit and card.rank == Rank.JACK:
                base_value += 15
                
        return base_value
        
    def update_tricks_won(self, tricks: int) -> None:
        """Update the number of tricks won this round."""
        self.tricks_won_this_round = tricks 