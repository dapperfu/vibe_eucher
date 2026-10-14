"""Specific AI player profiles with different playing styles."""

from typing import List, Optional
from .base_ai import BaseAI
from ..models import Card, Suit


class AggressiveAI(BaseAI):
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
        super().__init__(name, max(0.6, risk_ratio))  # Aggressive players have minimum 0.6 risk
    
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Decide whether to order up the top card.
        
        Aggressive players use sophisticated hand evaluation to make strategic decisions.
        """
        # Use the hand evaluation system
        hand_strength = self._evaluate_hand_for_trump(top_card.suit, top_card, is_partner_dealing)
        
        # Base threshold varies by risk tolerance
        # Conservative: 15+, Balanced: 12+, Aggressive: 8+
        base_threshold = 15.0 - (self.risk_ratio * 7.0)
        
        # Additional risk factors
        if hand_strength < 0:
            # Hand is strategically weak (e.g., would give dealer left bower)
            return False
        
        # Consider the risk ratio for final decision
        if self.risk_ratio > 0.8:
            # Very aggressive: order up with decent hands
            return hand_strength >= base_threshold * 0.7
        elif self.risk_ratio > 0.6:
            # Aggressive: order up with good hands
            return hand_strength >= base_threshold * 0.8
        else:
            # Balanced: use standard threshold
            return hand_strength >= base_threshold
    
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play.
        
        Aggressive players lead with high cards and play aggressively.
        Must follow suit if possible.
        """
        if not self.hand:
            raise ValueError("AI player has no cards to play")
        
        # If leading (no lead suit), play highest card
        if not lead_suit:
            return max(self.hand, key=lambda c: self._card_value(c, trump_suit))
        
        # MUST follow suit if possible - this is a strict rule
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
            # Cannot follow suit - can play any card
            # Aggressive players play high cards
            if self.risk_ratio > 0.7:
                # Very aggressive: play highest card
                return max(self.hand, key=lambda c: self._card_value(c, trump_suit))
            else:
                # Moderate: play high card but not necessarily highest
                sorted_cards = sorted(self.hand, key=lambda c: self._card_value(c, trump_suit))
                top_cards = sorted_cards[int(len(sorted_cards) * 0.6):]
                return top_cards[0] if top_cards else sorted_cards[0]


class ConservativeAI(BaseAI):
    """Conservative AI player - low risk tolerance, plays safe."""
    
    def __init__(self, name: str, risk_ratio: float = 0.2) -> None:
        """Initialize conservative AI player.
        
        Parameters
        ----------
        name : str
            The player's name
        risk_ratio : float
            Risk tolerance (0.0 = conservative, 1.0 = aggressive)
        """
        super().__init__(name, min(0.4, risk_ratio))  # Conservative players have maximum 0.4 risk
    
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Decide whether to order up the top card.
        
        Conservative players only order up with very strong hands.
        """
        hand_strength = self._evaluate_hand_for_trump(top_card.suit, top_card, is_partner_dealing)
        
        # Conservative: only order up with very strong hands
        base_threshold = 18.0
        
        # Additional safety checks
        if hand_strength < 0:
            return False
        
        # Conservative players need higher thresholds
        return hand_strength >= base_threshold * 1.2
    
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play.
        
        Conservative players play safe, low cards when possible.
        """
        if not self.hand:
            raise ValueError("AI player has no cards to play")
        
        # If leading (no lead suit), play moderate card
        if not lead_suit:
            sorted_cards = sorted(self.hand, key=lambda c: self._card_value(c, trump_suit))
            # Choose from middle 60% of cards
            middle_start = int(len(sorted_cards) * 0.2)
            middle_end = int(len(sorted_cards) * 0.8)
            middle_cards = sorted_cards[middle_start:middle_end]
            return middle_cards[0] if middle_cards else sorted_cards[len(sorted_cards) // 2]
        
        # MUST follow suit if possible
        if self.has_suit(lead_suit):
            cards_of_suit = self.get_cards_of_suit(lead_suit)
            
            # Conservative: play moderate card of lead suit
            sorted_cards = sorted(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
            # Choose from middle 60% of cards
            middle_start = int(len(sorted_cards) * 0.2)
            middle_end = int(len(sorted_cards) * 0.8)
            middle_cards = sorted_cards[middle_start:middle_end]
            return middle_cards[0] if middle_cards else sorted_cards[len(sorted_cards) // 2]
        else:
            # Cannot follow suit - can play any card
            # Conservative players play low cards
            sorted_cards = sorted(self.hand, key=lambda c: self._card_value(c, trump_suit))
            # Choose from bottom 60% of cards
            bottom_cards = sorted_cards[:int(len(sorted_cards) * 0.6)]
            return bottom_cards[0] if bottom_cards else sorted_cards[0]


class BalancedAI(BaseAI):
    """Balanced AI player - moderate risk tolerance, balanced strategy."""
    
    def __init__(self, name: str, risk_ratio: float = 0.5) -> None:
        """Initialize balanced AI player.
        
        Parameters
        ----------
        name : str
            The player's name
        risk_ratio : float
            Risk tolerance (0.0 = conservative, 1.0 = aggressive)
        """
        super().__init__(name, risk_ratio)
    
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Decide whether to order up the top card.
        
        Balanced players use standard thresholds.
        """
        hand_strength = self._evaluate_hand_for_trump(top_card.suit, top_card, is_partner_dealing)
        
        # Balanced: use standard threshold
        base_threshold = 12.0
        
        # Adjust based on risk ratio
        adjusted_threshold = base_threshold - (self.risk_ratio - 0.5) * 4.0
        
        if hand_strength < 0:
            return False
        
        return hand_strength >= adjusted_threshold
    
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play.
        
        Balanced players use mixed strategy based on risk ratio.
        """
        if not self.hand:
            raise ValueError("AI player has no cards to play")
        
        # If leading (no lead suit), play based on risk ratio
        if not lead_suit:
            sorted_cards = sorted(self.hand, key=lambda c: self._card_value(c, trump_suit))
            if self.risk_ratio > 0.6:
                # More aggressive: play higher card
                top_cards = sorted_cards[int(len(sorted_cards) * 0.3):]
                return top_cards[0] if top_cards else sorted_cards[0]
            else:
                # More conservative: play moderate card
                middle_cards = sorted_cards[int(len(sorted_cards) * 0.3):int(len(sorted_cards) * 0.7)]
                return middle_cards[0] if middle_cards else sorted_cards[len(sorted_cards) // 2]
        
        # MUST follow suit if possible
        if self.has_suit(lead_suit):
            cards_of_suit = self.get_cards_of_suit(lead_suit)
            
            # Balanced: play based on risk ratio
            sorted_cards = sorted(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
            if self.risk_ratio > 0.6:
                # More aggressive: play higher card
                top_cards = sorted_cards[int(len(sorted_cards) * 0.4):]
                return top_cards[0] if top_cards else sorted_cards[0]
            else:
                # More conservative: play moderate card
                middle_cards = sorted_cards[int(len(sorted_cards) * 0.3):int(len(sorted_cards) * 0.7)]
                return middle_cards[0] if middle_cards else sorted_cards[len(sorted_cards) // 2]
        else:
            # Cannot follow suit - can play any card
            sorted_cards = sorted(self.hand, key=lambda c: self._card_value(c, trump_suit))
            if self.risk_ratio > 0.6:
                # More aggressive: play higher card
                top_cards = sorted_cards[int(len(sorted_cards) * 0.4):]
                return top_cards[0] if top_cards else sorted_cards[0]
            else:
                # More conservative: play lower card
                bottom_cards = sorted_cards[:int(len(sorted_cards) * 0.6)]
                return bottom_cards[0] if bottom_cards else sorted_cards[0]


class OpportunisticAI(BaseAI):
    """Opportunistic AI player - adapts strategy based on game state."""
    
    def __init__(self, name: str, risk_ratio: float = 0.5) -> None:
        """Initialize opportunistic AI player.
        
        Parameters
        ----------
        name : str
            The player's name
        risk_ratio : float
            Risk tolerance (0.0 = conservative, 1.0 = aggressive)
        """
        super().__init__(name, risk_ratio)
        self.game_context = {}  # Store game context for adaptive decisions
    
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Decide whether to order up the top card.
        
        Opportunistic players adapt based on game context.
        """
        hand_strength = self._evaluate_hand_for_trump(top_card.suit, top_card, is_partner_dealing)
        
        # Base threshold
        base_threshold = 12.0
        
        # Adapt based on game context
        if self.game_context.get('team_losing', False):
            # Team is losing - be more aggressive
            base_threshold *= 0.8
        elif self.game_context.get('team_winning', False):
            # Team is winning - be more conservative
            base_threshold *= 1.2
        
        if hand_strength < 0:
            return False
        
        return hand_strength >= base_threshold
    
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play.
        
        Opportunistic players adapt strategy based on game context.
        """
        if not self.hand:
            raise ValueError("AI player has no cards to play")
        
        # Adapt risk ratio based on game context
        adaptive_risk = self.risk_ratio
        
        if self.game_context.get('team_losing', False):
            # Team is losing - be more aggressive
            adaptive_risk = min(1.0, adaptive_risk * 1.3)
        elif self.game_context.get('team_winning', False):
            # Team is winning - be more conservative
            adaptive_risk = max(0.0, adaptive_risk * 0.7)
        
        # If leading (no lead suit), play based on adaptive risk
        if not lead_suit:
            sorted_cards = sorted(self.hand, key=lambda c: self._card_value(c, trump_suit))
            if adaptive_risk > 0.6:
                # More aggressive: play higher card
                top_cards = sorted_cards[int(len(sorted_cards) * 0.3):]
                return top_cards[0] if top_cards else sorted_cards[0]
            else:
                # More conservative: play moderate card
                middle_cards = sorted_cards[int(len(sorted_cards) * 0.3):int(len(sorted_cards) * 0.7)]
                return middle_cards[0] if middle_cards else sorted_cards[len(sorted_cards) // 2]
        
        # MUST follow suit if possible
        if self.has_suit(lead_suit):
            cards_of_suit = self.get_cards_of_suit(lead_suit)
            
            # Opportunistic: play based on adaptive risk
            sorted_cards = sorted(cards_of_suit, key=lambda c: self._card_value(c, trump_suit))
            if adaptive_risk > 0.6:
                # More aggressive: play higher card
                top_cards = sorted_cards[int(len(sorted_cards) * 0.4):]
                return top_cards[0] if top_cards else sorted_cards[0]
            else:
                # More conservative: play moderate card
                middle_cards = sorted_cards[int(len(sorted_cards) * 0.3):int(len(sorted_cards) * 0.7)]
                return middle_cards[0] if middle_cards else sorted_cards[len(sorted_cards) // 2]
        else:
            # Cannot follow suit - can play any card
            sorted_cards = sorted(self.hand, key=lambda c: self._card_value(c, trump_suit))
            if adaptive_risk > 0.6:
                # More aggressive: play higher card
                top_cards = sorted_cards[int(len(sorted_cards) * 0.4):]
                return top_cards[0] if top_cards else sorted_cards[0]
            else:
                # More conservative: play lower card
                bottom_cards = sorted_cards[:int(len(sorted_cards) * 0.6)]
                return bottom_cards[0] if bottom_cards else sorted_cards[0]
    
    def update_game_context(self, context: dict) -> None:
        """Update the game context for adaptive decision making.
        
        Parameters
        ----------
        context : dict
            Game context information
        """
        self.game_context.update(context) 