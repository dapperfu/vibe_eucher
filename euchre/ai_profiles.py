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
        
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Decide whether to order up the top card.
        
        Aggressive players use sophisticated hand evaluation to make strategic decisions.
        """
        # Use the new hand evaluation system
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
        elif self.risk_ratio < 0.3:
            # Conservative: only order up with very strong hands
            return hand_strength >= base_threshold * 1.2
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

    def _evaluate_hand_for_trump(self, potential_trump_suit: Suit, top_card: Card, is_partner_dealing: bool = False) -> float:
        """Evaluate the strength of the hand for a potential trump suit.
        
        This method considers:
        - Quality of trump cards (not just quantity)
        - Strategic implications of the top card
        - Risk of giving dealer powerful cards
        - Overall hand strength and potential
        - Partner dealing (partner picks up what you order)
        - First player strategy considerations
        
        Parameters
        ----------
        potential_trump_suit : Suit
            The suit being considered as trump
        top_card : Card
            The top card that could be ordered up
        is_partner_dealing : bool
            Whether the player's partner is the dealer
            
        Returns
        -------
        float
            Hand strength score (higher = stronger hand for this trump)
        """
        score = 0.0
        
        # Get all potential trump cards (including left bower)
        trump_cards = self.get_cards_of_suit(potential_trump_suit)
        left_bower_suit = self._get_left_bower_suit(potential_trump_suit)
        left_bower_cards = self.get_cards_of_suit(left_bower_suit)
        
        # Evaluate trump cards quality with proper suit consideration
        for card in trump_cards:
            score += self._evaluate_trump_card_value(card, potential_trump_suit)
                
        # Evaluate left bower cards quality with proper suit consideration
        for card in left_bower_cards:
            score += self._evaluate_left_bower_card_value(card, potential_trump_suit)
                
        # Strategic considerations for ordering up
        if top_card.suit == potential_trump_suit:
            # We're considering ordering up the top card
            score += self._evaluate_top_card_strategy(top_card, potential_trump_suit, is_partner_dealing)
            
        # Partner dealing bonus - if partner is dealing, they pick up what you order
        if is_partner_dealing:
            score += self._evaluate_partner_dealing_bonus(top_card, potential_trump_suit)
            
        # Bonus for having multiple trump cards (synergy)
        total_trump_potential = len(trump_cards) + len(left_bower_cards)
        if total_trump_potential >= 3:
            score += 5.0  # Bonus for having 3+ trump cards
        elif total_trump_potential >= 2:
            score += 2.0  # Bonus for having 2+ trump cards
            
        # Consider off-suit strength (cards that can win non-trump tricks)
        off_suit_cards = [c for c in self.hand if c.suit not in [potential_trump_suit, left_bower_suit]]
        off_suit_strength = self._evaluate_off_suit_strength(off_suit_cards, potential_trump_suit)
        score += off_suit_strength * 0.3  # Off-suit strength is valuable but not as much as trump
        
        # First player strategy consideration
        if self._is_first_player_after_dealer():
            score += self._evaluate_first_player_strategy(potential_trump_suit, left_bower_suit)
        
        return score
        
    def _evaluate_trump_card_value(self, card: Card, trump_suit: Suit) -> float:
        """Evaluate the value of a trump card based on the actual trump suit.
        
        Parameters
        ----------
        card : Card
            The card to evaluate
        trump_suit : Suit
            The trump suit
            
        Returns
        -------
        float
            Card value in this trump suit
        """
        if card.rank == Rank.JACK:
            return 15.0  # Right bower is always most powerful
        elif card.rank == Rank.ACE:
            return 12.0
        elif card.rank == Rank.KING:
            return 10.0
        elif card.rank == Rank.QUEEN:
            return 8.0
        elif card.rank == Rank.TEN:
            return 6.0
        elif card.rank == Rank.NINE:
            return 4.0
        else:
            return 0.0
            
    def _evaluate_left_bower_card_value(self, card: Card, trump_suit: Suit) -> float:
        """Evaluate the value of a left bower card based on the actual trump suit.
        
        Parameters
        ----------
        card : Card
            The card to evaluate
        trump_suit : Suit
            The trump suit
            
        Returns
        -------
        float
            Card value as left bower
        """
        if card.rank == Rank.JACK:
            return 14.0  # Left bower is second most powerful
        else:
            # Other cards in left bower suit are worth less
            return card.rank.value * 0.5
            
    def _evaluate_off_suit_strength(self, off_suit_cards: List[Card], trump_suit: Suit) -> float:
        """Evaluate the strength of off-suit cards considering first player strategy.
        
        Parameters
        ----------
        off_suit_cards : List[Card]
            Cards not in trump or left bower suits
        trump_suit : Suit
            The trump suit
            
        Returns
        -------
        float
            Off-suit strength score
        """
        if not off_suit_cards:
            return 0.0
            
        # Sort cards by rank (high to low)
        sorted_cards = sorted(off_suit_cards, key=lambda c: c.rank.value, reverse=True)
        
        # First player strategy: high off-suit cards are valuable for leading
        if self._is_first_player_after_dealer():
            # Value high cards more for leading
            score = sum(c.rank.value * (1.0 + (i * 0.1)) for i, c in enumerate(sorted_cards))
        else:
            # Standard off-suit evaluation
            score = sum(c.rank.value for c in sorted_cards)
            
        return score
        
    def _evaluate_partner_dealing_bonus(self, top_card: Card, potential_trump_suit: Suit) -> float:
        """Evaluate the bonus for having your partner as dealer.
        
        When your partner is dealing, they pick up what you order, so you can
        be more aggressive if you have strong trump potential.
        
        Parameters
        ----------
        top_card : Card
            The top card being considered
        potential_trump_suit : Suit
            The potential trump suit
            
        Returns
        -------
        float
            Bonus score for partner dealing
        """
        bonus = 0.0
        
        # If partner is dealing, they get the top card
        if top_card.suit == potential_trump_suit:
            # Partner gets a trump card - evaluate its value
            if top_card.rank == Rank.JACK:
                bonus += 8.0  # Partner gets right bower
            elif top_card.rank == Rank.ACE:
                bonus += 6.0  # Partner gets Ace
            elif top_card.rank == Rank.KING:
                bonus += 4.0  # Partner gets King
            elif top_card.rank == Rank.QUEEN:
                bonus += 2.0  # Partner gets Queen
            elif top_card.rank == Rank.TEN:
                bonus += 1.0  # Partner gets Ten
            elif top_card.rank == Rank.NINE:
                bonus += 0.0  # Partner gets Nine
                
        # Check if ordering up gives partner the left bower
        left_bower_suit = self._get_left_bower_suit(potential_trump_suit)
        if top_card.suit == left_bower_suit and top_card.rank == Rank.JACK:
            # Partner gets left bower - this is good for your team
            bonus += 10.0
            
        # Consider if you have strong trump cards yourself
        trump_cards = self.get_cards_of_suit(potential_trump_suit)
        left_bower_cards = self.get_cards_of_suit(left_bower_suit)
        
        # If you have strong trump, partner getting more trump is synergistic
        if any(c.rank == Rank.JACK for c in trump_cards):  # Right bower
            bonus += 5.0
        if any(c.rank == Rank.JACK for c in left_bower_cards):  # Left bower
            bonus += 4.0
        if any(c.rank == Rank.ACE for c in trump_cards):  # Ace
            bonus += 3.0
            
        return bonus
        
    def _evaluate_first_player_strategy(self, trump_suit: Suit, left_bower_suit: Suit) -> float:
        """Evaluate the bonus for being first player after dealer.
        
        First player often leads with high off-suit cards to try to win tricks.
        
        Parameters
        ----------
        trump_suit : Suit
            The trump suit
        left_bower_suit : Suit
            The left bower suit
            
        Returns
        -------
        float
            Bonus score for first player strategy
        """
        bonus = 0.0
        
        # Get off-suit cards (not trump or left bower)
        off_suit_cards = [c for c in self.hand if c.suit not in [trump_suit, left_bower_suit]]
        
        if not off_suit_cards:
            return 0.0
            
        # Sort by rank (high to low)
        sorted_cards = sorted(off_suit_cards, key=lambda c: c.rank.value, reverse=True)
        
        # First player strategy: high off-suit cards are valuable for leading
        # Ace is worth 4, King 3, Queen 2, etc.
        for i, card in enumerate(sorted_cards[:3]):  # Top 3 cards
            if card.rank == Rank.ACE:
                bonus += 4.0 - i
            elif card.rank == Rank.KING:
                bonus += 3.0 - i
            elif card.rank == Rank.QUEEN:
                bonus += 2.0 - i
            elif card.rank == Rank.JACK:
                bonus += 1.0 - i
                
        return bonus
        
    def _is_first_player_after_dealer(self) -> bool:
        """Check if this player is the first to play after the dealer.
        
        Returns
        -------
        bool
            True if this player is first after dealer
        """
        # This would need to be implemented based on game state
        # For now, return False - this should be passed from the game logic
        return False
        
    def _evaluate_top_card_strategy(self, top_card: Card, potential_trump_suit: Suit, is_partner_dealing: bool = False) -> float:
        """Evaluate the strategic implications of ordering up the top card.
        
        Parameters
        ----------
        top_card : Card
            The top card being considered for ordering up
        potential_trump_suit : Suit
            The suit that would become trump
        is_partner_dealing : bool
            Whether the player's partner is the dealer
            
        Returns
        -------
        float
            Strategic score (positive = good, negative = bad)
        """
        score = 0.0
        
        # Check if ordering up gives dealer the left bower
        left_bower_suit = self._get_left_bower_suit(potential_trump_suit)
        if top_card.suit == left_bower_suit and top_card.rank == Rank.JACK:
            if is_partner_dealing:
                # Partner gets left bower - this is good for your team
                score += 10.0
            else:
                # Opponent gets left bower - this is extremely bad strategy
                score -= 25.0  # This is extremely bad strategy
        elif top_card.suit == potential_trump_suit and top_card.rank == Rank.JACK:
            # Ordering up the right bower - this is good for us
            score += 10.0
        elif top_card.suit == potential_trump_suit:
            # Ordering up a non-Jack trump card
            if top_card.rank == Rank.ACE:
                score += 8.0
            elif top_card.rank == Rank.KING:
                score += 6.0
            elif top_card.rank == Rank.QUEEN:
                score += 4.0
            elif top_card.rank == Rank.TEN:
                score += 2.0
            elif top_card.rank == Rank.NINE:
                score += 0.0  # Neutral
        else:
            # Ordering up a non-trump card (shouldn't happen in normal euchre)
            score -= 5.0
            
        # Consider if we have the left bower ourselves
        left_bower_in_hand = any(c.suit == left_bower_suit and c.rank == Rank.JACK for c in self.hand)
        if left_bower_in_hand:
            # We have the left bower, so ordering up this trump suit is more valuable
            score += 8.0
            
        # Consider if we have the right bower
        right_bower_in_hand = any(c.suit == potential_trump_suit and c.rank == Rank.JACK for c in self.hand)
        if right_bower_in_hand:
            # We have the right bower, so this trump suit is very strong for us
            score += 12.0
            
        return score


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
        
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Decide whether to order up the top card.
        
        Conservative players use sophisticated hand evaluation and are very cautious.
        """
        # Use the new hand evaluation system
        hand_strength = self._evaluate_hand_for_trump(top_card.suit, top_card, is_partner_dealing)
        
        # Conservative players have higher thresholds
        # Very conservative: 20+, Moderate: 18+, Slightly aggressive: 15+
        base_threshold = 20.0 - (self.risk_ratio * 5.0)
        
        # Additional risk factors
        if hand_strength < 0:
            # Hand is strategically weak (e.g., would give dealer left bower)
            return False
            
        # Conservative players are very risk-averse
        if self.risk_ratio < 0.2:
            # Very conservative: only order up with extremely strong hands
            return hand_strength >= base_threshold * 1.3
        elif self.risk_ratio < 0.4:
            # Conservative: order up with very strong hands
            return hand_strength >= base_threshold * 1.1
        elif self.risk_ratio < 0.6:
            # Moderate: order up with strong hands
            return hand_strength >= base_threshold
        else:
            # Slightly aggressive: order up with good hands
            return hand_strength >= base_threshold * 0.9
        
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play.
        
        Conservative players lead with low cards and play defensively.
        Must follow suit if possible.
        """
        if not self.hand:
            raise ValueError("AI player has no cards to play")
            
        # If leading (no lead suit), play lowest card
        if not lead_suit:
            return min(self.hand, key=lambda c: self._card_value(c, trump_suit))
            
        # MUST follow suit if possible - this is a strict rule
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
            # Cannot follow suit - can play any card
            # Conservative players play low cards
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

    def _evaluate_hand_for_trump(self, potential_trump_suit: Suit, top_card: Card, is_partner_dealing: bool = False) -> float:
        """Evaluate the strength of the hand for a potential trump suit.
        
        Conservative players are more cautious about strategic risks.
        """
        score = 0.0
        
        # Get all potential trump cards (including left bower)
        trump_cards = self.get_cards_of_suit(potential_trump_suit)
        left_bower_suit = self._get_left_bower_suit(potential_trump_suit)
        left_bower_cards = self.get_cards_of_suit(left_bower_suit)
        
        # Evaluate trump cards quality with proper suit consideration
        for card in trump_cards:
            score += self._evaluate_trump_card_value(card, potential_trump_suit)
                
        # Evaluate left bower cards quality with proper suit consideration
        for card in left_bower_cards:
            score += self._evaluate_left_bower_card_value(card, potential_trump_suit)
                
        # Strategic considerations for ordering up
        if top_card.suit == potential_trump_suit:
            # We're considering ordering up the top card
            score += self._evaluate_top_card_strategy(top_card, potential_trump_suit, is_partner_dealing)
            
        # Partner dealing bonus - if partner is dealing, they pick up what you order
        if is_partner_dealing:
            score += self._evaluate_partner_dealing_bonus(top_card, potential_trump_suit)
            
        # Bonus for having multiple trump cards (synergy)
        total_trump_potential = len(trump_cards) + len(left_bower_cards)
        if total_trump_potential >= 4:
            score += 8.0  # Conservative players value 4+ trump cards highly
        elif total_trump_potential >= 3:
            score += 5.0  # Bonus for having 3+ trump cards
        elif total_trump_potential >= 2:
            score += 2.0  # Bonus for having 2+ trump cards
            
        # Consider off-suit strength (cards that can win non-trump tricks)
        off_suit_cards = [c for c in self.hand if c.suit not in [potential_trump_suit, left_bower_suit]]
        off_suit_strength = self._evaluate_off_suit_strength(off_suit_cards, potential_trump_suit)
        score += off_suit_strength * 0.2  # Conservative players value off-suit less
        
        return score
        
    def _evaluate_trump_card_value(self, card: Card, trump_suit: Suit) -> float:
        """Evaluate the value of a trump card based on the actual trump suit."""
        if card.rank == Rank.JACK:
            return 15.0  # Right bower is always most powerful
        elif card.rank == Rank.ACE:
            return 12.0
        elif card.rank == Rank.KING:
            return 10.0
        elif card.rank == Rank.QUEEN:
            return 8.0
        elif card.rank == Rank.TEN:
            return 6.0
        elif card.rank == Rank.NINE:
            return 4.0
        else:
            return 0.0
            
    def _evaluate_left_bower_card_value(self, card: Card, trump_suit: Suit) -> float:
        """Evaluate the value of a left bower card based on the actual trump suit."""
        if card.rank == Rank.JACK:
            return 14.0  # Left bower is second most powerful
        else:
            # Other cards in left bower suit are worth less
            return card.rank.value * 0.5
            
    def _evaluate_off_suit_strength(self, off_suit_cards: List[Card], trump_suit: Suit) -> float:
        """Evaluate the strength of off-suit cards considering first player strategy."""
        if not off_suit_cards:
            return 0.0
            
        # Sort cards by rank (high to low)
        sorted_cards = sorted(off_suit_cards, key=lambda c: c.rank.value, reverse=True)
        
        # First player strategy: high off-suit cards are valuable for leading
        if self._is_first_player_after_dealer():
            # Value high cards more for leading
            score = sum(c.rank.value * (1.0 + (i * 0.1)) for i, c in enumerate(sorted_cards))
        else:
            # Standard off-suit evaluation
            score = sum(c.rank.value for c in sorted_cards)
            
        return score
        
    def _evaluate_partner_dealing_bonus(self, top_card: Card, potential_trump_suit: Suit) -> float:
        """Evaluate the bonus for having your partner as dealer."""
        bonus = 0.0
        
        # If partner is dealing, they get the top card
        if top_card.suit == potential_trump_suit:
            # Partner gets a trump card - evaluate its value
            if top_card.rank == Rank.JACK:
                bonus += 8.0  # Partner gets right bower
            elif top_card.rank == Rank.ACE:
                bonus += 6.0  # Partner gets Ace
            elif top_card.rank == Rank.KING:
                bonus += 4.0  # Partner gets King
            elif top_card.rank == Rank.QUEEN:
                bonus += 2.0  # Partner gets Queen
            elif top_card.rank == Rank.TEN:
                bonus += 1.0  # Partner gets Ten
            elif top_card.rank == Rank.NINE:
                bonus += 0.0  # Partner gets Nine
                
        # Check if ordering up gives partner the left bower
        left_bower_suit = self._get_left_bower_suit(potential_trump_suit)
        if top_card.suit == left_bower_suit and top_card.rank == Rank.JACK:
            # Partner gets left bower - this is good for your team
            bonus += 10.0
            
        # Consider if you have strong trump cards yourself
        trump_cards = self.get_cards_of_suit(potential_trump_suit)
        left_bower_cards = self.get_cards_of_suit(left_bower_suit)
        
        # If you have strong trump, partner getting more trump is synergistic
        if any(c.rank == Rank.JACK for c in trump_cards):  # Right bower
            bonus += 5.0
        if any(c.rank == Rank.JACK for c in left_bower_cards):  # Left bower
            bonus += 4.0
        if any(c.rank == Rank.ACE for c in trump_cards):  # Ace
            bonus += 3.0
            
        return bonus
        
    def _evaluate_first_player_strategy(self, trump_suit: Suit, left_bower_suit: Suit) -> float:
        """Evaluate the bonus for being first player after dealer."""
        bonus = 0.0
        
        # Get off-suit cards (not trump or left bower)
        off_suit_cards = [c for c in self.hand if c.suit not in [trump_suit, left_bower_suit]]
        
        if not off_suit_cards:
            return 0.0
            
        # Sort by rank (high to low)
        sorted_cards = sorted(off_suit_cards, key=lambda c: c.rank.value, reverse=True)
        
        # First player strategy: high off-suit cards are valuable for leading
        for i, card in enumerate(sorted_cards[:3]):  # Top 3 cards
            if card.rank == Rank.ACE:
                bonus += 4.0 - i
            elif card.rank == Rank.KING:
                bonus += 3.0 - i
            elif card.rank == Rank.QUEEN:
                bonus += 2.0 - i
            elif card.rank == Rank.JACK:
                bonus += 1.0 - i
                
        return bonus
        
    def _is_first_player_after_dealer(self) -> bool:
        """Check if this player is the first to play after the dealer."""
        # This would need to be implemented based on game state
        # For now, return False - this should be passed from the game logic
        return False
        
    def _evaluate_top_card_strategy(self, top_card: Card, potential_trump_suit: Suit, is_partner_dealing: bool = False) -> float:
        """Evaluate the strategic implications of ordering up the top card.
        
        Conservative players are very cautious about strategic risks.
        """
        score = 0.0
        
        # Check if ordering up gives dealer the left bower
        left_bower_suit = self._get_left_bower_suit(potential_trump_suit)
        if top_card.suit == left_bower_suit and top_card.rank == Rank.JACK:
            # CRITICAL: Ordering up a Jack of the left bower suit gives dealer left bower
            score -= 30.0  # Conservative players consider this extremely bad
        elif top_card.suit == potential_trump_suit and top_card.rank == Rank.JACK:
            # Ordering up the right bower - this is good for us
            score += 12.0
        elif top_card.suit == potential_trump_suit:
            # Ordering up a non-Jack trump card
            if top_card.rank == Rank.ACE:
                score += 10.0
            elif top_card.rank == Rank.KING:
                score += 8.0
            elif top_card.rank == Rank.QUEEN:
                score += 6.0
            elif top_card.rank == Rank.TEN:
                score += 4.0
            elif top_card.rank == Rank.NINE:
                score += 2.0  # Conservative players are more cautious about low cards
        else:
            # Ordering up a non-trump card (shouldn't happen in normal euchre)
            score -= 8.0
            
        # Consider if we have the left bower ourselves
        left_bower_in_hand = any(c.suit == left_bower_suit and c.rank == Rank.JACK for c in self.hand)
        if left_bower_in_hand:
            # We have the left bower, so ordering up this trump suit is more valuable
            score += 10.0
            
        # Consider if we have the right bower
        right_bower_in_hand = any(c.suit == potential_trump_suit and c.rank == Rank.JACK for c in self.hand)
        if right_bower_in_hand:
            # We have the right bower, so this trump suit is very strong for us
            score += 15.0
            
        return score


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
        
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Decide whether to order up the top card.
        
        Balanced players use sophisticated hand evaluation with moderate risk tolerance.
        """
        # Use the new hand evaluation system
        hand_strength = self._evaluate_hand_for_trump(top_card.suit, top_card, is_partner_dealing)
        
        # Balanced players have moderate thresholds
        # Conservative: 18+, Balanced: 15+, Aggressive: 12+
        base_threshold = 18.0 - (self.risk_ratio * 6.0)
        
        # Additional risk factors
        if hand_strength < 0:
            # Hand is strategically weak (e.g., would give dealer left bower)
            return False
            
        # Balanced approach based on risk ratio
        if self.risk_ratio < 0.3:
            # More conservative: order up with strong hands
            return hand_strength >= base_threshold * 1.1
        elif self.risk_ratio > 0.7:
            # More aggressive: order up with decent hands
            return hand_strength >= base_threshold * 0.9
        else:
            # Balanced: use standard threshold
            return hand_strength >= base_threshold
        
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play.
        
        Balanced players adapt their strategy based on the situation.
        Must follow suit if possible.
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
            
        # MUST follow suit if possible - this is a strict rule
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
            # Cannot follow suit - can play any card
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

    def _evaluate_hand_for_trump(self, potential_trump_suit: Suit, top_card: Card, is_partner_dealing: bool = False) -> float:
        """Evaluate the strength of the hand for a potential trump suit.
        
        Balanced players consider both risk and reward equally.
        """
        score = 0.0
        
        # Get all potential trump cards (including left bower)
        trump_cards = self.get_cards_of_suit(potential_trump_suit)
        left_bower_suit = self._get_left_bower_suit(potential_trump_suit)
        left_bower_cards = self.get_cards_of_suit(left_bower_suit)
        
        # Evaluate trump cards quality
        for card in trump_cards:
            if card.rank == Rank.JACK:
                score += 15.0  # Right bower is very powerful
            elif card.rank == Rank.ACE:
                score += 12.0
            elif card.rank == Rank.KING:
                score += 10.0
            elif card.rank == Rank.QUEEN:
                score += 8.0
            elif card.rank == Rank.TEN:
                score += 6.0
            elif card.rank == Rank.NINE:
                score += 4.0
                
        # Evaluate left bower cards quality
        for card in left_bower_cards:
            if card.rank == Rank.JACK:
                score += 14.0  # Left bower is second most powerful
            else:
                # Other cards in left bower suit are worth less
                score += card.rank.value * 0.5
                
        # Strategic considerations for ordering up
        if top_card.suit == potential_trump_suit:
            # We're considering ordering up the top card
            score += self._evaluate_top_card_strategy(top_card, potential_trump_suit, is_partner_dealing)
            
        # Bonus for having multiple trump cards (synergy)
        total_trump_potential = len(trump_cards) + len(left_bower_cards)
        if total_trump_potential >= 3:
            score += 5.0  # Bonus for having 3+ trump cards
        elif total_trump_potential >= 2:
            score += 2.0  # Bonus for having 2+ trump cards
            
        # Consider off-suit strength (cards that can win non-trump tricks)
        off_suit_cards = [c for c in self.hand if c.suit not in [potential_trump_suit, left_bower_suit]]
        off_suit_strength = sum(c.rank.value for c in off_suit_cards)
        score += off_suit_strength * 0.25  # Balanced players value off-suit moderately
        
        return score
        
    def _evaluate_trump_card_value(self, card: Card, trump_suit: Suit) -> float:
        """Evaluate the value of a trump card based on the actual trump suit."""
        if card.rank == Rank.JACK:
            return 15.0  # Right bower is always most powerful
        elif card.rank == Rank.ACE:
            return 12.0
        elif card.rank == Rank.KING:
            return 10.0
        elif card.rank == Rank.QUEEN:
            return 8.0
        elif card.rank == Rank.TEN:
            return 6.0
        elif card.rank == Rank.NINE:
            return 4.0
        else:
            return 0.0
            
    def _evaluate_left_bower_card_value(self, card: Card, trump_suit: Suit) -> float:
        """Evaluate the value of a left bower card based on the actual trump suit."""
        if card.rank == Rank.JACK:
            return 14.0  # Left bower is second most powerful
        else:
            # Other cards in left bower suit are worth less
            return card.rank.value * 0.5
            
    def _evaluate_off_suit_strength(self, off_suit_cards: List[Card], trump_suit: Suit) -> float:
        """Evaluate the strength of off-suit cards considering first player strategy."""
        if not off_suit_cards:
            return 0.0
            
        # Sort cards by rank (high to low)
        sorted_cards = sorted(off_suit_cards, key=lambda c: c.rank.value, reverse=True)
        
        # First player strategy: high off-suit cards are valuable for leading
        if self._is_first_player_after_dealer():
            # Value high cards more for leading
            score = sum(c.rank.value * (1.0 + (i * 0.1)) for i, c in enumerate(sorted_cards))
        else:
            # Standard off-suit evaluation
            score = sum(c.rank.value for c in sorted_cards)
            
        return score
        
    def _evaluate_partner_dealing_bonus(self, top_card: Card, potential_trump_suit: Suit) -> float:
        """Evaluate the bonus for having your partner as dealer."""
        bonus = 0.0
        
        # If partner is dealing, they get the top card
        if top_card.suit == potential_trump_suit:
            # Partner gets a trump card - evaluate its value
            if top_card.rank == Rank.JACK:
                bonus += 8.0  # Partner gets right bower
            elif top_card.rank == Rank.ACE:
                bonus += 6.0  # Partner gets Ace
            elif top_card.rank == Rank.KING:
                bonus += 4.0  # Partner gets King
            elif top_card.rank == Rank.QUEEN:
                bonus += 2.0  # Partner gets Queen
            elif top_card.rank == Rank.TEN:
                bonus += 1.0  # Partner gets Ten
            elif top_card.rank == Rank.NINE:
                bonus += 0.0  # Partner gets Nine
                
        # Check if ordering up gives partner the left bower
        left_bower_suit = self._get_left_bower_suit(potential_trump_suit)
        if top_card.suit == left_bower_suit and top_card.rank == Rank.JACK:
            # Partner gets left bower - this is good for your team
            bonus += 10.0
            
        # Consider if you have strong trump cards yourself
        trump_cards = self.get_cards_of_suit(potential_trump_suit)
        left_bower_cards = self.get_cards_of_suit(left_bower_suit)
        
        # If you have strong trump, partner getting more trump is synergistic
        if any(c.rank == Rank.JACK for c in trump_cards):  # Right bower
            bonus += 5.0
        if any(c.rank == Rank.JACK for c in left_bower_cards):  # Left bower
            bonus += 4.0
        if any(c.rank == Rank.ACE for c in trump_cards):  # Ace
            bonus += 3.0
            
        return bonus
        
    def _evaluate_first_player_strategy(self, trump_suit: Suit, left_bower_suit: Suit) -> float:
        """Evaluate the bonus for being first player after dealer."""
        bonus = 0.0
        
        # Get off-suit cards (not trump or left bower)
        off_suit_cards = [c for c in self.hand if c.suit not in [trump_suit, left_bower_suit]]
        
        if not off_suit_cards:
            return 0.0
            
        # Sort by rank (high to low)
        sorted_cards = sorted(off_suit_cards, key=lambda c: c.rank.value, reverse=True)
        
        # First player strategy: high off-suit cards are valuable for leading
        for i, card in enumerate(sorted_cards[:3]):  # Top 3 cards
            if card.rank == Rank.ACE:
                bonus += 4.0 - i
            elif card.rank == Rank.KING:
                bonus += 3.0 - i
            elif card.rank == Rank.QUEEN:
                bonus += 2.0 - i
            elif card.rank == Rank.JACK:
                bonus += 1.0 - i
                
        return bonus
        
    def _is_first_player_after_dealer(self) -> bool:
        """Check if this player is the first to play after the dealer."""
        # This would need to be implemented based on game state
        # For now, return False - this should be passed from the game logic
        return False
        
    def _evaluate_top_card_strategy(self, top_card: Card, potential_trump_suit: Suit, is_partner_dealing: bool = False) -> float:
        """Evaluate the strategic implications of ordering up the top card.
        
        Balanced players weigh risks and rewards equally.
        """
        score = 0.0
        
        # Check if ordering up gives dealer the left bower
        left_bower_suit = self._get_left_bower_suit(potential_trump_suit)
        if top_card.suit == left_bower_suit and top_card.rank == Rank.JACK:
            # CRITICAL: Ordering up a Jack of the left bower suit gives dealer left bower
            score -= 25.0  # This is extremely bad strategy
        elif top_card.suit == potential_trump_suit and top_card.rank == Rank.JACK:
            # Ordering up the right bower - this is good for us
            score += 10.0
        elif top_card.suit == potential_trump_suit:
            # Ordering up a non-Jack trump card
            if top_card.rank == Rank.ACE:
                score += 8.0
            elif top_card.rank == Rank.KING:
                score += 6.0
            elif top_card.rank == Rank.QUEEN:
                score += 4.0
            elif top_card.rank == Rank.TEN:
                score += 2.0
            elif top_card.rank == Rank.NINE:
                score += 0.0  # Neutral
        else:
            # Ordering up a non-trump card (shouldn't happen in normal euchre)
            score -= 5.0
            
        # Consider if we have the left bower ourselves
        left_bower_in_hand = any(c.suit == left_bower_suit and c.rank == Rank.JACK for c in self.hand)
        if left_bower_in_hand:
            # We have the left bower, so ordering up this trump suit is more valuable
            score += 8.0
            
        # Consider if we have the right bower
        right_bower_in_hand = any(c.suit == potential_trump_suit and c.rank == Rank.JACK for c in self.hand)
        if right_bower_in_hand:
            # We have the right bower, so this trump suit is very strong for us
            score += 12.0
            
        return score


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
        
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Decide whether to order up the top card.
        
        Opportunistic players use sophisticated hand evaluation and consider game situation.
        """
        # Use the new hand evaluation system
        hand_strength = self._evaluate_hand_for_trump(top_card.suit, top_card, is_partner_dealing)
        
        # Opportunistic players have moderate thresholds that adjust based on game situation
        # Base threshold: 12-18 depending on risk ratio
        base_threshold = 18.0 - (self.risk_ratio * 6.0)
        
        # Additional risk factors
        if hand_strength < 0:
            # Hand is strategically weak (e.g., would give dealer left bower)
            return False
            
        # Adjust threshold based on game situation
        if self.tricks_won_this_round >= 2:
            # Ahead in tricks - be more conservative
            adjusted_threshold = base_threshold * 1.2
        elif self.tricks_won_this_round == 0:
            # Behind in tricks - be more aggressive
            adjusted_threshold = base_threshold * 0.8
        else:
            # Balanced situation - use standard threshold
            adjusted_threshold = base_threshold
            
        # Consider risk ratio for final decision
        if self.risk_ratio > 0.7:
            # More aggressive: order up with decent hands
            adjusted_threshold *= 0.9
        elif self.risk_ratio < 0.3:
            # More conservative: order up with strong hands
            adjusted_threshold *= 1.1
            
        return hand_strength >= adjusted_threshold
        
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play.
        
        Opportunistic players adapt based on game situation.
        Must follow suit if possible.
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
            
        # MUST follow suit if possible - this is a strict rule
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
            # Cannot follow suit - can play any card
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

    def _evaluate_hand_for_trump(self, potential_trump_suit: Suit, top_card: Card, is_partner_dealing: bool = False) -> float:
        """Evaluate the strength of the hand for a potential trump suit.
        
        Opportunistic players consider both hand strength and game situation.
        """
        score = 0.0
        
        # Get all potential trump cards (including left bower)
        trump_cards = self.get_cards_of_suit(potential_trump_suit)
        left_bower_suit = self._get_left_bower_suit(potential_trump_suit)
        left_bower_cards = self.get_cards_of_suit(left_bower_suit)
        
        # Evaluate trump cards quality with proper suit consideration
        for card in trump_cards:
            score += self._evaluate_trump_card_value(card, potential_trump_suit)
                
        # Evaluate left bower cards quality with proper suit consideration
        for card in left_bower_cards:
            score += self._evaluate_left_bower_card_value(card, potential_trump_suit)
                
        # Strategic considerations for ordering up
        if top_card.suit == potential_trump_suit:
            # We're considering ordering up the top card
            score += self._evaluate_top_card_strategy(top_card, potential_trump_suit, is_partner_dealing)
            
        # Partner dealing bonus - if partner is dealing, they pick up what you order
        if is_partner_dealing:
            score += self._evaluate_partner_dealing_bonus(top_card, potential_trump_suit)
            
        # Bonus for having multiple trump cards (synergy)
        total_trump_potential = len(trump_cards) + len(left_bower_cards)
        if total_trump_potential >= 3:
            score += 6.0  # Opportunistic players value 3+ trump cards highly
        elif total_trump_potential >= 2:
            score += 3.0  # Bonus for having 2+ trump cards
            
        # Consider off-suit strength (cards that can win non-trump tricks)
        off_suit_cards = [c for c in self.hand if c.suit not in [potential_trump_suit, left_bower_suit]]
        off_suit_strength = self._evaluate_off_suit_strength(off_suit_cards, potential_trump_suit)
        score += off_suit_strength * 0.3  # Opportunistic players value off-suit moderately
        
        # Game situation adjustments
        if self.tricks_won_this_round >= 2:
            # Ahead in tricks - be more conservative
            score *= 0.8
        elif self.tricks_won_this_round == 0:
            # Behind in tricks - be more aggressive
            score *= 1.2
            
        # First player strategy consideration
        if self._is_first_player_after_dealer():
            score += self._evaluate_first_player_strategy(potential_trump_suit, left_bower_suit)
        
        return score
        
    def _evaluate_trump_card_value(self, card: Card, trump_suit: Suit) -> float:
        """Evaluate the value of a trump card based on the actual trump suit."""
        if card.rank == Rank.JACK:
            return 15.0  # Right bower is always most powerful
        elif card.rank == Rank.ACE:
            return 12.0
        elif card.rank == Rank.KING:
            return 10.0
        elif card.rank == Rank.QUEEN:
            return 8.0
        elif card.rank == Rank.TEN:
            return 6.0
        elif card.rank == Rank.NINE:
            return 4.0
        else:
            return 0.0
            
    def _evaluate_left_bower_card_value(self, card: Card, trump_suit: Suit) -> float:
        """Evaluate the value of a left bower card based on the actual trump suit."""
        if card.rank == Rank.JACK:
            return 14.0  # Left bower is second most powerful
        else:
            # Other cards in left bower suit are worth less
            return card.rank.value * 0.5
            
    def _evaluate_off_suit_strength(self, off_suit_cards: List[Card], trump_suit: Suit) -> float:
        """Evaluate the strength of off-suit cards considering first player strategy."""
        if not off_suit_cards:
            return 0.0
            
        # Sort cards by rank (high to low)
        sorted_cards = sorted(off_suit_cards, key=lambda c: c.rank.value, reverse=True)
        
        # First player strategy: high off-suit cards are valuable for leading
        if self._is_first_player_after_dealer():
            # Value high cards more for leading
            score = sum(c.rank.value * (1.0 + (i * 0.1)) for i, c in enumerate(sorted_cards))
        else:
            # Standard off-suit evaluation
            score = sum(c.rank.value for c in sorted_cards)
            
        return score
        
    def _evaluate_partner_dealing_bonus(self, top_card: Card, potential_trump_suit: Suit) -> float:
        """Evaluate the bonus for having your partner as dealer."""
        bonus = 0.0
        
        # If partner is dealing, they get the top card
        if top_card.suit == potential_trump_suit:
            # Partner gets a trump card - evaluate its value
            if top_card.rank == Rank.JACK:
                bonus += 8.0  # Partner gets right bower
            elif top_card.rank == Rank.ACE:
                bonus += 6.0  # Partner gets Ace
            elif top_card.rank == Rank.KING:
                bonus += 4.0  # Partner gets King
            elif top_card.rank == Rank.QUEEN:
                bonus += 2.0  # Partner gets Queen
            elif top_card.rank == Rank.TEN:
                bonus += 1.0  # Partner gets Ten
            elif top_card.rank == Rank.NINE:
                bonus += 0.0  # Partner gets Nine
                
        # Check if ordering up gives partner the left bower
        left_bower_suit = self._get_left_bower_suit(potential_trump_suit)
        if top_card.suit == left_bower_suit and top_card.rank == Rank.JACK:
            # Partner gets left bower - this is good for your team
            bonus += 10.0
            
        # Consider if you have strong trump cards yourself
        trump_cards = self.get_cards_of_suit(potential_trump_suit)
        left_bower_cards = self.get_cards_of_suit(left_bower_suit)
        
        # If you have strong trump, partner getting more trump is synergistic
        if any(c.rank == Rank.JACK for c in trump_cards):  # Right bower
            bonus += 5.0
        if any(c.rank == Rank.JACK for c in left_bower_cards):  # Left bower
            bonus += 4.0
        if any(c.rank == Rank.ACE for c in trump_cards):  # Ace
            bonus += 3.0
            
        return bonus
        
    def _evaluate_first_player_strategy(self, trump_suit: Suit, left_bower_suit: Suit) -> float:
        """Evaluate the bonus for being first player after dealer."""
        bonus = 0.0
        
        # Get off-suit cards (not trump or left bower)
        off_suit_cards = [c for c in self.hand if c.suit not in [trump_suit, left_bower_suit]]
        
        if not off_suit_cards:
            return 0.0
            
        # Sort by rank (high to low)
        sorted_cards = sorted(off_suit_cards, key=lambda c: c.rank.value, reverse=True)
        
        # First player strategy: high off-suit cards are valuable for leading
        for i, card in enumerate(sorted_cards[:3]):  # Top 3 cards
            if card.rank == Rank.ACE:
                bonus += 4.0 - i
            elif card.rank == Rank.KING:
                bonus += 3.0 - i
            elif card.rank == Rank.QUEEN:
                bonus += 2.0 - i
            elif card.rank == Rank.JACK:
                bonus += 1.0 - i
                
        return bonus
        
    def _is_first_player_after_dealer(self) -> bool:
        """Check if this player is the first to play after the dealer."""
        # This would need to be implemented based on game state
        # For now, return False - this should be passed from the game logic
        return False
        
    def _evaluate_top_card_strategy(self, top_card: Card, potential_trump_suit: Suit, is_partner_dealing: bool = False) -> float:
        """Evaluate the strategic implications of ordering up the top card.
        
        Opportunistic players consider both risks and game situation.
        """
        score = 0.0
        
        # Check if ordering up gives dealer the left bower
        left_bower_suit = self._get_left_bower_suit(potential_trump_suit)
        if top_card.suit == left_bower_suit and top_card.rank == Rank.JACK:
            # CRITICAL: Ordering up a Jack of the left bower suit gives dealer left bower
            score -= 25.0  # This is extremely bad strategy
        elif top_card.suit == potential_trump_suit and top_card.rank == Rank.JACK:
            # Ordering up the right bower - this is good for us
            score += 10.0
        elif top_card.suit == potential_trump_suit:
            # Ordering up a non-Jack trump card
            if top_card.rank == Rank.ACE:
                score += 8.0
            elif top_card.rank == Rank.KING:
                score += 6.0
            elif top_card.rank == Rank.QUEEN:
                score += 4.0
            elif top_card.rank == Rank.TEN:
                score += 2.0
            elif top_card.rank == Rank.NINE:
                score += 0.0  # Neutral
        else:
            # Ordering up a non-trump card (shouldn't happen in normal euchre)
            score -= 5.0
            
        # Consider if we have the left bower ourselves
        left_bower_in_hand = any(c.suit == left_bower_suit and c.rank == Rank.JACK for c in self.hand)
        if left_bower_in_hand:
            # We have the left bower, so ordering up this trump suit is more valuable
            score += 8.0
            
        # Consider if we have the right bower
        right_bower_in_hand = any(c.suit == potential_trump_suit and c.rank == Rank.JACK for c in self.hand)
        if right_bower_in_hand:
            # We have the right bower, so this trump suit is very strong for us
            score += 12.0
            
        return score 