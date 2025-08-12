"""Trump selection logic for the euchre game."""

from typing import List, Optional, Tuple
from ..models import Player, Card, Suit
from ..ai.ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI


class TrumpSelectionManager:
    """Manages trump selection during the game."""
    
    def __init__(self) -> None:
        """Initialize the trump selection manager."""
        pass
    
    def select_trump_suit(self, players: List[Player], top_card: Card, dealer: Player) -> Tuple[Optional[Suit], Optional[Player]]:
        """Handle trump selection for a round.
        
        Parameters
        ----------
        players : List[Player]
            List of players in the game
        top_card : Card
            The top card flipped up
        dealer : Player
            The current dealer
            
        Returns
        -------
        Tuple[Optional[Suit], Optional[Player]]
            The selected trump suit and the player who called it
        """
        # First round: players can order up the top card
        trump_suit, caller = self._first_round_selection(players, top_card, dealer)
        
        if trump_suit:
            return trump_suit, caller
        
        # Second round: dealer picks suit if no one ordered up
        trump_suit = self._dealer_suit_selection(dealer, top_card)
        return trump_suit, dealer
    
    def _first_round_selection(self, players: List[Player], top_card: Card, dealer: Player) -> Tuple[Optional[Suit], Optional[Player]]:
        """Handle first round of trump selection.
        
        Parameters
        ----------
        players : List[Player]
            List of players in the game
        top_card : Card
            The top card flipped up
        dealer : Player
            The current dealer
            
        Returns
        -------
        Tuple[Optional[Suit], Optional[Player]]
            The selected trump suit and the player who called it
        """
        # Start with the player to the left of the dealer
        dealer_index = players.index(dealer)
        start_index = (dealer_index + 1) % 4
        
        for i in range(4):
            player_index = (start_index + i) % 4
            player = players[player_index]
            
            if player.player_type.name == "AI":
                if self._ai_should_order_up(player, top_card, player_index == dealer_index):
                    return top_card.suit, player
            else:
                # Human player - would prompt here
                pass
        
        return None, None
    
    def _ai_should_order_up(self, player: Player, top_card: Card, is_dealer: bool) -> bool:
        """Determine if an AI player should order up the top card.
        
        Parameters
        ----------
        player : Player
            The AI player
        top_card : Card
            The top card flipped up
        is_dealer : bool
            Whether this player is the dealer
            
        Returns
        -------
        bool
            True if the AI should order up
        """
        # Check if player has the AI profile methods
        if hasattr(player, 'should_order_up'):
            return player.should_order_up(top_card, is_dealer)
        
        # Fallback to basic AI logic
        return self._basic_ai_trump_decision(player, top_card, is_dealer)
    
    def _basic_ai_trump_decision(self, player: Player, top_card: Card, is_dealer: bool) -> bool:
        """Basic AI trump decision logic.
        
        Parameters
        ----------
        player : Player
            The AI player
        top_card : Card
            The top card flipped up
        is_dealer : bool
            Whether this player is the dealer
            
        Returns
        -------
        bool
            True if the AI should order up
        """
        # Count cards of the potential trump suit
        trump_suit = top_card.suit
        cards_of_suit = [card for card in player.hand if card.suit == trump_suit]
        
        # Count high cards (10, J, Q, K, A)
        high_cards = [card for card in cards_of_suit if card.rank.value >= 10]
        
        # Basic decision: order up if you have 2+ cards of the suit or 1+ high cards
        return len(cards_of_suit) >= 2 or len(high_cards) >= 1
    
    def _dealer_suit_selection(self, dealer: Player, top_card: Card) -> Suit:
        """Handle dealer's suit selection when no one orders up.
        
        Parameters
        ----------
        dealer : Player
            The dealer
        top_card : Card
            The top card that was flipped up
            
        Returns
        -------
        Suit
            The suit selected by the dealer
        """
        if dealer.player_type.name == "AI":
            return self._ai_dealer_suit_selection(dealer)
        else:
            # Human dealer - would prompt here
            # For now, return a default suit
            return Suit.HEARTS
    
    def _ai_dealer_suit_selection(self, dealer: Player) -> Suit:
        """AI dealer's suit selection logic.
        
        Parameters
        ----------
        dealer : Player
            The AI dealer
            
        Returns
        -------
        Suit
            The suit selected by the dealer
        """
        # Count cards by suit
        suit_counts = {}
        for suit in Suit:
            suit_counts[suit] = len([card for card in dealer.hand if card.suit == suit])
        
        # Choose suit with most cards, or highest cards if tied
        best_suit = max(suit_counts.keys(), key=lambda s: (
            suit_counts[s], 
            max([card.rank.value for card in dealer.hand if card.suit == s] or [0])
        ))
        
        return best_suit
    
    def get_trump_selection_summary(self, trump_suit: Suit, caller: Player, top_card: Card) -> str:
        """Get a formatted summary of trump selection.
        
        Parameters
        ----------
        trump_suit : Suit
            The selected trump suit
        caller : Player
            The player who called trump
        top_card : Card
            The top card that was flipped up
            
        Returns
        -------
        str
            Formatted trump selection summary
        """
        if caller:
            return f"✅ Trump called by: {caller.name}\n🎯 Trump suit: {trump_suit.value.title()}"
        else:
            return f"🎯 Trump suit: {trump_suit.value.title()}"
    
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
    
    def is_partner_dealing(self, player: Player, dealer: Player) -> bool:
        """Check if a player's partner is dealing.
        
        Parameters
        ----------
        player : Player
            The player to check
        dealer : Player
            The current dealer
            
        Returns
        -------
        bool
            True if player's partner is dealing
        """
        # Players are partners if they have the same index parity (0,2 or 1,3)
        player_index = None
        dealer_index = None
        
        # Find indices (this would be better handled by the game state)
        # For now, assume we can determine this from the game context
        return False  # Placeholder 