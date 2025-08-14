"""Trump selection logic for the euchre game."""

from typing import List, Optional, Tuple
from ..models import Player, Card, Suit
from ..ai.ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI
from ..ai.ai_adapter import AIAdapter


class TrumpSelectionManager:
    """Manages trump selection during the game."""
    
    def __init__(self, logger=None) -> None:
        """Initialize the trump selection manager.
        
        Parameters
        ----------
        logger : Optional
            Logger instance for verbose output
        """
        self.logger = logger
    
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
        if self.logger:
            self.logger.info(f"🎯 Trump Selection Process Begins")
            self.logger.info(f"   Top card: {top_card.unicode_str()}")
            self.logger.info(f"   Dealer: {dealer.name}")
            self.logger.info(f"   Starting with player to dealer's left...")
        
        # First round: players can order up the top card
        trump_suit, caller = self._first_round_selection(players, top_card, dealer)
        
        if trump_suit:
            if self.logger:
                self.logger.info(f"✅ First round trump selection successful!")
            return trump_suit, caller
        
        # Second round: dealer picks suit if no one ordered up
        if self.logger:
            self.logger.info("🔄 No one ordered up. Second round begins.")
        trump_suit = self._second_round_selection(players, dealer, top_card)
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
            
            if self.logger:
                self.logger.info(f"   {player.name} considering...")
            
            if player.player_type.name == "AI":
                if AIAdapter.should_order_up(player, top_card, player_index == dealer_index):
                    if self.logger:
                        self.logger.info(f"   🎯 {player.name} ORDERS IT UP!")
                    return top_card.suit, player
                else:
                    if self.logger:
                        self.logger.info(f"   ❌ {player.name} passes")
            else:
                # Human player - would prompt here
                if self.logger:
                    self.logger.info(f"   ❌ {player.name} passes")
        
        return None, None
    
    def _second_round_selection(self, players: List[Player], dealer: Player, top_card: Card) -> Suit:
        """Handle second round of trump selection.
        
        Parameters
        ----------
        players : List[Player]
            List of players in the game
        top_card : Card
            The top card that was flipped up
        dealer : Player
            The current dealer
            
        Returns
        -------
        Suit
            The suit selected by the dealer
        """
        # Start with the player to the left of the dealer
        dealer_index = players.index(dealer)
        start_index = (dealer_index + 1) % 4
        
        for i in range(4):
            player_index = (start_index + i) % 4
            player = players[player_index]
            
            if self.logger:
                self.logger.info(f"   {player.name} considering second round...")
            
            if player.player_type.name == "AI":
                # AI players can call any suit as trump (excluding the turned down suit)
                if hasattr(player, 'choose_trump_suit'):
                    trump_suit = player.choose_trump_suit(top_card)
                    if trump_suit and trump_suit != top_card.suit:
                        if self.logger:
                            self.logger.info(f"   🎯 {player.name} calls {trump_suit.name} as trump!")
                        return trump_suit
                
                # Fallback: pass
                if self.logger:
                    self.logger.info(f"   ❌ {player.name} passes")
            else:
                # Human player - would prompt here
                if self.logger:
                    self.logger.info(f"   ❌ {player.name} passes")
        
        # If no one calls, dealer must pick
        if self.logger:
            self.logger.info(f"   🎲 {dealer.name} must pick a suit (dealer's choice).")
        trump_suit = self._dealer_suit_selection(dealer, top_card)
        if self.logger:
            self.logger.info(f"   👑 {dealer.name} picks {trump_suit.name} as trump.")
        return trump_suit
    
    # AI method calls now use AIAdapter.should_order_up() directly
    
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
            return self._ai_dealer_suit_selection(dealer, top_card)
        else:
            # Human dealer - would prompt here
            # For now, return a default suit (excluding the turned down suit)
            available_suits = [suit for suit in Suit if suit != top_card.suit]
            return available_suits[0] if available_suits else Suit.HEARTS
    
    def _ai_dealer_suit_selection(self, dealer: Player, top_card: Card) -> Suit:
        """AI dealer's suit selection logic.
        
        Parameters
        ----------
        dealer : Player
            The AI dealer
        top_card : Card
            The top card that was turned down (cannot be selected as trump)
            
        Returns
        -------
        Suit
            The suit selected by the dealer
        """
        # Count cards by suit (excluding the turned down suit)
        suit_counts = {}
        for suit in Suit:
            if suit != top_card.suit:  # Cannot select the turned down suit
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