"""
AI Adapter for Game Engine Integration

This module provides adapter methods that bridge the old game engine interface
with the new BaseAIInterface, allowing seamless integration without breaking
existing game logic.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-13
Context: Creating AI adapter to fix core game engine integration
"""

from typing import Optional, List, Tuple
from .base_ai_interface import BaseAIInterface, GameContext, DecisionResult, DecisionType
from ..models import Player, Card, Suit, Trick


class AIAdapter:
    """Adapter class that bridges old game engine interface with new BaseAIInterface."""
    
    @staticmethod
    def should_order_up(player: Player, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """
        Adapter method for should_order_up that works with both old and new AI interfaces.
        
        Parameters
        ----------
        player : Player
            The AI player
        top_card : Card
            The top card flipped up
        is_partner_dealing : bool
            Whether the dealer is the player's partner
            
        Returns
        -------
        bool
            True if the AI should order up
        """
        # Check if player implements the new BaseAIInterface
        if isinstance(player, BaseAIInterface):
            # Create GameContext for new interface
            context = AIAdapter._create_order_up_context(player, top_card, is_partner_dealing)
            
            # Call new interface method
            decision = player.should_order_up(context)
            
            # Return boolean result
            return decision.decision_type == DecisionType.ORDER_UP
        
        # Check if player has old interface method
        elif hasattr(player, 'should_order_up'):
            # Call old interface method
            return player.should_order_up(top_card, is_partner_dealing)
        
        # Fallback to basic logic
        return AIAdapter._basic_order_up_logic(player, top_card, is_partner_dealing)
    
    @staticmethod
    def should_call_trump(player: Player, hand: List[Card], top_card: Card, game_state: dict) -> bool:
        """
        Adapter method for should_call_trump that works with both old and new AI interfaces.
        
        Parameters
        ----------
        player : Player
            The AI player
        hand : List[Card]
            The player's hand
        top_card : Card
            The top card that was turned down
        game_state : dict
            Current game state information
            
        Returns
        -------
        bool
            True if the AI should call trump
        """
        # Check if player implements the new BaseAIInterface
        if isinstance(player, BaseAIInterface):
            # Create GameContext for new interface
            context = AIAdapter._create_call_trump_context(player, hand, top_card, game_state)
            
            # Call new interface method
            decision = player.should_call_trump(context)
            
            # Return boolean result
            return decision.decision_type == DecisionType.CALL_TRUMP
        
        # Check if player has old interface method
        elif hasattr(player, 'should_call_trump'):
            # Call old interface method
            return player.should_call_trump(hand, top_card, game_state)
        
        # Fallback to basic logic
        return AIAdapter._basic_call_trump_logic(player, hand, top_card, game_state)
    
    @staticmethod
    def select_trump_suit(player: Player, hand: List[Card], top_card: Card, game_state: dict) -> Suit:
        """
        Adapter method for select_trump_suit that works with both old and new AI interfaces.
        
        Parameters
        ----------
        player : Player
            The AI player
        hand : List[Card]
            The player's hand
        top_card : Card
            The top card that was turned down
        game_state : dict
            Current game state information
            
        Returns
        -------
        Suit
            The selected trump suit
        """
        # Check if player implements the new BaseAIInterface
        if isinstance(player, BaseAIInterface):
            # Create GameContext for new interface
            context = AIAdapter._create_select_trump_context(player, hand, top_card, game_state)
            
            # Call new interface method
            decision = player.select_trump_suit(context)
            
            # Extract suit from decision metadata
            if decision.metadata and 'selected_suit' in decision.metadata:
                suit_name = decision.metadata['selected_suit']
                return Suit[suit_name]
            
            # Fallback to basic logic if metadata is missing
            return AIAdapter._basic_trump_suit_selection(player, hand, top_card, game_state)
        
        # Check if player has old interface method
        elif hasattr(player, 'select_trump_suit'):
            # Call old interface method
            return player.select_trump_suit(hand, top_card, game_state)
        
        # Fallback to basic logic
        return AIAdapter._basic_trump_suit_selection(player, hand, top_card, game_state)
    
    @staticmethod
    def play_card(player: Player, hand: List[Card], lead_suit: Optional[Suit], 
                  trump_suit: Optional[Suit], current_trick: Optional[Trick], 
                  game_state: dict) -> Card:
        """
        Adapter method for play_card that works with both old and new AI interfaces.
        
        Parameters
        ----------
        player : Player
            The AI player
        hand : List[Card]
            The player's hand
        lead_suit : Optional[Suit]
            The lead suit of the current trick
        trump_suit : Optional[Suit]
            The trump suit for this round
        current_trick : Optional[Trick]
            The current trick being played
        game_state : dict
            Current game state information
            
        Returns
        -------
        Card
            The card to play
        """
        # Check if player implements the new BaseAIInterface
        if isinstance(player, BaseAIInterface):
            # Create GameContext for new interface
            context = AIAdapter._create_play_card_context(player, hand, lead_suit, trump_suit, current_trick, game_state)
            
            # Call new interface method
            decision = player.play_card(context)
            
            # Extract card from decision metadata
            if decision.metadata and 'selected_card' in decision.metadata:
                card_str = decision.metadata['selected_card']
                # Parse card string (e.g., "Hearts of Ace")
                return AIAdapter._parse_card_string(card_str, hand)
            
            # Fallback to basic logic if metadata is missing
            return AIAdapter._basic_card_selection(player, hand, lead_suit, trump_suit)
        
        # Check if player has old interface method
        elif hasattr(player, 'choose_card_to_play'):
            # Call old interface method
            return player.choose_card_to_play(lead_suit, trump_suit)
        
        # Fallback to basic logic
        return AIAdapter._basic_card_selection(player, hand, lead_suit, trump_suit)
    
    @staticmethod
    def discard_card(player: Player, hand: List[Card], trump_suit: Suit, game_state: dict) -> Card:
        """
        Adapter method for discard_card that works with both old and new AI interfaces.
        
        Parameters
        ----------
        player : Player
            The AI player
        hand : List[Card]
            The player's hand (including the flipped card)
        trump_suit : Suit
            The trump suit that was called
        game_state : dict
            Current game state information
            
        Returns
        -------
        Card
            The card to discard
        """
        # Check if player implements the new BaseAIInterface
        if isinstance(player, BaseAIInterface):
            # Create GameContext for new interface
            context = AIAdapter._create_discard_context(player, hand, trump_suit, game_state)
            
            # Call new interface method
            decision = player.discard_card(context)
            
            # Extract card from decision metadata
            if decision.metadata and 'discarded_card' in decision.metadata:
                card_str = decision.metadata['discarded_card']
                # Parse card string
                return AIAdapter._parse_card_string(card_str, hand)
            
            # Fallback to basic logic if metadata is missing
            return AIAdapter._basic_discard_selection(player, hand, trump_suit)
        
        # Check if player has old interface method
        elif hasattr(player, 'discard_card'):
            # Call old interface method
            return player.discard_card(hand, trump_suit, game_state)
        
        # Fallback to basic logic
        return AIAdapter._basic_discard_selection(player, hand, trump_suit)
    
    # Helper methods for creating GameContext objects
    @staticmethod
    def _create_order_up_context(player: Player, top_card: Card, is_partner_dealing: bool) -> GameContext:
        """Create GameContext for order up decision."""
        # This is a simplified context - in a real implementation you'd have more game state
        return GameContext(
            hand=player.hand,
            position=0,  # Would need to get actual position
            is_dealer=False,  # Would need to get actual dealer status
            partner_position=2,  # Would need to get actual partner position
            flipped_card=top_card,
            trump_suit=None,
            current_trick=[],
            trick_suit=None,
            team1_score=0,
            team2_score=0,
            tricks_won_team1=0,
            tricks_won_team2=0,
            current_trick_number=1,
            partner_is_dealer=is_partner_dealing,
            partner_hand_size=5,
            opponent1_hand_size=5,
            opponent2_hand_size=5
        )
    
    @staticmethod
    def _create_call_trump_context(player: Player, hand: List[Card], top_card: Card, game_state: dict) -> GameContext:
        """Create GameContext for call trump decision."""
        return GameContext(
            hand=hand,
            position=0,
            is_dealer=False,
            partner_position=2,
            flipped_card=top_card,
            trump_suit=None,
            current_trick=[],
            trick_suit=None,
            team1_score=game_state.get('team1_score', 0),
            team2_score=game_state.get('team2_score', 0),
            tricks_won_team1=game_state.get('tricks_won_team1', 0),
            tricks_won_team2=game_state.get('tricks_won_team2', 0),
            current_trick_number=game_state.get('current_trick_number', 1),
            partner_is_dealer=False,
            partner_hand_size=5,
            opponent1_hand_size=5,
            opponent2_hand_size=5
        )
    
    @staticmethod
    def _create_select_trump_context(player: Player, hand: List[Card], top_card: Card, game_state: dict) -> GameContext:
        """Create GameContext for select trump suit decision."""
        return GameContext(
            hand=hand,
            position=0,
            is_dealer=False,
            partner_position=2,
            flipped_card=top_card,
            trump_suit=None,
            current_trick=[],
            trick_suit=None,
            team1_score=game_state.get('team1_score', 0),
            team2_score=game_state.get('team2_score', 0),
            tricks_won_team1=game_state.get('tricks_won_team1', 0),
            tricks_won_team2=game_state.get('tricks_won_team2', 0),
            current_trick_number=game_state.get('current_trick_number', 1),
            partner_is_dealer=False,
            partner_hand_size=5,
            opponent1_hand_size=5,
            opponent2_hand_size=5
        )
    
    @staticmethod
    def _create_play_card_context(player: Player, hand: List[Card], lead_suit: Optional[Suit], 
                                 trump_suit: Optional[Suit], current_trick: Optional[Trick], 
                                 game_state: dict) -> GameContext:
        """Create GameContext for play card decision."""
        # Convert current trick to the format expected by GameContext
        trick_data = []
        if current_trick and hasattr(current_trick, 'cards_played'):
            for i, (player_name, card) in enumerate(current_trick.cards_played):
                trick_data.append((i, card))
        
        return GameContext(
            hand=hand,
            position=0,
            is_dealer=False,
            partner_position=2,
            flipped_card=None,
            trump_suit=trump_suit,
            current_trick=trick_data,
            trick_suit=lead_suit,
            team1_score=game_state.get('team1_score', 0),
            team2_score=game_state.get('team2_score', 0),
            tricks_won_team1=game_state.get('tricks_won_team1', 0),
            tricks_won_team2=game_state.get('tricks_won_team2', 0),
            current_trick_number=game_state.get('current_trick_number', 1),
            partner_is_dealer=False,
            partner_hand_size=5,
            opponent1_hand_size=5,
            opponent2_hand_size=5
        )
    
    @staticmethod
    def _create_discard_context(player: Player, hand: List[Card], trump_suit: Suit, game_state: dict) -> GameContext:
        """Create GameContext for discard card decision."""
        return GameContext(
            hand=hand,
            position=0,
            is_dealer=False,
            partner_position=2,
            flipped_card=None,
            trump_suit=trump_suit,
            current_trick=[],
            trick_suit=None,
            team1_score=game_state.get('team1_score', 0),
            team2_score=game_state.get('team2_score', 0),
            tricks_won_team1=game_state.get('tricks_won_team1', 0),
            tricks_won_team2=game_state.get('tricks_won_team2', 0),
            current_trick_number=game_state.get('current_trick_number', 1),
            partner_is_dealer=False,
            partner_hand_size=5,
            opponent1_hand_size=5,
            opponent2_hand_size=5
        )
    
    # Fallback logic methods
    @staticmethod
    def _basic_order_up_logic(player: Player, top_card: Card, is_partner_dealing: bool) -> bool:
        """Basic logic for order up decision."""
        # Count cards of the potential trump suit
        trump_suit = top_card.suit
        cards_of_suit = [card for card in player.hand if card.suit == trump_suit]
        
        # Count high cards (10, J, Q, K, A)
        high_cards = [card for card in cards_of_suit if card.rank.value >= 10]
        
        # Basic decision: order up if you have 2+ cards of the suit or 1+ high cards
        return len(cards_of_suit) >= 2 or len(high_cards) >= 1
    
    @staticmethod
    def _basic_call_trump_logic(player: Player, hand: List[Card], top_card: Card, game_state: dict) -> bool:
        """Basic logic for call trump decision."""
        # Similar to order up logic but for second round
        # Count cards by suit (excluding the turned down suit)
        suit_counts = {}
        for card in hand:
            if card.suit != top_card.suit:
                suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
        
        # Find suit with most cards
        if suit_counts:
            best_suit = max(suit_counts, key=suit_counts.get)
            best_count = suit_counts[best_suit]
            
            # Call trump if you have 3+ cards of a suit
            return best_count >= 3
        
        return False
    
    @staticmethod
    def _basic_trump_suit_selection(player: Player, hand: List[Card], top_card: Card, game_state: dict) -> Suit:
        """Basic logic for trump suit selection."""
        # Count cards by suit (excluding the turned down suit)
        suit_counts = {}
        for card in hand:
            if card.suit != top_card.suit:
                suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
        
        # Return suit with most cards, or default to hearts
        if suit_counts:
            return max(suit_counts, key=suit_counts.get)
        
        return Suit.HEARTS
    
    @staticmethod
    def _basic_card_selection(player: Player, hand: List[Card], lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Basic logic for card selection."""
        if not lead_suit:
            # Leading - play highest card
            return max(hand, key=lambda c: c.rank.value)
        
        # Must follow suit if possible
        cards_of_suit = [card for card in hand if card.suit == lead_suit]
        if cards_of_suit:
            return max(cards_of_suit, key=lambda c: c.rank.value)
        else:
            # Can't follow suit - play any card
            return max(hand, key=lambda c: c.rank.value)
    
    @staticmethod
    def _basic_discard_selection(player: Player, hand: List[Card], trump_suit: Suit) -> Card:
        """Basic logic for discard selection."""
        # Find the lowest value non-trump card
        non_trump_cards = [card for card in hand if card.suit != trump_suit]
        
        if non_trump_cards:
            return min(non_trump_cards, key=lambda c: c.rank.value)
        
        # If all cards are trump, discard the lowest trump
        return min(hand, key=lambda c: c.rank.value)
    
    @staticmethod
    def _parse_card_string(card_str: str, hand: List[Card]) -> Card:
        """Parse card string and find matching card in hand."""
        # Simple parsing for "Suit of Rank" format
        try:
            parts = card_str.split(' of ')
            if len(parts) == 2:
                suit_name, rank_name = parts
                suit = Suit[suit_name.upper()]
                rank = None
                
                # Map rank names to Rank enum
                rank_map = {
                    'ACE': 14, 'KING': 13, 'QUEEN': 12, 'JACK': 11,
                    'TEN': 10, 'NINE': 9
                }
                
                if rank_name.upper() in rank_map:
                    rank_value = rank_map[rank_name.upper()]
                    # Find the rank enum with this value
                    for r in [r for r in dir(rank) if not r.startswith('_')]:
                        if hasattr(rank, r) and getattr(rank, r).value == rank_value:
                            rank = getattr(rank, r)
                            break
                
                if suit and rank:
                    # Find matching card in hand
                    for card in hand:
                        if card.suit == suit and card.rank == rank:
                            return card
        except:
            pass
        
        # Fallback: return first card in hand
        return hand[0] if hand else None 