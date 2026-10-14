"""Trick management and scoring for the euchre game."""

from typing import List, Optional, Tuple
from ..models import Player, Card, Trick, Suit


class TrickManager:
    """Manages tricks during gameplay."""
    
    def __init__(self) -> None:
        """Initialize the trick manager."""
        self.current_trick: Optional[Trick] = None
        self.tricks_this_round: List[Trick] = []
        self.renege_count: int = 0
        self.players: List[Player] = []
        self.trump_suit: Optional[Suit] = None
    
    def set_players(self, players: List[Player]) -> None:
        """Set the players for this trick manager.
        
        Parameters
        ----------
        players : List[Player]
            List of players in the game
        """
        self.players = players
    
    def set_trump_suit(self, trump_suit: Suit) -> None:
        """Set the trump suit for this round.
        
        Parameters
        ----------
        trump_suit : Suit
            The trump suit for this round
        """
        self.trump_suit = trump_suit
    
    def start_new_trick(self) -> None:
        """Start a new trick."""
        try:
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"Starting new trick")
            
            self.current_trick = Trick()
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"Created new trick: {self.current_trick} (type: {type(self.current_trick)})")
                self.logger.debug(f"Trick lead_suit: {self.current_trick.lead_suit} (type: {type(self.current_trick.lead_suit)})")
                self.logger.debug(f"Trick cards_played: {self.current_trick.cards_played} (type: {type(self.current_trick.cards_played)})")
                
        except Exception as e:
            # Catch and rethrow with debug context
            import traceback
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"ERROR in start_new_trick: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                import logging
                logging.error(f"ERROR in start_new_trick: {e}")
                logging.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def get_current_trick(self) -> Optional[Trick]:
        """Get the current trick.
        
        Returns
        -------
        Optional[Trick]
            The current trick, or None if no trick is in progress
        """
        return self.current_trick
    
    def play_card(self, player: Player, card: Card) -> None:
        """Play a card in the current trick.
        
        Parameters
        ----------
        player : Player
            The player playing the card
        card : Card
            The card being played
        """
        if not self.current_trick:
            raise ValueError("No current trick to play in")
        
        if player not in self.players:
            raise ValueError("Player not in this trick")
        
        if card not in player.hand:
            raise ValueError("Card not in player's hand")
        
        # Set lead suit if this is the first card
        if not self.current_trick.cards_played:
            try:
                self.current_trick.lead_suit = card.suit
            except Exception as e:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.error(f"Error setting lead suit: {e}")
                else:
                    import logging
                    logging.error(f"Error setting lead suit: {e}")
                raise
        
        # Add card to trick
        self.current_trick.add_card(player, card)
        
        # Remove card from player's hand
        player.hand.remove(card)
    
    def complete_trick(self, trump_suit: Optional[Suit] = None) -> Player:
        """Complete the current trick and determine the winner.
        
        Parameters
        ----------
        trump_suit : Optional[Suit]
            The trump suit for this round
            
        Returns
        -------
        Player
            The player who won the trick
        """
        if not self.current_trick:
            raise ValueError("No current trick to complete")
        
        # Determine winner
        winner = self._determine_trick_winner(trump_suit)
        winner.tricks_won += 1
        
        # Set winner on the trick object
        self.current_trick.winner = winner
        
        # Add completed trick to round
        self.tricks_this_round.append(self.current_trick)
        
        # Cards already removed from hands in play_card, no need to remove again
        
        # Clear current trick
        self.current_trick = None
        
        return winner
    
    def _determine_trick_winner(self, trump_suit: Optional[Suit] = None) -> Player:
        """Determine who won the current trick.
        
        Parameters
        ----------
        trump_suit : Optional[Suit]
            The trump suit for this round
            
        Returns
        -------
        Player
            The player who won the trick
        """
        if not self.current_trick or not self.current_trick.cards_played:
            raise ValueError("No cards played in current trick")
        
        winning_card = self.current_trick.cards_played[0][1]
        winning_player = self.current_trick.cards_played[0][0]
        
        for player, card in self.current_trick.cards_played[1:]:
            if self._card_beats(card, winning_card, trump_suit):
                winning_card = card
                winning_player = player
        
        return winning_player
    
    def _card_beats(self, card1: Card, card2: Card, trump_suit: Optional[Suit] = None) -> bool:
        """Determine if card1 beats card2.
        
        Parameters
        ----------
        card1 : Card
            The first card
        card2 : Card
            The second card
        trump_suit : Optional[Suit]
            The trump suit for this round
            
        Returns
        -------
        bool
            True if card1 beats card2, False otherwise
        """
        try:
            # Safety check: ensure lead_suit is a Suit object
            if hasattr(self, 'current_trick') and self.current_trick and hasattr(self.current_trick, 'lead_suit'):
                lead_suit = self.current_trick.lead_suit
                if not isinstance(lead_suit, Suit) and lead_suit is not None:
                    # If lead_suit is corrupted, log it and use None
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.error(f"ERROR: lead_suit is corrupted: {lead_suit} (type: {type(lead_suit)})")
                    else:
                        import logging
                        logging.error(f"ERROR: lead_suit is corrupted: {lead_suit} (type: {type(lead_suit)})")
                    lead_suit = None
            else:
                lead_suit = None
            
            # Use the Card.beats method which properly implements bower rules
            return card1.beats(card2, trump_suit, lead_suit)
            
        except Exception as e:
            # Catch and rethrow with debug context
            import traceback
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"ERROR in _card_beats method: {e}")
                self.logger.error(f"Card1: {card1} (type: {type(card1)})")
                self.logger.error(f"Card2: {card2} (type: {type(card2)})")
                self.logger.error(f"Trump suit: {trump_suit} (type: {type(trump_suit)})")
                self.logger.error(f"Lead suit: {lead_suit if 'lead_suit' in locals() else 'Not set'} (type: {type(lead_suit) if 'lead_suit' in locals() else 'Not set'})")
                self.logger.error(f"Current trick: {self.current_trick if hasattr(self, 'current_trick') else 'No current_trick'}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                import logging
                logging.error(f"ERROR in _card_beats method: {e}")
                logging.error(f"Card1: {card1} (type: {type(card1)})")
                logging.error(f"Card2: {card2} (type: {type(card2)})")
                logging.error(f"Trump suit: {trump_suit} (type: {type(trump_suit)})")
                logging.error(f"Lead suit: {lead_suit if 'lead_suit' in locals() else 'Not set'} (type: {type(lead_suit) if 'lead_suit' in locals() else 'Not set'})")
                logging.error(f"Current trick: {self.current_trick if hasattr(self, 'current_trick') else 'No current_trick'}")
                logging.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    # Method removed - cards are now removed immediately when played
    
    def _is_renege(self, player: Player, card: Card, lead_suit: Suit) -> bool:
        """Check if a player reneged (didn't follow suit when possible).
        
        Parameters
        ----------
        player : Player
            The player who played the card
        card : Card
            The card played
        lead_suit : Suit
            The lead suit of the trick
            
        Returns
        -------
        bool
            True if player reneged
        """
        # If card follows lead suit, no renege
        if card.suit == lead_suit:
            return False
        
        # Check if player has cards of lead suit
        return player.has_suit(lead_suit)
    
    def get_round_results(self) -> List[int]:
        """Get the trick counts for each player in the current round.
        
        Returns
        -------
        List[int]
            List of trick counts for each player
        """
        if not self.tricks_this_round:
            return [0, 0, 0, 0]
        
        # Get players from the first trick
        players = [player for player, _ in self.tricks_this_round[0].cards_played]
        results = [0] * len(players)
        
        for trick in self.tricks_this_round:
            for player, _ in trick.cards_played:
                if player in players:
                    player_index = players.index(player)
                    results[player_index] = player.tricks_won
        
        return results
    
    def clear_round(self) -> None:
        """Clear the current round's tricks."""
        self.tricks_this_round.clear()
        self.current_trick = None
    
    def get_renege_count(self) -> int:
        """Get the total number of reneges in the game.
        
        Returns
        -------
        int
            Total renege count
        """
        return self.renege_count
    
    def reset(self) -> None:
        """Reset the trick manager."""
        self.current_trick = None
        self.tricks_this_round.clear()
        self.renege_count = 0 