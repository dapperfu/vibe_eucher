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
    
    def start_new_trick(self) -> None:
        """Start a new trick."""
        self.current_trick = Trick()
    
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
            raise ValueError("No current trick")
        
        # Set lead suit if this is the first card
        if not self.current_trick.lead_suit:
            self.current_trick.lead_suit = card.suit
        
        # Add card to trick
        self.current_trick.cards_played.append((player, card))
        
        # Remove card from player's hand immediately
        if card in player.hand:
            player.hand.remove(card)
        
        # Check for reneging
        if self._is_renege(player, card, self.current_trick.lead_suit):
            self.renege_count += 1
    
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
        """Check if card1 beats card2.
        
        Parameters
        ----------
        card1 : Card
            First card
        card2 : Card
            Second card
        trump_suit : Optional[Suit]
            The trump suit for this round
            
        Returns
        -------
        bool
            True if card1 beats card2
        """
        # Check if cards are trump cards
        card1_is_trump = card1.is_trump_card(trump_suit) if trump_suit else False
        card2_is_trump = card2.is_trump_card(trump_suit) if trump_suit else False
        
        # Trump cards always beat non-trump cards
        if card1_is_trump and not card2_is_trump:
            return True
        if not card1_is_trump and card2_is_trump:
            return False
        
        # If both are trump or both are non-trump, compare values
        if card1_is_trump and card2_is_trump:
            return card1.get_trump_value(trump_suit) > card2.get_trump_value(trump_suit)
        
        # Both non-trump - must follow lead suit
        if card1.suit == self.current_trick.lead_suit and card2.suit == self.current_trick.lead_suit:
            return card1.rank.value > card2.rank.value
        
        # If card1 doesn't follow lead suit, it can't win
        if card1.suit != self.current_trick.lead_suit:
            return False
        
        # card1 follows lead suit, card2 doesn't
        return True
    
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