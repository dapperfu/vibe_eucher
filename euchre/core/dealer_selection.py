"""
Traditional Euchre dealer selection using the "Black Jack" method.

This module implements the classic Euchre tradition where the first dealer
is determined by shuffling the deck and dealing cards until the first
"Black Jack" (Jack of Clubs or Jack of Spades) is turned up.
"""

from typing import List, Tuple
from ..models import Card, Suit, Rank, Player
from .deck import Deck


class DealerSelection:
    """Handles traditional Euchre dealer selection using the 'Black Jack' method."""
    
    def __init__(self, players: List[Player]) -> None:
        """Initialize the dealer selection process.
        
        Parameters
        ----------
        players : List[Player]
            List of players participating in the dealer selection
        """
        if len(players) != 4:
            raise ValueError("Euchre requires exactly 4 players")
        
        self.players = players
        self.deck = Deck()
    
    def select_first_dealer(self, method: str = "black_jack", verbose: bool = True) -> Tuple[Player, List[Card]]:
        """Select the first dealer using the specified method.
        
        Parameters
        ----------
        method : str
            Selection method: "black_jack" or "high_card"
        verbose : bool
            Whether to display the selection process
            
        Returns
        -------
        Tuple[Player, List[Card]]
            The selected first dealer and the list of cards dealt during selection
        """
        if method == "black_jack":
            return self._black_jack_selection(verbose)
        elif method == "high_card":
            return self._high_card_selection(verbose)
        else:
            raise ValueError(f"Unknown dealer selection method: {method}")
    
    def _black_jack_selection(self, verbose: bool = True) -> Tuple[Player, List[Card]]:
        """Select the first dealer using the traditional 'Black Jack' method.
        
        This method:
        1. Shuffles the deck thoroughly
        2. Deals cards one by one to players in rotation
        3. Identifies the first "Black Jack" (Jack of Clubs or Jack of Spades)
        4. Returns the player who receives that card as the first dealer
        
        Parameters
        ----------
        verbose : bool
            Whether to display the selection process
            
        Returns
        -------
        Tuple[Player, List[Card]]
            The selected first dealer and the list of cards dealt during selection
        """
        # Reset and shuffle the deck
        self.deck.reset_and_shuffle()
        
        # Shuffle multiple times for thorough randomization
        self.deck.shuffle_times(3)
        
        if verbose:
            print("🎲 Selecting first dealer using traditional 'Black Jack' method...")
            print("🃏 Shuffling deck thoroughly...")
        
        # Deal cards one by one until we find a Black Jack
        dealt_cards = []
        current_player_index = 0
        
        while not self.deck.is_empty:
            # Deal one card to the current player
            card = self.deck.draw_top_card()
            if card is None:
                break
                
            current_player = self.players[current_player_index]
            dealt_cards.append(card)
            
            if verbose:
                print(f"  {current_player.name} receives: {card.unicode_str()}")
            
            # Check if this is a Black Jack
            if self._is_black_jack(card):
                if verbose:
                    print(f"\n🎯 {current_player.name} receives the first Black Jack: {card.unicode_str()}")
                    print(f"👑 {current_player.name} is selected as the first dealer!")
                
                return current_player, dealt_cards
            
            # Move to next player
            current_player_index = (current_player_index + 1) % 4
        
        # Fallback: if no Black Jack found (shouldn't happen with a proper deck)
        if verbose:
            print("⚠️  No Black Jack found - using fallback selection")
        
        # Return first player as fallback
        return self.players[0], dealt_cards
    
    def _high_card_selection(self, verbose: bool = True) -> Tuple[Player, List[Card]]:
        """Select the first dealer using the 'High Card Draw' method.
        
        This method:
        1. Shuffles the deck thoroughly
        2. Deals one card to each player
        3. Player with highest card becomes dealer
        4. In case of ties, only tied players draw additional cards until a winner is determined
        
        Parameters
        ----------
        verbose : bool
            Whether to display the selection process
            
        Returns
        -------
        Tuple[Player, List[Card]]
            The selected first dealer and the list of cards dealt during selection
        """
        # Reset and shuffle the deck
        self.deck.reset_and_shuffle()
        
        # Shuffle multiple times for thorough randomization
        self.deck.shuffle_times(3)
        
        if verbose:
            print("🎲 Selecting first dealer using 'High Card Draw' method...")
            print("🃏 Shuffling deck thoroughly...")
        
        dealt_cards = []
        round_number = 1
        current_players = self.players.copy()  # Start with all players
        
        while not self.deck.is_empty:
            if verbose:
                print(f"\n--- Round {round_number} ---")
            
            # Deal one card to each current player (all players in round 1, tied players in subsequent rounds)
            round_cards = []
            for player in current_players:
                if self.deck.is_empty:
                    break
                    
                card = self.deck.draw_top_card()
                if card is None:
                    break
                    
                round_cards.append((player, card))
                dealt_cards.append(card)
                
                if verbose:
                    print(f"  {player.name} draws: {card.unicode_str()}")
            
            if not round_cards:
                break
            
            # Find the highest card in this round
            highest_card = max(round_cards, key=lambda x: x[1].rank.value)
            highest_rank = highest_card[1].rank.value
            
            # Check for ties
            tied_players = [(player, card) for player, card in round_cards if card.rank.value == highest_rank]
            
            if len(tied_players) == 1:
                # No tie - we have a winner
                winner = tied_players[0][0]
                if verbose:
                    print(f"\n🎯 {winner.name} has the highest card: {highest_card[1].unicode_str()}")
                    print(f"👑 {winner.name} is selected as the first dealer!")
                
                return winner, dealt_cards
            else:
                # Tie - show tied players and continue to next round with only tied players
                if verbose:
                    tied_names = [player.name for player, card in tied_players]
                    print(f"🤝 Tie between: {', '.join(tied_names)} with {highest_card[1].rank.name}")
                    print("🔄 Drawing additional cards to break tie...")
                
                # Update current_players to only include tied players for the next round
                current_players = [player for player, card in tied_players]
                round_number += 1
        
        # Fallback: if we run out of cards, pick the first player
        if verbose:
            print("⚠️  Ran out of cards - using fallback selection")
        
        return self.players[0], dealt_cards
    
    def _is_black_jack(self, card: Card) -> bool:
        """Check if a card is a 'Black Jack'.
        
        In Euchre tradition, a 'Black Jack' is either:
        - Jack of Clubs (♣️)
        - Jack of Spades (♠️)
        
        Parameters
        ----------
        card : Card
            The card to check
            
        Returns
        -------
        bool
            True if the card is a Black Jack
        """
        return (card.rank == Rank.JACK and 
                card.suit in [Suit.CLUBS, Suit.SPADES])
    
    def get_selection_summary(self, dealer: Player, dealt_cards: List[Card]) -> str:
        """Get a formatted summary of the dealer selection process.
        
        Parameters
        ----------
        dealer : Player
            The selected first dealer
        dealt_cards : List[Card]
            The cards dealt during selection
            
        Returns
        -------
        str
            Formatted summary of the selection process
        """
        summary = [
            "🎲 Dealer Selection Complete!",
            f"👑 First Dealer: {dealer.name}",
            f"🃏 Cards Dealt: {len(dealt_cards)}",
            "📋 Selection Process:",
        ]
        
        # Show the cards in order they were dealt
        current_player_index = 0
        for i, card in enumerate(dealt_cards):
            player = self.players[current_player_index]
            summary.append(f"  {i+1:2d}. {player.name:>8}: {card.unicode_str()}")
            
            # Move to next player
            current_player_index = (current_player_index + 1) % 4
        
        return "\n".join(summary)
    
    def reset_deck(self) -> None:
        """Reset the deck to its original state after dealer selection."""
        self.deck.reset() 