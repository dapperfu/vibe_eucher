"""Game state management for the euchre game."""

from typing import List, Optional, Dict
from ..models import Player, GameState, Suit


class GameStateManager:
    """Manages the overall game state."""
    
    def __init__(self):
        """Initialize the game state manager."""
        self.players: List[Player] = []
        self.dealer: Optional[Player] = None
        self.current_player_index: int = 0
        self.round_number: int = 1
        self.game_scores: Dict[str, int] = {"Team 1": 0, "Team 2": 0}
        self.trump_suit: Optional[Suit] = None
        self.trump_caller: Optional[Player] = None
        self.trump_caller_team: Optional[int] = None
    
    def initialize_game(self, players: List[Player]) -> None:
        """Initialize the game with players.
        
        Parameters
        ----------
        players : List[Player]
            List of players in the game
        """
        if len(players) != 4:
            raise ValueError("Euchre requires exactly 4 players")
        
        self.players = players
        self.current_player_index = 0
        self.round_number = 1
        
        # Reset game-specific state
        self.trump_caller = None
        self.trump_caller_team = None
    
    def start_new_round(self) -> None:
        """Start a new round of the game."""
        self.round_number += 1
        
        # Rotate dealer
        if self.dealer:
            dealer_index = next(i for i, p in enumerate(self.players) if p.name == self.dealer.name)
            next_dealer_index = (dealer_index + 1) % 4
            self.dealer = self.players[next_dealer_index]
        else:
            # First round, set dealer to first player
            self.dealer = self.players[0]
        
        # Reset round-specific state (but preserve trump caller team for scoring)
        self.trump_suit = None
        self.trump_caller = None
        # Don't reset trump_caller_team here - it's needed for scoring the previous round
    
    def set_trump_suit(self, trump_suit: Suit, caller: Player) -> None:
        """Set the trump suit and caller.
        
        Parameters
        ----------
        trump_suit : Suit
            The trump suit for this round
        caller : Player
            The player who called trump
        """
        self.trump_suit = trump_suit
        self.trump_caller = caller
        
        # Determine which team called trump
        caller_index = self.players.index(caller)
        self.trump_caller_team = caller_index % 2
    
    def get_trump_caller(self) -> Optional[Player]:
        """Get the player who called trump.
        
        Returns
        -------
        Optional[Player]
            The trump caller, or None if no trump has been called
        """
        return self.trump_caller
    
    def next_player(self) -> None:
        """Move to the next player's turn."""
        self.current_player_index = (self.current_player_index + 1) % 4
    
    def get_current_player(self) -> Player:
        """Get the current player.
        
        Returns
        -------
        Player
            The current player
        """
        return self.players[self.current_player_index]
    
    def get_dealer(self) -> Player:
        """Get the current dealer.
        
        Returns
        -------
        Player
            The current dealer
        """
        return self.dealer
    
    def set_dealer(self, dealer: Player) -> None:
        """Set the current dealer.
        
        Parameters
        ----------
        dealer : Player
            The player to set as dealer
        """
        self.dealer = dealer
    
    def update_scores(self, team1_score: int, team2_score: int) -> None:
        """Update the team scores.
        
        Parameters
        ----------
        team1_score : int
            New score for team 1
        team2_score : int
            New score for team 2
        """
        self.game_scores["Team 1"] = team1_score
        self.game_scores["Team 2"] = team2_score
    
    def is_game_over(self) -> bool:
        """Check if the game is over (a team has 10+ points).
        
        Returns
        -------
        bool
            True if game is over, False otherwise
        """
        return self.game_scores["Team 1"] >= 10 or self.game_scores["Team 2"] >= 10
    
    def get_winner(self) -> str:
        """Get the winning team.
        
        Returns
        -------
        str
            "Team 1" or "Team 2"
        """
        if self.game_scores["Team 1"] >= 10:
            return "Team 1"
        elif self.game_scores["Team 2"] >= 10:
            return "Team 2"
        else:
            return "Game in progress"
    
    def get_state(self) -> Optional[GameState]:
        """Get the current game state.
        
        Returns
        -------
        Optional[GameState]
            The current game state, or None if not initialized
        """
        # Return a GameState object constructed from current state
        if not self.players:
            return None
        
        return GameState(
            players=self.players,
            current_player_index=self.current_player_index,
            dealer_index=next(i for i, p in enumerate(self.players) if p.name == self.dealer.name) if self.dealer else 0,
            team1_score=self.game_scores["Team 1"],
            team2_score=self.game_scores["Team 2"],
            trump_suit=self.trump_suit,
            round_number=self.round_number
        )
    
    def reset(self) -> None:
        """Reset the game state manager."""
        self.players = []
        self.dealer = None
        self.current_player_index = 0
        self.trump_caller = None
        self.trump_caller_team = None
        self.round_number = 1
        self.game_scores = {"Team 1": 0, "Team 2": 0} 