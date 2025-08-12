"""Game state management for the euchre game."""

from typing import List, Optional
from ..models import Player, GameState, Suit


class GameStateManager:
    """Manages the current state of the euchre game."""
    
    def __init__(self) -> None:
        """Initialize the game state manager."""
        self.game_state: Optional[GameState] = None
        self.current_dealer_index: int = 0
        self.trump_caller: Optional[Player] = None
        self.trump_caller_team: Optional[int] = None
        self.round_number: int = 1
    
    def initialize_game(self, players: List[Player]) -> None:
        """Initialize a new game with the given players.
        
        Parameters
        ----------
        players : List[Player]
            List of players in the game
        """
        if len(players) != 4:
            raise ValueError("Euchre requires exactly 4 players")
        
        self.game_state = GameState(
            players=players,
            current_player_index=0,
            dealer_index=self.current_dealer_index,
            team1_score=0,
            team2_score=0,
            trump_suit=None,
            round_number=self.round_number
        )
        
        # Reset game-specific state
        self.trump_caller = None
        self.trump_caller_team = None
    
    def start_new_round(self) -> None:
        """Start a new round of the game."""
        if self.game_state:
            self.round_number += 1
            self.game_state.round_number = self.round_number
            
            # Rotate dealer
            self.current_dealer_index = (self.current_dealer_index + 1) % 4
            self.game_state.dealer_index = self.current_dealer_index
            
            # Reset round-specific state
            self.game_state.trump_suit = None
            self.trump_caller = None
            self.trump_caller_team = None
    
    def set_trump_suit(self, trump_suit: Suit, caller: Player) -> None:
        """Set the trump suit and track who called it.
        
        Parameters
        ----------
        trump_suit : Suit
            The trump suit for this round
        caller : Player
            The player who called trump
        """
        if self.game_state:
            self.game_state.trump_suit = trump_suit
            self.trump_caller = caller
            
            # Determine which team called trump
            caller_index = self.game_state.players.index(caller)
            self.trump_caller_team = caller_index % 2
    
    def next_player(self) -> None:
        """Move to the next player's turn."""
        if self.game_state:
            self.game_state.current_player_index = (self.game_state.current_player_index + 1) % 4
    
    def get_current_player(self) -> Player:
        """Get the current player.
        
        Returns
        -------
        Player
            The current player
            
        Raises
        ------
        ValueError
            If game state is not initialized
        """
        if not self.game_state:
            raise ValueError("Game state not initialized")
        return self.game_state.players[self.game_state.current_player_index]
    
    def get_dealer(self) -> Player:
        """Get the current dealer.
        
        Returns
        -------
        Player
            The current dealer
            
        Raises
        ------
        ValueError
            If game state is not initialized
        """
        if not self.game_state:
            raise ValueError("Game state not initialized")
        return self.game_state.players[self.game_state.dealer_index]
    
    def update_scores(self, team1_score: int, team2_score: int) -> None:
        """Update the team scores.
        
        Parameters
        ----------
        team1_score : int
            New score for team 1
        team2_score : int
            New score for team 2
        """
        if self.game_state:
            self.game_state.team1_score = team1_score
            self.game_state.team2_score = team2_score
    
    def is_game_over(self) -> bool:
        """Check if the game is over (a team has 10+ points).
        
        Returns
        -------
        bool
            True if game is over, False otherwise
        """
        if not self.game_state:
            return False
        return self.game_state.team1_score >= 10 or self.game_state.team2_score >= 10
    
    def get_winner(self) -> str:
        """Get the winning team.
        
        Returns
        -------
        str
            "Team 1" or "Team 2"
        """
        if not self.game_state:
            return "No winner"
        
        if self.game_state.team1_score >= 10:
            return "Team 1"
        elif self.game_state.team2_score >= 10:
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
        return self.game_state
    
    def reset(self) -> None:
        """Reset the game state manager."""
        self.game_state = None
        self.current_dealer_index = 0
        self.trump_caller = None
        self.trump_caller_team = None
        self.round_number = 1 