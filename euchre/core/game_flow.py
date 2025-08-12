"""
Game flow management for the euchre game.

This module manages the overall flow of the game, including
state transitions, round management, and game progression.
"""

from typing import List, Optional, Callable, Dict, Any
from enum import Enum, auto
from dataclasses import dataclass

from ..models import Player, Card, Suit, Trick
from ..utils.logging_config import GameLogger


class GamePhase(Enum):
    """Phases of the euchre game."""
    
    SETUP = auto()
    DEALING = auto()
    TRUMP_SELECTION = auto()
    PLAYING_TRICKS = auto()
    SCORING = auto()
    GAME_OVER = auto()


@dataclass
class GameFlowState:
    """Current state of the game flow."""
    
    phase: GamePhase
    current_round: int
    current_trick: Optional[Trick]
    dealer_index: int
    trump_suit: Optional[Suit]
    trump_caller: Optional[Player]
    round_winner: Optional[Player]
    game_winner: Optional[str]
    
    def __str__(self) -> str:
        """String representation of the game flow state."""
        return f"Phase: {self.phase.name}, Round: {self.current_round}, Trump: {self.trump_suit.name if self.trump_suit else 'None'}"


class GameFlow:
    """Manages the overall flow of the euchre game."""
    
    def __init__(self, players: List[Player], verbose: bool = False, very_verbose: bool = False) -> None:
        """Initialize the game flow manager.
        
        Parameters
        ----------
        players : List[Player]
            List of players in the game
        verbose : bool
            Enable verbose logging
        very_verbose : bool
            Enable very verbose logging
        """
        self.players = players
        self.logger = GameLogger(verbose, very_verbose)
        
        # Game flow state
        self.state = GameFlowState(
            phase=GamePhase.SETUP,
            current_round=1,
            current_trick=None,
            dealer_index=0,
            trump_suit=None,
            trump_caller=None,
            round_winner=None,
            game_winner=None
        )
        
        # Flow callbacks
        self.phase_callbacks: Dict[GamePhase, List[Callable[[GameFlowState], None]]] = {
            phase: [] for phase in GamePhase
        }
        
        # Game statistics
        self.round_history: List[Dict[str, Any]] = []
        self.trick_history: List[Trick] = []
        
        self.logger.debug("GameFlow initialized")
    
    def add_phase_callback(self, phase: GamePhase, callback: Callable[[GameFlowState], None]) -> None:
        """Add a callback to be executed when entering a phase.
        
        Parameters
        ----------
        phase : GamePhase
            The phase to add the callback for
        callback : Callable[[GameFlowState], None]
            The callback function to execute
        """
        self.phase_callbacks[phase].append(callback)
    
    def transition_to_phase(self, new_phase: GamePhase) -> None:
        """Transition to a new game phase.
        
        Parameters
        ----------
        new_phase : GamePhase
            The new phase to transition to
        """
        old_phase = self.state.phase
        self.state.phase = new_phase
        
        self.logger.debug(f"Game phase transition: {old_phase.name} -> {new_phase.name}")
        
        # Execute phase callbacks
        for callback in self.phase_callbacks[new_phase]:
            try:
                callback(self.state)
            except Exception as e:
                self.logger.error(f"Phase callback failed: {e}")
    
    def start_game(self) -> None:
        """Start the game flow."""
        self.logger.info("Starting game flow")
        self.transition_to_phase(GamePhase.SETUP)
        self.transition_to_phase(GamePhase.DEALING)
    
    def start_round(self) -> None:
        """Start a new round."""
        self.logger.info(f"Starting round {self.state.current_round}")
        self.state.current_trick = None
        self.transition_to_phase(GamePhase.TRUMP_SELECTION)
    
    def start_trick(self) -> None:
        """Start a new trick."""
        if not self.state.current_trick:
            self.state.current_trick = Trick()
        else:
            self.state.current_trick.reset()
        
        self.transition_to_phase(GamePhase.PLAYING_TRICKS)
        self.logger.debug(f"Started new trick in round {self.state.current_round}")
    
    def complete_trick(self, winner: Player, winning_card: Card) -> None:
        """Complete the current trick.
        
        Parameters
        ----------
        winner : Player
            The player who won the trick
        winning_card : Card
            The winning card
        """
        if self.state.current_trick:
            self.state.current_trick.winner = winner
            self.trick_history.append(self.state.current_trick)
            
            # Update player statistics
            winner.tricks_won += 1
            
            self.logger.debug(f"Trick completed by {winner.name} with {winning_card}")
    
    def complete_round(self) -> None:
        """Complete the current round."""
        self.logger.info(f"Completing round {self.state.current_round}")
        
        # Record round statistics
        round_stats = {
            "round": self.state.current_round,
            "trump_suit": self.state.trump_suit.name if self.state.trump_suit else None,
            "trump_caller": self.state.trump_caller.name if self.state.trump_caller else None,
            "tricks": [trick for trick in self.trick_history if trick],
            "player_tricks": {player.name: player.tricks_won for player in self.players}
        }
        self.round_history.append(round_stats)
        
        # Reset round state
        self.state.current_round += 1
        self.state.current_trick = None
        self.state.trump_suit = None
        self.state.trump_caller = None
        
        # Reset player round statistics
        for player in self.players:
            player.reset_round_stats()
        
        self.transition_to_phase(GamePhase.SCORING)
    
    def set_trump_suit(self, trump_suit: Suit, caller: Player) -> None:
        """Set the trump suit for the current round.
        
        Parameters
        ----------
        trump_suit : Suit
            The trump suit
        caller : Player
            The player who called trump
        """
        self.state.trump_suit = trump_suit
        self.state.trump_caller = caller
        self.logger.info(f"Trump suit set to {trump_suit.name} by {caller.name}")
    
    def set_dealer(self, dealer_index: int) -> None:
        """Set the dealer for the current round.
        
        Parameters
        ----------
        dealer_index : int
            Index of the dealer in the players list
        """
        self.state.dealer_index = dealer_index
        for i, player in enumerate(self.players):
            player.is_dealer = (i == dealer_index)
            player.set_team(i % 2)  # Team 0 or 1
        
        self.logger.debug(f"Dealer set to {self.players[dealer_index].name}")
    
    def get_current_player(self) -> Player:
        """Get the current player to act.
        
        Returns
        -------
        Player
            The current player
        """
        if not self.state.current_trick or not self.state.current_trick.is_started:
            # First player in the trick (left of dealer)
            return self.players[(self.state.dealer_index + 1) % len(self.players)]
        else:
            # Next player in the trick
            num_played = self.state.current_trick.num_cards_played
            start_idx = (self.state.dealer_index + 1) % len(self.players)
            current_idx = (start_idx + num_played) % len(self.players)
            return self.players[current_idx]
    
    def get_next_player(self) -> Player:
        """Get the next player to act.
        
        Returns
        -------
        Player
            The next player
        """
        current_player = self.get_current_player()
        current_idx = self.players.index(current_player)
        next_idx = (current_idx + 1) % len(self.players)
        return self.players[next_idx]
    
    def is_round_complete(self) -> bool:
        """Check if the current round is complete.
        
        Returns
        -------
        bool
            True if the round is complete
        """
        if not self.state.current_trick:
            return False
        
        return self.state.current_trick.is_complete()
    
    def is_game_over(self) -> bool:
        """Check if the game is over.
        
        Returns
        -------
        bool
            True if the game is over
        """
        # Check if any team has reached 10 points
        team_scores = {"Team 1": 0, "Team 2": 0}
        for player in self.players:
            team_name = player.get_team_name()
            if team_name in team_scores:
                team_scores[team_name] += player.score
        
        return any(score >= 10 for score in team_scores.values())
    
    def get_game_winner(self) -> Optional[str]:
        """Get the winning team.
        
        Returns
        -------
        Optional[str]
            The winning team name, or None if game not over
        """
        if not self.is_game_over():
            return None
        
        team_scores = {"Team 1": 0, "Team 2": 0}
        for player in self.players:
            team_name = player.get_team_name()
            if team_name in team_scores:
                team_scores[team_name] += player.score
        
        if team_scores["Team 1"] >= 10:
            return "Team 1"
        elif team_scores["Team 2"] >= 10:
            return "Team 2"
        
        return None
    
    def get_game_statistics(self) -> Dict[str, Any]:
        """Get comprehensive game statistics.
        
        Returns
        -------
        Dict[str, Any]
            Game statistics
        """
        team_scores = {"Team 1": 0, "Team 2": 0}
        for player in self.players:
            team_name = player.get_team_name()
            if team_name in team_scores:
                team_scores[team_name] += player.score
        
        return {
            "current_round": self.state.current_round,
            "current_phase": self.state.phase.name,
            "trump_suit": self.state.trump_suit.name if self.state.trump_suit else None,
            "trump_caller": self.state.trump_caller.name if self.state.trump_caller else None,
            "team_scores": team_scores,
            "player_scores": {player.name: player.score for player in self.players},
            "player_tricks": {player.name: player.tricks_won for player in self.players},
            "rounds_completed": len(self.round_history),
            "tricks_completed": len(self.trick_history),
            "game_over": self.is_game_over(),
            "winner": self.get_game_winner()
        }
    
    def reset_game(self) -> None:
        """Reset the game to initial state."""
        self.logger.info("Resetting game")
        
        # Reset state
        self.state = GameFlowState(
            phase=GamePhase.SETUP,
            current_round=1,
            current_trick=None,
            dealer_index=0,
            trump_suit=None,
            trump_caller=None,
            round_winner=None,
            game_winner=None
        )
        
        # Reset players
        for player in self.players:
            player.reset_game_stats()
        
        # Clear history
        self.round_history.clear()
        self.trick_history.clear()
        
        self.logger.debug("Game reset complete") 