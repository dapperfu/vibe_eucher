"""Scoring logic for the euchre game."""

from typing import List, Tuple
from ..models import Player


class ScoringManager:
    """Manages scoring for the euchre game."""
    
    def __init__(self) -> None:
        """Initialize the scoring manager."""
        pass
    
    def score_round(self, players: List[Player], trump_caller_team: int) -> Tuple[int, int]:
        """Score a round based on trick counts.
        
        Parameters
        ----------
        players : List[Player]
            List of players in the game
        trump_caller_team : int
            Team that called trump (0 for team 1, 1 for team 2)
            
        Returns
        -------
        Tuple[int, int]
            New scores for team 1 and team 2
        """
        # Count tricks for each team
        team1_tricks = sum(players[i].tricks_won for i in range(0, 4, 2))
        team2_tricks = sum(players[i].tricks_won for i in range(1, 4, 2))
        
        # Determine scoring
        if trump_caller_team == 0:  # Team 1 called trump
            team1_score, team2_score = self._score_trump_calling_team(team1_tricks, team2_tricks)
        else:  # Team 2 called trump
            team2_score, team1_score = self._score_trump_calling_team(team2_tricks, team1_tricks)
        
        return team1_score, team2_score
    
    def _score_trump_calling_team(self, trump_team_tricks: int, other_team_tricks: int) -> Tuple[int, int]:
        """Score when a team calls trump.
        
        Parameters
        ----------
        trump_team_tricks : int
            Number of tricks won by trump calling team
        other_team_tricks : int
            Number of tricks won by other team
            
        Returns
        -------
        Tuple[int, int]
            Scores for trump team and other team
        """
        trump_score = 0
        other_score = 0
        
        if trump_team_tricks >= 3:
            # Trump team gets points
            if trump_team_tricks == 5:
                # March - 2 points
                trump_score = 2
            else:
                # 3 or 4 tricks - 1 point
                trump_score = 1
        else:
            # Trump team gets set - other team gets 2 points
            other_score = 2
        
        return trump_score, other_score
    
    def is_team_set(self, players: List[Player], trump_caller_team: int) -> bool:
        """Check if the trump calling team got set (won 0-2 tricks).
        
        Parameters
        ----------
        players : List[Player]
            List of players in the game
        trump_caller_team : int
            Team that called trump (0 for team 1, 1 for team 2)
            
        Returns
        -------
        bool
            True if trump calling team got set
        """
        if trump_caller_team == 0:  # Team 1 called trump
            team_tricks = sum(players[i].tricks_won for i in range(0, 4, 2))
        else:  # Team 2 called trump
            team_tricks = sum(players[i].tricks_won for i in range(1, 4, 2))
        
        return team_tricks <= 2
    
    def get_round_summary(self, players: List[Player]) -> str:
        """Get a summary of the round results.
        
        Parameters
        ----------
        players : List[Player]
            List of players in the game
            
        Returns
        -------
        str
            Summary of the round
        """
        # Count tricks for each team
        team1_tricks = sum(players[i].tricks_won for i in range(0, 4, 2))
        team2_tricks = sum(players[i].tricks_won for i in range(1, 4, 2))
        
        return f"Team 1: {team1_tricks} tricks, Team 2: {team2_tricks} tricks"
    
    def is_game_over(self, players: List[Player]) -> bool:
        """Check if the game is over.
        
        Parameters
        ----------
        players : List[Player]
            List of players in the game
            
        Returns
        -------
        bool
            True if the game is over
        """
        # Check if any team has reached 10 points
        team1_score = sum(players[i].score for i in range(0, 4, 2))
        team2_score = sum(players[i].score for i in range(1, 4, 2))
        
        return team1_score >= 10 or team2_score >= 10
    
    def get_team_scores(self, players: List[Player]) -> Tuple[int, int]:
        """Get the current team scores.
        
        Parameters
        ----------
        players : List[Player]
            List of players in the game
            
        Returns
        -------
        Tuple[int, int]
            Scores for team 1 and team 2
        """
        team1_score = sum(players[i].score for i in range(0, 4, 2))
        team2_score = sum(players[i].score for i in range(1, 4, 2))
        
        return team1_score, team2_score
    
    def reset_player_tricks(self, players: List[Player]) -> None:
        """Reset trick counts for all players.
        
        Parameters
        ----------
        players : List[Player]
            List of players in the game
        """
        for player in players:
            player.tricks_won = 0 