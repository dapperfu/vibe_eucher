"""
Balanced Scoring System for Euchre

This module implements a balanced scoring system that helps create more
competitive gameplay by adjusting scoring based on team performance,
handicapping, and dynamic balance adjustments.
"""

import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..models import Player, Suit
from ..core.deck import Deck


class ScoringMode(Enum):
    """Different scoring modes for balancing gameplay."""
    STANDARD = "standard"           # Traditional Euchre scoring
    BALANCED = "balanced"           # Balanced with handicaps
    ADAPTIVE = "adaptive"           # Dynamically adjusts based on performance
    TEAM_HANDICAP = "team_handicap" # Handicaps based on team strength


@dataclass
class ScoringConfig:
    """Configuration for the balanced scoring system."""
    
    mode: ScoringMode = ScoringMode.BALANCED
    
    # Base scoring
    base_points_per_trick: int = 1
    euchre_bonus: int = 2  # Bonus for winning 5 tricks after opponent calls trump
    sweep_bonus: int = 1   # Bonus for winning all 5 tricks
    
    # Balancing parameters
    handicap_threshold: float = 0.6  # Win rate threshold for applying handicaps
    max_handicap: int = 2            # Maximum handicap points
    performance_window: int = 10     # Games to consider for performance
    
    # Team balance
    team_strength_analysis: bool = True
    dynamic_adjustment: bool = True
    random_factor: float = 0.1       # Small random factor for unpredictability


class BalancedScoring:
    """
    Balanced scoring system that adjusts scoring to create more competitive gameplay.
    """
    
    def __init__(self, config: ScoringConfig):
        """
        Initialize the balanced scoring system.
        
        Parameters
        ----------
        config : ScoringConfig
            Configuration for the scoring system
        """
        self.config = config
        self.team_performance = {}  # Track team performance
        self.game_history = []      # Track game history
        self.adjustment_history = [] # Track scoring adjustments
        
        # Initialize team tracking
        self._init_team_tracking()
    
    def _init_team_tracking(self):
        """Initialize team performance tracking."""
        self.team_performance = {
            "team1": {
                "games_played": 0,
                "games_won": 0,
                "total_score": 0,
                "average_score": 0.0,
                "recent_performance": []
            },
            "team2": {
                "games_played": 0,
                "games_won": 0,
                "total_score": 0,
                "average_score": 0.0,
                "recent_performance": []
            }
        }
    
    def calculate_round_score(self, team1_tricks: int, team2_tricks: int, 
                             trump_caller_team: int, team1_players: List[Player],
                             team2_players: List[Player]) -> Tuple[int, int]:
        """
        Calculate balanced scores for a round.
        
        Parameters
        ----------
        team1_tricks : int
            Number of tricks won by team 1
        team2_tricks : int
            Number of tricks won by team 2
        trump_caller_team : int
            Team that called trump (0 for team 1, 1 for team 2)
        team1_players : List[Player]
            Players in team 1
        team2_players : List[Player]
            Players in team 2
        
        Returns
        -------
        Tuple[int, int]
            (team1_score, team2_score) after balancing
        """
        # Calculate base scores
        base_team1_score = team1_tricks * self.config.base_points_per_trick
        base_team2_score = team2_tricks * self.config.base_points_per_trick
        
        # Apply bonuses
        if team1_tricks == 5:
            base_team1_score += self.config.sweep_bonus
        if team2_tricks == 5:
            base_team2_score += self.config.sweep_bonus
        
        # Apply euchre bonus
        if trump_caller_team == 0 and team2_tricks == 5:  # Team 1 called trump, team 2 won all
            base_team2_score += self.config.euchre_bonus
        elif trump_caller_team == 1 and team1_tricks == 5:  # Team 2 called trump, team 1 won all
            base_team1_score += self.config.euchre_bonus
        
        # Apply balancing adjustments
        if self.config.mode != ScoringMode.STANDARD:
            adjusted_team1, adjusted_team2 = self._apply_balancing(
                base_team1_score, base_team2_score, team1_players, team2_players
            )
            return adjusted_team1, adjusted_team2
        
        return base_team1_score, base_team2_score
    
    def _apply_balancing(self, team1_score: int, team2_score: int,
                         team1_players: List[Player], team2_players: List[Player]) -> Tuple[int, int]:
        """
        Apply balancing adjustments to scores.
        
        Parameters
        ----------
        team1_score : int
            Base score for team 1
        team2_score : int
            Base score for team 2
        team1_players : List[Player]
            Players in team 1
        team2_players : List[Player]
            Players in team 2
        
        Returns
        -------
        Tuple[int, int]
            Balanced scores for both teams
        """
        adjusted_team1 = team1_score
        adjusted_team2 = team2_score
        
        # Calculate team strength
        team1_strength = self._calculate_team_strength(team1_players)
        team2_strength = self._calculate_team_strength(team2_players)
        
        # Get recent performance
        team1_performance = self._get_team_performance("team1")
        team2_performance = self._get_team_performance("team2")
        
        # Apply handicaps based on performance
        if self.config.mode == ScoringMode.TEAM_HANDICAP:
            adjusted_team1, adjusted_team2 = self._apply_team_handicaps(
                adjusted_team1, adjusted_team2, team1_performance, team2_performance
            )
        
        # Apply adaptive balancing
        if self.config.mode == ScoringMode.ADAPTIVE:
            adjusted_team1, adjusted_team2 = self._apply_adaptive_balancing(
                adjusted_team1, adjusted_team2, team1_performance, team2_performance
            )
        
        # Apply balanced mode adjustments
        if self.config.mode == ScoringMode.BALANCED:
            adjusted_team1, adjusted_team2 = self._apply_balanced_adjustments(
                adjusted_team1, adjusted_team2, team1_strength, team2_strength,
                team1_performance, team2_performance
            )
        
        # Add small random factor for unpredictability
        if self.config.random_factor > 0:
            random_team1 = random.uniform(-self.config.random_factor, self.config.random_factor)
            random_team2 = random.uniform(-self.config.random_factor, self.config.random_factor)
            
            adjusted_team1 = max(0, adjusted_team1 + random_team1)
            adjusted_team2 = max(0, adjusted_team2 + random_team2)
        
        return int(adjusted_team1), int(adjusted_team2)
    
    def _calculate_team_strength(self, players: List[Player]) -> float:
        """
        Calculate the relative strength of a team based on player characteristics.
        
        Parameters
        ----------
        players : List[Player]
            Players in the team
        
        Returns
        -------
        float
            Team strength score (0.0 to 1.0)
        """
        if not self.config.team_strength_analysis:
            return 0.5  # Neutral strength
        
        total_strength = 0.0
        
        for player in players:
            # Analyze player characteristics
            if hasattr(player, 'risk_ratio'):
                # Higher risk ratio means more aggressive (potentially stronger)
                risk_strength = player.risk_ratio
            else:
                risk_strength = 0.5
            
            if hasattr(player, 'player_type'):
                # AI players might be stronger than human players
                type_strength = 0.7 if player.player_type.name == "AI" else 0.5
            else:
                type_strength = 0.5
            
            # Combine factors
            player_strength = (risk_strength + type_strength) / 2
            total_strength += player_strength
        
        # Average team strength
        return total_strength / len(players)
    
    def _get_team_performance(self, team_key: str) -> Dict[str, float]:
        """
        Get recent performance metrics for a team.
        
        Parameters
        ----------
        team_key : str
            Team identifier ("team1" or "team2")
        
        Returns
        -------
        Dict[str, float]
            Performance metrics
        """
        team_data = self.team_performance.get(team_key, {})
        
        games_played = team_data.get("games_played", 0)
        if games_played == 0:
            return {"win_rate": 0.5, "average_score": 0.0, "recent_trend": 0.0}
        
        win_rate = team_data.get("games_won", 0) / games_played
        average_score = team_data.get("average_score", 0.0)
        
        # Calculate recent trend (last N games)
        recent_performance = team_data.get("recent_performance", [])
        if len(recent_performance) >= 2:
            recent_trend = recent_performance[-1] - recent_performance[-2]
        else:
            recent_trend = 0.0
        
        return {
            "win_rate": win_rate,
            "average_score": average_score,
            "recent_trend": recent_trend
        }
    
    def _apply_team_handicaps(self, team1_score: int, team2_score: int,
                              team1_performance: Dict[str, float],
                              team2_performance: Dict[str, float]) -> Tuple[int, int]:
        """
        Apply handicaps based on team performance.
        
        Parameters
        ----------
        team1_score : int
            Current score for team 1
        team2_score : int
            Current score for team 2
        team1_performance : Dict[str, float]
            Performance metrics for team 1
        team2_performance : Dict[str, float]
            Performance metrics for team 2
        
        Returns
        -------
        Tuple[int, int]
            Scores after applying handicaps
        """
        adjusted_team1 = team1_score
        adjusted_team2 = team2_score
        
        # Apply handicap to stronger team
        if team1_performance["win_rate"] > self.config.handicap_threshold:
            handicap = min(self.config.max_handicap, 
                          int((team1_performance["win_rate"] - 0.5) * 4))
            adjusted_team1 = max(0, adjusted_team1 - handicap)
            adjusted_team2 = adjusted_team2 + handicap
        
        elif team2_performance["win_rate"] > self.config.handicap_threshold:
            handicap = min(self.config.max_handicap, 
                          int((team2_performance["win_rate"] - 0.5) * 4))
            adjusted_team2 = max(0, adjusted_team2 - handicap)
            adjusted_team1 = adjusted_team1 + handicap
        
        return adjusted_team1, adjusted_team2
    
    def _apply_adaptive_balancing(self, team1_score: int, team2_score: int,
                                 team1_performance: Dict[str, float],
                                 team2_performance: Dict[str, float]) -> Tuple[int, int]:
        """
        Apply adaptive balancing based on recent performance trends.
        
        Parameters
        ----------
        team1_score : int
            Current score for team 1
        team2_score : int
            Current score for team 2
        team1_performance : Dict[str, float]
            Performance metrics for team 1
        team2_performance : Dict[str, float]
            Performance metrics for team 2
        
        Returns
        -------
        Tuple[int, int]
            Scores after adaptive balancing
        """
        adjusted_team1 = team1_score
        adjusted_team2 = team2_score
        
        # Adjust based on recent trends
        team1_trend = team1_performance["recent_trend"]
        team2_trend = team2_performance["recent_trend"]
        
        # If one team is trending up, give slight advantage to the other
        if team1_trend > 0.1 and team2_trend < 0:
            # Team 1 trending up, give slight advantage to team 2
            adjustment = min(1, int(team1_trend * 2))
            adjusted_team1 = max(0, adjusted_team1 - adjustment)
            adjusted_team2 = adjusted_team2 + adjustment
        
        elif team2_trend > 0.1 and team1_trend < 0:
            # Team 2 trending up, give slight advantage to team 1
            adjustment = min(1, int(team2_trend * 2))
            adjusted_team2 = max(0, adjusted_team2 - adjustment)
            adjusted_team1 = adjusted_team1 + adjustment
        
        return adjusted_team1, adjusted_team2
    
    def _apply_balanced_adjustments(self, team1_score: int, team2_score: int,
                                   team1_strength: float, team2_strength: float,
                                   team1_performance: Dict[str, float],
                                   team2_performance: Dict[str, float]) -> Tuple[int, int]:
        """
        Apply balanced mode adjustments.
        
        Parameters
        ----------
        team1_score : int
            Current score for team 1
        team2_score : int
            Current score for team 2
        team1_strength : float
            Calculated strength of team 1
        team2_strength : float
            Calculated strength of team 2
        team1_performance : Dict[str, float]
            Performance metrics for team 1
        team2_performance : Dict[str, float]
            Performance metrics for team 2
        
        Returns
        -------
        Tuple[int, int]
            Scores after balanced adjustments
        """
        adjusted_team1 = team1_score
        adjusted_team2 = team2_score
        
        # Calculate strength difference
        strength_diff = team1_strength - team2_strength
        
        # Calculate performance difference
        performance_diff = team1_performance["win_rate"] - team2_performance["win_rate"]
        
        # Apply adjustments based on strength and performance
        if abs(strength_diff) > 0.2:  # Significant strength difference
            # Weaker team gets slight boost
            if strength_diff > 0:
                # Team 1 is stronger, boost team 2
                boost = min(1, int(strength_diff * 2))
                adjusted_team2 = adjusted_team2 + boost
            else:
                # Team 2 is stronger, boost team 1
                boost = min(1, int(abs(strength_diff) * 2))
                adjusted_team1 = adjusted_team1 + boost
        
        if abs(performance_diff) > 0.3:  # Significant performance difference
            # Underperforming team gets slight boost
            if performance_diff > 0:
                # Team 1 performing better, boost team 2
                boost = min(1, int(performance_diff * 2))
                adjusted_team2 = adjusted_team2 + boost
            else:
                # Team 2 performing better, boost team 1
                boost = min(1, int(abs(performance_diff) * 2))
                adjusted_team1 = adjusted_team1 + boost
        
        return adjusted_team1, adjusted_team2
    
    def update_team_performance(self, team1_score: int, team2_score: int,
                               team1_won: bool, team2_won: bool):
        """
        Update team performance tracking after a game.
        
        Parameters
        ----------
        team1_score : int
            Final score for team 1
        team2_score : int
            Final score for team 2
        team1_won : bool
            Whether team 1 won the game
        team2_won : bool
            Whether team 2 won the game
        """
        # Update team 1 performance
        self.team_performance["team1"]["games_played"] += 1
        if team1_won:
            self.team_performance["team1"]["games_won"] += 1
        
        self.team_performance["team1"]["total_score"] += team1_score
        self.team_performance["team1"]["average_score"] = (
            self.team_performance["team1"]["total_score"] / 
            self.team_performance["team1"]["games_played"]
        )
        
        # Update recent performance window
        recent_perf = self.team_performance["team1"]["recent_performance"]
        recent_perf.append(team1_score)
        if len(recent_perf) > self.config.performance_window:
            recent_perf.pop(0)
        
        # Update team 2 performance
        self.team_performance["team2"]["games_played"] += 1
        if team2_won:
            self.team_performance["team2"]["games_won"] += 1
        
        self.team_performance["team2"]["total_score"] += team2_score
        self.team_performance["team2"]["average_score"] = (
            self.team_performance["team2"]["total_score"] / 
            self.team_performance["team2"]["games_played"]
        )
        
        # Update recent performance window
        recent_perf = self.team_performance["team2"]["recent_performance"]
        recent_perf.append(team2_score)
        if len(recent_perf) > self.config.performance_window:
            recent_perf.pop(0)
        
        # Record adjustment if any were made
        if self.config.mode != ScoringMode.STANDARD:
            self._record_adjustment(team1_score, team2_score, team1_won, team2_won)
    
    def _record_adjustment(self, team1_score: int, team2_score: int,
                           team1_won: bool, team2_won: bool):
        """Record scoring adjustments for analysis."""
        adjustment_info = {
            "timestamp": len(self.game_history),
            "team1_score": team1_score,
            "team2_score": team2_score,
            "team1_won": team1_won,
            "team2_won": team2_won,
            "team1_performance": self._get_team_performance("team1"),
            "team2_performance": self._get_team_performance("team2")
        }
        
        self.adjustment_history.append(adjustment_info)
    
    def get_balance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the balancing system's current state.
        
        Returns
        -------
        Dict[str, Any]
            Summary of balancing system
        """
        return {
            "scoring_mode": self.config.mode.value,
            "team_performance": self.team_performance,
            "total_adjustments": len(self.adjustment_history),
            "config": {
                "handicap_threshold": self.config.handicap_threshold,
                "max_handicap": self.config.max_handicap,
                "performance_window": self.config.performance_window,
                "random_factor": self.config.random_factor
            }
        }
    
    def reset_balance(self):
        """Reset all balancing data and return to initial state."""
        self._init_team_tracking()
        self.game_history.clear()
        self.adjustment_history.clear()
        print("🔄 Balanced scoring system reset to initial state")


# Factory function for creating different scoring systems
def create_scoring_system(mode: ScoringMode = ScoringMode.BALANCED, 
                         **kwargs) -> BalancedScoring:
    """
    Create a scoring system with the specified mode.
    
    Parameters
    ----------
    mode : ScoringMode
        The scoring mode to use
    **kwargs
        Additional configuration parameters
    
    Returns
    -------
    BalancedScoring
        Configured scoring system
    """
    config = ScoringConfig(mode=mode, **kwargs)
    return BalancedScoring(config) 