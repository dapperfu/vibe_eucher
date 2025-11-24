"""Evaluation system for transformer RL with metrics collection and tournament evaluation.

Tracks performance metrics including:
- Average tricks per hand
- Hands won per thousand
- Sweep and set rates
- Bidding accuracy
- Play accuracy
- Tournament win rate
"""

from collections import defaultdict
from typing import Dict, List, Optional

from eucher.game import Game
from src.ai_players.transformer_rl.transformer_rl_player import TransformerRLPlayer


class EvaluationMetrics:
    """Collects and tracks evaluation metrics.

    Parameters
    ----------
    None

    Attributes
    ----------
    hands_played : int
        Total number of hands played.
    hands_won : int
        Total number of hands won.
    tricks_won : List[int]
        List of tricks won per hand.
    sweeps : int
        Number of sweeps (won all 5 tricks).
    sets : int
        Number of sets (opponents won 0 tricks).
    bidding_correct : int
        Number of correct bidding decisions.
    bidding_total : int
        Total number of bidding decisions.
    play_correct : int
        Number of correct play decisions.
    play_total : int
        Total number of play decisions.
    """

    def __init__(self) -> None:
        """Initialize evaluation metrics."""
        self.hands_played = 0
        self.hands_won = 0
        self.tricks_won: List[int] = []
        self.sweeps = 0
        self.sets = 0
        self.bidding_correct = 0
        self.bidding_total = 0
        self.play_correct = 0
        self.play_total = 0
        self.tournament_wins = 0
        self.tournament_games = 0

    def record_hand(
        self,
        tricks_won: int,
        hand_won: bool,
        was_sweep: bool = False,
        was_set: bool = False,
    ) -> None:
        """Record a hand result.

        Parameters
        ----------
        tricks_won : int
            Number of tricks won.
        hand_won : bool
            Whether hand was won.
        was_sweep : bool
            Whether it was a sweep.
        was_set : bool
            Whether opponents were set.
        """
        self.hands_played += 1
        if hand_won:
            self.hands_won += 1
        self.tricks_won.append(tricks_won)
        if was_sweep:
            self.sweeps += 1
        if was_set:
            self.sets += 1

    def record_bidding_decision(self, correct: bool) -> None:
        """Record a bidding decision.

        Parameters
        ----------
        correct : bool
            Whether decision was correct.
        """
        self.bidding_total += 1
        if correct:
            self.bidding_correct += 1

    def record_play_decision(self, correct: bool) -> None:
        """Record a play decision.

        Parameters
        ----------
        correct : bool
            Whether decision was correct.
        """
        self.play_total += 1
        if correct:
            self.play_correct += 1

    def record_tournament_game(self, won: bool) -> None:
        """Record a tournament game result.

        Parameters
        ----------
        won : bool
            Whether game was won.
        """
        self.tournament_games += 1
        if won:
            self.tournament_wins += 1

    def get_stats(self) -> Dict:
        """Get current statistics.

        Returns
        -------
        Dict
            Dictionary of statistics.
        """
        stats: Dict = {
            "hands_played": self.hands_played,
            "hands_won": self.hands_won,
            "win_rate": self.hands_won / self.hands_played if self.hands_played > 0 else 0.0,
            "avg_tricks_per_hand": sum(self.tricks_won) / len(self.tricks_won) if self.tricks_won else 0.0,
            "sweep_rate": self.sweeps / self.hands_played if self.hands_played > 0 else 0.0,
            "set_rate": self.sets / self.hands_played if self.hands_played > 0 else 0.0,
            "bidding_accuracy": self.bidding_correct / self.bidding_total if self.bidding_total > 0 else 0.0,
            "play_accuracy": self.play_correct / self.play_total if self.play_total > 0 else 0.0,
            "tournament_win_rate": self.tournament_wins / self.tournament_games if self.tournament_games > 0 else 0.0,
        }

        # Hands won per thousand
        if self.hands_played >= 1000:
            stats["hands_won_per_thousand"] = (self.hands_won / self.hands_played) * 1000
        else:
            stats["hands_won_per_thousand"] = (self.hands_won / max(1, self.hands_played)) * 1000

        return stats

    def reset(self) -> None:
        """Reset all metrics."""
        self.hands_played = 0
        self.hands_won = 0
        self.tricks_won.clear()
        self.sweeps = 0
        self.sets = 0
        self.bidding_correct = 0
        self.bidding_total = 0
        self.play_correct = 0
        self.play_total = 0
        self.tournament_wins = 0
        self.tournament_games = 0


def evaluate_agent(
    agent: TransformerRLPlayer,
    num_hands: int = 1000,
    opponent_type: str = "random",
    metrics: Optional[EvaluationMetrics] = None,
) -> EvaluationMetrics:
    """Evaluate transformer RL agent against opponents.

    Parameters
    ----------
    agent : TransformerRLPlayer
        The agent to evaluate.
    num_hands : int
        Number of hands to play.
    opponent_type : str
        Type of opponents: "random", "heuristic", or "fixed_bots".
    metrics : Optional[EvaluationMetrics]
        Existing metrics object to update. If None, creates new one.

    Returns
    -------
    EvaluationMetrics
        Evaluation metrics.
    """
    if metrics is None:
        metrics = EvaluationMetrics()

    # Create opponent configuration
    if opponent_type == "random":
        player_config = [
            ("RL Agent", "transformer_rl"),
            ("Random 1", "random"),
            ("Random 2", "random"),
            ("Random 3", "random"),
        ]
    elif opponent_type == "heuristic":
        player_config = [
            ("RL Agent", "transformer_rl"),
            ("Heuristic 1", "heuristic"),
            ("Heuristic 2", "heuristic"),
            ("Heuristic 3", "heuristic"),
        ]
    else:  # fixed_bots
        player_config = [
            ("RL Agent", "transformer_rl"),
            ("Bot 1", "heuristic"),
            ("Bot 2", "heuristic"),
            ("Bot 3", "heuristic"),
        ]

    # Play hands
    for hand_num in range(num_hands):
        try:
            game = Game(player_config)
            continue_game = game.play_hand()

            # Get hand results
            tricks_won = getattr(game, "_current_tricks_won", [0, 0])
            hand_won = tricks_won[0] >= 3
            was_sweep = tricks_won[0] == 5
            was_set = tricks_won[1] == 0

            # Record metrics
            metrics.record_hand(tricks_won[0], hand_won, was_sweep, was_set)

        except Exception as e:
            print(f"Error evaluating hand {hand_num}: {e}")
            continue

    return metrics


def tournament_evaluation(
    agent: TransformerRLPlayer,
    num_games: int = 100,
    opponent_types: Optional[List[str]] = None,
) -> Dict:
    """Run tournament evaluation against multiple opponent types.

    Parameters
    ----------
    agent : TransformerRLPlayer
        The agent to evaluate.
    num_games : int
        Number of games per opponent type.
    opponent_types : Optional[List[str]]
        List of opponent types to evaluate against. If None, uses default set.

    Returns
    -------
    Dict
        Tournament results.
    """
    if opponent_types is None:
        opponent_types = ["random", "heuristic", "fixed_bots"]

    results: Dict = {}
    overall_metrics = EvaluationMetrics()

    for opponent_type in opponent_types:
        metrics = evaluate_agent(agent, num_hands=num_games, opponent_type=opponent_type)
        results[opponent_type] = metrics.get_stats()
        overall_metrics.tournament_games += metrics.hands_played
        overall_metrics.tournament_wins += metrics.hands_won

    results["overall"] = overall_metrics.get_stats()
    return results


