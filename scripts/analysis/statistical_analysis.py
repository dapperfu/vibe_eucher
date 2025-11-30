#!/usr/bin/env python3
"""Statistical analysis script for PyTorch AI players.

Runs Monte Carlo simulations with different risk factor combinations and performs
statistical analysis including bell curves, card distributions, and design of experiments.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eucher.cards import Card, Deck, Rank, Suit
from eucher.game import Game
from eucher.players import Player
from eucher.players.profiles import PlayerProfile, HeuristicPlayer

# Conditional import for PyTorch player (only needed for full game simulations)
try:
    from eucher.players.computer.ml.pytorch.pytorch_player import PyTorchStrategicPlayer
except ImportError:
    PyTorchStrategicPlayer = None  # type: ignore

try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False


class AlwaysPassProfile(PlayerProfile):
    """Profile that always passes in trump selection (except when forced)."""

    def __init__(self, base_profile: PlayerProfile) -> None:
        """
        Initialize the always-pass profile.

        Parameters
        ----------
        base_profile : PlayerProfile
            Base profile to use for gameplay decisions.
        """
        self.base_profile = base_profile

    def decide_order_up(
        self, player: Player, turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """Always pass in order up."""
        return False

    def decide_call_trump(
        self,
        player: Player,
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """Pass unless forced to choose."""
        if must_choose:
            # When forced, use base profile to choose
            return self.base_profile.decide_call_trump(player, turned_card, trump_suit, must_choose)
        return None

    def choose_card_to_discard(
        self, player: Player, turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
    ) -> Card:
        """Use base profile for discard."""
        return self.base_profile.choose_card_to_discard(player, turned_card, ordered_up_by)

    def play_card(
        self,
        player: Player,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """Use base profile for gameplay."""
        return self.base_profile.play_card(player, led_suit, trump_suit, trick_cards, trick_player_ids)


class PyTorchPlayerProfile(PlayerProfile):
    """Wrapper profile that uses PyTorchStrategicPlayer with risk factors."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        trump_selection_risk: float = 0.5,
        gameplay_risk: float = 0.5,
        player_name: str = "PyTorch AI",
    ) -> None:
        """Initialize PyTorch player profile."""
        if PyTorchStrategicPlayer is None:
            raise ImportError("PyTorchStrategicPlayer not available. Install PyTorch dependencies.")
        super().__init__()
        self.player_name = player_name
        # Note: PyTorchStrategicPlayer doesn't support risk factors in current implementation
        self.pytorch_player = PyTorchStrategicPlayer(
            model_path=model_path,
            device=device,
        )

    def decide_order_up(
        self, player: Player, turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """Decide whether to order up."""
        return self.pytorch_player.decide_order_up(player, turned_card, dealer_id, trump_suit)

    def decide_call_trump(
        self,
        player: Player,
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """Decide which suit to call as trump."""
        return self.pytorch_player.decide_call_trump(player, turned_card, trump_suit, must_choose)

    def choose_card_to_discard(
        self, player: Player, turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
    ) -> Card:
        """Choose a card to discard."""
        return self.pytorch_player.choose_card_to_discard(player, turned_card, ordered_up_by)

    def play_card(
        self,
        player: Player,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """Play a card."""
        return self.pytorch_player.play_card(
            player, led_suit, trump_suit, trick_cards, trick_player_ids
        )


class GameStatistics:
    """Collects statistics from game runs."""

    def __init__(self) -> None:
        """Initialize statistics collector."""
        self.games: List[Dict] = []
        self.turned_cards: List[Card] = []
        self.risk_combinations: Dict[Tuple, List[Dict]] = defaultdict(list)
        # Track "screw the dealer" statistics
        self.screw_dealer_hands: List[Dict] = []  # List of hands where screw the dealer occurred

    def record_game(
        self,
        winner: Optional[int],
        scores: Tuple[int, int],
        hands_played: int,
        risk_factors: List[Tuple[float, float]],
        turned_card: Optional[Card] = None,
        screw_dealer_hands: Optional[List[Dict]] = None,
    ) -> None:
        """Record a completed game.

        Parameters
        ----------
        winner : Optional[int]
            Winning team (0 or 1) or None if tie.
        scores : Tuple[int, int]
            Final scores (team0, team1).
        hands_played : int
            Number of hands played.
        risk_factors : List[Tuple[float, float]]
            Risk factors for each player (trump, gameplay).
        turned_card : Optional[Card]
            First turned card of the game.
        screw_dealer_hands : Optional[List[Dict]]
            List of hands where screw the dealer occurred, each with:
            - dealer_team: int (team of the dealer)
            - dealer_team_won_hand: bool (whether dealer's team won the hand)
        """
        game_data = {
            "winner": winner,
            "scores": scores,
            "hands_played": hands_played,
            "risk_factors": risk_factors,
            "turned_card": str(turned_card) if turned_card else None,
            "timestamp": datetime.now().isoformat(),
        }
        self.games.append(game_data)

        # Track by risk combination
        risk_key = tuple(risk_factors)
        self.risk_combinations[risk_key].append(game_data)

        if turned_card:
            self.turned_cards.append(turned_card)

        # Track screw the dealer hands
        if screw_dealer_hands:
            self.screw_dealer_hands.extend(screw_dealer_hands)

    def get_win_rates(self) -> Dict[Tuple, Dict[str, float]]:
        """Get win rates by risk combination.

        Returns
        -------
        Dict[Tuple, Dict[str, float]]
            Win rates keyed by risk combination tuple.
        """
        win_rates = {}
        for risk_key, games in self.risk_combinations.items():
            if not games:
                continue

            team0_wins = sum(1 for g in games if g["winner"] == 0)
            team1_wins = sum(1 for g in games if g["winner"] == 1)
            total = len(games)

            win_rates[risk_key] = {
                "team0_rate": team0_wins / total if total > 0 else 0.0,
                "team1_rate": team1_wins / total if total > 0 else 0.0,
                "total_games": total,
            }

        return win_rates

    def get_turned_card_distribution(self) -> Dict[str, int]:
        """Get distribution of turned cards.

        Returns
        -------
        Dict[str, int]
            Count of each card turned up.
        """
        return Counter(str(card) for card in self.turned_cards)

    def get_screw_dealer_statistics(self) -> Dict:
        """Get statistics about "screw the dealer" rule.

        Returns
        -------
        Dict
            Statistics about screw the dealer occurrences and outcomes.
        """
        if not self.screw_dealer_hands:
            return {
                "total_occurrences": 0,
                "dealer_team_wins": 0,
                "dealer_team_losses": 0,
                "dealer_team_win_rate": 0.0,
            }

        total = len(self.screw_dealer_hands)
        dealer_team_wins = sum(1 for h in self.screw_dealer_hands if h.get("dealer_team_won_hand", False))
        dealer_team_losses = total - dealer_team_wins
        win_rate = dealer_team_wins / total if total > 0 else 0.0

        # Break down by dealer team
        team0_dealer = [h for h in self.screw_dealer_hands if h.get("dealer_team", -1) == 0]
        team1_dealer = [h for h in self.screw_dealer_hands if h.get("dealer_team", -1) == 1]

        team0_wins = sum(1 for h in team0_dealer if h.get("dealer_team_won_hand", False))
        team1_wins = sum(1 for h in team1_dealer if h.get("dealer_team_won_hand", False))

        return {
            "total_occurrences": total,
            "dealer_team_wins": dealer_team_wins,
            "dealer_team_losses": dealer_team_losses,
            "dealer_team_win_rate": win_rate,
            "team0_dealer": {
                "occurrences": len(team0_dealer),
                "wins": team0_wins,
                "losses": len(team0_dealer) - team0_wins,
                "win_rate": team0_wins / len(team0_dealer) if team0_dealer else 0.0,
            },
            "team1_dealer": {
                "occurrences": len(team1_dealer),
                "wins": team1_wins,
                "losses": len(team1_dealer) - team1_wins,
                "win_rate": team1_wins / len(team1_dealer) if team1_dealer else 0.0,
            },
        }

    def get_statistics_summary(self) -> Dict:
        """Get comprehensive statistics summary.

        Returns
        -------
        Dict
            Statistics summary.
        """
        if not self.games:
            return {}

        winners = [g["winner"] for g in self.games if g["winner"] is not None]
        hands = [g["hands_played"] for g in self.games]
        scores_team0 = [g["scores"][0] for g in self.games]
        scores_team1 = [g["scores"][1] for g in self.games]

        return {
            "total_games": len(self.games),
            "team0_wins": sum(1 for w in winners if w == 0),
            "team1_wins": sum(1 for w in winners if w == 1),
            "hands_played": {
                "mean": np.mean(hands),
                "std": np.std(hands),
                "min": np.min(hands),
                "max": np.max(hands),
                "median": np.median(hands),
            },
            "scores_team0": {
                "mean": np.mean(scores_team0),
                "std": np.std(scores_team0),
                "min": np.min(scores_team0),
                "max": np.max(scores_team0),
            },
            "scores_team1": {
                "mean": np.mean(scores_team1),
                "std": np.std(scores_team1),
                "min": np.min(scores_team1),
                "max": np.max(scores_team1),
            },
            "turned_card_distribution": self.get_turned_card_distribution(),
            "win_rates_by_combination": self.get_win_rates(),
            "screw_dealer_statistics": self.get_screw_dealer_statistics(),
        }


def play_single_game(
    risk_factors: List[Tuple[float, float]],
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict:
    """Play a single game and return results.

    Parameters
    ----------
    risk_factors : List[Tuple[float, float]]
        Risk factors for each player.
    model_path : Optional[str]
        Path to model file.
    device : Optional[str]
        Device to use.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict
        Game results.
    """
    if seed is not None:
        np.random.seed(seed)
        import random

        random.seed(seed)

    # Create profiles
    profiles: List[PlayerProfile] = []
    for i, (trump_risk, gameplay_risk) in enumerate(risk_factors):
        profile = PyTorchPlayerProfile(
            model_path=model_path,
            device=device,
            trump_selection_risk=trump_risk,
            gameplay_risk=gameplay_risk,
            player_name=f"PyTorch AI {i+1}",
        )
        profiles.append(profile)

    # Create game
    player_config = [
        ("PyTorch AI 1", "heuristic"),
        ("PyTorch AI 2", "heuristic"),
        ("PyTorch AI 3", "heuristic"),
        ("PyTorch AI 4", "heuristic"),
    ]
    game = Game(player_config)

    # Replace profiles
    for i, profile in enumerate(profiles):
        game.players[i].profile = profile

    # Play game
    hands_played = 0
    first_turned_card = None
    screw_dealer_hands: List[Dict] = []

    while True:
        hands_played += 1

        # Get turned card from first hand
        if hands_played == 1 and hasattr(game, "turned_card"):
            first_turned_card = game.turned_card

        # Store dealer info before playing hand
        dealer_id_before = game.dealer_id
        dealer_team = dealer_id_before % 2  # Teams: 0,2 -> team 0; 1,3 -> team 1
        scores_before = game.get_scores()

        continue_game = game.play_hand()

        # Check if screw the dealer occurred
        if game.trump_selector is not None and hasattr(game.trump_selector, "screw_the_dealer_occurred"):
            if game.trump_selector.screw_the_dealer_occurred:
                # When screw the dealer occurs, the dealer is forced to make trump
                # So the dealer's team is the "making" team
                # We need to check if they won 3+ tricks
                
                # Get tricks won from the hand (stored in _current_tricks_won)
                tricks_won = getattr(game, "_current_tricks_won", [0, 0])
                making_team_tricks = tricks_won[dealer_team]
                
                # In Euchre, the making team needs 3+ tricks to win the hand
                dealer_team_won_hand = making_team_tricks >= 3

                screw_dealer_hands.append({
                    "dealer_team": dealer_team,
                    "dealer_team_won_hand": dealer_team_won_hand,
                    "dealer_team_tricks": making_team_tricks,
                    "hand_number": hands_played,
                })

        winner = game.get_winner()
        if winner is not None:
            break

        if not continue_game:
            break

    return {
        "winner": game.get_winner(),
        "scores": tuple(game.scores),
        "hands_played": hands_played,
        "risk_factors": risk_factors,
        "turned_card": first_turned_card,
        "screw_dealer_hands": screw_dealer_hands,
    }


def simulate_screw_dealer_hand(
    seed: Optional[int] = None,
    dealer_id: int = 0,
    profile_type: str = "heuristic",
    max_attempts: int = 1000,
) -> Dict:
    """
    Simulate a single hand where "screw the dealer" occurs naturally.

    This function deals cards and checks if all non-dealer players naturally
    pass (because their hands are bad), then forces the dealer to pick trump.
    Only proceeds if all players naturally turn down selection.

    Parameters
    ----------
    seed : Optional[int]
        Random seed for reproducibility. If None, uses random seed.
    dealer_id : int
        ID of the dealer (0-3).
    profile_type : str
        Profile type for all players (default: "heuristic").
        Options: "heuristic", "ai", "random", "euchre_zero", etc.
        All players use the same type for consistency.
    max_attempts : int
        Maximum number of attempts to find a natural screw the dealer scenario.

    Returns
    -------
    Dict
        Result containing dealer_team, dealer_team_won_hand, and tricks_won.
        If "error" key is present, indicates the scenario didn't occur naturally.
    """
    import random
    from eucher.trump import TrumpSelector

    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    else:
        # Use random seed for this attempt
        seed = random.randint(0, 2**31 - 1)
        np.random.seed(seed)
        random.seed(seed)

    # Try to find a natural screw the dealer scenario
    for attempt in range(max_attempts):
        # Create game with same profile type for all players
        player_config = [
            ("Player 0", profile_type),
            ("Player 1", profile_type),
            ("Player 2", profile_type),
            ("Player 3", profile_type),
        ]
        
        try:
            game = Game(player_config, seed=seed + attempt)
        except Exception as e:
            # If profile type not supported, return error
            return {
                "dealer_team": dealer_id % 2,
                "dealer_team_won_hand": False,
                "dealer_team_tricks": 0,
                "error": f"Profile type '{profile_type}' not supported: {e}",
            }

        game.dealer_id = dealer_id

        # Deal cards
        deck = Deck()
        deck.shuffle()
        for player in game.players:
            cards = deck.deal(5)
            player.receive_hand(cards)

        # Turn up one card
        turned_card = deck.draw_one()
        game.turned_card = turned_card

        # Check Round 1: Order Up
        # Start with player left of dealer
        start_idx = (dealer_id + 1) % 4
        all_passed_round1 = True
        
        for i in range(3):  # 3 non-dealer players
            player_idx = (start_idx + i) % 4
            player = game.players[player_idx]
            
            # Let player naturally decide
            decision = player.decide_order_up(turned_card, dealer_id, None)
            if decision:
                # Someone ordered up - not a screw the dealer scenario
                all_passed_round1 = False
                break

        # If someone ordered up, try again
        if not all_passed_round1:
            continue

        # Round 2: Call Trump
        # All passed in round 1, now check round 2
        forbidden_suit = turned_card.suit
        all_passed_round2 = True
        
        for i in range(3):  # 3 non-dealer players
            player_idx = (start_idx + i) % 4
            player = game.players[player_idx]
            
            # Let player naturally decide
            decision = player.decide_call_trump(turned_card, None, must_choose=False)
            if decision is not None and decision != forbidden_suit:
                # Someone called trump - not a screw the dealer scenario
                all_passed_round2 = False
                break

        # If someone called trump, try again
        if not all_passed_round2:
            continue

        # All players naturally passed! Now dealer must pick (screw the dealer)
        dealer = game.players[dealer_id]
        dealer_decision = dealer.decide_call_trump(turned_card, None, must_choose=True)
        
        if dealer_decision is None or dealer_decision == forbidden_suit:
            # Dealer couldn't choose (shouldn't happen), pick a valid suit
            suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
            for suit in suits:
                if suit != forbidden_suit:
                    dealer_decision = suit
                    break

        game.trump_suit = dealer_decision

        # Play 5 tricks
        tricks_won = [0, 0]  # Team 0 and Team 1
        game._current_trick_number = 0

        # Reset trick winner for new hand
        if hasattr(game, "_last_trick_winner"):
            delattr(game, "_last_trick_winner")

        for trick_num in range(5):
            game._current_trick_number = trick_num
            winner_id = game._play_trick()
            winner = game.players[winner_id]
            tricks_won[winner.team] += 1
            game._current_tricks_won = tricks_won.copy()

        # Determine if dealer's team won
        dealer_team = dealer_id % 2
        dealer_team_tricks = tricks_won[dealer_team]
        dealer_team_won_hand = dealer_team_tricks >= 3

        return {
            "dealer_team": dealer_team,
            "dealer_team_won_hand": dealer_team_won_hand,
            "dealer_team_tricks": dealer_team_tricks,
            "tricks_won": tricks_won,
            "attempts": attempt + 1,
            "profile_type": profile_type,
        }

    # Couldn't find a natural screw the dealer scenario
    return {
        "dealer_team": dealer_id % 2,
        "dealer_team_won_hand": False,
        "dealer_team_tricks": 0,
        "error": f"Could not find natural screw the dealer scenario after {max_attempts} attempts",
        "profile_type": profile_type,
    }


def generate_doe_combinations() -> List[List[Tuple[float, float]]]:
    """Generate Design of Experiments combinations.

    Uses 5 risk levels: 0, 0.25, 0.5, 0.75, 1.0
    Creates balanced combinations for statistical analysis.

    Returns
    -------
    List[List[Tuple[float, float]]]
        List of risk factor combinations.
    """
    risk_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    combinations = []

    # Conservative vs Conservative
    combinations.append([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])

    # Balanced combinations
    # Team 0: Low risk, Team 1: High risk
    combinations.append([(0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (1.0, 1.0)])

    # Team 0: High risk, Team 1: Low risk
    combinations.append([(1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (0.0, 0.0)])

    # Mixed teams: One conservative, one aggressive per team
    combinations.append([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.5, 0.5)])

    # All combinations of risk levels (factorial design)
    # This creates a full factorial: 5^4 = 625 combinations
    # We'll sample a subset for efficiency
    for trump_p0 in risk_levels:
        for gameplay_p0 in risk_levels:
            for trump_p1 in risk_levels:
                for gameplay_p1 in risk_levels:
                    # Create symmetric teams
                    combinations.append(
                        [
                            (trump_p0, gameplay_p0),
                            (trump_p1, gameplay_p1),
                            (trump_p0, gameplay_p0),
                            (trump_p1, gameplay_p1),
                        ]
                    )

    # Add some asymmetric combinations
    for i, risk1 in enumerate(risk_levels):
        for j, risk2 in enumerate(risk_levels):
            if i != j:
                combinations.append(
                    [
                        (risk1, risk1),
                        (risk2, risk2),
                        (risk1, risk1),
                        (risk2, risk2),
                    ]
                )

    return combinations


def run_screw_dealer_simulation(
    num_hands: int,
    profile_type: str = "heuristic",
    parallel: bool = True,
    max_attempts_per_hand: int = 1000,
) -> GameStatistics:
    """
    Run Monte Carlo simulation specifically for "screw the dealer" scenarios.

    Simulates hands where all players naturally pass (because their hands are bad),
    then the dealer is forced to pick trump. Only includes scenarios that occur
    naturally through player decision-making.

    Parameters
    ----------
    num_hands : int
        Number of hands to simulate (will attempt until this many natural scenarios found).
    profile_type : str
        Profile type for all players (default: "heuristic").
        Options: "heuristic", "ai", "random", "euchre_zero", etc.
        All players use the same type for consistency.
    parallel : bool
        Whether to use parallel processing.
    max_attempts_per_hand : int
        Maximum attempts per hand to find a natural screw the dealer scenario.

    Returns
    -------
    GameStatistics
        Collected statistics.
    """
    stats = GameStatistics()

    # Distribute hands across dealer positions
    hands_per_dealer = max(1, num_hands // 4)
    remaining_hands = num_hands % 4

    hand_tasks = []
    for dealer_id in range(4):
        num_hands_this_dealer = hands_per_dealer
        if dealer_id < remaining_hands:
            num_hands_this_dealer += 1

        for hand_num in range(num_hands_this_dealer):
            seed = dealer_id * 1000000 + hand_num * 1000  # Deterministic seeds with space for attempts
            hand_tasks.append((dealer_id, seed, profile_type, max_attempts_per_hand))

    # Shuffle for better distribution
    np.random.shuffle(hand_tasks)

    # Helper function for parallel processing
    def _simulate_wrapper(args: Tuple[int, int, str, int]) -> Dict:
        """Wrapper for parallel processing."""
        dealer_id, seed, prof_type, max_attempts = args
        return simulate_screw_dealer_hand(seed, dealer_id, prof_type, max_attempts)

    # Run simulations
    if parallel:
        try:
            from multiprocessing import Pool
            import multiprocessing as mp

            num_workers = max(1, mp.cpu_count() - 1)
            with Pool(processes=num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(_simulate_wrapper, hand_tasks),
                        total=len(hand_tasks),
                        desc=f"Simulating screw the dealer ({profile_type})",
                    )
                )
        except Exception:
            # Fall back to sequential if parallel fails
            results = [
                simulate_screw_dealer_hand(seed, dealer_id, profile_type, max_attempts_per_hand)
                for dealer_id, seed, _, _ in tqdm(hand_tasks, desc=f"Simulating screw the dealer ({profile_type})")
            ]
    else:
        results = [
            simulate_screw_dealer_hand(seed, dealer_id, profile_type, max_attempts_per_hand)
            for dealer_id, seed, _, _ in tqdm(hand_tasks, desc=f"Simulating screw the dealer ({profile_type})")
        ]

    # Collect results
    successful = 0
    failed = 0
    for result in results:
        if "error" not in result:
            stats.screw_dealer_hands.append(result)
            successful += 1
        else:
            failed += 1

    if failed > 0:
        print(f"\nWarning: {failed} hands failed to find natural screw the dealer scenario")
        print(f"Successfully simulated: {successful} hands")

    return stats


def run_monte_carlo_simulation(
    num_games: int,
    risk_combinations: List[List[Tuple[float, float]]],
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    parallel: bool = True,
) -> GameStatistics:
    """Run Monte Carlo simulation.

    Parameters
    ----------
    num_games : int
        Number of games to play.
    risk_combinations : List[List[Tuple[float, float]]]
        Risk factor combinations to test.
    model_path : Optional[str]
        Path to model file.
    device : Optional[str]
        Device to use.
    parallel : bool
        Whether to use parallel processing.

    Returns
    -------
    GameStatistics
        Collected statistics.
    """
    stats = GameStatistics()

    # Distribute games across combinations
    games_per_combination = max(1, num_games // len(risk_combinations))
    remaining_games = num_games % len(risk_combinations)

    game_tasks = []
    for i, risk_factors in enumerate(risk_combinations):
        num_games_this_combo = games_per_combination
        if i < remaining_games:
            num_games_this_combo += 1

        for game_num in range(num_games_this_combo):
            seed = i * 10000 + game_num  # Deterministic seeds
            game_tasks.append((risk_factors, seed))

    # Shuffle for better distribution
    np.random.shuffle(game_tasks)

    # Play games
    if parallel and GPU_AVAILABLE:
        # Use GPU-accelerated batch processing if available
        print("Using GPU-accelerated simulation...")
        for risk_factors, seed in tqdm(game_tasks, desc="Playing games"):
            result = play_single_game(risk_factors, model_path, device, seed)
            stats.record_game(
                result["winner"],
                result["scores"],
                result["hands_played"],
                result["risk_factors"],
                result["turned_card"],
                result.get("screw_dealer_hands", []),
            )
    else:
        # Sequential processing
        for risk_factors, seed in tqdm(game_tasks, desc="Playing games"):
            result = play_single_game(risk_factors, model_path, device, seed)
            stats.record_game(
                result["winner"],
                result["scores"],
                result["hands_played"],
                result["risk_factors"],
                result["turned_card"],
                result.get("screw_dealer_hands", []),
            )

    return stats


def analyze_bell_curve(data: List[float], name: str) -> Dict:
    """Analyze if data follows a bell curve (normal distribution).

    Parameters
    ----------
    data : List[float]
        Data to analyze.
    name : str
        Name of the dataset.

    Returns
    -------
    Dict
        Analysis results.
    """
    if len(data) < 3:
        return {"error": "Insufficient data"}

    data_array = np.array(data)
    mean = np.mean(data_array)
    std = np.std(data_array)

    # Shapiro-Wilk test for normality
    if len(data_array) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data_array)
    else:
        # For large samples, use Anderson-Darling
        anderson_result = stats.anderson(data_array, dist="norm")
        shapiro_stat = anderson_result.statistic
        shapiro_p = None  # Anderson-Darling doesn't provide p-value directly

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(data_array, "norm", args=(mean, std))

    # Check for outliers (beyond 3 standard deviations)
    outliers = np.sum(np.abs(data_array - mean) > 3 * std)

    return {
        "name": name,
        "mean": float(mean),
        "std": float(std),
        "min": float(np.min(data_array)),
        "max": float(np.max(data_array)),
        "median": float(np.median(data_array)),
        "shapiro_wilk_statistic": float(shapiro_stat),
        "shapiro_wilk_p_value": float(shapiro_p) if shapiro_p is not None else None,
        "kolmogorov_smirnov_statistic": float(ks_stat),
        "kolmogorov_smirnov_p_value": float(ks_p),
        "outliers_3sigma": int(outliers),
        "outlier_percentage": float(outliers / len(data_array) * 100),
        "is_normal": ks_p > 0.05,  # p > 0.05 suggests normal distribution
    }


def analyze_card_distribution(turned_cards: List[Card]) -> Dict:
    """Analyze distribution of turned cards.

    Parameters
    ----------
    turned_cards : List[Card]
        List of turned cards.

    Returns
    -------
    Dict
        Distribution analysis.
    """
    if not turned_cards:
        return {"error": "No turned cards recorded"}

    card_counts = Counter(str(card) for card in turned_cards)
    total = len(turned_cards)
    expected_per_card = total / 24  # 24 possible cards

    # Chi-square test for uniform distribution
    observed = [card_counts.get(str(card), 0) for card in _all_cards()]
    chi2_stat, chi2_p = stats.chisquare(observed)

    # Find cards that never appeared
    all_card_strs = {str(card) for card in _all_cards()}
    never_appeared = all_card_strs - set(card_counts.keys())

    return {
        "total_turned": total,
        "expected_per_card": expected_per_card,
        "card_counts": dict(card_counts),
        "never_appeared": list(never_appeared),
        "chi_square_statistic": float(chi2_stat),
        "chi_square_p_value": float(chi2_p),
        "is_uniform": chi2_p > 0.05,  # p > 0.05 suggests uniform distribution
        "most_common": card_counts.most_common(5),
        "least_common": card_counts.most_common()[-5:] if len(card_counts) >= 5 else [],
    }


def _all_cards() -> List[Card]:
    """Get all possible Euchre cards."""
    cards = []
    for suit in Suit:
        for rank in [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]:
            cards.append(Card(suit, rank))
    return cards


def generate_report(stats: GameStatistics, output_file: Path) -> None:
    """Generate comprehensive statistical report.

    Parameters
    ----------
    stats : GameStatistics
        Collected statistics.
    output_file : Path
        Path to output file.
    """
    summary = stats.get_statistics_summary()
    screw_dealer_stats = stats.get_screw_dealer_statistics()

    # Generate report - handle case where we only have screw the dealer data
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "screw_dealer_analysis": screw_dealer_stats,
    }

    # Only add game statistics if we have games
    if stats.games:
        # Analyze bell curves
        hands_played = [g["hands_played"] for g in stats.games]
        scores_team0 = [g["scores"][0] for g in stats.games]
        scores_team1 = [g["scores"][1] for g in stats.games]

        bell_curve_hands = analyze_bell_curve(hands_played, "Hands Played")
        bell_curve_scores0 = analyze_bell_curve(scores_team0, "Team 0 Scores")
        bell_curve_scores1 = analyze_bell_curve(scores_team1, "Team 1 Scores")

        # Analyze card distribution
        card_dist = analyze_card_distribution(stats.turned_cards)

        report.update({
            "bell_curve_analysis": {
                "hands_played": bell_curve_hands,
                "team0_scores": bell_curve_scores0,
                "team1_scores": bell_curve_scores1,
            },
            "card_distribution_analysis": card_dist,
            "win_rates_by_combination": stats.get_win_rates(),
        })

    # Save JSON report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Generate human-readable report
    txt_file = output_file.with_suffix(".txt")
    with open(txt_file, "w") as f:
        f.write("=" * 80 + "\n")
        if stats.games:
            f.write("PyTorch AI Statistical Analysis Report\n")
        else:
            f.write("Screw the Dealer Statistical Analysis Report\n")
        f.write("=" * 80 + "\n\n")

        if stats.games:
            f.write(f"Total Games: {summary['total_games']}\n")
            f.write(f"Team 0 Wins: {summary['team0_wins']}\n")
            f.write(f"Team 1 Wins: {summary['team1_wins']}\n\n")

        if stats.games:
            bell_curve_hands = report.get("bell_curve_analysis", {}).get("hands_played", {})
            bell_curve_scores0 = report.get("bell_curve_analysis", {}).get("team0_scores", {})
            bell_curve_scores1 = report.get("bell_curve_analysis", {}).get("team1_scores", {})
            card_dist = report.get("card_distribution_analysis", {})

            f.write("Bell Curve Analysis:\n")
            f.write("-" * 80 + "\n")
            for name, analysis in [
                ("Hands Played", bell_curve_hands),
                ("Team 0 Scores", bell_curve_scores0),
                ("Team 1 Scores", bell_curve_scores1),
            ]:
                if analysis and "error" not in analysis:
                    f.write(f"\n{name}:\n")
                    f.write(f"  Mean: {analysis['mean']:.2f}\n")
                    f.write(f"  Std: {analysis['std']:.2f}\n")
                    f.write(f"  Normal Distribution: {analysis['is_normal']}\n")
                    f.write(f"  Outliers (3σ): {analysis['outliers_3sigma']} ({analysis['outlier_percentage']:.2f}%)\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Card Distribution Analysis:\n")
            f.write("-" * 80 + "\n")
            if card_dist and "error" not in card_dist:
                f.write(f"Total Cards Turned: {card_dist['total_turned']}\n")
                f.write(f"Expected per card: {card_dist['expected_per_card']:.2f}\n")
                f.write(f"Uniform Distribution: {card_dist['is_uniform']}\n")
                f.write(f"Chi-square p-value: {card_dist['chi_square_p_value']:.4f}\n\n")

                f.write("Most Common Cards:\n")
                for card, count in card_dist.get("most_common", []):
                    f.write(f"  {card}: {count} times\n")

                f.write("\nNever Appeared:\n")
                if card_dist.get("never_appeared"):
                    for card in card_dist["never_appeared"]:
                        f.write(f"  {card}\n")
                else:
                    f.write("  (All cards appeared at least once)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Screw the Dealer Analysis:\n")
        f.write("-" * 80 + "\n")
        screw_dealer_stats = stats.get_screw_dealer_statistics()
        f.write(f"Total Occurrences: {screw_dealer_stats['total_occurrences']}\n")
        if screw_dealer_stats['total_occurrences'] > 0:
            f.write(f"Dealer's Team Wins: {screw_dealer_stats['dealer_team_wins']}\n")
            f.write(f"Dealer's Team Losses: {screw_dealer_stats['dealer_team_losses']}\n")
            f.write(f"Dealer's Team Win Rate: {screw_dealer_stats['dealer_team_win_rate']:.2%}\n\n")
            
            f.write("Breakdown by Dealer Team:\n")
            f.write(f"  Team 0 as Dealer:\n")
            f.write(f"    Occurrences: {screw_dealer_stats['team0_dealer']['occurrences']}\n")
            f.write(f"    Wins: {screw_dealer_stats['team0_dealer']['wins']}\n")
            f.write(f"    Losses: {screw_dealer_stats['team0_dealer']['losses']}\n")
            f.write(f"    Win Rate: {screw_dealer_stats['team0_dealer']['win_rate']:.2%}\n")
            f.write(f"  Team 1 as Dealer:\n")
            f.write(f"    Occurrences: {screw_dealer_stats['team1_dealer']['occurrences']}\n")
            f.write(f"    Wins: {screw_dealer_stats['team1_dealer']['wins']}\n")
            f.write(f"    Losses: {screw_dealer_stats['team1_dealer']['losses']}\n")
            f.write(f"    Win Rate: {screw_dealer_stats['team1_dealer']['win_rate']:.2%}\n")
            
            # Statistical conclusion
            win_rate = screw_dealer_stats['dealer_team_win_rate']
            if win_rate < 0.5:
                f.write(f"\nCONCLUSION: Screw the dealer statistically disadvantages the dealer's team.\n")
                f.write(f"The dealer's team wins only {win_rate:.2%} of hands when forced to make trump.\n")
            elif win_rate > 0.5:
                f.write(f"\nCONCLUSION: Screw the dealer does NOT disadvantage the dealer's team.\n")
                f.write(f"The dealer's team wins {win_rate:.2%} of hands when forced to make trump.\n")
            else:
                f.write(f"\nCONCLUSION: Screw the dealer appears neutral.\n")
                f.write(f"The dealer's team wins {win_rate:.2%} of hands when forced to make trump.\n")
        else:
            f.write("No 'screw the dealer' occurrences recorded in this simulation.\n")

        if stats.games:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Win Rates by Risk Combination:\n")
            f.write("-" * 80 + "\n")
            for risk_key, rates in sorted(stats.get_win_rates().items()):
                f.write(f"\nRisk Factors: {risk_key}\n")
                f.write(f"  Team 0 Win Rate: {rates['team0_rate']:.2%}\n")
                f.write(f"  Team 1 Win Rate: {rates['team1_rate']:.2%}\n")
                f.write(f"  Total Games: {rates['total_games']}\n")

    print(f"\nReport saved to: {output_file}")
    print(f"Human-readable report: {txt_file}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run statistical analysis on PyTorch AI players"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1000,
        help="Number of games to play (default: 1000)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to PyTorch model file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cpu' or 'cuda' (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stats/analysis_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--doe-only",
        action="store_true",
        help="Only run Design of Experiments combinations",
    )
    parser.add_argument(
        "--screw-dealer-only",
        action="store_true",
        help="Run fast simulation only for 'screw the dealer' scenarios",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=10000,
        help="Number of hands to simulate (for --screw-dealer-only, default: 10000)",
    )
    parser.add_argument(
        "--profile-type",
        type=str,
        default="heuristic",
        help="Player profile type for gameplay (default: 'heuristic'). Options: heuristic, ai, random, euchre_zero, etc.",
    )
    parser.add_argument(
        "--all-profile-types",
        action="store_true",
        help="Run simulation for all profile types (heuristic, ai, random, euchre_zero)",
    )

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    elif GPU_AVAILABLE:
        device = "cuda"
        print("GPU detected, using CUDA acceleration")
    else:
        device = "cpu"
        print("Using CPU")

    # Generate risk combinations
    if args.doe_only:
        risk_combinations = generate_doe_combinations()
        print(f"Generated {len(risk_combinations)} DOE combinations")
    else:
        # Include some standard combinations plus DOE
        risk_combinations = [
            # Conservative vs Conservative
            [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            # Balanced
            [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
            # Mixed
            [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (1.0, 1.0)],
        ]
        # Add key DOE combinations (sample subset for efficiency)
        doe_combos = generate_doe_combinations()
        # Sample strategically: take first few, then sample evenly
        key_combos = doe_combos[:10]  # First 10 key combinations
        if len(doe_combos) > 10:
            sample_rate = max(1, (len(doe_combos) - 10) // 40)  # Sample 40 more
            key_combos.extend(doe_combos[10::sample_rate])
        risk_combinations.extend(key_combos)

    # Run simulation
    if args.screw_dealer_only:
        if args.all_profile_types:
            # Run for all profile types
            profile_types = ["heuristic", "ai", "random", "euchre_zero"]
            all_stats = GameStatistics()
            
            for prof_type in profile_types:
                print(f"\n{'='*80}")
                print(f"Running 'screw the dealer' simulation for {prof_type} players")
                print(f"{'='*80}")
                print(f"Number of hands: {args.num_hands}")
                
                stats = run_screw_dealer_simulation(
                    args.num_hands,
                    prof_type,
                    parallel=True,
                )
                
                # Merge stats
                all_stats.screw_dealer_hands.extend(stats.screw_dealer_hands)
                
                # Print summary for this profile type
                prof_stats = stats.get_screw_dealer_statistics()
                if prof_stats['total_occurrences'] > 0:
                    print(f"\n{prof_type} Results:")
                    print(f"  Total occurrences: {prof_stats['total_occurrences']}")
                    print(f"  Dealer's team win rate: {prof_stats['dealer_team_win_rate']:.2%}")
            
            stats = all_stats
        else:
            print(f"Running fast 'screw the dealer' simulation with {args.num_hands} hands...")
            print(f"Profile type: {args.profile_type}")
            stats = run_screw_dealer_simulation(
                args.num_hands,
                args.profile_type,
                parallel=True,
            )
    else:
        print(f"Total risk combinations: {len(risk_combinations)}")
        stats = run_monte_carlo_simulation(
            args.num_games,
            risk_combinations,
            args.model_path,
            device,
        )

    # Generate report
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path("stats") / f"analysis_{timestamp}.json"

    generate_report(stats, output_file)


if __name__ == "__main__":
    main()

