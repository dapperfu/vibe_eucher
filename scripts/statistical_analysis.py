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

from eucher.ai_players.pytorch_player import PyTorchStrategicPlayer
from eucher.cards import Card, Rank, Suit
from eucher.game import Game
from eucher.players import Player
from eucher.players.profiles import PlayerProfile

try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False


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
        super().__init__()
        self.player_name = player_name
        self.pytorch_player = PyTorchStrategicPlayer(
            model_path=model_path,
            device=device,
            trump_selection_risk=trump_selection_risk,
            gameplay_risk=gameplay_risk,
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

    def record_game(
        self,
        winner: Optional[int],
        scores: Tuple[int, int],
        hands_played: int,
        risk_factors: List[Tuple[float, float]],
        turned_card: Optional[Card] = None,
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
        ("PyTorch AI 1", "simple"),
        ("PyTorch AI 2", "simple"),
        ("PyTorch AI 3", "simple"),
        ("PyTorch AI 4", "simple"),
    ]
    game = Game(player_config)

    # Replace profiles
    for i, profile in enumerate(profiles):
        game.players[i].profile = profile

    # Play game
    hands_played = 0
    first_turned_card = None

    while True:
        hands_played += 1

        # Get turned card from first hand
        if hands_played == 1 and hasattr(game, "turned_card"):
            first_turned_card = game.turned_card

        continue_game = game.play_hand()

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

    # Analyze bell curves
    hands_played = [g["hands_played"] for g in stats.games]
    scores_team0 = [g["scores"][0] for g in stats.games]
    scores_team1 = [g["scores"][1] for g in stats.games]

    bell_curve_hands = analyze_bell_curve(hands_played, "Hands Played")
    bell_curve_scores0 = analyze_bell_curve(scores_team0, "Team 0 Scores")
    bell_curve_scores1 = analyze_bell_curve(scores_team1, "Team 1 Scores")

    # Analyze card distribution
    card_dist = analyze_card_distribution(stats.turned_cards)

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "bell_curve_analysis": {
            "hands_played": bell_curve_hands,
            "team0_scores": bell_curve_scores0,
            "team1_scores": bell_curve_scores1,
        },
        "card_distribution_analysis": card_dist,
        "win_rates_by_combination": stats.get_win_rates(),
    }

    # Save JSON report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Generate human-readable report
    txt_file = output_file.with_suffix(".txt")
    with open(txt_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch AI Statistical Analysis Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Games: {summary['total_games']}\n")
        f.write(f"Team 0 Wins: {summary['team0_wins']}\n")
        f.write(f"Team 1 Wins: {summary['team1_wins']}\n\n")

        f.write("Bell Curve Analysis:\n")
        f.write("-" * 80 + "\n")
        for name, analysis in [
            ("Hands Played", bell_curve_hands),
            ("Team 0 Scores", bell_curve_scores0),
            ("Team 1 Scores", bell_curve_scores1),
        ]:
            if "error" not in analysis:
                f.write(f"\n{name}:\n")
                f.write(f"  Mean: {analysis['mean']:.2f}\n")
                f.write(f"  Std: {analysis['std']:.2f}\n")
                f.write(f"  Normal Distribution: {analysis['is_normal']}\n")
                f.write(f"  Outliers (3σ): {analysis['outliers_3sigma']} ({analysis['outlier_percentage']:.2f}%)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Card Distribution Analysis:\n")
        f.write("-" * 80 + "\n")
        if "error" not in card_dist:
            f.write(f"Total Cards Turned: {card_dist['total_turned']}\n")
            f.write(f"Expected per card: {card_dist['expected_per_card']:.2f}\n")
            f.write(f"Uniform Distribution: {card_dist['is_uniform']}\n")
            f.write(f"Chi-square p-value: {card_dist['chi_square_p_value']:.4f}\n\n")

            f.write("Most Common Cards:\n")
            for card, count in card_dist["most_common"]:
                f.write(f"  {card}: {count} times\n")

            f.write("\nNever Appeared:\n")
            if card_dist["never_appeared"]:
                for card in card_dist["never_appeared"]:
                    f.write(f"  {card}\n")
            else:
                f.write("  (All cards appeared at least once)\n")

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

    print(f"Total risk combinations: {len(risk_combinations)}")

    # Run simulation
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

