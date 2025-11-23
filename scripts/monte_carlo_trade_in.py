"""Monte Carlo simulation for evaluating 9-10 trade-in strategy.

This script randomly generates hands until a player has a 9-10 trade-in situation,
then forks the game to simulate both trading in and not trading in, playing both
games to completion to evaluate if trading in is a good strategy.
"""

import copy
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from eucher.cards import Card, Deck, Rank, Suit
from eucher.game import Game
from eucher.players import Player
from eucher.players.base import PlayerProfile
from eucher.trade_in import TradeInHandler


@dataclass
class TradeInResult:
    """Results from a single trade-in simulation."""

    player_id: int
    player_type: str
    eligible_cards: List[Card]
    kitty_cards: List[Card]
    traded_in: bool
    team_won: bool
    tricks_won_team: int
    tricks_won_opponent: int
    final_scores: Tuple[int, int]


class ForcedTradeInProfile(PlayerProfile):
    """Profile that forces a specific trade-in decision."""

    def __init__(self, force_trade_in: bool, base_profile: PlayerProfile) -> None:
        """
        Initialize the forced trade-in profile.

        Parameters
        ----------
        force_trade_in : bool
            True to force trade-in, False to force no trade-in.
        base_profile : PlayerProfile
            The base profile to delegate other decisions to.
        """
        self.force_trade_in = force_trade_in
        self.base_profile = base_profile

    def decide_trade_in(self, player: Player, eligible_cards: List[Card]) -> bool:
        """Force the trade-in decision."""
        return self.force_trade_in

    def decide_order_up(
        self, player: Player, turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """Delegate to base profile."""
        return self.base_profile.decide_order_up(player, turned_card, dealer_id, trump_suit)

    def decide_call_trump(
        self,
        player: Player,
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """Delegate to base profile."""
        return self.base_profile.decide_call_trump(player, turned_card, trump_suit, must_choose)

    def choose_card_to_discard(
        self, player: Player, turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
    ) -> Card:
        """Delegate to base profile."""
        return self.base_profile.choose_card_to_discard(player, turned_card, ordered_up_by)

    def play_card(
        self,
        player: Player,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """Delegate to base profile."""
        return self.base_profile.play_card(player, led_suit, trump_suit, trick_cards, trick_player_ids)

    def decide_going_alone(self, player: Player, trump_suit: Suit) -> bool:
        """Delegate to base profile."""
        return self.base_profile.decide_going_alone(player, trump_suit)


def find_trade_in_situation(game: Game) -> Optional[Tuple[Player, List[Card]]]:
    """
    Find if any player has a 9-10 trade-in situation.

    Parameters
    ----------
    game : Game
        The game to check.

    Returns
    -------
    Optional[Tuple[Player, List[Card]]]
        Tuple of (player, eligible_cards) if found, None otherwise.
    """
    trade_in_handler = TradeInHandler(game.players, game.kitty, game.dealer_id, None)
    for player in game.players:
        eligible_cards = trade_in_handler.check_trade_in_eligible(player)
        if eligible_cards is not None:
            return (player, eligible_cards)
    return None


def generate_random_hand_with_trade_in(
    max_attempts: int = 10000, seed: Optional[int] = None, show_progress: bool = False
) -> Optional[Tuple[Game, Player, List[Card]]]:
    """
    Generate a random hand until someone has a 9-10 trade-in situation.

    Parameters
    ----------
    max_attempts : int
        Maximum number of attempts before giving up.
    seed : Optional[int]
        Random seed for reproducibility.
    show_progress : bool
        Whether to show progress indicators.

    Returns
    -------
    Optional[Tuple[Game, Player, List[Card]]]
        Tuple of (game, player, eligible_cards) if found, None otherwise.
    """
    if seed is not None:
        random.seed(seed)

    for attempt in range(max_attempts):
        if show_progress and attempt > 0 and attempt % 1000 == 0:
            print(f"    Attempt {attempt}/{max_attempts}...", end="\r", flush=True)

        # Create a game with random players (we'll test different types later)
        player_config = [
            ("Player0", "random"),
            ("Player1", "random"),
            ("Player2", "random"),
            ("Player3", "random"),
        ]
        game = Game(player_config, seed=None, trade_in_enabled=True)

        # Deal a hand
        deck = Deck()
        deck.shuffle()

        # Deal 5 cards to each player
        for player in game.players:
            cards = deck.deal(5)
            player.receive_hand(cards)

        # Turn up one card
        game.turned_card = deck.draw_one()

        # Store kitty
        game.kitty = deck.cards.copy()

        # Check for trade-in situation
        trade_in_situation = find_trade_in_situation(game)
        if trade_in_situation is not None:
            if show_progress:
                print(f"    Found trade-in situation after {attempt + 1} attempts!")
            player, eligible_cards = trade_in_situation
            return (game, player, eligible_cards)

    if show_progress:
        print(f"    No trade-in situation found after {max_attempts} attempts.")
    return None


def simulate_game_with_trade_in_decision(
    base_game: Game,
    trade_in_player: Player,
    eligible_cards: List[Card],
    force_trade_in: bool,
    player_type: str,
) -> TradeInResult:
    """
    Simulate a game with a forced trade-in decision.

    Parameters
    ----------
    base_game : Game
        The base game state (at the point of trade-in decision).
    trade_in_player : Player
        The player making the trade-in decision.
    eligible_cards : List[Card]
        The cards eligible for trade-in.
    force_trade_in : bool
        True to force trade-in, False to force no trade-in.
    player_type : str
        Type of player making the decision (for tracking).

    Returns
    -------
    TradeInResult
        The results of the simulation.
    """
    # Deep copy the game state
    game = copy.deepcopy(base_game)

    # Initialize going_alone attributes if not present
    if not hasattr(game, "going_alone"):
        game.going_alone = False
    if not hasattr(game, "going_alone_player_id"):
        game.going_alone_player_id = None

    # Find the corresponding player in the copied game
    copied_player = game.players[trade_in_player.player_id]

    # Override the trade-in decision
    original_profile = copied_player.profile
    forced_profile = ForcedTradeInProfile(force_trade_in, original_profile)
    copied_player.profile = forced_profile

    # Process trade-in if forced
    if force_trade_in:
        trade_in_handler = TradeInHandler(game.players, game.kitty, game.dealer_id, None)
        trade_in_handler._execute_trade_in(copied_player, eligible_cards)
        game.kitty = trade_in_handler.kitty.copy()

    # Play the hand to completion
    initial_scores = game.get_scores()
    tricks_won = [0, 0]

    # Select trump
    from eucher.trump import TrumpSelector

    game.trump_selector = TrumpSelector(game.players, None)
    game.trump_suit = game.trump_selector.select_trump(game.turned_card, game.dealer_id)

    # If all passed, this hand doesn't count
    if game.trump_suit is None:
        return TradeInResult(
            player_id=trade_in_player.player_id,
            player_type=player_type,
            eligible_cards=eligible_cards.copy(),
            kitty_cards=base_game.kitty.copy(),
            traded_in=force_trade_in,
            team_won=False,
            tricks_won_team=0,
            tricks_won_opponent=0,
            final_scores=(initial_scores[0], initial_scores[1]),
        )

    # Check if going alone
    if game.trump_selector.going_alone:
        game.going_alone = True
        game.going_alone_player_id = game.trump_selector.going_alone_player_id

    # Play all 5 tricks
    for _ in range(5):
        winner_id = game._play_trick()
        winner = game.players[winner_id]
        tricks_won[winner.team] += 1

    # Score the hand using the game's scoring method
    game._score_hand(tricks_won)

    final_scores = game.get_scores()
    trade_in_player_team = trade_in_player.team
    team_won = game.scores[trade_in_player_team] > initial_scores[trade_in_player_team]

    return TradeInResult(
        player_id=trade_in_player.player_id,
        player_type=player_type,
        eligible_cards=eligible_cards.copy(),
        kitty_cards=base_game.kitty.copy(),
        traded_in=force_trade_in,
        team_won=team_won,
        tricks_won_team=tricks_won[trade_in_player_team],
        tricks_won_opponent=tricks_won[1 - trade_in_player_team],
        final_scores=(final_scores[0], final_scores[1]),
    )


def run_monte_carlo_simulation(
    player_types: List[str] = ["random", "heuristic", "ai"],
    num_simulations: int = 100,
    max_attempts_per_sim: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, Dict]:
    """
    Run Monte Carlo simulation for 9-10 trade-in strategy.

    Parameters
    ----------
    player_types : List[str]
        List of player types to test.
    num_simulations : int
        Number of simulations to run per player type.
    max_attempts_per_sim : int
        Maximum attempts to find a trade-in situation per simulation.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Dict]
        Statistics for each player type.
    """
    if seed is not None:
        random.seed(seed)

    results: Dict[str, List[Tuple[TradeInResult, TradeInResult]]] = defaultdict(list)
    stats: Dict[str, Dict] = {}

    for player_type in player_types:
        print(f"\n{'='*80}")
        print(f"Testing player type: {player_type}")
        print(f"{'='*80}")

        simulation_count = 0
        attempt_count = 0
        last_progress_update = 0

        print(f"  Searching for trade-in situations (this may take a while)...")
        while simulation_count < num_simulations and attempt_count < num_simulations * max_attempts_per_sim:
            attempt_count += 1

            # Show progress every 5000 attempts
            if attempt_count - last_progress_update >= 5000:
                print(f"  Attempt {attempt_count}: Found {simulation_count}/{num_simulations} situations...")
                last_progress_update = attempt_count

            # Generate a random hand with trade-in situation
            result = generate_random_hand_with_trade_in(
                max_attempts=max_attempts_per_sim, seed=None, show_progress=False
            )

            if result is None:
                # If we exhausted attempts, break
                if attempt_count >= num_simulations * max_attempts_per_sim:
                    print(f"  Warning: Only found {simulation_count} situations after {attempt_count} attempts")
                    break
                continue

            base_game, trade_in_player, eligible_cards = result

            # Replace the trade-in player with the specified type
            player_config = [
                ("Player0", player_type if i == trade_in_player.player_id else "random")
                for i in range(4)
            ]
            new_game = Game(player_config, seed=None, trade_in_enabled=True)

            # Copy the hand state
            new_game.dealer_id = base_game.dealer_id
            new_game.turned_card = base_game.turned_card
            new_game.kitty = base_game.kitty.copy()

            for i, player in enumerate(base_game.players):
                new_game.players[i].receive_hand(player.hand.copy())

            # Find the trade-in player in the new game
            new_trade_in_player = new_game.players[trade_in_player.player_id]

            # Simulate both paths
            result_trade_in = simulate_game_with_trade_in_decision(
                new_game, new_trade_in_player, eligible_cards, True, player_type
            )
            result_no_trade_in = simulate_game_with_trade_in_decision(
                new_game, new_trade_in_player, eligible_cards, False, player_type
            )

            results[player_type].append((result_trade_in, result_no_trade_in))
            simulation_count += 1

            print(f"  Found situation {simulation_count}/{num_simulations} (after {attempt_count} attempts)")
            if simulation_count % 10 == 0:
                print(f"  Completed {simulation_count}/{num_simulations} simulations...")

        # Calculate statistics
        trade_in_wins = sum(1 for r1, r2 in results[player_type] if r1.team_won)
        no_trade_in_wins = sum(1 for r1, r2 in results[player_type] if r2.team_won)
        trade_in_better = sum(1 for r1, r2 in results[player_type] if r1.team_won and not r2.team_won)
        no_trade_in_better = sum(1 for r1, r2 in results[player_type] if r2.team_won and not r1.team_won)
        ties = sum(1 for r1, r2 in results[player_type] if r1.team_won == r2.team_won)

        avg_tricks_trade_in = sum(r1.tricks_won_team for r1, r2 in results[player_type]) / len(
            results[player_type]
        )
        avg_tricks_no_trade_in = sum(r2.tricks_won_team for r1, r2 in results[player_type]) / len(
            results[player_type]
        )

        stats[player_type] = {
            "total_simulations": len(results[player_type]),
            "trade_in_wins": trade_in_wins,
            "no_trade_in_wins": no_trade_in_wins,
            "trade_in_better": trade_in_better,
            "no_trade_in_better": no_trade_in_better,
            "ties": ties,
            "trade_in_win_rate": trade_in_wins / len(results[player_type]) if results[player_type] else 0,
            "no_trade_in_win_rate": no_trade_in_wins / len(results[player_type])
            if results[player_type]
            else 0,
            "avg_tricks_trade_in": avg_tricks_trade_in,
            "avg_tricks_no_trade_in": avg_tricks_no_trade_in,
            "trade_in_advantage": trade_in_better - no_trade_in_better,
        }

        print(f"\n  Statistics for {player_type}:")
        print(f"    Total simulations: {stats[player_type]['total_simulations']}")
        print(f"    Trade-in wins: {stats[player_type]['trade_in_wins']} ({stats[player_type]['trade_in_win_rate']:.1%})")
        print(f"    No trade-in wins: {stats[player_type]['no_trade_in_wins']} ({stats[player_type]['no_trade_in_win_rate']:.1%})")
        print(f"    Trade-in better: {stats[player_type]['trade_in_better']}")
        print(f"    No trade-in better: {stats[player_type]['no_trade_in_better']}")
        print(f"    Ties: {stats[player_type]['ties']}")
        print(f"    Average tricks (trade-in): {stats[player_type]['avg_tricks_trade_in']:.2f}")
        print(f"    Average tricks (no trade-in): {stats[player_type]['avg_tricks_no_trade_in']:.2f}")
        print(f"    Trade-in advantage: {stats[player_type]['trade_in_advantage']}")

    return stats


def main() -> None:
    """Run the Monte Carlo simulation."""
    import argparse

    parser = argparse.ArgumentParser(description="Monte Carlo simulation for 9-10 trade-in strategy")
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=100,
        help="Number of simulations per player type (default: 100)",
    )
    parser.add_argument(
        "--player-types",
        nargs="+",
        default=["random", "heuristic", "ai"],
        choices=["random", "heuristic", "ai"],
        help="Player types to test (default: random heuristic ai)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: print to stdout)",
    )

    args = parser.parse_args()

    print("Monte Carlo Simulation: 9-10 Trade-In Strategy Evaluation")
    print("=" * 80)
    print(f"Player types: {', '.join(args.player_types)}")
    print(f"Simulations per type: {args.num_simulations}")
    print(f"Seed: {args.seed}")

    stats = run_monte_carlo_simulation(
        player_types=args.player_types,
        num_simulations=args.num_simulations,
        seed=args.seed,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for player_type, stat in stats.items():
        print(f"\n{player_type.upper()}:")
        print(f"  Trade-in is better: {stat['trade_in_better']} times")
        print(f"  No trade-in is better: {stat['no_trade_in_better']} times")
        print(f"  Net advantage: {stat['trade_in_advantage']:+d}")
        if stat["trade_in_advantage"] > 0:
            print(f"  → Trade-in is a GOOD strategy")
        elif stat["trade_in_advantage"] < 0:
            print(f"  → Trade-in is a BAD strategy")
        else:
            print(f"  → Trade-in is NEUTRAL")

    # Save to file if requested
    if args.output:
        import json
        from datetime import datetime

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_simulations": args.num_simulations,
                "player_types": args.player_types,
                "seed": args.seed,
            },
            "statistics": stats,
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

