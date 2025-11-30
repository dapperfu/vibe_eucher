#!/usr/bin/env python3
"""Monte Carlo simulation to test discard strategies in Euchre.

Tests whether discarding a low trump card can be optimal compared to
discarding high off-suit cards.
"""

import argparse
import multiprocessing as mp
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from eucher.players.base import Player

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eucher.cards import Card, Deck, Rank, Suit
from eucher.game import Game


class ForcedDiscardProfile:
    """Profile that forces a specific discard decision."""

    def __init__(self, discard_card: Card, base_profile) -> None:
        """Initialize with forced discard card and base profile."""
        self.discard_card = discard_card
        self.base_profile = base_profile

    def choose_card_to_discard(
        self, player: "Player", turned_card=None, ordered_up_by=None
    ) -> Card:
        """Force discard of specified card."""
        if self.discard_card in player.hand:
            return self.discard_card
        # Fallback if card not in hand (shouldn't happen)
        return min(player.hand, key=lambda c: c.rank.value)

    def __getattr__(self, name):
        """Delegate all other methods to base profile."""
        return getattr(self.base_profile, name)


def simulate_hand_with_discard(
    test_hand: List[Card],
    discard_card: Card,
    trump_suit: Suit,
    simulation_seed: int,
    dealer_id: int = 0,
    test_player_id: int = 0,
    opponent_type: str = "heuristic",
) -> Tuple[bool, int, int]:
    """
    Simulate a hand with a specific discard decision.

    Parameters
    ----------
    test_hand : List[Card]
        The test player's hand (6 cards before discard).
    discard_card : Card
        The card to discard.
    trump_suit : Suit
        The trump suit.
    simulation_seed : int
        Random seed for this simulation (for opponent hands).
    dealer_id : int
        ID of the dealer (default: 0).
    test_player_id : int
        ID of the test player (default: 0, the dealer).
    opponent_type : str
        Type of opponent bots to use.

    Returns
    -------
    Tuple[bool, int, int]
        Tuple of (test_player_team_won, tricks_won_team0, tricks_won_team1).
    """
    # Set seed for this simulation
    random.seed(simulation_seed)
    np.random.seed(simulation_seed)

    # Create game with specified opponent types
    player_config = [
        ("Player0", opponent_type),
        ("Player1", opponent_type),
        ("Player2", opponent_type),
        ("Player3", opponent_type),
    ]

    game = Game(player_config, seed=simulation_seed)
    game.dealer_id = dealer_id
    game.trump_suit = trump_suit
    game.turned_card = Card(trump_suit, Rank.TEN)  # Assume 10 was turned and ordered up

    # Create full deck and remove test hand and turned card
    full_deck = Deck()
    all_cards = set(full_deck.cards)

    # Remove test hand cards and turned card from available cards
    # Note: test_hand has 6 cards (before discard), turned_card is the 10♥ that was ordered up
    used_cards = set(test_hand) | {game.turned_card}
    remaining_cards = [card for card in all_cards if card not in used_cards]

    # Shuffle remaining cards for opponent hands and kitty
    random.shuffle(remaining_cards)

    # Create base profile for test player
    from eucher.plugins import get_registry
    registry = get_registry()
    base_metadata = registry.get(opponent_type)
    if base_metadata:
        if base_metadata.requires_game:
            base_profile = base_metadata.factory(game=game, player_id=test_player_id)
        else:
            base_profile = base_metadata.factory()
    else:
        # Fallback to heuristic
        from eucher.players.computer.heuristic import HeuristicPlayer
        base_profile = HeuristicPlayer()

    # Set test player's profile (will use for playing cards)
    game.players[test_player_id].profile = base_profile

    # Deal remaining cards to opponents (5 cards each)
    # Total: 24 cards in deck
    # Used: 6 (test hand) + 1 (turned card) = 7
    # Remaining: 24 - 7 = 17 cards
    # Opponents: 3 × 5 = 15 cards
    # Kitty: 17 - 15 = 2 cards (but actually kitty should have 3, so we'll use 2 for opponents)
    card_idx = 0
    for i in range(4):
        if i != test_player_id:
            opponent_hand = remaining_cards[card_idx : card_idx + 5]
            card_idx += 5
            game.players[i].receive_hand(opponent_hand)

    # Simulate discard (dealer picks up turned card and discards)
    # Remove the discard card from hand
    final_hand = [card for card in test_hand if card != discard_card]
    if len(final_hand) != 5:
        raise ValueError(f"Expected 5 cards after discard, got {len(final_hand)}")
    game.players[test_player_id].receive_hand(final_hand)

    # Initialize game state for playing tricks
    game._current_trick_number = 0
    game._current_tricks_won = [0, 0]
    game.going_alone = False
    game.going_alone_player_id = None
    
    # Reset trick winner - first trick starts with player left of dealer
    if hasattr(game, "_last_trick_winner"):
        delattr(game, "_last_trick_winner")

    # Verify all players have exactly 5 cards
    for i, player in enumerate(game.players):
        if len(player.hand) != 5:
            if simulation_seed % 1000 == 0:
                print(f"Warning: Player {i} has {len(player.hand)} cards (expected 5)", file=sys.stderr)
            return (False, 0, 0)

    # Play 5 tricks using game's built-in method
    tricks_won = [0, 0]
    try:
        for trick_num in range(5):
            game._current_trick_number = trick_num
            winner_id = game._play_trick()
            winner = game.players[winner_id]
            tricks_won[winner.team] += 1
            game._last_trick_winner = winner_id

        # Check if test player's team won (need 3+ tricks)
        test_player_team = test_player_id % 2
        return (tricks_won[test_player_team] >= 3, tricks_won[0], tricks_won[1])

    except Exception as e:
        # If simulation fails, return False (conservative)
        if simulation_seed % 1000 == 0:  # Only print error occasionally
            print(f"Warning: Simulation failed (seed {simulation_seed}): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        return (False, 0, 0)


def _simulate_hand_wrapper(args: Tuple) -> Tuple[bool, int, int]:
    """Wrapper for multiprocessing."""
    (
        test_hand,
        discard_card,
        trump_suit,
        simulation_seed,
        dealer_id,
        test_player_id,
        opponent_type,
    ) = args
    return simulate_hand_with_discard(
        test_hand,
        discard_card,
        trump_suit,
        simulation_seed,
        dealer_id,
        test_player_id,
        opponent_type,
    )


def run_monte_carlo_comparison(
    test_hand: List[Card],
    discard_strategies: List[Tuple[str, Card]],
    trump_suit: Suit,
    num_simulations: int,
    base_seed: int,
    dealer_id: int = 0,
    test_player_id: int = 0,
    opponent_type: str = "heuristic",
    num_workers: Optional[int] = None,
) -> dict:
    """
    Run Monte Carlo comparison of discard strategies.

    Parameters
    ----------
    test_hand : List[Card]
        The test player's hand (6 cards before discard).
    discard_strategies : List[Tuple[str, Card]]
        List of (strategy_name, card_to_discard) tuples.
    trump_suit : Suit
        The trump suit.
    num_simulations : int
        Number of simulations to run per strategy.
    base_seed : int
        Base seed for generating simulation seeds.
    dealer_id : int
        ID of the dealer.
    test_player_id : int
        ID of the test player.
    opponent_type : str
        Type of opponent bots.
    num_workers : Optional[int]
        Number of parallel workers. If None, uses CPU count - 1.

    Returns
    -------
    dict
        Dictionary with results for each strategy.
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    results = {}

    for strategy_name, discard_card in discard_strategies:
        print(f"\nTesting strategy: {strategy_name} (discard {discard_card})")

        # Prepare arguments for each simulation
        simulation_args = [
            (
                test_hand,
                discard_card,
                trump_suit,
                base_seed + sim_num,
                dealer_id,
                test_player_id,
                opponent_type,
            )
            for sim_num in range(num_simulations)
        ]

        wins = 0
        tricks_won_list: List[int] = []
        opponent_tricks_list: List[int] = []

        # Run simulations in parallel
        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                simulation_results = list(
                    tqdm(
                        pool.imap(_simulate_hand_wrapper, simulation_args),
                        total=num_simulations,
                        desc=f"  {strategy_name}",
                    )
                )
        else:
            simulation_results = [
                _simulate_hand_wrapper(args)
                for args in tqdm(simulation_args, desc=f"  {strategy_name}")
            ]

        for team_won, tricks_team0, tricks_team1 in simulation_results:
            if team_won:
                wins += 1
            test_player_team = test_player_id % 2
            if test_player_team == 0:
                tricks_won_list.append(tricks_team0)
                opponent_tricks_list.append(tricks_team1)
            else:
                tricks_won_list.append(tricks_team1)
                opponent_tricks_list.append(tricks_team0)

        win_rate = wins / num_simulations if num_simulations > 0 else 0.0
        avg_tricks_won = np.mean(tricks_won_list) if tricks_won_list else 0.0
        avg_opponent_tricks = (
            np.mean(opponent_tricks_list) if opponent_tricks_list else 0.0
        )

        results[strategy_name] = {
            "win_rate": win_rate,
            "wins": wins,
            "total_simulations": num_simulations,
            "avg_tricks_won": avg_tricks_won,
            "avg_opponent_tricks": avg_opponent_tricks,
            "discard_card": str(discard_card),
        }

        print(f"  Win rate: {win_rate:.2%} ({wins}/{num_simulations})")
        print(f"  Avg tricks won: {avg_tricks_won:.2f}")
        print(f"  Avg opponent tricks: {avg_opponent_tricks:.2f}")

    return results


def main() -> None:
    """Main function to run Monte Carlo discard strategy comparison."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation to compare discard strategies in Euchre"
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=10000,
        help="Number of simulations to run per strategy (default: 10000)",
    )
    parser.add_argument(
        "--opponent-type",
        type=str,
        default="heuristic",
        choices=["random", "heuristic", "ai"],
        help="Type of opponent bots (default: heuristic)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1). Set to 1 to disable parallelization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )

    args = parser.parse_args()

    # Define the exact hand from the user's scenario
    # Hand: 10♥, Q♥, K♥, Q♦, K♦, A♣
    # Hearts is trump
    trump_suit = Suit.HEARTS
    test_hand = [
        Card(Suit.HEARTS, Rank.TEN),  # 10♥ (low trump)
        Card(Suit.HEARTS, Rank.QUEEN),  # Q♥ (trump)
        Card(Suit.HEARTS, Rank.KING),  # K♥ (trump)
        Card(Suit.DIAMONDS, Rank.QUEEN),  # Q♦ (off-suit)
        Card(Suit.DIAMONDS, Rank.KING),  # K♦ (off-suit)
        Card(Suit.CLUBS, Rank.ACE),  # A♣ (off-suit, high)
    ]

    print("=" * 80)
    print("Monte Carlo Discard Strategy Comparison")
    print("=" * 80)
    print(f"\nTest Hand (6 cards, dealer after ordering up):")
    for i, card in enumerate(test_hand, 1):
        is_trump = (
            card.suit == trump_suit
            or (card.rank == Rank.JACK and card.is_same_color(Card(trump_suit, Rank.ACE)))
        )
        trump_marker = " [TRUMP]" if is_trump else ""
        print(f"  {i}. {card}{trump_marker}")
    print(f"\nTrump Suit: {trump_suit.value}")
    print(f"Number of simulations per strategy: {args.num_simulations:,}")
    print(f"Opponent type: {args.opponent_type}")
    print()

    # Define discard strategies to test
    discard_strategies = [
        ("Discard Low Trump (10♥)", Card(Suit.HEARTS, Rank.TEN)),
        ("Discard High Off-Suit (A♣)", Card(Suit.CLUBS, Rank.ACE)),
        ("Discard Off-Suit (Q♦)", Card(Suit.DIAMONDS, Rank.QUEEN)),
        ("Discard Off-Suit (K♦)", Card(Suit.DIAMONDS, Rank.KING)),
    ]

    # Run comparison
    results = run_monte_carlo_comparison(
        test_hand=test_hand,
        discard_strategies=discard_strategies,
        trump_suit=trump_suit,
        num_simulations=args.num_simulations,
        base_seed=args.seed,
        dealer_id=0,
        test_player_id=0,
        opponent_type=args.opponent_type,
        num_workers=args.num_workers,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Strategy':<30} {'Win Rate':<12} {'Avg Tricks':<12} {'Discard Card':<15}")
    print("-" * 80)

    # Sort by win rate
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["win_rate"], reverse=True
    )

    for strategy_name, stats in sorted_results:
        print(
            f"{strategy_name:<30} {stats['win_rate']:>10.2%}  {stats['avg_tricks_won']:>10.2f}  {stats['discard_card']:<15}"
        )

    # Determine best strategy
    best_strategy = sorted_results[0]
    print(f"\nBest Strategy: {best_strategy[0]}")
    print(f"  Win Rate: {best_strategy[1]['win_rate']:.2%}")
    print(f"  Avg Tricks Won: {best_strategy[1]['avg_tricks_won']:.2f}")

    # Check if discarding trump is optimal
    trump_strategy = results.get("Discard Low Trump (10♥)")
    if trump_strategy:
        best_non_trump = max(
            [
                (name, stats)
                for name, stats in results.items()
                if "Trump" not in name
            ],
            key=lambda x: x[1]["win_rate"],
        )
        if trump_strategy["win_rate"] > best_non_trump[1]["win_rate"]:
            print(
                f"\n✓ Discarding low trump IS optimal in this scenario!"
            )
            print(
                f"  Trump strategy beats best non-trump strategy by {trump_strategy['win_rate'] - best_non_trump[1]['win_rate']:.2%}"
            )
        else:
            print(
                f"\n✗ Discarding low trump is NOT optimal in this scenario."
            )
            print(
                f"  Best non-trump strategy ({best_non_trump[0]}) beats trump strategy by {best_non_trump[1]['win_rate'] - trump_strategy['win_rate']:.2%}"
            )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

