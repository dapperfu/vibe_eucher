"""Monte Carlo simulation to compare first trick strategies in Euchre.

This script tests whether it's better to play a high off-trump card (like an Ace)
or a low trump card (like a 9 trump) when leading the first trick after the dealer.

Performance:
- Uses CPU multiprocessing for parallelization (default: CPU count - 1 workers)
- GPU acceleration is not practical for this use case due to:
  * Complex Python-based game logic with branching
  * Stateful game simulation requiring sequential decision-making
  * Overhead of GPU memory transfers would outweigh benefits
- For best performance, use --num-workers to match your CPU core count
"""

import argparse
import multiprocessing as mp
import random
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eucher.cards import Card, Deck, Rank, Suit
from eucher.game import Game
from eucher.players.profiles import AIPlayer, HeuristicPlayer, PlayerProfile
from eucher.rules import RulesEngine


class ForcedFirstTrickProfile(PlayerProfile):
    """Player profile that forces a specific card on the first trick."""

    def __init__(self, forced_card: Optional[Card], base_profile: PlayerProfile) -> None:
        """
        Initialize the forced first trick profile.

        Parameters
        ----------
        forced_card : Optional[Card]
            The card to play on the first trick. If None, uses base profile strategy.
        base_profile : PlayerProfile
            The base profile to use for all other decisions.
        """
        self.forced_card: Optional[Card] = forced_card
        self.base_profile: PlayerProfile = base_profile
        self.trick_number: int = 0
        self.rules = RulesEngine()

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """Delegate to base profile."""
        return self.base_profile.decide_order_up(player, turned_card, dealer_id, trump_suit)

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """Delegate to base profile."""
        return self.base_profile.decide_call_trump(player, turned_card, trump_suit, must_choose)

    def choose_card_to_discard(
        self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
    ) -> Card:
        """Delegate to base profile."""
        return self.base_profile.choose_card_to_discard(player, turned_card, ordered_up_by)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Play card, forcing the specified card on the first trick.

        Parameters
        ----------
        player : Player
            The player making the decision.
        led_suit : Optional[Suit]
            The suit that was led, if any.
        trump_suit : Optional[Suit]
            The current trump suit, if any.
        trick_cards : List[Card]
            Cards already played in the trick.
        trick_player_ids : List[int]
            Player IDs who played each card.

        Returns
        -------
        Card
            The card to play.
        """
        # Check if this is the first trick (no led suit, no cards played)
        is_first_trick = led_suit is None and len(trick_cards) == 0

        if is_first_trick and self.forced_card is not None and self.forced_card in player.hand:
            # Validate that the forced card is a legal play
            valid_cards = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)
            if self.forced_card in valid_cards:
                return self.forced_card

        # Use base profile for all other cases
        return self.base_profile.play_card(player, led_suit, trump_suit, trick_cards, trick_player_ids)


def is_card_trump(card: Card, trump_suit: Suit) -> bool:
    """
    Check if a card is a trump card.

    Parameters
    ----------
    card : Card
        The card to check.
    trump_suit : Suit
        The current trump suit.

    Returns
    -------
    bool
        True if card is trump, False otherwise.
    """
    # Right Bower (Jack of trump suit)
    if card.rank == Rank.JACK and card.suit == trump_suit:
        return True
    # Left Bower (Jack of same color as trump)
    if card.rank == Rank.JACK:
        trump_card = Card(trump_suit, Rank.ACE)  # Dummy card for color check
        if card.is_same_color(trump_card):
            return True
    # Regular trump suit card
    return card.suit == trump_suit


def find_high_off_trump_cards(hand: List[Card], trump_suit: Optional[Suit]) -> List[Card]:
    """
    Find high off-trump cards (Ace or King) that are not trump.

    Parameters
    ----------
    hand : List[Card]
        The player's hand.
    trump_suit : Optional[Suit]
        The current trump suit.

    Returns
    -------
    List[Card]
        List of high off-trump cards, sorted by rank (Ace first, then King).
    """
    if trump_suit is None:
        # No trump yet, return all high cards
        high_cards = [card for card in hand if card.rank in (Rank.ACE, Rank.KING)]
        return sorted(high_cards, key=lambda c: c.rank.value, reverse=True)

    high_off_trump = []
    for card in hand:
        # Check if it's a high card (Ace or King)
        if card.rank in (Rank.ACE, Rank.KING):
            # Check if it's NOT trump
            if not is_card_trump(card, trump_suit):
                high_off_trump.append(card)

    # Sort by rank (Ace first, then King)
    return sorted(high_off_trump, key=lambda c: c.rank.value, reverse=True)


def find_low_trump_cards(hand: List[Card], trump_suit: Optional[Suit]) -> List[Card]:
    """
    Find low trump cards (9 or 10) that are trump.

    Parameters
    ----------
    hand : List[Card]
        The player's hand.
    trump_suit : Optional[Suit]
        The current trump suit.

    Returns
    -------
    List[Card]
        List of low trump cards, sorted by rank (9 first, then 10).
    """
    if trump_suit is None:
        return []

    low_trump = []
    for card in hand:
        # Check if it's a low card (9 or 10)
        if card.rank in (Rank.NINE, Rank.TEN):
            # Check if it IS trump
            if is_card_trump(card, trump_suit):
                low_trump.append(card)

    # Sort by rank (9 first, then 10)
    return sorted(low_trump, key=lambda c: c.rank.value)


def generate_test_hand(seed: int) -> Tuple[List[Card], Card]:
    """
    Generate a reproducible test hand and turned card.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[List[Card], Card]
        Tuple of (test_player_hand, turned_card).
    """
    random.seed(seed)
    np.random.seed(seed)

    deck = Deck()
    deck.shuffle()

    # Deal 5 cards to test player
    test_hand = deck.deal(5)

    # Deal 5 cards to each of the other 3 players (15 cards)
    for _ in range(3):
        _ = deck.deal(5)

    # Turn up one card
    turned_card = deck.draw_one()

    return test_hand, turned_card


def simulate_hand(
    test_hand: List[Card],
    turned_card: Card,
    forced_card: Optional[Card],
    simulation_seed: int,
    dealer_id: int = 0,
    test_player_id: Optional[int] = None,
    show_hands: bool = False,
) -> Tuple[bool, int, int]:
    """
    Simulate a single hand with a forced first trick card.

    Parameters
    ----------
    test_hand : List[Card]
        The test player's hand.
    turned_card : Card
        The turned card.
    forced_card : Optional[Card]
        The card to force on the first trick. If None, uses normal strategy.
    simulation_seed : int
        Random seed for this simulation (for opponent hands).
    dealer_id : int
        ID of the dealer (default: 0).
    test_player_id : Optional[int]
        ID of the test player. If None, calculated as (dealer_id + 1) % 4.

    Returns
    -------
    Tuple[bool, int, int]
        Tuple of (test_player_team_won, tricks_won_team0, tricks_won_team1).
    """
    # Set seed for this simulation
    random.seed(simulation_seed)
    np.random.seed(simulation_seed)

    # Determine test player ID (first to play after dealer)
    if test_player_id is None:
        test_player_id = (dealer_id + 1) % 4

    # Create game with random opponents
    # We'll place test player at the correct position
    player_config = [
        ("Player0", "random"),
        ("Player1", "random"),
        ("Player2", "random"),
        ("Player3", "random"),
    ]
    # Set test player to use AI profile
    player_config[test_player_id] = ("TestPlayer", "ai")

    game = Game(player_config)
    game.dealer_id = dealer_id

    # Create full deck and remove test hand and turned card
    full_deck = Deck()
    all_cards = set(full_deck.cards)
    
    # Remove test hand cards and turned card from available cards
    used_cards = set(test_hand) | {turned_card}
    remaining_cards = [card for card in all_cards if card not in used_cards]
    
    # Shuffle remaining cards for opponent hands
    random.shuffle(remaining_cards)

    # Set test player's hand
    game.players[test_player_id].receive_hand(test_hand.copy())

    # Deal to opponents from remaining cards
    card_idx = 0
    for i in range(4):
        if i != test_player_id:
            opponent_hand = remaining_cards[card_idx : card_idx + 5]
            game.players[i].receive_hand(opponent_hand)
            card_idx += 5

    # Set turned card
    game.turned_card = turned_card

    # Display all players' hands if requested
    if show_hands:
        print("\n" + "=" * 80)
        print(f"Simulation #{simulation_seed} - All Players' Hands")
        print("=" * 80)
        for i, player in enumerate(game.players):
            player_type = "Test Player" if i == test_player_id else "Opponent"
            print(f"\nPlayer {i} ({player.name}) - {player_type}:")
            for j, card in enumerate(player.hand, 1):
                print(f"  {j}. {card}")
        print(f"\nTurned Card: {turned_card}")
        print("=" * 80 + "\n")

    # Create forced profile for test player
    base_profile = game.players[test_player_id].profile
    forced_profile = ForcedFirstTrickProfile(forced_card, base_profile)
    game.players[test_player_id].profile = forced_profile

    # Store initial scores
    initial_scores = game.get_scores()

    # Play the hand
    try:
        # Use game's play_hand method, but we need to track tricks won
        # We'll manually play through to track results
        from eucher.trump import TrumpSelector

        # Select trump
        trump_selector = TrumpSelector(game.players, None)
        trump_suit = trump_selector.select_trump(turned_card, dealer_id)

        if trump_suit is None:
            # All passed, redeal - return as loss
            return False, 0, 0

        game.trump_suit = trump_suit

        # Play 5 tricks and track winners
        tricks_won = [0, 0]  # Team 0 and Team 1

        # Reset trick winner for new hand
        if hasattr(game, "_last_trick_winner"):
            delattr(game, "_last_trick_winner")

        for trick_num in range(5):
            game._current_trick_number = trick_num
            winner_id = game._play_trick()
            winner = game.players[winner_id]
            tricks_won[winner.team] += 1

        # Test player's team
        test_player_team = test_player_id % 2
        test_player_team_won = tricks_won[test_player_team] >= 3

        return test_player_team_won, tricks_won[0], tricks_won[1]

    except Exception as e:
        # If simulation fails, return as loss
        print(f"Simulation error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False, 0, 0


def _simulate_hand_wrapper(args: Tuple) -> Tuple[bool, int, int]:
    """
    Wrapper function for multiprocessing.

    Parameters
    ----------
    args : Tuple
        Tuple of (test_hand, turned_card, forced_card, simulation_seed, dealer_id, test_player_id, show_hands).

    Returns
    -------
    Tuple[bool, int, int]
        Tuple of (test_player_team_won, tricks_won_team0, tricks_won_team1).
    """
    test_hand, turned_card, forced_card, simulation_seed, dealer_id, test_player_id, show_hands = args
    return simulate_hand(test_hand, turned_card, forced_card, simulation_seed, dealer_id, test_player_id, show_hands)


def run_monte_carlo_simulation(
    test_hand: List[Card],
    turned_card: Card,
    strategy_card: Optional[Card],
    num_simulations: int,
    base_seed: int,
    dealer_id: int = 0,
    test_player_id: Optional[int] = None,
    num_workers: Optional[int] = None,
    show_hands: bool = False,
) -> Tuple[float, float, List[int], List[int]]:
    """
    Run Monte Carlo simulation for a strategy.

    Parameters
    ----------
    test_hand : List[Card]
        The test player's hand.
    strategy_card : Optional[Card]
        The card to play on first trick for this strategy.
    num_simulations : int
        Number of simulations to run.
    base_seed : int
        Base seed for generating simulation seeds.
    dealer_id : int
        ID of the dealer.
    test_player_id : Optional[int]
        ID of the test player.
    num_workers : Optional[int]
        Number of parallel workers. If None, uses CPU count - 1.

    Returns
    -------
    Tuple[float, float, List[int], List[int]]
        Tuple of (win_rate, avg_tricks_won, tricks_won_list, opponent_tricks_list).
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    # Prepare arguments for each simulation
    # Only show hands for the first simulation if requested
    simulation_args = [
        (test_hand, turned_card, strategy_card, base_seed + sim_num, dealer_id, test_player_id, show_hands and sim_num == 0)
        for sim_num in range(num_simulations)
    ]

    wins = 0
    tricks_won_list: List[int] = []
    opponent_tricks_list: List[int] = []

    # Run simulations in parallel
    if num_workers > 1 and num_simulations > 10:
        # Use multiprocessing for larger simulations
        with mp.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_simulate_hand_wrapper, simulation_args),
                    total=num_simulations,
                    desc="Simulations",
                    unit="sim",
                )
            )
    else:
        # Sequential for small simulations or single worker
        results = [
            simulate_hand(test_hand, turned_card, strategy_card, base_seed + sim_num, dealer_id, test_player_id, show_hands and sim_num == 0)
            for sim_num in tqdm(range(num_simulations), desc="Simulations", unit="sim")
        ]

    # Process results
    for team_won, tricks_team0, tricks_team1 in results:
        if team_won:
            wins += 1
        tricks_won_list.append(tricks_team0)
        opponent_tricks_list.append(tricks_team1)

    win_rate = wins / num_simulations
    avg_tricks_won = statistics.mean(tricks_won_list) if tricks_won_list else 0.0

    return win_rate, avg_tricks_won, tricks_won_list, opponent_tricks_list


def main() -> None:
    """Main function to run Monte Carlo simulation."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation to compare first trick strategies in Euchre"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generating test hand (default: 42)",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=10000,
        help="Number of simulations to run per strategy (default: 10000)",
    )
    parser.add_argument(
        "--dealer-id",
        type=int,
        default=0,
        help="ID of the dealer (default: 0). Test player is first to play after dealer.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1). Set to 1 to disable parallelization.",
    )
    parser.add_argument(
        "--no-show-hands",
        action="store_true",
        help="Don't display all players' cards for the first simulation (default: show hands).",
    )

    args = parser.parse_args()

    # Generate test hand
    print("=" * 80)
    print("Monte Carlo First Trick Strategy Comparison")
    print("=" * 80)
    print(f"\nUsing seed: {args.seed}")
    print(f"Number of simulations per strategy: {args.num_simulations:,}")
    test_player_id = (args.dealer_id + 1) % 4
    print(f"Dealer ID: {args.dealer_id}")
    print(f"Test player ID: {test_player_id} (first to play after dealer)")
    print()

    test_hand, turned_card = generate_test_hand(args.seed)

    print("Test Hand:")
    for i, card in enumerate(test_hand, 1):
        print(f"  {i}. {card}")
    print(f"\nTurned Card: {turned_card}")
    print()

    # We need to determine trump first to identify strategy cards
    # For now, we'll simulate trump selection by creating a temporary game
    temp_game = Game([("P0", "random"), ("P1", "random"), ("P2", "random"), ("P3", "random")])
    temp_game.dealer_id = args.dealer_id
    temp_game.players[0].receive_hand(test_hand.copy())

    # Set turned card
    temp_game.turned_card = turned_card

    # Select trump
    from eucher.trump import TrumpSelector

    trump_selector = TrumpSelector(temp_game.players, None)
    trump_suit = trump_selector.select_trump(turned_card, args.dealer_id)

    if trump_suit is None:
        print("ERROR: All players passed on trump. Cannot run simulation.")
        print("Try a different seed.")
        sys.exit(1)

    print(f"Trump Suit: {trump_suit.value}")
    print()

    # Find strategy cards
    high_off_trump_cards = find_high_off_trump_cards(test_hand, trump_suit)
    low_trump_cards = find_low_trump_cards(test_hand, trump_suit)

    print("Strategy Cards Found:")
    if high_off_trump_cards:
        print(f"  High off-trump: {[str(c) for c in high_off_trump_cards]}")
    else:
        print("  High off-trump: None found")

    if low_trump_cards:
        print(f"  Low trump: {[str(c) for c in low_trump_cards]}")
    else:
        print("  Low trump: None found")
    print()

    # Select cards for each strategy
    strategy1_card = high_off_trump_cards[0] if high_off_trump_cards else None
    strategy2_card = low_trump_cards[0] if low_trump_cards else None

    if strategy1_card is None and strategy2_card is None:
        print("ERROR: No strategy cards found in hand. Cannot run simulation.")
        print("Try a different seed.")
        sys.exit(1)

    # Run simulations for each strategy
    base_seed = args.seed * 1000000  # Large multiplier to avoid overlap

    results: List[Tuple[str, Optional[Card], float, float, List[int], List[int]]] = []

    # Determine number of workers
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Using {num_workers} parallel worker(s) for simulations")
    print()

    show_hands = not args.no_show_hands

    if strategy1_card:
        print(f"Running simulations for Strategy 1: High off-trump ({strategy1_card})...")
        win_rate, avg_tricks, tricks_list, opp_tricks_list = run_monte_carlo_simulation(
            test_hand, turned_card, strategy1_card, args.num_simulations, base_seed, args.dealer_id, test_player_id, num_workers, show_hands
        )
        results.append(("High off-trump", strategy1_card, win_rate, avg_tricks, tricks_list, opp_tricks_list))
        print(f"  Win rate: {win_rate:.2%}")
        print(f"  Avg tricks won: {avg_tricks:.2f}")
        print()

    if strategy2_card:
        print(f"Running simulations for Strategy 2: Low trump ({strategy2_card})...")
        win_rate, avg_tricks, tricks_list, opp_tricks_list = run_monte_carlo_simulation(
            test_hand, turned_card, strategy2_card, args.num_simulations, base_seed + 1, args.dealer_id, test_player_id, num_workers, show_hands
        )
        results.append(("Low trump", strategy2_card, win_rate, avg_tricks, tricks_list, opp_tricks_list))
        print(f"  Win rate: {win_rate:.2%}")
        print(f"  Avg tricks won: {avg_tricks:.2f}")
        print()

    # Print comparison
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    if len(results) == 2:
        strategy1_name, strategy1_card, win_rate1, avg_tricks1, tricks_list1, _ = results[0]
        strategy2_name, strategy2_card, win_rate2, avg_tricks2, tricks_list2, _ = results[1]

        print(f"Strategy 1: {strategy1_name} ({strategy1_card})")
        print(f"  Win Rate: {win_rate1:.2%}")
        print(f"  Avg Tricks Won: {avg_tricks1:.2f}")
        if tricks_list1:
            print(f"  Std Dev: {statistics.stdev(tricks_list1):.2f}")
        print()

        print(f"Strategy 2: {strategy2_name} ({strategy2_card})")
        print(f"  Win Rate: {win_rate2:.2%}")
        print(f"  Avg Tricks Won: {avg_tricks2:.2f}")
        if tricks_list2:
            print(f"  Std Dev: {statistics.stdev(tricks_list2):.2f}")
        print()

        win_rate_diff = win_rate1 - win_rate2
        tricks_diff = avg_tricks1 - avg_tricks2

        print("Comparison:")
        if win_rate_diff > 0:
            print(f"  Strategy 1 wins {abs(win_rate_diff):.2%} more often")
        elif win_rate_diff < 0:
            print(f"  Strategy 2 wins {abs(win_rate_diff):.2%} more often")
        else:
            print("  Strategies are tied")

        if tricks_diff > 0:
            print(f"  Strategy 1 wins {abs(tricks_diff):.2f} more tricks on average")
        elif tricks_diff < 0:
            print(f"  Strategy 2 wins {abs(tricks_diff):.2f} more tricks on average")
        else:
            print("  Strategies win the same number of tricks on average")

        # Statistical significance test (simple t-test approximation)
        if len(tricks_list1) > 1 and len(tricks_list2) > 1:
            try:
                from scipy import stats

                t_stat, p_value = stats.ttest_ind(tricks_list1, tricks_list2)
                print()
                print(f"Statistical Test (t-test):")
                print(f"  t-statistic: {t_stat:.4f}")
                print(f"  p-value: {p_value:.4f}")
                if p_value < 0.05:
                    print("  Result: Statistically significant difference (p < 0.05)")
                else:
                    print("  Result: No statistically significant difference (p >= 0.05)")
            except ImportError:
                pass  # scipy not available

    elif len(results) == 1:
        strategy_name, strategy_card, win_rate, avg_tricks, tricks_list, _ = results[0]
        print(f"Strategy: {strategy_name} ({strategy_card})")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg Tricks Won: {avg_tricks:.2f}")
        if tricks_list:
            print(f"  Std Dev: {statistics.stdev(tricks_list):.2f}")

    print()
    print("=" * 80)
    print(f"Test hand seed: {args.seed}")
    print("To reproduce, run with: --seed", args.seed)
    print("=" * 80)


if __name__ == "__main__":
    main()

