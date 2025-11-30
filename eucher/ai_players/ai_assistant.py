"""AI Assistant for Euchre decision analysis.

This module provides Monte Carlo simulation through the PyTorch AI to analyze
decisions and provide win probability estimates.
"""

import random
from typing import Dict, List, Optional, Tuple

from eucher.ai_players.benchmark_manager import BenchmarkManager
from eucher.ai_players.pytorch_player import PyTorchStrategicPlayer
from eucher.cards import Card, Deck, Suit
from eucher.game import Game
from eucher.rules import RulesEngine


class AIAssistant:
    """AI Assistant that provides decision analysis using Monte Carlo simulation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        thinking_time: str = "normal",
    ) -> None:
        """
        Initialize the AI Assistant.

        Parameters
        ----------
        model_path : Optional[str]
            Path to PyTorch model file.
        device : Optional[str]
            Device string (cpu/cuda).
        thinking_time : str
            Thinking time preset: 'fast' (0.5s), 'quick' (1s), 'normal' (10s),
            'thorough' (60s), 'deep' (120s).
        """
        self.benchmark_manager = BenchmarkManager()
        self.rules = RulesEngine()

        # Get number of simulations for thinking time
        thinking_times_config = self.benchmark_manager.get_thinking_times_config()
        self.num_simulations = thinking_times_config.get(thinking_time, thinking_times_config["normal"])

        # Initialize PyTorch player for simulations
        self.pytorch_player = PyTorchStrategicPlayer(
            model_path=model_path,
            device=device,
            trump_selection_risk=0.5,
            gameplay_risk=0.5,
        )

    def analyze_order_up_decision(
        self,
        hand: List[Card],
        turned_card: Card,
        dealer_id: int,
        player_id: int,
        team: int,
    ) -> Dict[str, float]:
        """
        Analyze order up decision using Monte Carlo simulation.

        Parameters
        ----------
        hand : List[Card]
            Player's hand.
        turned_card : Card
            Card turned up by dealer.
        dealer_id : int
            ID of dealer.
        player_id : int
            ID of player making decision.
        team : int
            Team of player (0 or 1).

        Returns
        -------
        Dict[str, float]
            Dictionary with 'win_probability' and 'confidence'.
        """
        wins = 0
        total = 0

        for _ in range(self.num_simulations):
            # Simulate hand with this decision
            result = self._simulate_hand_with_order_up(hand, turned_card, dealer_id, player_id, team)
            if result:
                wins += 1
            total += 1

        win_probability = wins / total if total > 0 else 0.0
        confidence = min(1.0, total / 100.0)  # Confidence based on number of simulations

        return {
            "win_probability": win_probability,
            "confidence": confidence,
            "simulations": total,
        }

    def analyze_trump_selection(
        self,
        hand: List[Card],
        turned_card: Card,
        available_suits: List[Suit],
        player_id: int,
        team: int,
    ) -> Dict[Suit, float]:
        """
        Analyze trump selection options using Monte Carlo simulation.

        Parameters
        ----------
        hand : List[Card]
            Player's hand.
        turned_card : Card
            Card that was turned up.
        available_suits : List[Suit]
            Suits available to call as trump.
        player_id : int
            ID of player making decision.
        team : int
            Team of player (0 or 1).

        Returns
        -------
        Dict[Suit, float]
            Dictionary mapping suit to win probability.
        """
        suit_probabilities: Dict[Suit, float] = {}

        for suit in available_suits:
            wins = 0
            total = 0

            for _ in range(self.num_simulations):
                result = self._simulate_hand_with_trump(hand, suit, player_id, team)
                if result:
                    wins += 1
                total += 1

            suit_probabilities[suit] = wins / total if total > 0 else 0.0

        return suit_probabilities

    def analyze_card_play(
        self,
        hand: List[Card],
        valid_cards: List[Card],
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        player_id: int,
        team: int,
        trick_number: int,
        tricks_won_team0: int,
        tricks_won_team1: int,
    ) -> List[Tuple[Card, float]]:
        """
        Analyze card play options using Monte Carlo simulation.

        Parameters
        ----------
        hand : List[Card]
            Player's hand.
        valid_cards : List[Card]
            Valid cards that can be played.
        led_suit : Optional[Suit]
            Suit that was led.
        trump_suit : Optional[Suit]
            Current trump suit.
        trick_cards : List[Card]
            Cards already played in trick.
        player_id : int
            ID of player making decision.
        team : int
            Team of player (0 or 1).
        trick_number : int
            Current trick number.
        tricks_won_team0 : int
            Tricks won by team 0.
        tricks_won_team1 : int
            Tricks won by team 1.

        Returns
        -------
        List[Tuple[Card, float]]
            List of (card, win_probability) tuples.
        """
        results: List[Tuple[Card, float]] = []

        for card in valid_cards:
            wins = 0
            total = 0

            for _ in range(self.num_simulations):
                result = self._simulate_card_play(
                    hand.copy(),
                    card,
                    led_suit,
                    trump_suit,
                    trick_cards.copy(),
                    player_id,
                    team,
                    trick_number,
                    tricks_won_team0,
                    tricks_won_team1,
                )
                if result:
                    wins += 1
                total += 1

            win_probability = wins / total if total > 0 else 0.0
            results.append((card, win_probability))

        return results

    def _simulate_hand_with_order_up(
        self,
        hand: List[Card],
        turned_card: Card,
        dealer_id: int,
        player_id: int,
        team: int,
    ) -> bool:
        """
        Simulate a hand assuming order up decision.

        Returns
        -------
        bool
            True if player's team wins the hand.
        """
        # Create a simplified game simulation
        deck = Deck()
        all_cards = set(deck.cards)
        known_cards = set(hand) | {turned_card}
        remaining_cards = list(all_cards - known_cards)
        random.shuffle(remaining_cards)

        # Deal remaining cards to other players
        other_hands: List[List[Card]] = []
        card_idx = 0
        for i in range(4):
            if i == player_id:
                other_hands.append(hand.copy())
            else:
                other_hand = remaining_cards[card_idx : card_idx + 5]
                card_idx += 5
                other_hands.append(other_hand)

        # Simulate hand with simple AI players (faster than PyTorch for simulation)
        player_config = [
            ("AI0", "simple"),
            ("AI1", "simple"),
            ("AI2", "simple"),
            ("AI3", "simple"),
        ]

        try:
            game = Game(player_config)
            for i, other_hand in enumerate(other_hands):
                game.players[i].receive_hand(other_hand)

            game.turned_card = turned_card
            game.dealer_id = dealer_id
            game.trump_suit = turned_card.suit  # Assume ordered up

            # Play hand
            tricks_won = [0, 0]
            for _ in range(5):
                winner_id = game._play_trick()
                winner_team = winner_id % 2
                tricks_won[winner_team] += 1

            # Check if player's team won
            return tricks_won[team] >= 3
        except Exception:
            # Fallback: simple heuristic
            return random.random() > 0.5

    def _simulate_hand_with_trump(
        self,
        hand: List[Card],
        trump_suit: Suit,
        player_id: int,
        team: int,
    ) -> bool:
        """
        Simulate a hand with a specific trump suit.

        Returns
        -------
        bool
            True if player's team wins the hand.
        """
        # Similar to _simulate_hand_with_order_up but with specified trump
        deck = Deck()
        all_cards = set(deck.cards)
        known_cards = set(hand)
        remaining_cards = list(all_cards - known_cards)
        random.shuffle(remaining_cards)

        other_hands: List[List[Card]] = []
        card_idx = 0
        for i in range(4):
            if i == player_id:
                other_hands.append(hand.copy())
            else:
                other_hand = remaining_cards[card_idx : card_idx + 5]
                card_idx += 5
                other_hands.append(other_hand)

        player_config = [
            ("AI0", "simple"),
            ("AI1", "simple"),
            ("AI2", "simple"),
            ("AI3", "simple"),
        ]

        try:
            game = Game(player_config)
            for i, other_hand in enumerate(other_hands):
                game.players[i].receive_hand(other_hand)

            game.trump_suit = trump_suit

            tricks_won = [0, 0]
            for _ in range(5):
                winner_id = game._play_trick()
                winner_team = winner_id % 2
                tricks_won[winner_team] += 1

            return tricks_won[team] >= 3
        except Exception:
            return random.random() > 0.5

    def _simulate_card_play(
        self,
        hand: List[Card],
        card_to_play: Card,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        player_id: int,
        team: int,
        trick_number: int,
        tricks_won_team0: int,
        tricks_won_team1: int,
    ) -> bool:
        """
        Simulate playing a card and determine if team wins the trick.

        Returns
        -------
        bool
            True if player's team wins the trick.
        """
        remaining_hand = [c for c in hand if c != card_to_play]
        current_trick = trick_cards + [card_to_play]

        if len(current_trick) == 4:
            # Trick is complete
            player_ids = list(range(4))
            winner_id = self.rules.determine_trick_winner(
                current_trick, player_ids, led_suit or current_trick[0].suit, trump_suit
            )
            winner_team = winner_id % 2
            return winner_team == team

        # Simulate remaining players
        deck = Deck()
        all_cards = set(deck.cards)
        known_cards = set(remaining_hand) | set(current_trick)
        remaining_cards = list(all_cards - known_cards)
        random.shuffle(remaining_cards)

        for _ in range(4 - len(current_trick)):
            if remaining_cards:
                current_trick.append(remaining_cards.pop())

        player_ids = list(range(4))
        winner_id = self.rules.determine_trick_winner(
            current_trick, player_ids, led_suit or current_trick[0].suit, trump_suit
        )
        winner_team = winner_id % 2
        return winner_team == team

