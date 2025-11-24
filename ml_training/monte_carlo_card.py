"""Monte Carlo simulation for card selection training."""

import random
from typing import List, Optional

from src.cards import Card, Suit
from src.game import Game
from src.ml_training.data_collector import GameDataCollector, TrainingExample
from src.rules import RulesEngine


class MonteCarloCardSelector:
    """Uses Monte Carlo simulation to evaluate card selections."""

    def __init__(self, num_simulations: int = 1000) -> None:
        """
        Initialize the Monte Carlo selector.

        Parameters
        ----------
        num_simulations : int
            Number of Monte Carlo simulations to run per card.
        """
        self.num_simulations = num_simulations
        self.collector = GameDataCollector()
        self.rules = RulesEngine()

    def evaluate_card_selection(
        self,
        hand: List[Card],
        valid_cards: List[Card],
        trump_suit: Optional[Suit],
        led_suit: Optional[Suit],
        trick_cards: List[Card],
        player_id: int,
        dealer_id: int,
        team: int,
        trick_number: int,
        tricks_won_team0: int,
        tricks_won_team1: int,
    ) -> List[tuple[Card, float]]:
        """
        Evaluate each valid card using Monte Carlo simulation.

        Parameters
        ----------
        hand : List[Card]
            Player's current hand.
        valid_cards : List[Card]
            Valid cards that can be played.
        trump_suit : Optional[Suit]
            Current trump suit.
        led_suit : Optional[Suit]
            Suit that was led.
        trick_cards : List[Card]
            Cards already played in trick.
        player_id : int
            ID of the player.
        dealer_id : int
            ID of the dealer.
        team : int
            Team of the player.
        trick_number : int
            Current trick number.
        tricks_won_team0 : int
            Tricks won by team 0 so far.
        tricks_won_team1 : int
            Tricks won by team 1 so far.

        Returns
        -------
        List[tuple[Card, float]]
            List of (card, win_probability) tuples.
        """
        results = []

        for card in valid_cards:
            win_count = 0

            for _ in range(self.num_simulations):
                # Simulate playing this card with random opponents
                if self._simulate_card_play(
                    hand.copy(),
                    card,
                    trump_suit,
                    led_suit,
                    trick_cards.copy(),
                    player_id,
                    dealer_id,
                    team,
                    trick_number,
                    tricks_won_team0,
                    tricks_won_team1,
                ):
                    win_count += 1

            win_probability = win_count / self.num_simulations
            results.append((card, win_probability))

            # Record training example
            self.collector.record_card_play(
                hand=hand,
                trump_suit=trump_suit,
                led_suit=led_suit,
                trick_cards=trick_cards,
                player_id=player_id,
                dealer_id=dealer_id,
                team=team,
                trick_number=trick_number,
                tricks_won_team0=tricks_won_team0,
                tricks_won_team1=tricks_won_team1,
                card_played=card,
            )

        return results

    def _simulate_card_play(
        self,
        hand: List[Card],
        card_to_play: Card,
        trump_suit: Optional[Suit],
        led_suit: Optional[Suit],
        trick_cards: List[Card],
        player_id: int,
        dealer_id: int,
        team: int,
        trick_number: int,
        tricks_won_team0: int,
        tricks_won_team1: int,
    ) -> bool:
        """
        Simulate playing a card and determine if team wins the trick.

        Parameters
        ----------
        hand : List[Card]
            Player's hand.
        card_to_play : Card
            Card to play.
        trump_suit : Optional[Suit]
            Current trump suit.
        led_suit : Optional[Suit]
            Suit that was led.
        trick_cards : List[Card]
            Cards already played.
        player_id : int
            ID of the player.
        dealer_id : int
            ID of the dealer.
        team : int
            Team of the player.
        trick_number : int
            Current trick number.
        tricks_won_team0 : int
            Tricks won by team 0.
        tricks_won_team1 : int
            Tricks won by team 1.

        Returns
        -------
        bool
            True if player's team wins the trick, False otherwise.
        """
        # Remove card from hand
        remaining_hand = [c for c in hand if c != card_to_play]
        current_trick = trick_cards + [card_to_play]

        # Determine how many more cards need to be played
        cards_remaining = 4 - len(current_trick)

        if cards_remaining == 0:
            # Trick is complete, determine winner
            player_ids = list(range(4))
            winner_id = self.rules.determine_trick_winner(current_trick, player_ids, led_suit or current_trick[0].suit, trump_suit)
            winner_team = winner_id % 2
            return winner_team == team

        # Simulate remaining players with random cards
        # This is simplified - in reality we'd need to track what cards are left
        # For training purposes, we'll use a simplified simulation
        remaining_cards_pool = self._generate_remaining_cards(remaining_hand, current_trick)

        for i in range(cards_remaining):
            # Random opponent plays a random card
            if remaining_cards_pool:
                played_card = random.choice(remaining_cards_pool)
                remaining_cards_pool.remove(played_card)
                current_trick.append(played_card)

        # Determine winner
        player_ids = list(range(4))
        winner_id = self.rules.determine_trick_winner(current_trick, player_ids, led_suit or current_trick[0].suit, trump_suit)
        winner_team = winner_id % 2
        return winner_team == team

    def _generate_remaining_cards(self, hand: List[Card], trick_cards: List[Card]) -> List[Card]:
        """
        Generate a pool of remaining cards for simulation.

        Parameters
        ----------
        hand : List[Card]
            Player's remaining hand.
        trick_cards : List[Card]
            Cards already played.

        Returns
        -------
        List[Card]
            Pool of remaining cards.
        """
        # Simplified: generate random valid cards
        # In a full implementation, we'd track all played cards
        from src.cards import Deck

        deck = Deck()
        all_played = set(hand) | set(trick_cards)
        remaining = [card for card in deck.cards if card not in all_played]
        return remaining[:20]  # Limit for simulation

    def save_data(self, filepath: str) -> None:
        """
        Save collected training data.

        Parameters
        ----------
        filepath : str
            Path to save file.
        """
        self.collector.save(filepath)


def main() -> None:
    """Run Monte Carlo card selection."""
    selector = MonteCarloCardSelector(num_simulations=100)

    # Example usage
    from src.cards import Deck

    deck = Deck()
    deck.shuffle()
    hand = deck.deal(5)
    valid_cards = hand  # All cards valid when leading

    results = selector.evaluate_card_selection(
        hand=hand,
        valid_cards=valid_cards,
        trump_suit=None,
        led_suit=None,
        trick_cards=[],
        player_id=0,
        dealer_id=3,
        team=0,
        trick_number=0,
        tricks_won_team0=0,
        tricks_won_team1=0,
    )

    for card, prob in results:
        print(f"{card}: {prob:.2%}")

    selector.save_data("training_data/monte_carlo_cards.json")


if __name__ == "__main__":
    main()

