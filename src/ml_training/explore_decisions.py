"""Decision tree exploration for training data generation."""

import random
from typing import List, Optional

from src.cards import Card, Deck, Suit
from src.game import Game
from src.ml_training.data_collector import GameDataCollector


class DecisionExplorer:
    """Explores all decision paths for a given hand."""

    def __init__(self, num_simulations: int = 1000) -> None:
        """
        Initialize the decision explorer.

        Parameters
        ----------
        num_simulations : int
            Number of simulations to run for each decision path.
        """
        self.num_simulations = num_simulations
        self.collector = GameDataCollector()

    def explore_hand(self, hand: List[Card], turned_card: Card, dealer_id: int) -> None:
        """
        Explore all decision paths for a given hand.

        Parameters
        ----------
        hand : List[Card]
            The hand to explore (5 cards for one player).
        turned_card : Card
            The turned card.
        dealer_id : int
            ID of the dealer.
        """
        # Round 1: Explore order_up decisions
        self._explore_order_up(hand, turned_card, dealer_id)

        # Round 2: Explore call_trump decisions (if no one ordered up)
        self._explore_call_trump(hand, turned_card, dealer_id)

    def _explore_order_up(self, hand: List[Card], turned_card: Card, dealer_id: int) -> None:
        """
        Explore all order_up decision paths.

        Parameters
        ----------
        hand : List[Card]
            The hand to explore.
        turned_card : Card
            The turned card.
        dealer_id : int
            ID of the dealer.
        """
        # For each player (except dealer), try both order_up and pass
        for player_id in range(4):
            if player_id == dealer_id:
                continue

            for decision in [True, False]:
                # Simulate this decision path
                for _ in range(self.num_simulations):
                    self._simulate_order_up_path(hand, turned_card, dealer_id, player_id, decision)

    def _explore_call_trump(self, hand: List[Card], turned_card: Card, dealer_id: int) -> None:
        """
        Explore all call_trump decision paths.

        Parameters
        ----------
        hand : List[Card]
            The hand to explore.
        turned_card : Card
            The turned card.
        dealer_id : int
            ID of the dealer.
        """
        forbidden_suit = turned_card.suit
        available_suits = [suit for suit in Suit if suit != forbidden_suit]

        # For each player, try all suits + pass
        for player_id in range(4):
            if player_id == dealer_id:
                # Dealer must choose (screw the dealer rule)
                for suit in available_suits:
                    for _ in range(self.num_simulations):
                        self._simulate_call_trump_path(hand, turned_card, dealer_id, player_id, suit)
            else:
                # Other players can pass or choose a suit
                for decision in [None] + available_suits:
                    for _ in range(self.num_simulations):
                        self._simulate_call_trump_path(hand, turned_card, dealer_id, player_id, decision)

    def _simulate_order_up_path(
        self, hand: List[Card], turned_card: Card, dealer_id: int, player_id: int, decision: bool
    ) -> None:
        """
        Simulate a single order_up decision path.

        Parameters
        ----------
        hand : List[Card]
            The hand to test.
        turned_card : Card
            The turned card.
        dealer_id : int
            ID of the dealer.
        player_id : int
            ID of the player making the decision.
        decision : bool
            True to order up, False to pass.
        """
        # Create a game with random opponents
        player_config = [
            ("Player0", "random"),
            ("Player1", "random"),
            ("Player2", "random"),
            ("Player3", "random"),
        ]

        game = Game(player_config)

        # Set the specific hand for the player we're testing
        game.players[player_id].receive_hand(hand.copy())

        # Deal remaining cards randomly
        deck = Deck()
        deck.shuffle()

        # Remove cards already dealt
        all_cards = set()
        for card in hand:
            all_cards.add(card)
        all_cards.add(turned_card)

        remaining_cards = [card for card in deck.cards if card not in all_cards]
        random.shuffle(remaining_cards)

        # Deal to other players
        card_idx = 0
        for i in range(4):
            if i == player_id:
                continue
            player_hand = []
            for _ in range(5):
                if card_idx < len(remaining_cards):
                    player_hand.append(remaining_cards[card_idx])
                    card_idx += 1
            game.players[i].receive_hand(player_hand)

        # Set dealer and turned card
        game.dealer_id = dealer_id
        game.turned_card = turned_card

        # Override the decision for the test player
        original_profile = game.players[player_id].profile

        class ForcedDecisionProfile:
            """Profile that forces a specific decision."""

            def decide_order_up(self, player, turned_card, dealer_id, trump_suit):
                return decision

            def decide_call_trump(self, player, turned_card, trump_suit, must_choose):
                return None

            def choose_card_to_discard(self, player):
                return player.hand[0]

            def play_card(self, player, led_suit, trump_suit, trick_cards, trick_player_ids):
                return random.choice(player.hand)

        game.players[player_id].profile = ForcedDecisionProfile()

        # Record decision
        self.collector.record_order_up_decision(
            hand=hand,
            turned_card=turned_card,
            player_id=player_id,
            dealer_id=dealer_id,
            team=player_id % 2,
            decision=decision,
        )

        # Play the hand
        try:
            game.play_hand()
            tricks_won = [0, 0]
            # Extract tricks won from game state (simplified)
            scores = game.get_scores()
            self.collector.finish_hand(tricks_won, list(scores))
        except Exception:
            # If hand fails, skip
            pass

        # Restore original profile
        game.players[player_id].profile = original_profile

    def _simulate_call_trump_path(
        self, hand: List[Card], turned_card: Card, dealer_id: int, player_id: int, decision: Optional[Suit]
    ) -> None:
        """
        Simulate a single call_trump decision path.

        Parameters
        ----------
        hand : List[Card]
            The hand to test.
        turned_card : Card
            The turned card.
        dealer_id : int
            ID of the dealer.
        player_id : int
            ID of the player making the decision.
        decision : Optional[Suit]
            Suit to call, or None to pass.
        """
        # Similar to _simulate_order_up_path but for call_trump
        # Implementation would be similar but for round 2
        pass  # Placeholder - full implementation would follow same pattern

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
    """Run decision exploration."""
    explorer = DecisionExplorer(num_simulations=100)

    # Explore a sample hand
    deck = Deck()
    deck.shuffle()
    hand = deck.deal(5)
    turned_card = deck.draw_one()

    explorer.explore_hand(hand, turned_card, dealer_id=0)
    explorer.save_data("training_data/explored_decisions.json")


if __name__ == "__main__":
    main()

