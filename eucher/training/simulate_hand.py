"""Full hand simulation for training data generation."""

from typing import List, Optional

from src.cards import Card, Deck, Suit
from src.game import Game
from src.ml_training.data_collector import GameDataCollector


class HandSimulator:
    """Simulates full hands to generate training data."""

    def __init__(self) -> None:
        """Initialize the hand simulator."""
        self.collector = GameDataCollector()

    def simulate_hand_with_all_decisions(self) -> None:
        """
        Simulate a hand, trying all valid decisions at each decision point.

        This generates training data by exploring the decision space.
        """
        # Deal a hand
        deck = Deck()
        deck.shuffle()

        # Deal to all players
        hands = []
        for _ in range(4):
            hands.append(deck.deal(5))

        turned_card = deck.draw_one()

        # Create game with random opponents
        player_config = [
            ("Player0", "random"),
            ("Player1", "random"),
            ("Player2", "random"),
            ("Player3", "random"),
        ]

        game = Game(player_config)

        # Set hands
        for i, hand in enumerate(hands):
            game.players[i].receive_hand(hand)

        game.turned_card = turned_card
        game.dealer_id = 0

        # Try all order_up decisions
        self._try_all_order_up_decisions(game, hands, turned_card)

        # Try all call_trump decisions (if no one ordered up)
        self._try_all_call_trump_decisions(game, hands, turned_card)

        # For each trick, try all card plays
        # This would be done during actual gameplay

    def _try_all_order_up_decisions(self, game: Game, hands: List[List[Card]], turned_card: Card) -> None:
        """
        Try all possible order_up decisions.

        Parameters
        ----------
        game : Game
            The game instance.
        hands : List[List[Card]]
            Hands for all players.
        turned_card : Card
            The turned card.
        """
        dealer_id = game.dealer_id

        # For each non-dealer player, try both decisions
        for player_id in range(4):
            if player_id == dealer_id:
                continue

            for decision in [True, False]:
                # Reset game state
                for i, hand in enumerate(hands):
                    game.players[i].receive_hand(hand.copy())
                game.turned_card = turned_card
                game.dealer_id = dealer_id

                # Override decision
                original_profile = game.players[player_id].profile

                class ForcedOrderUpProfile:
                    """Profile that forces order_up decision."""

                    def decide_order_up(self, player, turned_card, dealer_id, trump_suit):
                        return decision

                    def decide_call_trump(self, player, turned_card, trump_suit, must_choose):
                        return None

                    def choose_card_to_discard(self, player):
                        return player.hand[0]

                    def play_card(self, player, led_suit, trump_suit, trick_cards, trick_player_ids):
                        from src.rules import RulesEngine
                        rules = RulesEngine()
                        valid = rules.get_valid_plays(player.hand, led_suit, trump_suit)
                        import random
                        return random.choice(valid) if valid else player.hand[0]

                game.players[player_id].profile = ForcedOrderUpProfile()

                # Record decision
                self.collector.record_order_up_decision(
                    hand=hands[player_id].copy(),
                    turned_card=turned_card,
                    player_id=player_id,
                    dealer_id=dealer_id,
                    team=player_id % 2,
                    decision=decision,
                )

                # Play hand
                try:
                    game.play_hand()
                    scores = game.get_scores()
                    self.collector.finish_hand([0, 0], list(scores))
                except Exception:
                    pass

                # Restore profile
                game.players[player_id].profile = original_profile

    def _try_all_call_trump_decisions(self, game: Game, hands: List[List[Card]], turned_card: Card) -> None:
        """
        Try all possible call_trump decisions.

        Parameters
        ----------
        game : Game
            The game instance.
        hands : List[List[Card]]
            Hands for all players.
        turned_card : Card
            The turned card.
        """
        dealer_id = game.dealer_id
        forbidden_suit = turned_card.suit
        available_suits = [suit for suit in Suit if suit != forbidden_suit]

        # For each player, try all suits + pass (except dealer must choose)
        for player_id in range(4):
            decisions = available_suits if player_id == dealer_id else [None] + available_suits

            for decision in decisions:
                # Reset game state
                for i, hand in enumerate(hands):
                    game.players[i].receive_hand(hand.copy())
                game.turned_card = turned_card
                game.dealer_id = dealer_id

                # Override decision
                original_profile = game.players[player_id].profile

                class ForcedCallTrumpProfile:
                    """Profile that forces call_trump decision."""

                    def decide_order_up(self, player, turned_card, dealer_id, trump_suit):
                        return False

                    def decide_call_trump(self, player, turned_card, trump_suit, must_choose):
                        return decision

                    def choose_card_to_discard(self, player):
                        return player.hand[0]

                    def play_card(self, player, led_suit, trump_suit, trick_cards, trick_player_ids):
                        from src.rules import RulesEngine
                        rules = RulesEngine()
                        valid = rules.get_valid_plays(player.hand, led_suit, trump_suit)
                        import random
                        return random.choice(valid) if valid else player.hand[0]

                game.players[player_id].profile = ForcedCallTrumpProfile()

                # Record decision
                self.collector.record_call_trump_decision(
                    hand=hands[player_id].copy(),
                    turned_card=turned_card,
                    player_id=player_id,
                    dealer_id=dealer_id,
                    team=player_id % 2,
                    decision=decision,
                )

                # Play hand
                try:
                    game.play_hand()
                    scores = game.get_scores()
                    self.collector.finish_hand([0, 0], list(scores))
                except Exception:
                    pass

                # Restore profile
                game.players[player_id].profile = original_profile

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
    """Run hand simulation."""
    simulator = HandSimulator()
    simulator.simulate_hand_with_all_decisions()
    simulator.save_data("training_data/simulated_hands.json")


if __name__ == "__main__":
    main()

