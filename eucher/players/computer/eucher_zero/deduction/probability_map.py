"""Probability map for hidden card inference."""

from typing import Dict, List

import numpy as np

from eucher.cards import Card
from eucher.players.computer.eucher_zero.deduction.card_tracker import CardTracker


class ProbabilityMap:
    """Maintain probability distributions for hidden cards."""

    NUM_CARDS = 24
    NUM_PLAYERS = 4

    def __init__(self, card_tracker: CardTracker, player_id: int) -> None:
        """
        Initialize probability map.

        Parameters
        ----------
        card_tracker : CardTracker
            Card tracker instance.
        player_id : int
            ID of the player (0-3).
        """
        self.card_tracker = card_tracker
        self.player_id = player_id

        # Probability map: card -> [prob_player_0, prob_player_1, prob_player_2, prob_player_3]
        # For player_id, we know their hand, so probability is 1.0 for them, 0.0 for others
        # For unknown cards, we distribute uniformly among other players
        self.prob_map: Dict[Card, np.ndarray] = {}

    def initialize(self, all_cards: List[Card], player_hand: List[Card]) -> None:
        """
        Initialize probability map.

        Parameters
        ----------
        all_cards : List[Card]
            All 24 cards.
        player_hand : List[Card]
            Cards in player's hand.
        """
        # Get unknown cards
        unknown_cards = self.card_tracker.get_unknown_cards()

        # Count how many cards each other player should have
        # Each player has 5 cards, we have player_hand, so others have 5 each
        cards_per_player = 5
        num_other_players = 3

        for card in all_cards:
            if card in player_hand:
                # We have this card
                probs = np.zeros(4)
                probs[self.player_id] = 1.0
                self.prob_map[card] = probs
            elif card in unknown_cards:
                # Unknown card: uniform distribution among other players
                probs = np.zeros(4)
                prob_per_player = 1.0 / num_other_players
                for i in range(4):
                    if i != self.player_id:
                        probs[i] = prob_per_player
                self.prob_map[card] = probs
            else:
                # Card is visible (played/discarded)
                probs = np.zeros(4)
                self.prob_map[card] = probs

    def update_from_play(self, card: Card, player_id: int) -> None:
        """
        Update probabilities when a card is played.

        Parameters
        ----------
        card : Card
            Card that was played.
        player_id : int
            ID of player who played it.
        """
        # Mark this card as definitely with this player
        if card in self.prob_map:
            probs = np.zeros(4)
            probs[player_id] = 1.0
            self.prob_map[card] = probs

        # Re-normalize other unknown cards
        self._renormalize()

    def _renormalize(self) -> None:
        """Renormalize probabilities for unknown cards."""
        # Simplified: maintain uniform distribution
        # In full implementation, would use Bayesian inference
        pass

    def get_probability_map(self) -> np.ndarray:
        """
        Get probability map as array (24 cards × 3 players).

        Parameters
        ----------
        np.ndarray
            Probability map [24, 3] excluding current player.
        """
        # Get all cards in order
        from eucher.cards import Deck

        deck = Deck()
        all_cards = sorted(deck.cards.copy(), key=lambda c: (c.suit.value, c.rank.value))

        prob_array = np.zeros((self.NUM_CARDS, 3))
        player_indices = [i for i in range(4) if i != self.player_id]

        for i, card in enumerate(all_cards):
            if card in self.prob_map:
                probs = self.prob_map[card]
                for j, player_idx in enumerate(player_indices):
                    prob_array[i, j] = probs[player_idx]

        return prob_array

    def sample_hidden_state(self) -> Dict[int, List[Card]]:
        """
        Sample hidden card assignments.

        Parameters
        ----------
        Dict[int, List[Card]]
            Map of player_id -> list of cards.
        """
        # Simplified: uniform sampling
        # In full implementation, would sample according to probabilities
        from eucher.cards import Deck

        deck = Deck()
        all_cards = deck.cards.copy()
        unknown_cards = self.card_tracker.get_unknown_cards()

        # Distribute unknown cards randomly
        np.random.shuffle(unknown_cards)
        assignments: Dict[int, List[Card]] = {i: [] for i in range(4)}

        # We know our hand
        from eucher.players.computer.eucher_zero.deduction.card_tracker import CardLocation

        for card in self.card_tracker.location_cards[CardLocation.IN_HAND]:
            assignments[self.player_id].append(card)

        # Distribute unknown cards
        cards_per_player = 5
        for i, card in enumerate(unknown_cards):
            player_idx = (i % 3)  # Distribute among other 3 players
            if player_idx >= self.player_id:
                player_idx += 1
            assignments[player_idx].append(card)

        return assignments

