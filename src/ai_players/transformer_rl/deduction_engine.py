"""Perfect memory and card deduction system for Euchre RL.

This module tracks all seen cards and calculates probability distributions
for each player's unknown cards based on perfect memory and play patterns.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from eucher.cards import Card, Rank, Suit


class DeductionEngine:
    """Tracks perfect memory and calculates card probability distributions.

    This engine maintains perfect memory of all cards seen during gameplay
    and calculates probability distributions for each player's unknown cards.

    Parameters
    ----------
    None

    Attributes
    ----------
    all_cards : Set[Card]
        Set of all 24 cards in Euchre deck.
    seen_cards : Set[Card]
        Set of all cards that have been seen (dealt or played).
    played_cards : Set[Card]
        Set of all cards that have been played.
    player_known_cards : Dict[int, Set[Card]]
        Cards known to be held by each player (from deals, discards, plays).
    player_cannot_have : Dict[int, Set[Card]]
        Cards that each player cannot have (seen in other players' hands).
    """

    def __init__(self) -> None:
        """Initialize deduction engine."""
        # Generate all 24 cards in Euchre deck
        self.all_cards: Set[Card] = set()
        for suit in Suit:
            for rank in Rank:
                self.all_cards.add(Card(suit, rank))

        # Tracking sets
        self.seen_cards: Set[Card] = set()
        self.played_cards: Set[Card] = set()
        self.player_known_cards: Dict[int, Set[Card]] = defaultdict(set)
        self.player_cannot_have: Dict[int, Set[Card]] = defaultdict(set)

        # Track cards dealt to each player
        self.dealt_cards: Dict[int, List[Card]] = defaultdict(list)

    def reset(self) -> None:
        """Reset deduction state for a new hand."""
        self.seen_cards.clear()
        self.played_cards.clear()
        self.player_known_cards.clear()
        self.player_cannot_have.clear()
        self.dealt_cards.clear()

    def record_deal(self, player_id: int, cards: List[Card]) -> None:
        """Record cards dealt to a player.

        Parameters
        ----------
        player_id : int
            ID of the player receiving cards.
        cards : List[Card]
            Cards dealt to the player.
        """
        self.dealt_cards[player_id] = cards.copy()
        for card in cards:
            self.seen_cards.add(card)
            self.player_known_cards[player_id].add(card)

    def record_play(self, player_id: int, card: Card) -> None:
        """Record a card being played.

        Parameters
        ----------
        player_id : int
            ID of the player who played the card.
        card : Card
            Card that was played.
        """
        self.played_cards.add(card)
        self.seen_cards.add(card)
        self.player_known_cards[player_id].discard(card)  # Remove from hand

        # All other players cannot have this card
        for pid in range(4):
            if pid != player_id:
                self.player_cannot_have[pid].add(card)

    def record_discard(self, player_id: int, card: Card) -> None:
        """Record a card being discarded.

        Parameters
        ----------
        player_id : int
            ID of the player who discarded.
        card : Card
            Card that was discarded.
        """
        self.played_cards.add(card)
        self.seen_cards.add(card)
        self.player_known_cards[player_id].discard(card)

        # All other players cannot have this card
        for pid in range(4):
            if pid != player_id:
                self.player_cannot_have[pid].add(card)

    def get_deduction_probabilities(
        self, player_id: int, current_hand_size: Optional[int] = None
    ) -> Dict[Card, float]:
        """Calculate probability distribution for a player's unknown cards.

        Parameters
        ----------
        player_id : int
            ID of the player.
        current_hand_size : Optional[int]
            Current hand size of the player. If None, assumes 5 cards.

        Returns
        -------
        Dict[Card, float]
            Dictionary mapping each card to its probability of being in player's hand.
        """
        if current_hand_size is None:
            current_hand_size = 5

        # Cards the player is known to have
        known_cards = self.player_known_cards[player_id]
        num_known = len(known_cards)

        # Cards the player cannot have
        cannot_have = self.player_cannot_have[player_id]

        # Remaining cards that could be in player's hand
        possible_cards: Set[Card] = self.all_cards - self.seen_cards - cannot_have

        # If we know all cards, return deterministic distribution
        if num_known >= current_hand_size:
            prob_dist: Dict[Card, float] = {}
            for card in known_cards:
                prob_dist[card] = 1.0
            for card in self.all_cards:
                if card not in prob_dist:
                    prob_dist[card] = 0.0
            return prob_dist

        # Calculate remaining slots
        remaining_slots = current_hand_size - num_known
        num_possible = len(possible_cards)

        if num_possible == 0:
            # No possible cards - return zeros
            prob_dist = {card: 0.0 for card in self.all_cards}
            for card in known_cards:
                prob_dist[card] = 1.0
            return prob_dist

        # Uniform probability for each possible card
        prob_per_card = remaining_slots / num_possible

        # Build probability distribution
        prob_dist: Dict[Card, float] = {}
        for card in self.all_cards:
            if card in known_cards:
                prob_dist[card] = 1.0
            elif card in cannot_have:
                prob_dist[card] = 0.0
            elif card in possible_cards:
                prob_dist[card] = prob_per_card
            else:
                prob_dist[card] = 0.0

        return prob_dist

    def get_deduction_map_tensor(
        self, current_hand_sizes: Optional[Dict[int, int]] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Get deduction probability maps as a tensor.

        Parameters
        ----------
        current_hand_sizes : Optional[Dict[int, int]]
            Current hand size for each player. If None, assumes 5 for all.
        device : Optional[torch.device]
            Device to create tensor on.

        Returns
        -------
        torch.Tensor
            Tensor of shape (24, 4) where [card_idx, player_id] is the probability
            that player has that card.
        """
        if current_hand_sizes is None:
            current_hand_sizes = {i: 5 for i in range(4)}

        # Create card index mapping
        card_to_idx: Dict[Card, int] = {}
        idx = 0
        for suit in Suit:
            for rank in Rank:
                card_to_idx[Card(suit, rank)] = idx
                idx += 1

        # Initialize tensor
        deduction_map = torch.zeros(24, 4, device=device)

        # Fill in probabilities for each player
        for player_id in range(4):
            hand_size = current_hand_sizes.get(player_id, 5)
            probs = self.get_deduction_probabilities(player_id, hand_size)
            for card, prob in probs.items():
                if card in card_to_idx:
                    card_idx = card_to_idx[card]
                    deduction_map[card_idx, player_id] = prob

        return deduction_map

    def get_perfect_memory_state(self) -> Dict[str, Set[Card]]:
        """Get current perfect memory state.

        Returns
        -------
        Dict[str, Set[Card]]
            Dictionary with keys:
            - 'seen': all seen cards
            - 'played': all played cards
            - 'unseen': all unseen cards
        """
        return {
            "seen": self.seen_cards.copy(),
            "played": self.played_cards.copy(),
            "unseen": self.all_cards - self.seen_cards,
        }

    def get_suit_void_probabilities(
        self, player_id: int, trump_suit: Optional[Suit] = None
    ) -> Dict[Suit, float]:
        """Calculate probability that a player is void in each suit.

        Parameters
        ----------
        player_id : int
            ID of the player.
        trump_suit : Optional[Suit]
            Current trump suit, if any.

        Returns
        -------
        Dict[Suit, float]
            Dictionary mapping each suit to probability player is void.
        """
        probs = self.get_deduction_probabilities(player_id)
        hand_size = len(self.player_known_cards[player_id])

        void_probs: Dict[Suit, float] = {}
        for suit in Suit:
            # Count cards of this suit that player could have
            suit_cards = [Card(suit, rank) for rank in Rank]
            prob_has_suit = sum(probs.get(card, 0.0) for card in suit_cards)

            # Probability of being void = 1 - probability of having at least one
            void_probs[suit] = max(0.0, 1.0 - prob_has_suit)

        return void_probs

    def get_trump_count_probability(
        self, player_id: int, trump_suit: Suit, count: int
    ) -> float:
        """Calculate probability that player has exactly 'count' trump cards.

        Parameters
        ----------
        player_id : int
            ID of the player.
        trump_suit : Suit
            Current trump suit.
        count : int
            Number of trump cards.

        Returns
        -------
        float
            Probability that player has exactly 'count' trump cards.
        """
        probs = self.get_deduction_probabilities(player_id)
        trump_cards = [Card(trump_suit, rank) for rank in Rank]

        # Get probabilities for each trump card
        trump_probs = [probs.get(card, 0.0) for card in trump_cards]

        # Simplified calculation: assume independence
        # More accurate would use hypergeometric distribution
        # For now, use binomial approximation
        total_trump_prob = sum(trump_probs)
        hand_size = len(self.player_known_cards[player_id])

        # Rough approximation
        if count == 0:
            return max(0.0, 1.0 - total_trump_prob)
        elif count <= len(trump_cards):
            # Use binomial probability
            p = total_trump_prob / len(trump_cards) if len(trump_cards) > 0 else 0.0
            # Simplified: return normalized probability
            return min(1.0, total_trump_prob / count) if count > 0 else 0.0
        else:
            return 0.0

