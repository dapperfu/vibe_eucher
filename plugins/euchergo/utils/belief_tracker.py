"""Belief tracker for unknown card probabilities."""

from typing import Dict, List, Optional

import torch

from eucher.cards import Card, Deck, Rank, Suit


class BeliefTracker:
    """Track probability distribution of unseen cards per player.

    Updates belief map after each trick using card constraints.
    """

    NUM_CARDS = 24

    def __init__(self) -> None:
        """Initialize belief tracker."""
        self._card_to_idx = self._build_card_index()
        self.belief_maps: Dict[int, torch.Tensor] = {}
        self.known_cards: Dict[int, List[Card]] = {}
        self.played_cards: List[Card] = []

    def _build_card_index(self) -> Dict[Card, int]:
        """
        Build mapping from Card to index (0-23).

        Returns
        -------
        Dict[Card, int]
            Mapping from Card to index.
        """
        idx = 0
        card_map: Dict[Card, int] = {}
        for suit in Suit:
            for rank in [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]:
                card = Card(suit, rank)
                card_map[card] = idx
                idx += 1
        return card_map

    def _card_to_index(self, card: Card) -> int:
        """
        Convert card to index.

        Parameters
        ----------
        card : Card
            Card to convert.

        Returns
        -------
        int
            Card index (0-23).
        """
        return self._card_to_idx[card]

    def initialize(self, player_hand: List[Card], known_cards: Optional[Dict[int, List[Card]]] = None) -> None:
        """
        Initialize belief maps.

        Parameters
        ----------
        player_hand : List[Card]
            Current player's hand.
        known_cards : Optional[Dict[int, List[Card]]]
            Known cards per player (e.g., from perfect memory).
        """
        # Initialize belief maps for all players
        for player_id in range(4):
            # Start with uniform distribution over all cards
            belief = torch.ones(self.NUM_CARDS, dtype=torch.float32) / self.NUM_CARDS
            self.belief_maps[player_id] = belief
            self.known_cards[player_id] = []

        # Remove known cards (current player's hand)
        for card in player_hand:
            card_idx = self._card_to_index(card)
            for player_id in range(4):
                self.belief_maps[player_id][card_idx] = 0.0

        # Remove known cards from other players if provided
        if known_cards:
            for player_id, cards in known_cards.items():
                for card in cards:
                    card_idx = self._card_to_index(card)
                    self.belief_maps[player_id][card_idx] = 0.0
                    self.known_cards[player_id].append(card)

        # Normalize
        self._normalize_beliefs()

    def update_after_card_played(self, player_id: int, card: Card) -> None:
        """
        Update belief map after a card is played.

        Parameters
        ----------
        player_id : int
            Player who played the card.
        card : Card
            Card that was played.
        """
        card_idx = self._card_to_index(card)
        # Set probability to 0 for this card for this player
        self.belief_maps[player_id][card_idx] = 0.0
        self.played_cards.append(card)

        # Update other players' beliefs (they can't have this card)
        for other_id in range(4):
            if other_id != player_id:
                self.belief_maps[other_id][card_idx] = 0.0

        self._normalize_beliefs()

    def update_after_trick(self, trick_cards: List[Card], trick_player_ids: List[int]) -> None:
        """
        Update belief maps after a trick is completed.

        Parameters
        ----------
        trick_cards : List[Card]
            Cards played in the trick.
        trick_player_ids : List[int]
            Player IDs who played each card.
        """
        for card, player_id in zip(trick_cards, trick_player_ids):
            self.update_after_card_played(player_id, card)

    def update_suit_void(self, player_id: int, suit: Suit) -> None:
        """
        Update belief map when a player is known to be void in a suit.

        Parameters
        ----------
        player_id : int
            Player ID.
        suit : Suit
            Suit the player is void in.
        """
        # Set probability to 0 for all cards of this suit
        for rank in [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]:
            card = Card(suit, rank)
            card_idx = self._card_to_index(card)
            self.belief_maps[player_id][card_idx] = 0.0

        self._normalize_beliefs()

    def get_belief_map(self, player_id: int) -> torch.Tensor:
        """
        Get belief map for a player.

        Parameters
        ----------
        player_id : int
            Player ID.

        Returns
        -------
        torch.Tensor
            Belief map (24-dim probability vector).
        """
        if player_id not in self.belief_maps:
            # Initialize if not present
            belief = torch.ones(self.NUM_CARDS, dtype=torch.float32) / self.NUM_CARDS
            self.belief_maps[player_id] = belief
        return self.belief_maps[player_id].clone()

    def get_all_belief_maps(self) -> Dict[int, torch.Tensor]:
        """
        Get all belief maps.

        Returns
        -------
        Dict[int, torch.Tensor]
            Dictionary mapping player_id to belief map.
        """
        return {pid: self.get_belief_map(pid) for pid in range(4)}

    def _normalize_beliefs(self) -> None:
        """Normalize belief maps to sum to 1."""
        for player_id in range(4):
            belief = self.belief_maps[player_id]
            total = belief.sum()
            if total > 0:
                self.belief_maps[player_id] = belief / total
            else:
                # All cards known: set to uniform (shouldn't happen)
                self.belief_maps[player_id] = torch.ones(self.NUM_CARDS, dtype=torch.float32) / self.NUM_CARDS



