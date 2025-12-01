"""State encoder for EucherGo with belief maps and fixed-length tensors."""

from typing import Dict, List, Optional

import torch

from eucher.cards import Card, Rank, Suit


class EucherGoStateEncoder:
    """Encode game state to fixed-length tensor representation for EucherGo.

    Encodes:
    - Player hand: binary 24-vector (one per card)
    - Trump suit: one-hot encoding
    - Dealer/leader indices: integer encoding
    - Current trick cards: binary vectors
    - Trick history: binary vectors for each trick
    - Belief map: 24-vector per player (unknown card probabilities)
    - Flags: can follow suit, must call trump, can go alone, etc.
    """

    NUM_CARDS = 24
    NUM_SUITS = 4
    NUM_RANKS = 6
    NUM_PLAYERS = 4
    MAX_TRICKS = 5  # Maximum tricks per hand

    def __init__(self) -> None:
        """Initialize state encoder."""
        self._card_to_idx = self._build_card_index()

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
        Convert card to index (0-23).

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

    def encode_state(
        self,
        player_hand: List[Card],
        trump_suit: Optional[Suit],
        dealer_id: int,
        leader_id: int,
        current_trick_cards: List[Card],
        trick_history: List[List[Card]],
        belief_map: Dict[int, torch.Tensor],
        can_follow_suit: bool,
        must_call_trump: bool,
        can_go_alone: bool,
        player_id: int,
    ) -> torch.Tensor:
        """
        Encode full game state to fixed-length tensor.

        Parameters
        ----------
        player_hand : List[Card]
            Player's hand.
        trump_suit : Optional[Suit]
            Current trump suit.
        dealer_id : int
            Dealer ID (0-3).
        leader_id : int
            Current trick leader ID (0-3).
        current_trick_cards : List[Card]
            Cards played in current trick.
        trick_history : List[List[Card]]
            History of tricks (list of cards per trick).
        belief_map : Dict[int, torch.Tensor]
            Belief map per player (player_id -> 24-vector probabilities).
        can_follow_suit : bool
            Whether player can follow suit.
        must_call_trump : bool
            Whether player must call trump.
        can_go_alone : bool
            Whether player can go alone.
        player_id : int
            Current player ID.

        Returns
        -------
        torch.Tensor
            Fixed-length state tensor.
        """
        # Player hand: 24-dim binary vector
        hand_vec = self._encode_hand(player_hand)

        # Trump suit: 5-dim one-hot (None + 4 suits)
        trump_vec = self._encode_trump(trump_suit)

        # Dealer/leader indices: 2-dim one-hot each (4 dims each = 8 total)
        dealer_vec = self._encode_player_index(dealer_id)
        leader_vec = self._encode_player_index(leader_id)

        # Current trick: 4 cards max × 24 dims = 96 dims
        trick_vec = self._encode_trick(current_trick_cards)

        # Trick history: 5 tricks max × 4 cards × 24 dims = 480 dims
        history_vec = self._encode_trick_history(trick_history)

        # Belief map: 4 players × 24 dims = 96 dims
        belief_vec = self._encode_belief_map(belief_map)

        # Flags: 3 dims
        flags_vec = torch.tensor(
            [float(can_follow_suit), float(must_call_trump), float(can_go_alone)], dtype=torch.float32
        )

        # Concatenate all vectors
        state_tensor = torch.cat(
            [
                hand_vec,  # 24
                trump_vec,  # 5
                dealer_vec,  # 4
                leader_vec,  # 4
                trick_vec,  # 96
                history_vec,  # 480
                belief_vec,  # 96
                flags_vec,  # 3
            ]
        )

        return state_tensor

    def _encode_hand(self, hand: List[Card]) -> torch.Tensor:
        """
        Encode hand as binary 24-vector.

        Parameters
        ----------
        hand : List[Card]
            Player's hand.

        Returns
        -------
        torch.Tensor
            24-dim binary vector.
        """
        vec = torch.zeros(self.NUM_CARDS, dtype=torch.float32)
        for card in hand:
            idx = self._card_to_index(card)
            vec[idx] = 1.0
        return vec

    def _encode_trump(self, trump_suit: Optional[Suit]) -> torch.Tensor:
        """
        Encode trump suit as one-hot (5-dim: None + 4 suits).

        Parameters
        ----------
        trump_suit : Optional[Suit]
            Trump suit or None.

        Returns
        -------
        torch.Tensor
            5-dim one-hot vector.
        """
        vec = torch.zeros(5, dtype=torch.float32)
        if trump_suit is None:
            vec[0] = 1.0
        else:
            suit_idx = list(Suit).index(trump_suit) + 1
            vec[suit_idx] = 1.0
        return vec

    def _encode_player_index(self, player_id: int) -> torch.Tensor:
        """
        Encode player index as one-hot (4-dim).

        Parameters
        ----------
        player_id : int
            Player ID (0-3).

        Returns
        -------
        torch.Tensor
            4-dim one-hot vector.
        """
        vec = torch.zeros(4, dtype=torch.float32)
        if 0 <= player_id < 4:
            vec[player_id] = 1.0
        return vec

    def _encode_trick(self, trick_cards: List[Card]) -> torch.Tensor:
        """
        Encode current trick cards (4 cards max × 24 dims = 96 dims).

        Parameters
        ----------
        trick_cards : List[Card]
            Cards in current trick.

        Returns
        -------
        torch.Tensor
            96-dim vector (4 × 24).
        """
        vec = torch.zeros(4 * self.NUM_CARDS, dtype=torch.float32)
        for i, card in enumerate(trick_cards[:4]):
            idx = self._card_to_index(card)
            vec[i * self.NUM_CARDS + idx] = 1.0
        return vec

    def _encode_trick_history(self, trick_history: List[List[Card]]) -> torch.Tensor:
        """
        Encode trick history (5 tricks × 4 cards × 24 dims = 480 dims).

        Parameters
        ----------
        trick_history : List[List[Card]]
            History of tricks.

        Returns
        -------
        torch.Tensor
            480-dim vector (5 × 4 × 24).
        """
        vec = torch.zeros(self.MAX_TRICKS * 4 * self.NUM_CARDS, dtype=torch.float32)
        for trick_idx, trick in enumerate(trick_history[: self.MAX_TRICKS]):
            for card_idx, card in enumerate(trick[:4]):
                card_pos = self._card_to_index(card)
                vec[trick_idx * 4 * self.NUM_CARDS + card_idx * self.NUM_CARDS + card_pos] = 1.0
        return vec

    def _encode_belief_map(self, belief_map: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Encode belief map (4 players × 24 dims = 96 dims).

        Parameters
        ----------
        belief_map : Dict[int, torch.Tensor]
            Belief map per player (player_id -> 24-vector probabilities).

        Returns
        -------
        torch.Tensor
            96-dim vector (4 × 24).
        """
        vec = torch.zeros(4 * self.NUM_CARDS, dtype=torch.float32)
        for player_id in range(4):
            if player_id in belief_map:
                belief = belief_map[player_id]
                if belief.shape[0] == self.NUM_CARDS:
                    vec[player_id * self.NUM_CARDS : (player_id + 1) * self.NUM_CARDS] = belief
        return vec

    def get_state_size(self) -> int:
        """
        Get the size of the encoded state tensor.

        Returns
        -------
        int
            State tensor size.
        """
        return 24 + 5 + 4 + 4 + 96 + 480 + 96 + 3  # 712



