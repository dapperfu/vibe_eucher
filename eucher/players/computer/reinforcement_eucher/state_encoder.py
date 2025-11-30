"""State encoder for ReinforcementEucher."""

from typing import Dict, List, Optional

import torch

from eucher.cards import Card, Rank, Suit
from eucher.game import Game


class StateEncoder:
    """Encode game state to tensor representation for ReinforcementEucher.

    State representation includes:
    - Player hand (binary 24)
    - Trump suit one-hot (5: none + 4 suits)
    - Dealer/leader indices (one-hot 4 each)
    - Current trick cards (binary 24 × 4 positions)
    - Trick history (binary 24 × 5 tricks × 4 positions)
    - Unseen card estimates (binary 24)
    - Legal move flags (binary action_space_size)
    """

    NUM_CARDS = 24
    NUM_SUITS = 4
    NUM_RANKS = 6
    NUM_PLAYERS = 4
    MAX_TRICKS = 5
    MAX_TRICK_CARDS = 4

    def __init__(self) -> None:
        """Initialize state encoder."""
        pass

    def encode_full_state(
        self,
        game: Game,
        player_id: int,
        legal_actions: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Encode full game state for a player.

        Parameters
        ----------
        game : Game
            The game instance.
        player_id : int
            ID of the player (0-3).
        legal_actions : Optional[List[int]]
            List of legal action indices for masking.

        Returns
        -------
        torch.Tensor
            Encoded state tensor (flattened).
        """
        player = game.players[player_id]

        # Hand encoding (24 dims)
        hand_encoding = self.encode_hand(player.hand)

        # Trump suit encoding (5 dims: none + 4 suits)
        trump_encoding = self.encode_trump(game.trump_suit)

        # Position encodings (4 dims each)
        dealer_encoding = self.encode_position(game.dealer_id)
        leader_encoding = self.encode_position(
            game.current_trick.leader_id if game.current_trick and game.current_trick.leader_id is not None else 0
        )

        # Current trick encoding (24 × 4 = 96 dims)
        trick_encoding = self.encode_current_trick(game)

        # Trick history encoding (24 × 5 × 4 = 480 dims, simplified to 24 × 5 = 120)
        history_encoding = self.encode_trick_history(game)

        # Unseen card estimates (24 dims)
        unseen_encoding = self.encode_unseen_cards(game, player_id)

        # Legal action mask (action_space_size dims)
        legal_mask = self.encode_legal_actions(legal_actions)

        # Concatenate all encodings
        state_tensor = torch.cat([
            hand_encoding,  # 24
            trump_encoding,  # 5
            dealer_encoding,  # 4
            leader_encoding,  # 4
            trick_encoding,  # 96
            history_encoding,  # 120
            unseen_encoding,  # 24
            legal_mask,  # 18
        ])

        return state_tensor

    def encode_hand(self, hand: List[Card]) -> torch.Tensor:
        """Encode hand as binary vector (24 dims).

        Parameters
        ----------
        hand : List[Card]
            Player's hand.

        Returns
        -------
        torch.Tensor
            Binary encoding of hand (24 dims).
        """
        tensor = torch.zeros(self.NUM_CARDS)
        for card in hand:
            idx = self._card_to_index(card)
            tensor[idx] = 1.0
        return tensor

    def encode_trump(self, trump_suit: Optional[Suit]) -> torch.Tensor:
        """Encode trump suit (5 dims: none + 4 suits).

        Parameters
        ----------
        trump_suit : Optional[Suit]
            Current trump suit, or None.

        Returns
        -------
        torch.Tensor
            One-hot encoding of trump (5 dims).
        """
        tensor = torch.zeros(5)
        if trump_suit is None:
            tensor[0] = 1.0
        else:
            suit_idx = list(Suit).index(trump_suit) + 1
            tensor[suit_idx] = 1.0
        return tensor

    def encode_position(self, position_id: int) -> torch.Tensor:
        """Encode player position (4 dims one-hot).

        Parameters
        ----------
        position_id : int
            Position ID (0-3).

        Returns
        -------
        torch.Tensor
            One-hot encoding (4 dims).
        """
        tensor = torch.zeros(4)
        if 0 <= position_id < 4:
            tensor[position_id] = 1.0
        return tensor

    def encode_current_trick(self, game: Game) -> torch.Tensor:
        """Encode current trick cards (24 × 4 = 96 dims).

        Parameters
        ----------
        game : Game
            Game instance.

        Returns
        -------
        torch.Tensor
            Encoding of current trick (96 dims flattened).
        """
        tensor = torch.zeros(self.NUM_CARDS, self.MAX_TRICK_CARDS)
        if game.current_trick and game.current_trick.cards:
            for i, card in enumerate(game.current_trick.cards):
                if i < self.MAX_TRICK_CARDS:
                    idx = self._card_to_index(card)
                    tensor[idx, i] = 1.0
        return tensor.flatten()

    def encode_trick_history(self, game: Game) -> torch.Tensor:
        """Encode trick history (simplified: 24 × 5 = 120 dims).

        Parameters
        ----------
        game : Game
            Game instance.

        Returns
        -------
        torch.Tensor
            Encoding of trick history (120 dims flattened).
        """
        tensor = torch.zeros(self.NUM_CARDS, self.MAX_TRICKS)
        if hasattr(game, "trick_history") and game.trick_history:
            for trick_num, trick in enumerate(game.trick_history[:self.MAX_TRICKS]):
                for card in trick.cards if hasattr(trick, "cards") else []:
                    idx = self._card_to_index(card)
                    tensor[idx, trick_num] = 1.0
        return tensor.flatten()

    def encode_unseen_cards(self, game: Game, player_id: int) -> torch.Tensor:
        """Encode estimated distribution of unseen cards (24 dims).

        Simple implementation: mark cards not in player's hand as potentially unseen.

        Parameters
        ----------
        game : Game
            Game instance.
        player_id : int
            Player ID.

        Returns
        -------
        torch.Tensor
            Binary encoding of unseen cards (24 dims).
        """
        tensor = torch.ones(self.NUM_CARDS)  # Start with all cards unseen

        # Mark cards in player's hand as seen
        player = game.players[player_id]
        for card in player.hand:
            idx = self._card_to_index(card)
            tensor[idx] = 0.0

        # Mark turned card as seen if present
        if game.turned_card:
            idx = self._card_to_index(game.turned_card)
            tensor[idx] = 0.0

        # Mark cards in current trick as seen
        if game.current_trick and game.current_trick.cards:
            for card in game.current_trick.cards:
                idx = self._card_to_index(card)
                tensor[idx] = 0.0

        return tensor

    def encode_legal_actions(self, legal_actions: Optional[List[int]]) -> torch.Tensor:
        """Encode legal action mask (action_space_size dims).

        Parameters
        ----------
        legal_actions : Optional[List[int]]
            List of legal action indices.

        Returns
        -------
        torch.Tensor
            Binary mask of legal actions (18 dims).
        """
        from eucher.players.computer.reinforcement_eucher.action_space import ACTION_SPACE_SIZE

        tensor = torch.zeros(ACTION_SPACE_SIZE)
        if legal_actions:
            for action_id in legal_actions:
                if 0 <= action_id < ACTION_SPACE_SIZE:
                    tensor[action_id] = 1.0
        else:
            # If no legal actions provided, assume all actions are legal
            tensor.fill_(1.0)
        return tensor

    def _card_to_index(self, card: Card) -> int:
        """Convert card to index (0-23).

        Parameters
        ----------
        card : Card
            Card to convert.

        Returns
        -------
        int
            Card index (0-23).
        """
        # Order: 9, 10, J, Q, K, A for each suit
        rank_order = [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]
        suit_order = list(Suit)

        rank_idx = rank_order.index(card.rank)
        suit_idx = suit_order.index(card.suit)

        return suit_idx * 6 + rank_idx

    def get_state_dim(self) -> int:
        """Get dimension of encoded state.

        Returns
        -------
        int
            State dimension.
        """
        return (
            24 +  # hand
            5 +  # trump
            4 +  # dealer
            4 +  # leader
            96 +  # current trick
            120 +  # trick history
            24 +  # unseen cards
            18  # legal actions
        )  # Total: 295


