"""State encoder for EuchreZero."""

from typing import Dict, List, Optional

import torch

from eucher.cards import Card, Rank, Suit
from eucher.game import Game


class StateEncoder:
    """Encode game state to tensor representation."""

    NUM_CARDS = 24
    NUM_SUITS = 4
    NUM_RANKS = 6

    def __init__(self) -> None:
        """Initialize state encoder."""
        pass

    def encode_full_state(
        self, game: Game, player_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        Encode full game state for a player.

        Parameters
        ----------
        game : Game
            The game instance.
        player_id : int
            ID of the player (0-3).

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of encoded state tensors.
        """
        player = game.players[player_id]

        return {
            "hand": self.encode_hand(player.hand),
            "upcard": self.encode_card(game.turned_card) if game.turned_card else self.encode_card(None),
            "is_dealer": torch.tensor([1.0 if game.dealer_id == player_id else 0.0]),
            "player_seat": self.encode_seat(player_id),
            "partner_seat": self.encode_seat((player_id + 2) % 4),
            "trump_context": self.encode_trump(game.trump_suit),
            "bidding_stage": self.encode_bidding_stage(game),
            "score": torch.tensor([float(game.scores[0]), float(game.scores[1])]),
            "trick_history": self.encode_trick_history(game),
            "perfect_memory": self.encode_perfect_memory(game, player_id),
            "deduction_map": self.encode_deduction_map(game, player_id),
            "suit_voids": self.encode_suit_voids(game, player_id),
            "trump_counts": self.encode_trump_counts(game, player_id),
            "risk_factor": torch.tensor([0.0]),  # Will be set by player
        }

    def encode_hand(self, hand: List[Card]) -> torch.Tensor:
        """
        Encode hand as one-hot (24 cards × 5 positions).

        Parameters
        ----------
        hand : List[Card]
            Player's hand.

        Returns
        -------
        torch.Tensor
            Encoded hand tensor (120 dims flattened).
        """
        tensor = torch.zeros(24, 5)
        for i, card in enumerate(hand[:5]):
            card_idx = self._card_to_index(card)
            tensor[card_idx, i] = 1.0
        return tensor.flatten()

    def encode_card(self, card: Optional[Card]) -> torch.Tensor:
        """
        Encode single card as one-hot (24 dims).

        Parameters
        ----------
        card : Optional[Card]
            Card to encode, or None.

        Returns
        -------
        torch.Tensor
            Encoded card tensor (24 dims).
        """
        tensor = torch.zeros(24)
        if card is not None:
            idx = self._card_to_index(card)
            tensor[idx] = 1.0
        return tensor

    def encode_seat(self, seat_id: int) -> torch.Tensor:
        """
        Encode player seat (4-dim one-hot).

        Parameters
        ----------
        seat_id : int
            Seat ID (0-3).

        Returns
        -------
        torch.Tensor
            Encoded seat tensor (4 dims).
        """
        tensor = torch.zeros(4)
        tensor[seat_id] = 1.0
        return tensor

    def encode_trump(self, trump_suit: Optional[Suit]) -> torch.Tensor:
        """
        Encode trump context (5-dim: None + 4 suits).

        Parameters
        ----------
        trump_suit : Optional[Suit]
            Current trump suit, or None.

        Returns
        -------
        torch.Tensor
            Encoded trump tensor (5 dims).
        """
        tensor = torch.zeros(5)
        if trump_suit is None:
            tensor[0] = 1.0
        else:
            suit_idx = list(Suit).index(trump_suit) + 1
            tensor[suit_idx] = 1.0
        return tensor

    def encode_bidding_stage(self, game: Game) -> torch.Tensor:
        """
        Encode bidding stage (3-dim: order_up, call_trump, playing).

        Parameters
        ----------
        game : Game
            Game instance.

        Returns
        -------
        torch.Tensor
            Encoded bidding stage tensor (3 dims).
        """
        tensor = torch.zeros(3)
        if game.trump_suit is None:
            # Check if we're in order_up or call_trump phase
            # Simplified: assume call_trump if no trump set
            tensor[1] = 1.0  # call_trump stage
        else:
            tensor[2] = 1.0  # playing stage
        return tensor

    def encode_trick_history(self, game: Game) -> torch.Tensor:
        """
        Encode trick history (simplified: 5 tricks × 4 cards × 2 features = 40 dims).

        Parameters
        ----------
        game : Game
            Game instance.

        Returns
        -------
        torch.Tensor
            Encoded trick history tensor (40 dims).
        """
        # Simplified encoding: just track if tricks have been played
        # In full implementation, would encode card indices and winners
        tensor = torch.zeros(40)
        # Placeholder - would track actual trick history
        return tensor

    def encode_perfect_memory(self, game: Game, player_id: int) -> torch.Tensor:
        """
        Encode perfect memory (24 cards × 4 locations = 96 dims).

        Parameters
        ----------
        game : Game
            Game instance.
        player_id : int
            Player ID.

        Returns
        -------
        torch.Tensor
            Encoded perfect memory tensor (96 dims).
        """
        # Track card locations: hand, played, discard, unknown
        tensor = torch.zeros(24, 4)
        player = game.players[player_id]

        # Mark cards in player's hand
        for card in player.hand:
            card_idx = self._card_to_index(card)
            tensor[card_idx, 0] = 1.0  # In hand

        # Mark turned card if exists
        if game.turned_card:
            card_idx = self._card_to_index(game.turned_card)
            tensor[card_idx, 1] = 1.0  # Visible (played/discard)

        # Unknown cards default to location 3
        for i in range(24):
            if tensor[i].sum() == 0:
                tensor[i, 3] = 1.0  # Unknown

        return tensor.flatten()

    def encode_deduction_map(self, game: Game, player_id: int) -> torch.Tensor:
        """
        Encode deduction probabilities (24 cards × 3 players = 72 dims).

        Parameters
        ----------
        game : Game
            Game instance.
        player_id : int
            Player ID.

        Returns
        -------
        torch.Tensor
            Encoded deduction map tensor (72 dims).
        """
        # Simplified: uniform probabilities initially
        # In full implementation, would use Bayesian inference
        tensor = torch.ones(24, 3) / 3.0  # Uniform distribution
        return tensor.flatten()

    def encode_suit_voids(self, game: Game, player_id: int) -> torch.Tensor:
        """
        Encode suit void inference (4 suits × 3 players = 12 dims).

        Parameters
        ----------
        game : Game
            Game instance.
        player_id : int
            Player ID.

        Returns
        -------
        torch.Tensor
            Encoded suit voids tensor (12 dims).
        """
        # Simplified: no void inference initially
        tensor = torch.zeros(4, 3)
        return tensor.flatten()

    def encode_trump_counts(self, game: Game, player_id: int) -> torch.Tensor:
        """
        Encode trump count inference (3 players).

        Parameters
        ----------
        game : Game
            Game instance.
        player_id : int
            Player ID.

        Returns
        -------
        torch.Tensor
            Encoded trump counts tensor (3 dims).
        """
        # Simplified: no inference initially
        tensor = torch.zeros(3)
        return tensor

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
        suits = list(Suit)
        ranks = [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]
        suit_idx = suits.index(card.suit)
        rank_idx = ranks.index(card.rank)
        return suit_idx * 6 + rank_idx

    def encode_state_dict_to_tensor(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Concatenate all state components into single tensor.

        Parameters
        ----------
        state_dict : Dict[str, torch.Tensor]
            Dictionary of state tensors.

        Returns
        -------
        torch.Tensor
            Concatenated state tensor (~400 dims).
        """
        components = [
            state_dict["hand"],  # 120
            state_dict["upcard"],  # 24
            state_dict["is_dealer"],  # 1
            state_dict["player_seat"],  # 4
            state_dict["partner_seat"],  # 4
            state_dict["trump_context"],  # 5
            state_dict["bidding_stage"],  # 3
            state_dict["score"],  # 2
            state_dict["trick_history"],  # 40
            state_dict["perfect_memory"],  # 96
            state_dict["deduction_map"],  # 72
            state_dict["suit_voids"],  # 12
            state_dict["trump_counts"],  # 3
            state_dict["risk_factor"],  # 1
        ]
        return torch.cat(components, dim=0)

