"""Enhanced feature encoder for transformer RL with deduction and perfect memory.

This module extends the base feature encoder with:
- Deduction probability maps
- Perfect memory card states
- Risk factor embedding
- Auxiliary features (suit void probabilities, trump counts, dominance rankings)
"""

from typing import Dict, List, Optional

import torch

from eucher.cards import Card, Rank, Suit

from eucher.players.computer.ml.pytorch.feature_encoder import EuchreFeatureEncoder
from eucher.players.computer.ml.pytorch.game_state_tracker import TrickHistoryTracker
from src.ai_players.transformer_rl.deduction_engine import DeductionEngine


class TransformerRLFeatureEncoder(EuchreFeatureEncoder):
    """Enhanced feature encoder for transformer RL with deduction and perfect memory.

    Extends the base EuchreFeatureEncoder with additional features required
    for the transformer RL system.

    Parameters
    ----------
    card_embedding_dim : int
        Dimension for card embeddings.
    max_trick_history : int
        Maximum tricks to include in history.
    device : Optional[torch.device]
        Device to create tensors on.
    """

    def __init__(self, card_embedding_dim: int = 32, max_trick_history: int = 5, device: Optional[torch.device] = None) -> None:
        """Initialize enhanced feature encoder.

        Parameters
        ----------
        card_embedding_dim : int
            Card embedding dimension.
        max_trick_history : int
            Maximum trick history length.
        device : Optional[torch.device]
            Device to create tensors on.
        """
        super().__init__(card_embedding_dim, max_trick_history)
        self.device = device

    def encode_upcard_features(
        self, turned_card: Optional[Card], trump_suit: Optional[Suit] = None
    ) -> torch.Tensor:
        """Encode upcard features including bower information.

        Parameters
        ----------
        turned_card : Optional[Card]
            The turned up card.
        trump_suit : Optional[Suit]
            Current trump suit, if any.

        Returns
        -------
        torch.Tensor
            Upcard features tensor of shape (upcard_feature_dim,).
        """
        # Upcard features:
        # - Card encoding (12 features)
        # - Is potential trump (1)
        # - Is bower (1)
        # - Is left bower (1)
        # Total: 15 features
        upcard_features = torch.zeros(15, device=self.device)

        if turned_card is not None:
            # Encode card
            card_features = self.encode_card(turned_card, trump_suit)
            upcard_features[:12] = card_features

            # Is potential trump (if turned card's suit could be trump)
            upcard_features[12] = 1.0

            # Is bower (right bower)
            if trump_suit is not None:
                if turned_card.suit == trump_suit and turned_card.rank == Rank.JACK:
                    upcard_features[13] = 1.0

                # Is left bower
                left_bower_suit = self._get_left_bower_suit(trump_suit)
                if turned_card.suit == left_bower_suit and turned_card.rank == Rank.JACK:
                    upcard_features[14] = 1.0

        return upcard_features

    def _get_left_bower_suit(self, trump_suit: Suit) -> Suit:
        """Get the suit of the left bower for a given trump suit.

        Parameters
        ----------
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        Suit
            The suit of the left bower.
        """
        # Left bower mapping: Hearts <-> Diamonds, Clubs <-> Spades
        left_bower_map = {
            Suit.HEARTS: Suit.DIAMONDS,
            Suit.DIAMONDS: Suit.HEARTS,
            Suit.CLUBS: Suit.SPADES,
            Suit.SPADES: Suit.CLUBS,
        }
        return left_bower_map[trump_suit]

    def encode_player_position(
        self, player_id: int, dealer_id: int, partner_id: int
    ) -> torch.Tensor:
        """Encode player seat and partner position.

        Parameters
        ----------
        player_id : int
            Current player ID (0-3).
        dealer_id : int
            Dealer ID (0-3).
        partner_id : int
            Partner ID (0-3).

        Returns
        -------
        torch.Tensor
            Position features tensor of shape (position_feature_dim,).
        """
        # Position features:
        # - Player position one-hot (4)
        # - Dealer position one-hot (4)
        # - Partner position one-hot (4)
        # - Relative positions (3)
        # Total: 15 features
        position_features = torch.zeros(15, device=self.device)

        # One-hot encodings
        position_features[player_id] = 1.0
        position_features[4 + dealer_id] = 1.0
        position_features[8 + partner_id] = 1.0

        # Relative positions
        position_features[12] = (player_id - dealer_id) % 4 / 4.0  # Position relative to dealer
        position_features[13] = (partner_id - player_id) % 4 / 4.0  # Position relative to partner
        position_features[14] = 1.0 if player_id % 2 == partner_id % 2 else 0.0  # Same team

        return position_features

    def encode_risk_factor(self, risk_factor: float) -> torch.Tensor:
        """Encode risk factor as a feature vector.

        Parameters
        ----------
        risk_factor : float
            Risk factor value (0.0-1.0).

        Returns
        -------
        torch.Tensor
            Risk factor encoding tensor of shape (risk_feature_dim,).
        """
        # Risk features:
        # - Risk value (1)
        # - Risk category one-hot (3: low, medium, high)
        # Total: 4 features
        risk_features = torch.zeros(4, device=self.device)

        risk_features[0] = risk_factor

        # Risk categories
        if risk_factor < 0.33:
            risk_features[1] = 1.0  # Low risk
        elif risk_factor < 0.67:
            risk_features[2] = 1.0  # Medium risk
        else:
            risk_features[3] = 1.0  # High risk

        return risk_features

    def encode_auxiliary_features(
        self,
        deduction_engine: DeductionEngine,
        player_id: int,
        trump_suit: Optional[Suit] = None,
        hand: Optional[List[Card]] = None,
    ) -> torch.Tensor:
        """Encode auxiliary features (suit void probabilities, trump counts, dominance).

        Parameters
        ----------
        deduction_engine : DeductionEngine
            Deduction engine for probability calculations.
        player_id : int
            Current player ID.
        trump_suit : Optional[Suit]
            Current trump suit, if any.
        hand : Optional[List[Card]]
            Player's hand.

        Returns
        -------
        torch.Tensor
            Auxiliary features tensor of shape (auxiliary_feature_dim,).
        """
        # Auxiliary features:
        # - Suit void probabilities (4)
        # - Trump count probabilities (7: 0-6 trump cards)
        # - Dominance ranking (1)
        # Total: 12 features
        aux_features = torch.zeros(12, device=self.device)

        if trump_suit is not None:
            # Suit void probabilities
            void_probs = deduction_engine.get_suit_void_probabilities(player_id, trump_suit)
            for suit_idx, suit in enumerate(Suit):
                aux_features[suit_idx] = void_probs.get(suit, 0.0)

            # Trump count probabilities
            for count in range(7):
                prob = deduction_engine.get_trump_count_probability(player_id, trump_suit, count)
                aux_features[4 + count] = prob

        # Dominance ranking (simplified: count high cards in hand)
        if hand is not None:
            high_cards = sum(1 for card in hand if card.rank.value >= Rank.KING.value)
            aux_features[11] = high_cards / len(hand) if hand else 0.0

        return aux_features

    def encode_full_state_with_deduction(
        self,
        hand: List[Card],
        trick_history: List[dict],
        tracker: TrickHistoryTracker,
        deduction_engine: DeductionEngine,
        trick_number: int,
        team_score: int,
        opponent_score: int,
        player_position: int,
        dealer_id: int,
        trump_suit: Optional[Suit] = None,
        current_trick: Optional[List[Card]] = None,
        turned_card: Optional[Card] = None,
        risk_factor: float = 0.5,
        current_hand_sizes: Optional[Dict[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode full game state with deduction and perfect memory.

        Parameters
        ----------
        hand : List[Card]
            Player's hand.
        trick_history : List[dict]
            History of completed tricks.
        tracker : TrickHistoryTracker
            Trick history tracker.
        deduction_engine : DeductionEngine
            Deduction engine for probability calculations.
        trick_number : int
            Current trick number.
        team_score : int
            Player's team score.
        opponent_score : int
            Opponent team score.
        player_position : int
            Player position.
        dealer_id : int
            Dealer ID.
        trump_suit : Optional[Suit]
            Current trump suit.
        current_trick : Optional[List[Card]]
            Cards in current trick, if any.
        turned_card : Optional[Card]
            Turned up card, if any.
        risk_factor : float
            Risk factor (0.0-1.0).
        current_hand_sizes : Optional[Dict[int, int]]
            Current hand size for each player.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing all encoded features:
            - 'hand': hand encoding
            - 'trick_history': trick history encoding
            - 'game_context': game context
            - 'current_trick': current trick encoding
            - 'deduction_maps': deduction probability maps (24, 4)
            - 'upcard_features': upcard features
            - 'position_features': player position features
            - 'risk_features': risk factor features
            - 'auxiliary_features': auxiliary features
        """
        # Get base encoding
        base_encoding = self.encode_full_state(
            hand=hand,
            trick_history=trick_history,
            tracker=tracker,
            trick_number=trick_number,
            team_score=team_score,
            opponent_score=opponent_score,
            player_position=player_position,
            dealer_id=dealer_id,
            trump_suit=trump_suit,
            current_trick=current_trick,
        )

        # Add deduction maps
        deduction_maps = deduction_engine.get_deduction_map_tensor(current_hand_sizes, device=self.device) if deduction_engine else None

        # Add upcard features
        upcard_features = self.encode_upcard_features(turned_card, trump_suit)

        # Add position features
        partner_id = (player_position + 2) % 4
        position_features = self.encode_player_position(player_position, dealer_id, partner_id)

        # Add risk features
        risk_features = self.encode_risk_factor(risk_factor)

        # Add auxiliary features
        auxiliary_features = self.encode_auxiliary_features(
            deduction_engine, player_position, trump_suit, hand
        )

        # Combine all features
        return {
            **base_encoding,
            "deduction_maps": deduction_maps,
            "upcard_features": upcard_features,
            "position_features": position_features,
            "risk_features": risk_features,
            "auxiliary_features": auxiliary_features,
        }

