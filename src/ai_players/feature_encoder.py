"""Feature encoder for converting game state to neural network inputs."""

from typing import List, Optional

import numpy as np
import torch

from src.cards import Card, Rank, Suit

from src.ai_players.game_state_tracker import TrickHistoryTracker


class EuchreFeatureEncoder:
    """Converts game state to neural network input features.

    Parameters
    ----------
    None

    Attributes
    ----------
    card_embedding_dim : int
        Dimension of card embeddings.
    max_trick_history : int
        Maximum number of tricks to include in history.
    """

    def __init__(self, card_embedding_dim: int = 32, max_trick_history: int = 5) -> None:
        """Initialize the feature encoder.

        Parameters
        ----------
        card_embedding_dim : int
            Dimension for card embeddings.
        max_trick_history : int
            Maximum tricks to include in history.
        """
        self.card_embedding_dim = card_embedding_dim
        self.max_trick_history = max_trick_history

        # Card vocabulary size: 24 cards (6 ranks × 4 suits)
        self.num_cards = 24
        # Suit vocabulary: 4 suits
        self.num_suits = 4
        # Rank vocabulary: 6 ranks
        self.num_ranks = 6

    def encode_hand(self, hand: List[Card]) -> torch.Tensor:
        """Encode player's hand as a feature vector.

        Parameters
        ----------
        hand : List[Card]
            Player's hand (should be 5 cards after discard, 6 before).

        Returns
        -------
        torch.Tensor
            Hand encoding tensor of shape (6, feature_dim).
            Padded with zeros if hand has fewer than 6 cards.
        """
        # One-hot encoding for each card position
        # Feature vector per card: [suit_one_hot(4), rank_one_hot(6), is_trump(1), trump_rank(1)]
        feature_dim = self.num_suits + self.num_ranks + 2  # 12 features per card
        hand_features = torch.zeros(6, feature_dim)

        for i, card in enumerate(hand[:6]):
            # Suit one-hot
            suit_idx = list(Suit).index(card.suit)
            hand_features[i, suit_idx] = 1.0

            # Rank one-hot
            rank_idx = list(Rank).index(card.rank)
            hand_features[i, self.num_suits + rank_idx] = 1.0

            # Additional features would be added here if needed

        return hand_features

    def encode_card(self, card: Card, trump_suit: Optional[Suit] = None) -> torch.Tensor:
        """Encode a single card as a feature vector.

        Parameters
        ----------
        card : Card
            The card to encode.
        trump_suit : Optional[Suit]
            Current trump suit, if any.

        Returns
        -------
        torch.Tensor
            Card encoding tensor of shape (feature_dim,).
        """
        feature_dim = self.num_suits + self.num_ranks + 2
        features = torch.zeros(feature_dim)

        # Suit one-hot
        suit_idx = list(Suit).index(card.suit)
        features[suit_idx] = 1.0

        # Rank one-hot
        rank_idx = list(Rank).index(card.rank)
        features[self.num_suits + rank_idx] = 1.0

        # Is trump
        if trump_suit is not None:
            features[self.num_suits + self.num_ranks] = 1.0 if card._is_trump(trump_suit) else 0.0
            # Trump rank (normalized 0-1)
            if card._is_trump(trump_suit):
                trump_rank = card._get_trump_rank(trump_suit)
                features[self.num_suits + self.num_ranks + 1] = trump_rank / 7.0

        return features

    def encode_trick_history(
        self, trick_history: List[dict], trump_suit: Optional[Suit] = None
    ) -> torch.Tensor:
        """Encode trick history as a sequence.

        Parameters
        ----------
        trick_history : List[dict]
            List of completed tricks, each with 'cards' and 'winner_team'.
        trump_suit : Optional[Suit]
            Current trump suit, if any.

        Returns
        -------
        torch.Tensor
            Trick history encoding tensor of shape (max_trick_history, 4, card_feature_dim).
            Each trick has 4 cards, padded if necessary.
        """
        card_feature_dim = self.num_suits + self.num_ranks + 2
        history_tensor = torch.zeros(self.max_trick_history, 4, card_feature_dim)

        for trick_idx, trick in enumerate(trick_history[: self.max_trick_history]):
            cards = trick.get("cards", [])
            for card_idx, card in enumerate(cards[:4]):
                history_tensor[trick_idx, card_idx] = self.encode_card(card, trump_suit)

        return history_tensor

    def encode_won_tricks_summary(
        self, tracker: TrickHistoryTracker, trump_suit: Optional[Suit] = None
    ) -> torch.Tensor:
        """Encode summary of won tricks.

        Parameters
        ----------
        tracker : TrickHistoryTracker
            Trick history tracker.
        trump_suit : Optional[Suit]
            Current trump suit, if any.

        Returns
        -------
        torch.Tensor
            Summary vector of shape (summary_dim,).
            Includes: cards seen count, suit distribution, team tricks won.
        """
        cards_seen = tracker.get_cards_seen()
        suit_dist = tracker.get_suit_distribution()
        team_tricks = tracker.get_team_tricks_won()

        # Summary features:
        # - Cards seen count (1)
        # - Suit distribution (4)
        # - Team tricks won (2)
        # - Trump cards seen (1)
        # Total: 8 features
        summary = torch.zeros(8)

        summary[0] = len(cards_seen) / 24.0  # Normalized cards seen

        # Suit distribution
        for suit_idx, suit in enumerate(Suit):
            summary[1 + suit_idx] = suit_dist.get(suit, 0) / 24.0

        # Team tricks won
        summary[5] = team_tricks.get(0, 0) / 5.0
        summary[6] = team_tricks.get(1, 0) / 5.0

        # Trump cards seen
        if trump_suit is not None:
            trump_cards_seen = sum(1 for card in cards_seen if card._is_trump(trump_suit))
            summary[7] = trump_cards_seen / 6.0  # Max 6 trump cards

        return summary

    def encode_game_context(
        self,
        trick_number: int,
        team_score: int,
        opponent_score: int,
        player_position: int,
        dealer_id: int,
        trump_suit: Optional[Suit] = None,
    ) -> torch.Tensor:
        """Encode game context features.

        Parameters
        ----------
        trick_number : int
            Current trick number (0-4).
        team_score : int
            Player's team score.
        opponent_score : int
            Opponent team score.
        player_position : int
            Player position (0-3).
        dealer_id : int
            Dealer ID (0-3).
        trump_suit : Optional[Suit]
            Current trump suit, if any.

        Returns
        -------
        torch.Tensor
            Context features tensor of shape (context_dim,).
        """
        # Context features:
        # - Trick number (1)
        # - Team score (1)
        # - Opponent score (1)
        # - Score differential (1)
        # - Player position one-hot (4)
        # - Is dealer (1)
        # - Trump suit one-hot (4)
        # Total: 13 features
        context = torch.zeros(13)

        context[0] = trick_number / 5.0  # Normalized trick number
        context[1] = team_score / 10.0  # Normalized score
        context[2] = opponent_score / 10.0
        context[3] = (team_score - opponent_score) / 10.0  # Score differential

        # Player position one-hot
        context[4 + player_position] = 1.0

        # Is dealer
        context[8] = 1.0 if player_position == dealer_id else 0.0

        # Trump suit one-hot
        if trump_suit is not None:
            trump_idx = list(Suit).index(trump_suit)
            context[9 + trump_idx] = 1.0

        return context

    def encode_full_state(
        self,
        hand: List[Card],
        trick_history: List[dict],
        tracker: TrickHistoryTracker,
        trick_number: int,
        team_score: int,
        opponent_score: int,
        player_position: int,
        dealer_id: int,
        trump_suit: Optional[Suit] = None,
        current_trick: Optional[List[Card]] = None,
    ) -> dict:
        """Encode full game state for neural network input.

        Parameters
        ----------
        hand : List[Card]
            Player's hand.
        trick_history : List[dict]
            History of completed tricks.
        tracker : TrickHistoryTracker
            Trick history tracker.
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

        Returns
        -------
        dict
            Dictionary containing all encoded features:
            - 'hand': hand encoding
            - 'trick_history': trick history encoding
            - 'won_tricks_summary': won tricks summary
            - 'game_context': game context
            - 'current_trick': current trick encoding (if provided)
        """
        return {
            "hand": self.encode_hand(hand),
            "trick_history": self.encode_trick_history(trick_history, trump_suit),
            "won_tricks_summary": self.encode_won_tricks_summary(tracker, trump_suit),
            "game_context": self.encode_game_context(
                trick_number, team_score, opponent_score, player_position, dealer_id, trump_suit
            ),
            "current_trick": (
                torch.stack([self.encode_card(card, trump_suit) for card in current_trick])
                if current_trick
                else torch.zeros(4, self.num_suits + self.num_ranks + 2)
            ),
        }

