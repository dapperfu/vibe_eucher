"""Feature extraction and encoding for ML models."""

from typing import List, Optional

import numpy as np
import torch

from eucher.cards import Card, Rank, Suit


class GameStateEncoder:
    """Encodes game state into tensor features for ML models."""

    # Total number of possible cards in Euchre deck
    NUM_CARDS = 24

    def __init__(self) -> None:
        """Initialize the game state encoder."""
        # Create mapping from card to index
        self.card_to_index: dict[Card, int] = {}
        index = 0
        for suit in Suit:
            for rank in Rank:
                card = Card(suit, rank)
                self.card_to_index[card] = index
                index += 1

        # Reverse mapping
        self.index_to_card: dict[int, Card] = {v: k for k, v in self.card_to_index.items()}

    def encode_card(self, card: Card) -> np.ndarray:
        """
        Encode a single card as a one-hot vector.

        Parameters
        ----------
        card : Card
            The card to encode.

        Returns
        -------
        np.ndarray
            One-hot encoded vector of shape (24,).
        """
        encoding = np.zeros(self.NUM_CARDS, dtype=np.float32)
        if card in self.card_to_index:
            encoding[self.card_to_index[card]] = 1.0
        return encoding

    def encode_hand(self, hand: List[Card]) -> np.ndarray:
        """
        Encode a hand (up to 5 cards) as a matrix.

        Parameters
        ----------
        hand : List[Card]
            The hand to encode.

        Returns
        -------
        np.ndarray
            Encoded hand matrix of shape (5, 24).
        """
        encoding = np.zeros((5, self.NUM_CARDS), dtype=np.float32)
        for i, card in enumerate(hand[:5]):
            encoding[i] = self.encode_card(card)
        return encoding

    def encode_trick_cards(self, trick_cards: List[Card]) -> np.ndarray:
        """
        Encode cards played in current trick (0-3 cards).

        Parameters
        ----------
        trick_cards : List[Card]
            Cards played in the trick so far.

        Returns
        -------
        np.ndarray
            Encoded trick cards matrix of shape (3, 24).
        """
        encoding = np.zeros((3, self.NUM_CARDS), dtype=np.float32)
        for i, card in enumerate(trick_cards[:3]):
            encoding[i] = self.encode_card(card)
        return encoding

    def encode_played_cards(self, played_cards: List[Card]) -> np.ndarray:
        """
        Encode cards that have been played in previous tricks.

        This is a binary vector indicating which cards have been played
        in previous tricks (not including current trick or player's hand).

        Parameters
        ----------
        played_cards : List[Card]
            Cards that have been played in previous tricks.

        Returns
        -------
        np.ndarray
            Binary vector of shape (24,) where 1 indicates the card was played.
        """
        encoding = np.zeros(self.NUM_CARDS, dtype=np.float32)
        for card in played_cards:
            if card in self.card_to_index:
                encoding[self.card_to_index[card]] = 1.0
        return encoding

    def encode_turned_card(self, turned_card: Optional[Card]) -> np.ndarray:
        """
        Encode the turned card.

        Parameters
        ----------
        turned_card : Optional[Card]
            The turned card, or None.

        Returns
        -------
        np.ndarray
            One-hot encoded vector of shape (24,).
        """
        if turned_card is None:
            return np.zeros(self.NUM_CARDS, dtype=np.float32)
        return self.encode_card(turned_card)

    def encode_suit(self, suit: Optional[Suit]) -> np.ndarray:
        """
        Encode a suit as a one-hot vector.

        Parameters
        ----------
        suit : Optional[Suit]
            The suit to encode, or None.

        Returns
        -------
        np.ndarray
            One-hot encoded vector of shape (4,).
        """
        encoding = np.zeros(4, dtype=np.float32)
        if suit is not None:
            suit_index = list(Suit).index(suit)
            encoding[suit_index] = 1.0
        return encoding

    def encode_positional_features(
        self,
        player_id: int,
        dealer_id: int,
        team: int,
        trick_number: int,
        tricks_won_team0: int,
        tricks_won_team1: int,
    ) -> np.ndarray:
        """
        Encode positional and game state features.

        Parameters
        ----------
        player_id : int
            ID of the current player (0-3).
        dealer_id : int
            ID of the dealer (0-3).
        team : int
            Team of the current player (0 or 1).
        trick_number : int
            Current trick number (0-4).
        tricks_won_team0 : int
            Tricks won by team 0 so far.
        tricks_won_team1 : int
            Tricks won by team 1 so far.

        Returns
        -------
        np.ndarray
            Positional features vector.
        """
        features = np.zeros(12, dtype=np.float32)

        # Player ID (one-hot)
        features[player_id] = 1.0

        # Dealer ID (one-hot)
        features[4 + dealer_id] = 1.0

        # Team (binary)
        features[8] = float(team)

        # Trick number (normalized)
        features[9] = float(trick_number) / 4.0

        # Tricks won (normalized)
        features[10] = float(tricks_won_team0) / 5.0
        features[11] = float(tricks_won_team1) / 5.0

        return features

    def encode_game_state(
        self,
        hand: List[Card],
        turned_card: Optional[Card],
        trump_suit: Optional[Suit],
        led_suit: Optional[Suit],
        trick_cards: List[Card],
        player_id: int,
        dealer_id: int,
        team: int,
        trick_number: int,
        tricks_won_team0: int,
        tricks_won_team1: int,
        played_cards: Optional[List[Card]] = None,
    ) -> torch.Tensor:
        """
        Encode full game state into a feature tensor.

        Parameters
        ----------
        hand : List[Card]
            Player's current hand (only cards the player can see).
        turned_card : Optional[Card]
            The turned card, if applicable.
        trump_suit : Optional[Suit]
            Current trump suit, if determined.
        led_suit : Optional[Suit]
            Suit that was led in current trick, if any.
        trick_cards : List[Card]
            Cards played in current trick.
        player_id : int
            ID of the current player.
        dealer_id : int
            ID of the dealer.
        team : int
            Team of the current player.
        trick_number : int
            Current trick number.
        tricks_won_team0 : int
            Tricks won by team 0.
        tricks_won_team1 : int
            Tricks won by team 1.
        played_cards : Optional[List[Card]]
            Cards that have been played in previous tricks (excluding current trick
            and player's hand). If None, defaults to empty list.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (feature_size,).
        """
        # Encode hand (5 * 24 = 120 features)
        # Note: Only includes player's own hand - player cannot see other players' hands
        hand_encoding = self.encode_hand(hand).flatten()

        # Encode turned card (24 features)
        turned_encoding = self.encode_turned_card(turned_card)

        # Encode trump suit (4 features)
        trump_encoding = self.encode_suit(trump_suit)

        # Encode led suit (4 features)
        led_encoding = self.encode_suit(led_suit)

        # Encode trick cards (3 * 24 = 72 features)
        trick_encoding = self.encode_trick_cards(trick_cards).flatten()

        # Encode played cards from previous tricks (24 features)
        # Binary vector indicating which cards were played in previous tricks
        played_encoding = self.encode_played_cards(played_cards or [])

        # Positional features (12 features)
        positional = self.encode_positional_features(
            player_id, dealer_id, team, trick_number, tricks_won_team0, tricks_won_team1
        )

        # Concatenate all features
        feature_vector = np.concatenate([
            hand_encoding,      # 120
            turned_encoding,   # 24
            trump_encoding,    # 4
            led_encoding,     # 4
            trick_encoding,    # 72
            played_encoding,   # 24
            positional,        # 12
        ])

        # Total: 260 features (was 236, added 24 for played cards)
        return torch.tensor(feature_vector, dtype=torch.float32)

    def decode_card_index(self, index: int) -> Optional[Card]:
        """
        Decode a card index back to a Card.

        Parameters
        ----------
        index : int
            Card index (0-23).

        Returns
        -------
        Optional[Card]
            The card, or None if index is invalid.
        """
        return self.index_to_card.get(index)

    def get_valid_card_indices(self, hand: List[Card]) -> List[int]:
        """
        Get valid card indices for a hand.

        Parameters
        ----------
        hand : List[Card]
            The player's hand.

        Returns
        -------
        List[int]
            List of valid card indices.
        """
        return [self.card_to_index[card] for card in hand if card in self.card_to_index]

