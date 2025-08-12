"""Game state encoder for converting euchre game states to neural network inputs."""

import torch
import numpy as np
from typing import List, Optional, Tuple
from ..models import Card, Suit, Rank, Player, GameState


class GameStateEncoder:
    """Encoder for converting euchre game states to neural network input tensors."""
    
    def __init__(self, input_size: int = 128) -> None:
        """Initialize the encoder.
        
        Parameters
        ----------
        input_size : int
            Size of the output feature vector
        """
        self.input_size = input_size
        
        # Card encoding dimensions
        self.num_suits = 4
        self.num_ranks = 6
        self.hand_size = 5
        
        # Feature breakdown
        self.card_features = self.num_suits * self.num_ranks  # 24
        self.hand_features = self.hand_size * self.card_features  # 120
        self.game_state_features = 8  # trump, dealer, position, etc.
        
        # Total features should match input_size
        assert self.hand_features + self.game_state_features <= input_size
        
    def encode_game_state_for_ordering(self, 
                                     player: Player, 
                                     top_card: Card) -> torch.Tensor:
        """Encode game state for trump ordering decision.
        
        Parameters
        ----------
        player : Player
            The player making the decision
        top_card : Card
            The top card that could be ordered up
            
        Returns
        -------
        torch.Tensor
            Encoded game state tensor
        """
        features = []
        
        # Encode player's hand
        hand_features = self._encode_hand(player.hand)
        features.extend(hand_features)
        
        # Encode top card
        top_card_features = self._encode_card(top_card)
        features.extend(top_card_features)
        
        # Encode game context
        context_features = self._encode_ordering_context(player)
        features.extend(context_features)
        
        # Pad to input_size
        features = self._pad_features(features)
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def encode_game_state_for_card_selection(self,
                                           player: Player,
                                           lead_suit: Optional[Suit],
                                           trump_suit: Optional[Suit]) -> torch.Tensor:
        """Encode game state for card selection decision.
        
        Parameters
        ----------
        player : Player
            The player making the decision
        lead_suit : Optional[Suit]
            The suit that was led (if any)
        trump_suit : Optional[Suit]
            The current trump suit
            
        Returns
        -------
        torch.Tensor
            Encoded game state tensor
        """
        features = []
        
        # Encode player's hand
        hand_features = self._encode_hand(player.hand)
        features.extend(hand_features)
        
        # Encode lead suit
        lead_suit_features = self._encode_suit(lead_suit) if lead_suit else [0] * self.num_suits
        features.extend(lead_suit_features)
        
        # Encode trump suit
        trump_suit_features = self._encode_suit(trump_suit) if trump_suit else [0] * self.num_suits
        features.extend(trump_suit_features)
        
        # Encode game context
        context_features = self._encode_card_selection_context(player)
        features.extend(context_features)
        
        # Pad to input_size
        features = self._pad_features(features)
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def encode_game_state_for_trump_selection(self, hand: List[Card]) -> torch.Tensor:
        """Encode game state for trump selection decision.
        
        Parameters
        ----------
        hand : List[Card]
            The dealer's hand
            
        Returns
        -------
        torch.Tensor
            Encoded game state tensor
        """
        features = []
        
        # Encode hand
        hand_features = self._encode_hand(hand)
        features.extend(hand_features)
        
        # Encode hand statistics
        hand_stats = self._encode_hand_statistics(hand)
        features.extend(hand_stats)
        
        # Pad to input_size
        features = self._pad_features(features)
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _encode_hand(self, hand: List[Card]) -> List[float]:
        """Encode a hand of cards.
        
        Parameters
        ----------
        hand : List[Card]
            List of cards in the hand
            
        Returns
        -------
        List[float]
            Encoded hand features
        """
        features = []
        
        # Encode each card position
        for i in range(self.hand_size):
            if i < len(hand):
                card_features = self._encode_card(hand[i])
            else:
                # No card in this position
                card_features = [0] * self.card_features
            features.extend(card_features)
            
        return features
    
    def _encode_card(self, card: Card) -> List[float]:
        """Encode a single card.
        
        Parameters
        ----------
        card : Card
            The card to encode
            
        Returns
        -------
        List[float]
            Encoded card features
        """
        features = []
        
        # One-hot encoding for suit
        suit_encoding = [0] * self.num_suits
        suit_encoding[card.suit.value - 1] = 1  # Suit values are 1-4
        features.extend(suit_encoding)
        
        # One-hot encoding for rank
        rank_encoding = [0] * self.num_ranks
        rank_encoding[card.rank.value - 9] = 1  # Rank values are 9-14
        features.extend(rank_encoding)
        
        # Trump indicator
        features.append(1.0 if card.is_trump else 0.0)
        
        # Card strength (normalized rank value)
        features.append((card.rank.value - 9) / 5.0)  # Normalize to [0, 1]
        
        return features
    
    def _encode_suit(self, suit: Suit) -> List[float]:
        """Encode a suit.
        
        Parameters
        ----------
        suit : Suit
            The suit to encode
            
        Returns
        -------
        List[float]
            One-hot encoded suit
        """
        encoding = [0] * self.num_suits
        encoding[suit.value - 1] = 1
        return encoding
    
    def _encode_ordering_context(self, player: Player) -> List[float]:
        """Encode context for trump ordering decision.
        
        Parameters
        ----------
        player : Player
            The player making the decision
            
        Returns
        -------
        List[float]
            Context features
        """
        features = []
        
        # Player position (0-3)
        features.append(player.position / 3.0 if hasattr(player, 'position') else 0.0)
        
        # Is dealer
        features.append(1.0 if player.is_dealer else 0.0)
        
        # Hand strength (sum of card values)
        hand_strength = sum(card.rank.value for card in player.hand)
        features.append(hand_strength / 70.0)  # Max possible: 5 * 14 = 70
        
        # Number of trump cards
        trump_count = sum(1 for card in player.hand if card.is_trump)
        features.append(trump_count / 5.0)
        
        # Number of high cards (J, Q, K, A)
        high_card_count = sum(1 for card in player.hand if card.rank.value >= 11)
        features.append(high_card_count / 5.0)
        
        return features
    
    def _encode_card_selection_context(self, player: Player) -> List[float]:
        """Encode context for card selection decision.
        
        Parameters
        ----------
        player : Player
            The player making the decision
            
        Returns
        -------
        List[float]
            Context features
        """
        features = []
        
        # Player position
        features.append(player.position / 3.0 if hasattr(player, 'position') else 0.0)
        
        # Is dealer
        features.append(1.0 if player.is_dealer else 0.0)
        
        # Cards remaining in hand
        features.append(len(player.hand) / 5.0)
        
        # Trick number (if available)
        features.append(0.0)  # Placeholder for trick number
        
        return features
    
    def _encode_hand_statistics(self, hand: List[Card]) -> List[float]:
        """Encode hand statistics for trump selection.
        
        Parameters
        ----------
        hand : List[Card]
            The hand to analyze
            
        Returns
        -------
        List[float]
            Hand statistics features
        """
        features = []
        
        # Count cards by suit
        for suit in Suit:
            count = sum(1 for card in hand if card.suit == suit)
            features.append(count / 5.0)
        
        # Average card strength by suit
        for suit in Suit:
            suit_cards = [card for card in hand if card.suit == suit]
            if suit_cards:
                avg_strength = sum(card.rank.value for card in suit_cards) / len(suit_cards)
                features.append((avg_strength - 9) / 5.0)  # Normalize
            else:
                features.append(0.0)
        
        # Overall hand strength
        total_strength = sum(card.rank.value for card in hand)
        features.append(total_strength / 70.0)
        
        return features
    
    def _pad_features(self, features: List[float]) -> List[float]:
        """Pad features to the required input size.
        
        Parameters
        ----------
        features : List[float]
            Current feature list
            
        Returns
        -------
        List[float]
            Padded feature list
        """
        if len(features) < self.input_size:
            features.extend([0.0] * (self.input_size - len(features)))
        elif len(features) > self.input_size:
            features = features[:self.input_size]
            
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features for interpretability.
        
        Returns
        -------
        List[str]
            List of feature names
        """
        names = []
        
        # Hand features
        for i in range(self.hand_size):
            for suit in Suit:
                for rank in Rank:
                    names.append(f"hand_{i}_{suit.value}_{rank.value}")
            names.append(f"hand_{i}_trump")
            names.append(f"hand_{i}_strength")
        
        # Top card features (for ordering)
        for suit in Suit:
            names.append(f"top_card_suit_{suit.value}")
        for rank in Rank:
            names.append(f"top_card_rank_{rank.value}")
        names.append("top_card_trump")
        names.append("top_card_strength")
        
        # Context features
        names.extend([
            "player_position", "is_dealer", "hand_strength", 
            "trump_count", "high_card_count", "cards_remaining"
        ])
        
        # Suit features
        for suit in Suit:
            names.append(f"lead_suit_{suit.value}")
            names.append(f"trump_suit_{suit.value}")
        
        # Hand statistics
        for suit in Suit:
            names.append(f"suit_{suit.value}_count")
            names.append(f"suit_{suit.value}_avg_strength")
        names.append("total_hand_strength")
        
        # Pad to input_size
        while len(names) < self.input_size:
            names.append(f"padding_{len(names)}")
            
        return names[:self.input_size] 