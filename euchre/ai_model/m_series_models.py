"""
M-Series PyTorch AI Models for Euchre

This module contains brand new, from-scratch PyTorch neural network models
designed specifically for Euchre gameplay. These models take into account:
- Dealer status (self, partner, or opposing team)
- All cards in hand with comprehensive encoding
- Trump selection decisions
- Card play decisions based on game context
- Partner coordination and team strategy

The models are designed to be trained on thousands of games to achieve
optimal weights for the best possible Euchre players.

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-12
Context: Creating brand new PyTorch AI models for Euchre players
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass

from ..models import Card, Suit, Rank, Player, PlayerType


@dataclass
class MSeriesRiskProfile:
    """Risk profile for M-Series AI models with Euchre-specific parameters."""
    
    # Trump calling parameters
    trump_calling_aggression: float = 0.5  # How aggressively to call trump
    partner_dealer_bonus: float = 0.2      # Bonus when partner is dealer
    ace_ordering_threshold: float = 0.6    # Threshold for ordering up aces
    
    # Card play parameters
    leading_aggression: float = 0.5        # How aggressively to lead
    trump_usage_strategy: float = 0.5      # When to use trump cards
    partner_coordination: float = 0.7      # How much to coordinate with partner
    
    # Game state adaptation
    score_adaptation: float = 0.6          # How much to adapt based on score
    set_avoidance: float = 0.8             # How much to avoid being set
    
    def to(self, device: torch.device) -> 'MSeriesRiskProfile':
        """Move risk profile to device."""
        return MSeriesRiskProfile(
            trump_calling_aggression=self.trump_calling_aggression,
            partner_dealer_bonus=self.partner_dealer_bonus,
            ace_ordering_threshold=self.ace_ordering_threshold,
            leading_aggression=self.leading_aggression,
            trump_usage_strategy=self.trump_usage_strategy,
            partner_coordination=self.partner_coordination,
            score_adaptation=self.score_adaptation,
            set_avoidance=self.set_avoidance
        )
    
    def get_risk_vector(self) -> torch.Tensor:
        """Get risk parameters as a tensor."""
        return torch.tensor([
            self.trump_calling_aggression,
            self.partner_dealer_bonus,
            self.ace_ordering_threshold,
            self.leading_aggression,
            self.trump_usage_strategy,
            self.partner_coordination,
            self.score_adaptation,
            self.set_avoidance
        ], dtype=torch.float32)


class MSeriesGameStateEncoder:
    """Advanced game state encoder for M-Series models with comprehensive Euchre understanding."""
    
    def __init__(self, input_size: int = 256):
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
        self.card_features = self.num_suits * self.num_ranks + 3  # 24 + trump + strength + bower
        self.hand_features = self.hand_size * self.card_features  # 135
        self.game_context_features = 32  # dealer, partner, position, score, etc.
        self.trick_context_features = 20  # current trick state
        self.historical_features = 20     # game history and patterns
        
        # Total features should match input_size
        total_features = (self.hand_features + self.game_context_features + 
                         self.trick_context_features + self.historical_features)
        assert total_features <= input_size, f"Input size {input_size} too small for {total_features} features"
        
    def encode_game_state_for_trump_decision(self, 
                                           player: Player, 
                                           top_card: Card,
                                           dealer: Player,
                                           team_scores: Dict[str, int],
                                           round_number: int) -> torch.Tensor:
        """Encode game state for trump calling decisions.
        
        Parameters
        ----------
        player : Player
            The player making the decision
        top_card : Card
            The top card that can be ordered up
        dealer : Player
            The current dealer
        team_scores : Dict[str, int]
            Current team scores
        round_number : int
            Current round number
            
        Returns
        -------
        torch.Tensor
            Encoded game state tensor
        """
        features = []
        
        # Encode hand cards with enhanced features
        hand_encoding = self._encode_hand_with_context(player.hand, top_card.suit)
        features.extend(hand_encoding)
        
        # Encode top card
        top_card_encoding = self._encode_card_with_context(top_card, top_card.suit)
        features.extend(top_card_encoding)
        
        # Encode dealer context
        dealer_encoding = self._encode_dealer_context(player, dealer)
        features.extend(dealer_encoding)
        
        # Encode game context
        game_context = self._encode_game_context(team_scores, round_number)
        features.extend(game_context)
        
        # Encode historical patterns
        historical = self._encode_historical_context(player, round_number)
        features.extend(historical)
        
        # Encode partner coordination signals
        partner_signals = self._encode_partner_coordination(player, round_number)
        features.extend(partner_signals)
        
        # Encode advanced card counting
        card_counting = self._encode_advanced_card_counting(player, round_number)
        features.extend(card_counting)
        
        # Encode strategic game state
        strategic_state = self._encode_strategic_game_state(player, team_scores, round_number)
        features.extend(strategic_state)
        
        # Pad to input_size if necessary
        if len(features) < self.input_size:
            features.extend([0.0] * (self.input_size - len(features)))
        
        return torch.tensor(features, dtype=torch.float32)
    
    def encode_game_state_for_card_play(self,
                                       player: Player,
                                       lead_suit: Optional[Suit],
                                       trump_suit: Optional[Suit],
                                       current_trick: List[Tuple[str, Card]],
                                       team_scores: Dict[str, int]) -> torch.Tensor:
        """Encode game state for card play decisions.
        
        Parameters
        ----------
        player : Player
            The player making the decision
        lead_suit : Optional[Suit]
            The lead suit of the current trick
        trump_suit : Optional[Suit]
            The current trump suit
        current_trick : List[Tuple[str, Card]]
            Cards played in current trick
        team_scores : Dict[str, int]
            Current team scores
            
        Returns
        -------
        torch.Tensor
            Encoded game state tensor
        """
        features = []
        
        # Encode hand cards
        hand_encoding = self._encode_hand_with_context(player.hand, trump_suit)
        features.extend(hand_encoding)
        
        # Encode current trick
        trick_encoding = self._encode_trick_context(current_trick, lead_suit, trump_suit)
        features.extend(trick_encoding)
        
        # Encode game context
        game_context = self._encode_game_context(team_scores, 1)  # Current round
        features.extend(game_context)
        
        # Encode player position and strategy
        position_encoding = self._encode_position_context(player, current_trick)
        features.extend(position_encoding)
        
        # Encode advanced card play strategy
        card_strategy = self._encode_card_play_strategy(player, current_trick, lead_suit, trump_suit)
        features.extend(card_strategy)
        
        # Encode partner coordination for card play
        partner_card_coordination = self._encode_partner_card_coordination(player, current_trick)
        features.extend(partner_card_coordination)
        
        # Pad to input_size if necessary
        if len(features) < self.input_size:
            features.extend([0.0] * (self.input_size - len(features)))
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _encode_hand_with_context(self, hand: List[Card], trump_suit: Optional[Suit]) -> List[float]:
        """Encode hand with comprehensive context including trump and bower status."""
        features = []
        
        for card in hand:
            card_features = self._encode_card_with_context(card, trump_suit)
            features.extend(card_features)
        
        # Add hand-level features
        hand_strength = self._calculate_hand_strength(hand, trump_suit)
        features.append(hand_strength)
        
        # Add trump count
        trump_count = sum(1 for card in hand if card.is_trump or 
                         (trump_suit and card.suit == trump_suit))
        features.append(trump_count / 5.0)  # Normalize
        
        # Add suit distribution
        suit_counts = [0] * 4
        for card in hand:
            suit_index = list(Suit).index(card.suit)
            suit_counts[suit_index] += 1
        features.extend([count / 5.0 for count in suit_counts])  # Normalize
        
        return features
    
    def _encode_card_with_context(self, card: Card, trump_suit: Optional[Suit]) -> List[float]:
        """Encode a single card with comprehensive context."""
        features = []
        
        # One-hot encoding for suit
        suit_encoding = [0] * self.num_suits
        suit_index = list(Suit).index(card.suit)
        suit_encoding[suit_index] = 1
        features.extend(suit_encoding)
        
        # One-hot encoding for rank
        rank_encoding = [0] * self.num_ranks
        rank_index = card.rank.value - 9  # 9=0, 10=1, J=2, Q=3, K=4, A=5
        rank_encoding[rank_index] = 1
        features.extend(rank_encoding)
        
        # Trump indicator
        is_trump = card.is_trump or (trump_suit and card.suit == trump_suit)
        features.append(1.0 if is_trump else 0.0)
        
        # Card strength (normalized)
        features.append((card.rank.value - 9) / 5.0)
        
        # Bower status
        if trump_suit:
            is_right_bower = (card.rank == Rank.JACK and card.suit == trump_suit)
            is_left_bower = (card.rank == Rank.JACK and 
                           card.suit == self._get_left_bower_suit(trump_suit))
            features.append(1.0 if is_right_bower else 0.0)
            features.append(1.0 if is_left_bower else 0.0)
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _encode_partner_coordination(self, player: Player, round_number: int) -> List[float]:
        """Encode partner coordination signals and intuition."""
        features = []
        
        # Partner's previous plays and signals
        if hasattr(player, 'game_history') and player.game_history:
            partner_plays = [play for play in player.game_history if play['player'] != player.name]
            
            # Analyze partner's trump calling patterns
            trump_calls = [play for play in partner_plays if play['action'] == 'trump_call']
            features.extend([
                float(len(trump_calls)),  # Number of trump calls by partner
                float(1.0 if any(play['suit'] == 'hearts' for play in trump_calls) else 0.0),  # Partner likes hearts
                float(1.0 if any(play['suit'] == 'diamonds' for play in trump_calls) else 0.0),  # Partner likes diamonds
                float(1.0 if any(play['suit'] == 'clubs' for play in trump_calls) else 0.0),  # Partner likes clubs
                float(1.0 if any(play['suit'] == 'spades' for play in trump_calls) else 0.0),  # Partner likes spades
            ])
            
            # Partner's card playing patterns
            card_plays = [play for play in partner_plays if play['action'] == 'card_play']
            if card_plays:
                # Partner's tendency to lead with high cards
                high_card_leads = [play for play in card_plays if play['is_lead'] and play['card_rank'] >= 12]
                features.extend([
                    float(len(high_card_leads) / len(card_plays)),  # High card lead ratio
                    float(1.0 if any(play['card_suit'] == 'hearts' for play in card_plays) else 0.0),  # Partner plays hearts
                    float(1.0 if any(play['card_suit'] == 'diamonds' for play in card_plays) else 0.0),  # Partner plays diamonds
                    float(1.0 if any(play['card_suit'] == 'clubs' for play in card_plays) else 0.0),  # Partner plays clubs
                    float(1.0 if any(play['card_suit'] == 'spades' for play in card_plays) else 0.0),  # Partner plays spades
                ])
            else:
                features.extend([0.0] * 5)
        else:
            features.extend([0.0] * 10)  # No history yet
        
        return features
    
    def _encode_advanced_card_counting(self, player: Player, round_number: int) -> List[float]:
        """Encode advanced card counting and memory features."""
        features = []
        
        # Track cards that have been played
        if hasattr(player, 'game_history') and player.game_history:
            played_cards = [play['card'] for play in player.game_history if 'card' in play]
            
            # Count remaining cards by suit
            remaining_hearts = 6 - len([c for c in played_cards if c.suit == Suit.HEARTS])
            remaining_diamonds = 6 - len([c for c in played_cards if c.suit == Suit.DIAMONDS])
            remaining_clubs = 6 - len([c for c in played_cards if c.suit == Suit.CLUBS])
            remaining_spades = 6 - len([c for c in played_cards if c.suit == Suit.SPADES])
            
            features.extend([
                float(remaining_hearts) / 6.0,    # Remaining hearts ratio
                float(remaining_diamonds) / 6.0,  # Remaining diamonds ratio
                float(remaining_clubs) / 6.0,     # Remaining clubs ratio
                float(remaining_spades) / 6.0,    # Remaining spades ratio
            ])
            
            # Track high cards remaining
            high_cards_remaining = 0
            for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
                for rank in [Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK]:
                    card = Card(rank, suit)
                    if card not in played_cards:
                        high_cards_remaining += 1
            
            features.extend([
                float(high_cards_remaining) / 16.0,  # High cards remaining ratio
                float(1.0 if remaining_hearts <= 2 else 0.0),  # Hearts running low
                float(1.0 if remaining_diamonds <= 2 else 0.0),  # Diamonds running low
                float(1.0 if remaining_clubs <= 2 else 0.0),  # Clubs running low
                float(1.0 if remaining_spades <= 2 else 0.0),  # Spades running low
            ])
        else:
            features.extend([1.0] * 4 + [1.0] * 5)  # All cards available initially
        
        return features
    
    def _encode_strategic_game_state(self, player: Player, team_scores: Dict[str, int], round_number: int) -> List[float]:
        """Encode strategic game state and decision context."""
        features = []
        
        # Team score analysis
        team1_score = team_scores.get("Team 1", 0)
        team2_score = team_scores.get("Team 2", 0)
        
        # Determine which team the player is on
        if player.name in ["Magnus", "Mentor"]:
            player_team_score = team1_score
            opponent_team_score = team2_score
        else:
            player_team_score = team2_score
            opponent_team_score = team1_score
        
        features.extend([
            float(player_team_score),           # Player's team score
            float(opponent_team_score),        # Opponent's team score
            float(player_team_score - opponent_team_score),  # Score difference
            float(1.0 if player_team_score >= 8 else 0.0),  # Close to winning
            float(1.0 if opponent_team_score >= 8 else 0.0),  # Opponent close to winning
        ])
        
        # Round-based strategy
        features.extend([
            float(round_number),               # Current round
            float(1.0 if round_number <= 2 else 0.0),  # Early game
            float(1.0 if 3 <= round_number <= 4 else 0.0),  # Mid game
            float(1.0 if round_number >= 5 else 0.0),  # Late game
        ])
        
        return features
    
    def _encode_dealer_context(self, player: Player, dealer: Player) -> List[float]:
        """Encode dealer context including partner relationships."""
        features = []
        
        # Dealer position (one-hot)
        dealer_encoding = [0] * 4
        dealer_index = self._get_player_index(dealer)
        dealer_encoding[dealer_index] = 1
        features.extend(dealer_encoding)
        
        # Player position (one-hot)
        player_encoding = [0] * 4
        player_index = self._get_player_index(player)
        player_encoding[player_index] = 1
        features.extend(player_encoding)
        
        # Partner relationship
        is_partner_dealing = self._are_partners(player, dealer)
        features.append(1.0 if is_partner_dealing else 0.0)
        
        # Opponent dealing
        is_opponent_dealing = not is_partner_dealing and player.name != dealer.name
        features.append(1.0 if is_opponent_dealing else 0.0)
        
        # Self dealing
        is_self_dealing = player.name == dealer.name
        features.append(1.0 if is_self_dealing else 0.0)
        
        return features
    
    def _encode_game_context(self, team_scores: Dict[str, int], round_number: int) -> List[float]:
        """Encode general game context."""
        features = []
        
        # Team scores (normalized)
        team1_score = team_scores.get("Team 1", 0)
        team2_score = team_scores.get("Team 2", 0)
        features.append(team1_score / 10.0)  # Normalize to [0, 1]
        features.append(team2_score / 10.0)
        
        # Score difference
        score_diff = (team1_score - team2_score) / 10.0
        features.append(score_diff)
        
        # Round number (normalized)
        features.append(round_number / 10.0)
        
        # Game phase (early/middle/late)
        if round_number <= 3:
            game_phase = 0.0  # Early
        elif round_number <= 6:
            game_phase = 0.5  # Middle
        else:
            game_phase = 1.0  # Late
        features.append(game_phase)
        
        return features
    
    def _encode_trick_context(self, current_trick: List[Tuple[str, Card]], 
                             lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> List[float]:
        """Encode current trick context."""
        features = []
        
        # Number of cards played
        features.append(len(current_trick) / 4.0)
        
        # Lead suit encoding
        lead_suit_encoding = [0] * 4
        if lead_suit:
            lead_index = list(Suit).index(lead_suit)
            lead_suit_encoding[lead_index] = 1
        features.extend(lead_suit_encoding)
        
        # Trump suit encoding
        trump_suit_encoding = [0] * 4
        if trump_suit:
            trump_index = list(Suit).index(trump_suit)
            trump_suit_encoding[trump_index] = 1
        features.extend(trump_suit_encoding)
        
        # Current winning card strength
        if current_trick:
            winning_card = max(current_trick, key=lambda x: self._get_card_value(x[1], trump_suit))[1]
            winning_strength = self._get_card_value(winning_card, trump_suit) / 20.0  # Normalize
            features.append(winning_strength)
        else:
            features.append(0.0)
        
        return features
    
    def _encode_card_play_strategy(self, player: Player, current_trick: List[Tuple[str, Card]], 
                                  lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> List[float]:
        """Encode advanced card play strategy and decision making."""
        features = []
        
        # Analyze current trick strength
        if current_trick:
            trick_cards = [card for _, card in current_trick]
            trick_suits = [card.suit for card in trick_cards]
            
            # Determine if we can win the trick
            can_win = False
            if lead_suit:
                # Check if we have a higher card of the lead suit
                lead_suit_cards = [card for card in player.hand if card.suit == lead_suit]
                if lead_suit_cards:
                    highest_lead = max(lead_suit_cards, key=lambda c: c.rank.value)
                    highest_trick = max([c for c in trick_cards if c.suit == lead_suit], key=lambda c: c.rank.value)
                    can_win = highest_lead.rank.value > highest_trick.rank.value
            
            # Check if we can trump to win
            can_trump_win = False
            if trump_suit and lead_suit != trump_suit:
                trump_cards = [card for card in player.hand if card.suit == trump_suit]
                if trump_cards:
                    can_trump_win = True
            
            features.extend([
                float(1.0 if can_win else 0.0),      # Can win with lead suit
                float(1.0 if can_trump_win else 0.0), # Can trump to win
                float(len(current_trick) / 4.0),     # Trick progress
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Strategic card counting
        if trump_suit:
            trump_cards_in_hand = sum(1 for card in player.hand if card.suit == trump_suit)
            features.extend([
                float(trump_cards_in_hand / 5.0),    # Trump cards in hand ratio
                float(1.0 if trump_cards_in_hand >= 3 else 0.0),  # Strong trump hand
                float(1.0 if trump_cards_in_hand <= 1 else 0.0),  # Weak trump hand
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return features
    
    def _encode_partner_card_coordination(self, player: Player, current_trick: List[Tuple[str, Card]]) -> List[float]:
        """Encode partner coordination signals during card play."""
        features = []
        
        # Analyze partner's card in current trick
        partner_card = None
        for player_name, card in current_trick:
            if player_name != player.name and self._is_partner(player_name, player.name):
                partner_card = card
                break
        
        if partner_card:
            # Partner's card strength and strategy
            features.extend([
                float(partner_card.rank.value - 9) / 5.0,  # Partner's card strength
                float(1.0 if partner_card.rank.value >= 12 else 0.0),  # Partner played high card
                float(1.0 if partner_card.rank.value <= 10 else 0.0),  # Partner played low card
                float(1.0 if partner_card.suit == Suit.HEARTS else 0.0),  # Partner played hearts
                float(1.0 if partner_card.suit == Suit.DIAMONDS else 0.0),  # Partner played diamonds
                float(1.0 if partner_card.suit == Suit.CLUBS else 0.0),  # Partner played clubs
                float(1.0 if partner_card.suit == Suit.SPADES else 0.0),  # Partner played spades
            ])
        else:
            features.extend([0.0] * 7)
        
        return features
    
    def _is_partner(self, player1_name: str, player2_name: str) -> bool:
        """Check if two players are partners."""
        team1 = ["Magnus", "Mentor"]
        team2 = ["Maverick", "Mystic"]
        return (player1_name in team1 and player2_name in team1) or \
               (player1_name in team2 and player2_name in team2)
    
    def _encode_position_context(self, player: Player, current_trick: List[Tuple[str, Card]]) -> List[float]:
        """Encode player position context."""
        features = []
        
        # Position in trick (0=lead, 1=second, 2=third, 3=last)
        position = len(current_trick)
        position_encoding = [0] * 4
        position_encoding[position] = 1
        features.extend(position_encoding)
        
        # Is leading
        features.append(1.0 if position == 0 else 0.0)
        
        # Is following
        features.append(1.0 if position > 0 else 0.0)
        
        return features
    
    def _encode_historical_context(self, player: Player, round_number: int) -> List[float]:
        """Encode historical context and patterns."""
        features = []
        
        # Player's tricks won this round (normalized)
        tricks_won = player.tricks_won
        features.append(tricks_won / 5.0)
        
        # Player's score (normalized)
        features.append(player.score / 10.0)
        
        # Round performance (placeholder for future enhancement)
        features.extend([0.0] * 18)  # Placeholder for historical patterns
        
        return features
    
    def _get_left_bower_suit(self, trump_suit: Suit) -> Suit:
        """Get the left bower suit for a given trump suit."""
        if trump_suit == Suit.HEARTS:
            return Suit.DIAMONDS
        elif trump_suit == Suit.DIAMONDS:
            return Suit.HEARTS
        elif trump_suit == Suit.CLUBS:
            return Suit.SPADES
        else:  # SPADES
            return Suit.CLUBS
    
    def _get_player_index(self, player: Player) -> int:
        """Get player index (0-3) based on name."""
        # This is a simplified mapping - in practice you'd want a more robust system
        name_to_index = {"Alice": 0, "Bob": 1, "Charlie": 2, "David": 3}
        return name_to_index.get(player.name, 0)
    
    def _are_partners(self, player1: Player, player2: Player) -> bool:
        """Check if two players are partners."""
        # Partners are players 0&2 and 1&3
        indices = [self._get_player_index(player1), self._get_player_index(player2)]
        return (indices[0] % 2 == 0 and indices[1] % 2 == 0) or \
               (indices[0] % 2 == 1 and indices[1] % 2 == 1)
    
    def _calculate_hand_strength(self, hand: List[Card], trump_suit: Optional[Suit]) -> float:
        """Calculate overall hand strength."""
        if not trump_suit:
            return sum(card.rank.value for card in hand) / 70.0  # Normalize
        
        strength = 0.0
        for card in hand:
            if card.is_trump or card.suit == trump_suit:
                # Trump cards get bonus
                if card.rank == Rank.JACK:
                    if card.suit == trump_suit:
                        strength += 20.0  # Right bower
                    else:
                        strength += 19.0  # Left bower
                else:
                    strength += card.rank.value + 5.0  # Trump bonus
            else:
                strength += card.rank.value
        
        return strength / 100.0  # Normalize
    
    def _get_card_value(self, card: Card, trump_suit: Optional[Suit]) -> float:
        """Get card value for trick comparison."""
        if not trump_suit:
            return card.rank.value
        
        if card.is_trump or card.suit == trump_suit:
            if card.rank == Rank.JACK:
                if card.suit == trump_suit:
                    return 20.0  # Right bower
                else:
                    return 19.0  # Left bower
            else:
                return card.rank.value + 5.0  # Trump bonus
        else:
            return card.rank.value


class MSeriesBaseModel(nn.Module):
    """Base class for M-Series Euchre AI models."""
    
    def __init__(self, input_size: int = 256, hidden_size: int = 512, 
                 risk_embedding_size: int = 64):
        """Initialize the base model.
        
        Parameters
        ----------
        input_size : int
            Size of input feature vector
        hidden_size : int
            Size of hidden layers
        risk_embedding_size : int
            Size of risk parameter embeddings
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.risk_embedding_size = risk_embedding_size
        
        # Risk parameter embedding
        self.risk_embedding = nn.Linear(8, risk_embedding_size)
        
        # Enhanced input processing with specialized layers
        self.input_layer = nn.Linear(input_size + risk_embedding_size, hidden_size)
        
        # Partner coordination and intuition layers
        self.partner_intuition_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.2)
        )
        
        # Advanced card counting and memory layers
        self.card_memory_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.2)
        )
        
        # Strategic planning and game theory layers
        self.strategic_planning_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.2)
        )
        
        # Trick analysis and pattern recognition layers
        self.trick_analysis_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.2)
        )
        
        # Fusion layer to combine all specialized features
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size // 2 * 4, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2)
        )
        
        # Enhanced hidden layers with residual connections
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size // 2)
        ])
        
        # Output heads with strategic context
        self.trump_decision_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 2)  # Order up or not
        )
        
        self.card_selection_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 5)  # 5 cards in hand
        )
        
        self.suit_selection_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 4)  # 4 suits for trump calling
        )
        
        # Regularization and normalization
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x: torch.Tensor, risk_params: MSeriesRiskProfile) -> Dict[str, torch.Tensor]:
        """Forward pass with risk parameters.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features [batch_size, input_size]
        risk_params : MSeriesRiskProfile
            Risk parameters for this forward pass
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing various outputs
        """
        batch_size = x.size(0)
        
        # Get risk embedding
        risk_vector = risk_params.get_risk_vector().unsqueeze(0).expand(batch_size, -1)
        risk_embedded = self.risk_embedding(risk_vector)
        
        # Combine input with risk embedding
        combined_input = torch.cat([x, risk_embedded], dim=1)
        
        # Enhanced forward pass through specialized layers
        h = F.relu(self.input_layer(combined_input))
        h = self.batch_norm(h)
        h = self.dropout(h)
        
        # Process through specialized intuition and strategy layers
        partner_features = self.partner_intuition_layer(h)
        card_memory_features = self.card_memory_layer(h)
        strategic_features = self.strategic_planning_layer(h)
        trick_features = self.trick_analysis_layer(h)
        
        # Fusion of all specialized features
        combined_features = torch.cat([
            partner_features, card_memory_features, 
            strategic_features, trick_features
        ], dim=1)
        
        h = self.feature_fusion(combined_features)
        h = self.layer_norm(h)
        h = self.dropout(h)
        
        # Process through enhanced hidden layers
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
            h = self.dropout(h)
        
        # Output heads
        trump_logits = self.trump_decision_head(h)
        card_logits = self.card_selection_head(h)
        suit_logits = self.suit_selection_head(h)
        
        return {
            'trump_decision': trump_logits,
            'card_selection': card_logits,
            'suit_selection': suit_logits,
            'hidden_features': h
        }
    
    def get_trump_decision_probs(self, x: torch.Tensor, risk_params: MSeriesRiskProfile) -> torch.Tensor:
        """Get trump decision probabilities."""
        outputs = self.forward(x, risk_params)
        return F.softmax(outputs['trump_decision'], dim=1)
    
    def get_card_selection_probs(self, x: torch.Tensor, risk_params: MSeriesRiskProfile) -> torch.Tensor:
        """Get card selection probabilities."""
        outputs = self.forward(x, risk_params)
        return F.softmax(outputs['card_selection'], dim=1)
    
    def get_suit_selection_probs(self, x: torch.Tensor, risk_params: MSeriesRiskProfile) -> torch.Tensor:
        """Get suit selection probabilities for trump calling."""
        outputs = self.forward(x, risk_params)
        return F.softmax(outputs['suit_selection'], dim=1)


class MagnusModel(MSeriesBaseModel):
    """Magnus - The Strategic Mastermind
    
    Magnus is designed to be the most strategic and analytical player,
    with deep understanding of partner coordination and long-term planning.
    """
    
    def __init__(self, input_size: int = 256, hidden_size: int = 512, 
                 risk_embedding_size: int = 64):
        super().__init__(input_size, hidden_size, risk_embedding_size)
        
        # Additional strategic layers
        self.strategy_encoder = nn.Linear(hidden_size // 2, 128)
        self.partner_coordination = nn.Linear(128, 64)
        self.long_term_planning = nn.Linear(64, 32)
        
        # Strategic output heads
        self.strategic_trump_head = nn.Linear(32, 2)
        self.strategic_card_head = nn.Linear(32, 5)
        
    def forward(self, x: torch.Tensor, risk_params: MSeriesRiskProfile) -> Dict[str, torch.Tensor]:
        """Forward pass with strategic enhancements."""
        base_outputs = super().forward(x, risk_params)
        
        # Strategic processing
        h = base_outputs['hidden_features']
        strategy = F.relu(self.strategy_encoder(h))
        partner_coord = F.relu(self.partner_coordination(strategy))
        long_term = F.relu(self.long_term_planning(partner_coord))
        
        # Strategic outputs
        strategic_trump = self.strategic_trump_head(long_term)
        strategic_card = self.strategic_card_head(long_term)
        
        base_outputs.update({
            'strategic_trump': strategic_trump,
            'strategic_card': strategic_card
        })
        
        return base_outputs


class MaverickModel(MSeriesBaseModel):
    """Maverick - The Aggressive Risk-Taker
    
    Maverick is designed to be bold and unpredictable, taking calculated risks
    and using aggressive strategies to dominate the game.
    """
    
    def __init__(self, input_size: int = 256, hidden_size: int = 512, 
                 risk_embedding_size: int = 64):
        super().__init__(input_size, hidden_size, risk_embedding_size)
        
        # Aggressive enhancement layers
        self.aggression_encoder = nn.Linear(hidden_size // 2, 128)
        self.risk_assessment = nn.Linear(128, 64)
        self.aggressive_strategy = nn.Linear(64, 32)
        
        # Aggressive output heads
        self.aggressive_trump_head = nn.Linear(32, 2)
        self.aggressive_card_head = nn.Linear(32, 5)
        
    def forward(self, x: torch.Tensor, risk_params: MSeriesRiskProfile) -> Dict[str, torch.Tensor]:
        """Forward pass with aggressive enhancements."""
        base_outputs = super().forward(x, risk_params)
        
        # Aggressive processing
        h = base_outputs['hidden_features']
        aggression = F.relu(self.aggression_encoder(h))
        risk_assess = F.relu(self.risk_assessment(aggression))
        aggressive_strat = F.relu(self.aggressive_strategy(risk_assess))
        
        # Aggressive outputs
        aggressive_trump = self.aggressive_trump_head(aggressive_strat)
        aggressive_card = self.aggressive_card_head(aggressive_strat)
        
        base_outputs.update({
            'aggressive_trump': aggressive_trump,
            'aggressive_card': aggressive_card
        })
        
        return base_outputs


class MentorModel(MSeriesBaseModel):
    """Mentor - The Balanced Teacher
    
    Mentor is designed to be the most balanced and adaptable player,
    learning from every game and teaching optimal strategies.
    """
    
    def __init__(self, input_size: int = 256, hidden_size: int = 512, 
                 risk_embedding_size: int = 64):
        super().__init__(input_size, hidden_size, risk_embedding_size)
        
        # Learning and adaptation layers
        self.learning_encoder = nn.Linear(hidden_size // 2, 128)
        self.adaptation_layer = nn.Linear(128, 64)
        self.balanced_strategy = nn.Linear(64, 32)
        
        # Learning output heads
        self.learning_trump_head = nn.Linear(32, 2)
        self.learning_card_head = nn.Linear(32, 5)
        
    def forward(self, x: torch.Tensor, risk_params: MSeriesRiskProfile) -> Dict[str, torch.Tensor]:
        """Forward pass with learning enhancements."""
        base_outputs = super().forward(x, risk_params)
        
        # Learning processing
        h = base_outputs['hidden_features']
        learning = F.relu(self.learning_encoder(h))
        adaptation = F.relu(self.adaptation_layer(learning))
        balanced_strat = F.relu(self.balanced_strategy(adaptation))
        
        # Learning outputs
        learning_trump = self.learning_trump_head(balanced_strat)
        learning_card = self.learning_card_head(balanced_strat)
        
        base_outputs.update({
            'learning_trump': learning_trump,
            'learning_card': learning_card
        })
        
        return base_outputs


class MysticModel(MSeriesBaseModel):
    """Mystic - The Intuitive Player
    
    Mystic is designed to have deep intuition about game patterns,
    using subtle cues and game flow to make optimal decisions.
    """
    
    def __init__(self, input_size: int = 256, hidden_size: int = 512, 
                 risk_embedding_size: int = 64):
        super().__init__(input_size, hidden_size, risk_embedding_size)
        
        # Intuition and pattern recognition layers
        self.intuition_encoder = nn.Linear(hidden_size // 2, 128)
        self.pattern_recognition = nn.Linear(128, 64)
        self.intuitive_strategy = nn.Linear(64, 32)
        
        # Intuition output heads
        self.intuitive_trump_head = nn.Linear(32, 2)
        self.intuitive_card_head = nn.Linear(32, 5)
        
    def forward(self, x: torch.Tensor, risk_params: MSeriesRiskProfile) -> Dict[str, torch.Tensor]:
        """Forward pass with intuitive enhancements."""
        base_outputs = super().forward(x, risk_params)
        
        # Intuition processing
        h = base_outputs['hidden_features']
        intuition = F.relu(self.intuition_encoder(h))
        pattern_rec = F.relu(self.pattern_recognition(intuition))
        intuitive_strat = F.relu(self.intuitive_strategy(pattern_rec))
        
        # Intuition outputs
        intuitive_trump = self.intuitive_trump_head(intuitive_strat)
        intuitive_card = self.intuitive_card_head(intuitive_strat)
        
        base_outputs.update({
            'intuitive_trump': intuitive_trump,
            'intuitive_card': intuitive_card
        })
        
        return base_outputs


def create_mseries_model(model_name: str, input_size: int = 256, hidden_size: int = 512, 
                        risk_embedding_size: int = 64) -> MSeriesBaseModel:
    """Create an M-Series model by name.
    
    Parameters
    ----------
    model_name : str
        Name of the model to create
    input_size : int
        Size of input feature vector
    hidden_size : int
        Size of hidden layers
    risk_embedding_size : int
        Size of risk parameter embeddings
        
    Returns
    -------
    MSeriesBaseModel
        The created model instance
    """
    model_name = model_name.lower()
    
    if model_name == "magnus":
        return MagnusModel(input_size, hidden_size, risk_embedding_size)
    elif model_name == "maverick":
        return MaverickModel(input_size, hidden_size, risk_embedding_size)
    elif model_name == "mentor":
        return MentorModel(input_size, hidden_size, risk_embedding_size)
    elif model_name == "mystic":
        return MysticModel(input_size, hidden_size, risk_embedding_size)
    else:
        raise ValueError(f"Unknown M-Series model: {model_name}")


def create_mseries_risk_profile(profile_name: str) -> MSeriesRiskProfile:
    """Create predefined risk profiles for M-Series models.
    
    Parameters
    ----------
    profile_name : str
        Name of the risk profile
        
    Returns
    -------
    MSeriesRiskProfile
        Configured risk profile
    """
    profiles = {
        'magnus': MSeriesRiskProfile(
            trump_calling_aggression=0.6,
            partner_dealer_bonus=0.3,
            ace_ordering_threshold=0.7,
            leading_aggression=0.5,
            trump_usage_strategy=0.6,
            partner_coordination=0.9,
            score_adaptation=0.8,
            set_avoidance=0.7
        ),
        'maverick': MSeriesRiskProfile(
            trump_calling_aggression=0.8,
            partner_dealer_bonus=0.1,
            ace_ordering_threshold=0.4,
            leading_aggression=0.9,
            trump_usage_strategy=0.8,
            partner_coordination=0.4,
            score_adaptation=0.3,
            set_avoidance=0.4
        ),
        'mentor': MSeriesRiskProfile(
            trump_calling_aggression=0.5,
            partner_dealer_bonus=0.2,
            ace_ordering_threshold=0.6,
            leading_aggression=0.5,
            trump_usage_strategy=0.5,
            partner_coordination=0.7,
            score_adaptation=0.6,
            set_avoidance=0.8
        ),
        'mystic': MSeriesRiskProfile(
            trump_calling_aggression=0.7,
            partner_dealer_bonus=0.2,
            ace_ordering_threshold=0.5,
            leading_aggression=0.6,
            trump_usage_strategy=0.7,
            partner_coordination=0.6,
            score_adaptation=0.7,
            set_avoidance=0.6
        )
    }
    
    if profile_name not in profiles:
        print(f"Warning: Unknown profile '{profile_name}', using 'mentor'")
        return profiles['mentor']
    
    return profiles[profile_name] 