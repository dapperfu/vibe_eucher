# Euchre AI Implementation Guide

This guide provides detailed technical information for implementing and using the two AI systems in the Euchre project.

## Table of Contents

1. [Rule-Based AI Implementation](#rule-based-ai-implementation)
2. [Neural Network AI Implementation](#neural-network-ai-implementation)
3. [Integration and Usage](#integration-and-usage)
4. [Performance Optimization](#performance-optimization)

## Rule-Based AI Implementation

### Class Hierarchy

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from ..models import Player, PlayerType, Card, Suit, Rank

class BaseAI(Player, ABC):
    """Abstract base class for all AI players."""
    
    def __init__(self, name: str, risk_ratio: float = 0.5) -> None:
        super().__init__(name, PlayerType.AI)
        self.risk_ratio = max(0.0, min(1.0, risk_ratio))
    
    @abstractmethod
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Decide whether to order up the top card."""
        pass
    
    @abstractmethod
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play in the current trick."""
        pass
```

### Implementing a Custom AI Profile

```python
class CustomAI(BaseAI):
    """Custom AI implementation with specific strategy."""
    
    def __init__(self, name: str, risk_ratio: float = 0.5, custom_param: float = 0.3) -> None:
        super().__init__(name, risk_ratio)
        self.custom_param = custom_param
    
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        # Custom trump calling logic
        hand_strength = self._evaluate_hand_for_trump(top_card.suit, top_card, is_partner_dealing)
        
        # Custom threshold calculation
        base_threshold = 12.0
        risk_adjustment = self.risk_ratio * 5.0
        custom_adjustment = self.custom_param * 3.0
        
        threshold = base_threshold - risk_adjustment + custom_adjustment
        return hand_strength >= threshold
    
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        if not lead_suit:
            # Leading - use custom strategy
            return self._custom_lead_strategy(trump_suit)
        elif self.has_suit(lead_suit):
            # Following suit - use custom follow strategy
            return self._custom_follow_strategy(lead_suit, trump_suit)
        else:
            # Cannot follow suit - use custom off-suit strategy
            return self._custom_off_suit_strategy(trump_suit)
    
    def _custom_lead_strategy(self, trump_suit: Optional[Suit]) -> Card:
        """Custom leading strategy."""
        # Implement your custom logic here
        pass
    
    def _custom_follow_strategy(self, lead_suit: Suit, trump_suit: Optional[Suit]) -> Card:
        """Custom follow suit strategy."""
        # Implement your custom logic here
        pass
    
    def _custom_off_suit_strategy(self, trump_suit: Optional[Suit]) -> Card:
        """Custom off-suit strategy."""
        # Implement your custom logic here
        pass
```

### Risk Parameter System

```python
class RiskProfile:
    """Advanced risk profile system for AI players."""
    
    def __init__(self, 
                 trump_calling: float = 0.5,
                 card_play: float = 0.5,
                 set_avoidance: float = 0.5,
                 partner_coordination: float = 0.5):
        self.trump_calling = max(0.0, min(1.0, trump_calling))
        self.card_play = max(0.0, min(1.0, card_play))
        self.set_avoidance = max(0.0, min(1.0, set_avoidance))
        self.partner_coordination = max(0.0, min(1.0, partner_coordination))
    
    def get_trump_threshold(self, base_threshold: float) -> float:
        """Calculate trump calling threshold based on risk profile."""
        risk_factor = (1.0 - self.trump_calling) * 2.0
        return base_threshold + risk_factor
    
    def get_card_play_aggression(self) -> float:
        """Get card play aggression factor."""
        return self.card_play
    
    def should_avoid_set(self, current_score: int, opponent_score: int) -> bool:
        """Determine if set avoidance is critical."""
        score_diff = opponent_score - current_score
        avoidance_threshold = 8.0 - (self.set_avoidance * 4.0)
        return score_diff >= avoidance_threshold
```

## Neural Network AI Implementation

### Model Architecture

#### EuchreNN (Original)

```python
class EuchreNN(nn.Module):
    """Enhanced neural network for euchre with risk-aware decision making."""
    
    def __init__(self, 
                 input_size: int = 128,
                 hidden_size: int = 256,
                 output_size: int = 64,
                 risk_embedding_size: int = 32,
                 use_risk_attention: bool = True):
        super().__init__()
        
        # Risk parameter embedding
        self.risk_embedding = nn.Linear(4, risk_embedding_size)
        
        # Main network layers
        self.input_layer = nn.Linear(input_size + risk_embedding_size, hidden_size)
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Risk-aware attention mechanism
        if use_risk_attention:
            self.risk_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                batch_first=True
            )
    
    def forward(self, x: torch.Tensor, risk_params: RiskParameters) -> Dict[str, torch.Tensor]:
        # Risk parameter embedding
        risk_embedding = self.risk_embedding(risk_params.get_risk_vector())
        
        # Combine input with risk embedding
        x_with_risk = torch.cat([x, risk_embedding], dim=1)
        
        # Forward pass through network
        h = F.relu(self.input_layer(x_with_risk))
        h = F.relu(self.hidden_layer1(h))
        h = F.relu(self.hidden_layer2(h))
        
        # Apply risk attention if enabled
        if hasattr(self, 'risk_attention'):
            h = h.unsqueeze(1)  # Add sequence dimension
            h, _ = self.risk_attention(h, h, h)
            h = h.squeeze(1)  # Remove sequence dimension
        
        output = self.output_layer(h)
        
        return {
            'trump_decision': output[:, :2],      # Binary classification
            'card_selection': output[:, 2:7],     # 5-class classification
            'suit_selection': output[:, 7:11]     # 4-class classification
        }
```

#### M-Series Base Model

```python
class MSeriesBaseModel(nn.Module):
    """Advanced M-Series base model with specialized layers."""
    
    def __init__(self, input_size: int = 256, hidden_size: int = 512, 
                 risk_embedding_size: int = 64):
        super().__init__()
        
        # Risk profile embedding
        self.risk_embedding = nn.Linear(8, risk_embedding_size)
        
        # Input processing
        self.input_layer = nn.Linear(input_size + risk_embedding_size, hidden_size)
        
        # Specialized layers
        self.partner_intuition_layer = nn.Linear(hidden_size, hidden_size // 2)
        self.card_memory_layer = nn.Linear(hidden_size, hidden_size // 2)
        self.strategic_planning_layer = nn.Linear(hidden_size, hidden_size // 2)
        self.trick_analysis_layer = nn.Linear(hidden_size, hidden_size // 2)
        
        # Feature fusion
        self.feature_fusion = nn.Linear(hidden_size * 2, hidden_size)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(3)
        ])
        
        # Output heads
        self.trump_decision_head = nn.Linear(hidden_size, 2)
        self.card_selection_head = nn.Linear(hidden_size, 5)
        self.suit_selection_head = nn.Linear(hidden_size, 4)
        
        # Regularization
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x: torch.Tensor, risk_params: MSeriesRiskProfile) -> Dict[str, torch.Tensor]:
        # Risk parameter embedding
        risk_embedding = self.risk_embedding(risk_params.get_risk_vector())
        
        # Combine input with risk embedding
        x_with_risk = torch.cat([x, risk_embedding], dim=1)
        
        # Initial processing
        h = F.relu(self.input_layer(x_with_risk))
        
        # Specialized feature processing
        partner_features = self.partner_intuition_layer(h)
        card_memory_features = self.card_memory_layer(h)
        strategic_features = self.strategic_planning_layer(h)
        trick_features = self.trick_analysis_layer(h)
        
        # Feature fusion
        combined_features = torch.cat([
            partner_features, card_memory_features, 
            strategic_features, trick_features
        ], dim=1)
        
        h = self.feature_fusion(combined_features)
        h = self.layer_norm(h)
        h = self.dropout(h)
        
        # Process through hidden layers
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
            h = self.dropout(h)
        
        # Generate outputs
        return {
            'trump_decision': self.trump_decision_head(h),
            'card_selection': self.card_selection_head(h),
            'suit_selection': self.suit_selection_head(h),
            'hidden_features': h
        }
```

### Game State Encoding

#### Basic Encoder

```python
class GameStateEncoder:
    """Converts euchre game states to neural network inputs."""
    
    def __init__(self, input_size: int = 128):
        self.input_size = input_size
        self.num_suits = 4
        self.num_ranks = 6
        self.hand_size = 5
    
    def encode_game_state(self, player: Player, top_card: Optional[Card], 
                         trump_suit: Optional[Suit], current_trick: List[Tuple[str, Card]],
                         team_scores: Dict[str, int], round_number: int) -> torch.Tensor:
        """Encode complete game state to feature vector."""
        features = []
        
        # Encode hand
        hand_encoding = self._encode_hand(player.hand, trump_suit)
        features.extend(hand_encoding)
        
        # Encode top card
        top_card_encoding = self._encode_card(top_card) if top_card else [0] * 24
        features.extend(top_card_encoding)
        
        # Encode current trick
        trick_encoding = self._encode_trick(current_trick, trump_suit)
        features.extend(trick_encoding)
        
        # Encode game context
        context_encoding = self._encode_context(team_scores, round_number)
        features.extend(context_encoding)
        
        # Pad to input size
        if len(features) < self.input_size:
            features.extend([0.0] * (self.input_size - len(features)))
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _encode_hand(self, hand: List[Card], trump_suit: Optional[Suit]) -> List[float]:
        """Encode player's hand."""
        features = []
        for card in hand:
            card_features = self._encode_card(card, trump_suit)
            features.extend(card_features)
        
        # Add hand-level features
        trump_count = sum(1 for card in hand if self._is_trump(card, trump_suit))
        features.append(trump_count / 5.0)
        
        return features
    
    def _encode_card(self, card: Card, trump_suit: Optional[Suit] = None) -> List[float]:
        """Encode a single card."""
        features = []
        
        # One-hot suit encoding
        suit_encoding = [0] * self.num_suits
        suit_index = list(Suit).index(card.suit)
        suit_encoding[suit_index] = 1
        features.extend(suit_encoding)
        
        # One-hot rank encoding
        rank_encoding = [0] * self.num_ranks
        rank_index = card.rank.value - 9  # 9=0, 10=1, J=2, Q=3, K=4, A=5
        rank_encoding[rank_index] = 1
        features.extend(rank_encoding)
        
        # Trump indicator
        is_trump = self._is_trump(card, trump_suit)
        features.append(1.0 if is_trump else 0.0)
        
        # Card strength
        features.append((card.rank.value - 9) / 5.0)
        
        return features
    
    def _is_trump(self, card: Card, trump_suit: Optional[Suit]) -> bool:
        """Check if card is trump."""
        if not trump_suit:
            return False
        return card.suit == trump_suit or card.is_trump
```

#### Advanced M-Series Encoder

```python
class MSeriesGameStateEncoder:
    """Advanced game state encoder for M-Series models."""
    
    def __init__(self, input_size: int = 256):
        self.input_size = input_size
        self.num_suits = 4
        self.num_ranks = 6
        self.hand_size = 5
        
        # Feature breakdown
        self.card_features = self.num_suits * self.num_ranks + 3
        self.hand_features = self.hand_size * self.card_features
        self.game_context_features = 32
        self.trick_context_features = 20
        self.historical_features = 20
        self.strategic_features = 49
    
    def encode_game_state(self, player: Player, top_card: Optional[Card],
                         trump_suit: Optional[Suit], current_trick: List[Tuple[str, Card]],
                         team_scores: Dict[str, int], round_number: int,
                         dealer: Player, game_history: List[Dict]) -> torch.Tensor:
        """Encode comprehensive game state."""
        features = []
        
        # Encode hand with context
        hand_encoding = self._encode_hand_with_context(player.hand, trump_suit)
        features.extend(hand_encoding)
        
        # Encode current trick
        trick_encoding = self._encode_trick_context(current_trick, trump_suit)
        features.extend(trick_encoding)
        
        # Encode game context
        game_context = self._encode_game_context(team_scores, round_number)
        features.extend(game_context)
        
        # Encode player position and strategy
        position_encoding = self._encode_position_context(player, dealer)
        features.extend(position_encoding)
        
        # Encode advanced strategy
        strategy_encoding = self._encode_strategic_context(player, current_trick, trump_suit)
        features.extend(strategy_encoding)
        
        # Encode historical patterns
        historical_encoding = self._encode_historical_context(game_history, player)
        features.extend(historical_encoding)
        
        # Pad to input size
        if len(features) < self.input_size:
            features.extend([0.0] * (self.input_size - len(features)))
        
        return torch.tensor(features, dtype=torch.float32)
```

## Integration and Usage

### Creating AI Players

```python
from euchre.ai.ai_factory import AIFactory
from euchre.ai_model.m_series_models import MagnusModel, MSeriesRiskProfile
from euchre.ai_model.game_state_encoder import GameStateEncoder

class AIPlayerManager:
    """Manages different types of AI players."""
    
    def __init__(self):
        self.rule_based_players = {}
        self.neural_players = {}
        self.encoders = {}
    
    def create_rule_based_player(self, name: str, ai_type: str, risk_ratio: float = 0.5):
        """Create a rule-based AI player."""
        player = AIFactory.create_ai_player(name, ai_type, risk_ratio)
        self.rule_based_players[name] = player
        return player
    
    def create_neural_player(self, name: str, model_path: str, risk_profile: MSeriesRiskProfile):
        """Create a neural network AI player."""
        model = MagnusModel.load_model(model_path)
        player = NeuralAIPlayer(name, model, risk_profile)
        self.neural_players[name] = player
        return player
    
    def get_player(self, name: str):
        """Get a player by name."""
        if name in self.rule_based_players:
            return self.rule_based_players[name]
        elif name in self.neural_players:
            return self.neural_players[name]
        else:
            raise ValueError(f"Player {name} not found")
```

### Game Integration

```python
class EuchreGame:
    """Main game class with AI integration."""
    
    def __init__(self, players: List[Player]):
        self.players = players
        self.ai_manager = AIPlayerManager()
        self.game_state = GameState()
    
    def play_round(self):
        """Play a single round."""
        # Deal cards
        self._deal_cards()
        
        # Trump selection phase
        trump_suit = self._trump_selection_phase()
        
        # Play tricks
        for trick_num in range(5):
            self._play_trick(trump_suit)
        
        # Update scores
        self._update_scores()
    
    def _trump_selection_phase(self) -> Optional[Suit]:
        """Handle trump selection with AI players."""
        top_card = self.game_state.top_card
        
        for player in self.players:
            if isinstance(player, BaseAI):
                # Rule-based AI
                should_order = player.should_order_up(top_card, self._is_partner_dealing(player))
            elif isinstance(player, NeuralAIPlayer):
                # Neural network AI
                game_state_tensor = self._encode_game_state_for_player(player)
                should_order = player.should_order_up(game_state_tensor, top_card)
            else:
                # Human player
                should_order = self._get_human_trump_decision(player, top_card)
            
            if should_order:
                return top_card.suit
        
        # Dealer calls trump
        dealer = self.players[self.game_state.dealer_index]
        if isinstance(dealer, NeuralAIPlayer):
            game_state_tensor = self._encode_game_state_for_player(dealer)
            trump_suit = dealer.choose_trump_suit(game_state_tensor)
        else:
            trump_suit = self._get_human_trump_decision(dealer, None)
        
        return trump_suit
    
    def _play_trick(self, trump_suit: Suit):
        """Play a single trick."""
        current_trick = []
        lead_suit = None
        
        for player in self.players:
            if isinstance(player, BaseAI):
                # Rule-based AI
                card = player.choose_card_to_play(lead_suit, trump_suit)
            elif isinstance(player, NeuralAIPlayer):
                # Neural network AI
                game_state_tensor = self._encode_game_state_for_player(player)
                card = player.choose_card_to_play(game_state_tensor, lead_suit, trump_suit)
            else:
                # Human player
                card = self._get_human_card_choice(player, lead_suit, trump_suit)
            
            if not lead_suit:
                lead_suit = card.suit
            
            current_trick.append((player.name, card))
            player.hand.remove(card)
        
        # Determine winner
        winner = self._determine_trick_winner(current_trick, trump_suit)
        self.game_state.trick_winners.append(winner)
```

## Performance Optimization

### Batch Processing

```python
class BatchAIPlayer:
    """Optimized AI player for batch processing."""
    
    def __init__(self, model: nn.Module, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        self.pending_decisions = []
    
    def add_decision_request(self, game_state: torch.Tensor, decision_type: str):
        """Add a decision request to the batch."""
        self.pending_decisions.append((game_state, decision_type))
        
        if len(self.pending_decisions) >= self.batch_size:
            self._process_batch()
    
    def _process_batch(self):
        """Process all pending decisions in a single batch."""
        if not self.pending_decisions:
            return
        
        # Prepare batch
        game_states = torch.stack([req[0] for req in self.pending_decisions])
        decision_types = [req[1] for req in self.pending_decisions]
        
        # Process batch
        with torch.no_grad():
            outputs = self.model(game_states)
        
        # Distribute results
        for i, (game_state, decision_type) in enumerate(self.pending_decisions):
            if decision_type == "trump":
                result = outputs['trump_decision'][i]
            elif decision_type == "card":
                result = outputs['card_selection'][i]
            else:
                result = outputs['suit_selection'][i]
            
            # Store result for retrieval
            self._store_decision_result(game_state, decision_type, result)
        
        # Clear pending decisions
        self.pending_decisions.clear()
```

### Model Quantization

```python
def quantize_model(model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
    """Quantize model for faster inference."""
    # Prepare model for quantization
    model.eval()
    
    # Set up quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare for quantization
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate with data
    with torch.no_grad():
        for data, _ in calibration_data:
            model(data)
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    return quantized_model
```

### Caching and Memoization

```python
class CachedAIPlayer:
    """AI player with decision caching for performance."""
    
    def __init__(self, base_player, cache_size: int = 10000):
        self.base_player = base_player
        self.cache = {}
        self.cache_size = cache_size
    
    def should_order_up(self, game_state_hash: str, top_card: Card, 
                       is_partner_dealing: bool = False) -> bool:
        """Cached trump decision."""
        cache_key = f"order_up_{game_state_hash}_{top_card}_{is_partner_dealing}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get decision from base player
        decision = self.base_player.should_order_up(top_card, is_partner_dealing)
        
        # Cache result
        self._add_to_cache(cache_key, decision)
        
        return decision
    
    def _add_to_cache(self, key: str, value: Any):
        """Add item to cache with size management."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest item (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
```

This implementation guide provides the technical foundation for working with both AI systems. The rule-based system offers immediate usability and customization, while the neural network system provides advanced learning capabilities and performance optimization opportunities. 