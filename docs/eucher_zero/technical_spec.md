# EuchreZero Technical Specification

## Code Structure Examples

### 1. Unified Action Space

```python
# eucher/players/computer/euchre_zero/action_space.py

from enum import IntEnum
from typing import Optional, Tuple
from eucher.cards import Card, Suit

class ActionType(IntEnum):
    """Action types in unified action space."""
    PASS = 0
    ORDER_UP = 1
    CALL_HEARTS = 2
    CALL_DIAMONDS = 3
    CALL_CLUBS = 4
    CALL_SPADES = 5
    STICK_DEALER = 6
    DISCARD_0 = 7
    DISCARD_1 = 8
    DISCARD_2 = 9
    DISCARD_3 = 10
    DISCARD_4 = 11
    DISCARD_5 = 12
    PLAY_0 = 13
    PLAY_1 = 14
    PLAY_2 = 15
    PLAY_3 = 16
    PLAY_4 = 17

ACTION_SPACE_SIZE = 18

class ActionEncoder:
    """Encode/decode actions in unified space."""
    
    @staticmethod
    def encode_bid_action(action: str, suit: Optional[Suit] = None) -> int:
        """Encode bidding action."""
        if action == "pass":
            return ActionType.PASS
        elif action == "order_up":
            return ActionType.ORDER_UP
        elif action == "call":
            suit_map = {
                Suit.HEARTS: ActionType.CALL_HEARTS,
                Suit.DIAMONDS: ActionType.CALL_DIAMONDS,
                Suit.CLUBS: ActionType.CALL_CLUBS,
                Suit.SPADES: ActionType.CALL_SPADES,
            }
            return suit_map[suit]
        elif action == "stick_dealer":
            return ActionType.STICK_DEALER
        raise ValueError(f"Unknown bid action: {action}")
    
    @staticmethod
    def encode_discard_action(card_index: int) -> int:
        """Encode discard action."""
        if not 0 <= card_index <= 5:
            raise ValueError(f"Invalid discard index: {card_index}")
        return ActionType.DISCARD_0 + card_index
    
    @staticmethod
    def encode_play_action(card_index: int) -> int:
        """Encode play action."""
        if not 0 <= card_index <= 4:
            raise ValueError(f"Invalid play index: {card_index}")
        return ActionType.PLAY_0 + card_index
    
    @staticmethod
    def decode_action(action_id: int) -> Tuple[str, Optional[int], Optional[Suit]]:
        """Decode action to (type, card_index, suit)."""
        if action_id == ActionType.PASS:
            return ("pass", None, None)
        elif action_id == ActionType.ORDER_UP:
            return ("order_up", None, None)
        elif ActionType.CALL_HEARTS <= action_id <= ActionType.CALL_SPADES:
            suit_map = {
                ActionType.CALL_HEARTS: Suit.HEARTS,
                ActionType.CALL_DIAMONDS: Suit.DIAMONDS,
                ActionType.CALL_CLUBS: Suit.CLUBS,
                ActionType.CALL_SPADES: Suit.SPADES,
            }
            return ("call", None, suit_map[action_id])
        elif action_id == ActionType.STICK_DEALER:
            return ("stick_dealer", None, None)
        elif ActionType.DISCARD_0 <= action_id <= ActionType.DISCARD_5:
            return ("discard", action_id - ActionType.DISCARD_0, None)
        elif ActionType.PLAY_0 <= action_id <= ActionType.PLAY_4:
            return ("play", action_id - ActionType.PLAY_0, None)
        raise ValueError(f"Unknown action ID: {action_id}")
```

### 2. State Encoder

```python
# eucher/players/computer/euchre_zero/state_encoder.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from eucher.cards import Card, Suit
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
        """Encode hand as one-hot (24 cards × 5 positions)."""
        tensor = torch.zeros(24, 5)
        for i, card in enumerate(hand[:5]):
            card_idx = self._card_to_index(card)
            tensor[card_idx, i] = 1.0
        return tensor.flatten()
    
    def encode_card(self, card: Optional[Card]) -> torch.Tensor:
        """Encode single card as one-hot (24 dims)."""
        tensor = torch.zeros(24)
        if card is not None:
            idx = self._card_to_index(card)
            tensor[idx] = 1.0
        return tensor
    
    def encode_seat(self, seat_id: int) -> torch.Tensor:
        """Encode player seat (4-dim one-hot)."""
        tensor = torch.zeros(4)
        tensor[seat_id] = 1.0
        return tensor
    
    def encode_trump(self, trump_suit: Optional[Suit]) -> torch.Tensor:
        """Encode trump context (5-dim: None + 4 suits)."""
        tensor = torch.zeros(5)
        if trump_suit is None:
            tensor[0] = 1.0
        else:
            suit_idx = list(Suit).index(trump_suit) + 1
            tensor[suit_idx] = 1.0
        return tensor
    
    def encode_bidding_stage(self, game: Game) -> torch.Tensor:
        """Encode bidding stage (3-dim: order_up, call_trump, playing)."""
        # Simplified - would track actual stage
        tensor = torch.zeros(3)
        if game.trump_suit is None:
            tensor[1] = 1.0  # Assume call_trump stage
        else:
            tensor[2] = 1.0  # Playing stage
        return tensor
    
    def encode_trick_history(self, game: Game) -> torch.Tensor:
        """Encode trick history (5 tricks × 4 cards × features)."""
        # Would encode completed tricks
        return torch.zeros(5 * 4 * 10)  # Placeholder
    
    def encode_perfect_memory(self, game: Game, player_id: int) -> torch.Tensor:
        """Encode perfect memory (24 cards × location)."""
        # Would track all card locations
        return torch.zeros(24 * 4)  # Placeholder
    
    def encode_deduction_map(self, game: Game, player_id: int) -> torch.Tensor:
        """Encode deduction probabilities (24 cards × 3 players)."""
        # Would use deduction system
        return torch.zeros(24 * 3)  # Placeholder
    
    def encode_suit_voids(self, game: Game, player_id: int) -> torch.Tensor:
        """Encode suit void inference (4 suits × 3 players)."""
        return torch.zeros(4 * 3)  # Placeholder
    
    def encode_trump_counts(self, game: Game, player_id: int) -> torch.Tensor:
        """Encode trump count inference (3 players)."""
        return torch.zeros(3)  # Placeholder
    
    def _card_to_index(self, card: Card) -> int:
        """Convert card to index (0-23)."""
        suits = list(Suit)
        ranks = [9, 10, 11, 12, 13, 14]  # NINE through ACE
        suit_idx = suits.index(card.suit)
        rank_idx = ranks.index(card.rank.value)
        return suit_idx * 6 + rank_idx
```

### 3. Representation Network

```python
# eucher/players/computer/euchre_zero/networks/representation.py

import torch
import torch.nn as nn
from typing import Dict

class RepresentationNetwork(nn.Module):
    """Encode game state to latent representation."""
    
    def __init__(self, latent_size: int = 512) -> None:
        """
        Initialize representation network.
        
        Parameters
        ----------
        latent_size : int
            Size of latent vector (default: 512).
        """
        super().__init__()
        
        # Input processing layers
        self.hand_encoder = nn.Sequential(
            nn.Linear(24 * 5, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        
        self.upcard_encoder = nn.Sequential(
            nn.Linear(24, 32),
            nn.ReLU(),
        )
        
        self.context_encoder = nn.Sequential(
            nn.Linear(4 + 4 + 5 + 3 + 2, 64),  # seat + partner + trump + bidding + score
            nn.ReLU(),
        )
        
        self.history_encoder = nn.Sequential(
            nn.Linear(5 * 4 * 10, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        
        self.memory_encoder = nn.Sequential(
            nn.Linear(24 * 4, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        
        self.deduction_encoder = nn.Sequential(
            nn.Linear(24 * 3, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        
        self.inference_encoder = nn.Sequential(
            nn.Linear(4 * 3 + 3, 32),  # suit voids + trump counts
            nn.ReLU(),
        )
        
        # Concatenate and process
        combined_size = 128 + 32 + 64 + 128 + 128 + 128 + 32 + 1  # +1 for risk
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, latent_size),
        )
    
    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        state_dict : Dict[str, torch.Tensor]
            Dictionary of state tensors.
        
        Returns
        -------
        torch.Tensor
            Latent state vector [batch_size, latent_size].
        """
        # Process each component
        hand = self.hand_encoder(state_dict["hand"])
        upcard = self.upcard_encoder(state_dict["upcard"])
        context = self.context_encoder(
            torch.cat([
                state_dict["player_seat"],
                state_dict["partner_seat"],
                state_dict["trump_context"],
                state_dict["bidding_stage"],
                state_dict["score"],
            ], dim=-1)
        )
        history = self.history_encoder(state_dict["trick_history"])
        memory = self.memory_encoder(state_dict["perfect_memory"])
        deduction = self.deduction_encoder(state_dict["deduction_map"])
        inference = self.inference_encoder(
            torch.cat([
                state_dict["suit_voids"],
                state_dict["trump_counts"],
            ], dim=-1)
        )
        
        # Concatenate all features
        combined = torch.cat([
            hand, upcard, context, history, memory, deduction, inference,
            state_dict["risk_factor"],
        ], dim=-1)
        
        # Generate latent representation
        latent = self.shared_layers(combined)
        return latent
```

### 4. MCTS Node

```python
# eucher/players/computer/euchre_zero/mcts/tree.py

from typing import Dict, Optional, List
import torch
import numpy as np

class MCTSNode:
    """Node in MCTS tree."""
    
    def __init__(
        self,
        state: torch.Tensor,
        prior: float = 0.0,
        hidden_state: Optional[Dict[int, List]] = None,
    ) -> None:
        """
        Initialize MCTS node.
        
        Parameters
        ----------
        state : torch.Tensor
            Latent state representation.
        prior : float
            Prior probability from policy network.
        hidden_state : Optional[Dict[int, List]]
            Sampled hidden card assignments.
        """
        self.state = state
        self.prior = prior
        self.hidden_state = hidden_state
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.risk_value_sum = 0.0
        
        self.children: Dict[int, "MCTSNode"] = {}
        self.parent: Optional["MCTSNode"] = None
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return len(self.children) == 0
    
    def get_value(self) -> float:
        """Get average value."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_risk_value(self) -> float:
        """Get average risk-adjusted value."""
        if self.visit_count == 0:
            return 0.0
        return self.risk_value_sum / self.visit_count
    
    def ucb_score(self, exploration_constant: float = 1.0) -> Dict[int, float]:
        """
        Calculate UCB scores for all children.
        
        Parameters
        ----------
        exploration_constant : float
            Exploration constant for UCB.
        
        Returns
        -------
        Dict[int, float]
            Action -> UCB score mapping.
        """
        scores = {}
        for action, child in self.children.items():
            if child.visit_count == 0:
                scores[action] = float('inf')
            else:
                exploitation = child.get_value()
                exploration = exploration_constant * child.prior * np.sqrt(
                    self.visit_count
                ) / (1 + child.visit_count)
                scores[action] = exploitation + exploration
        return scores
    
    def select_action(self, exploration_constant: float = 1.0) -> int:
        """
        Select action using UCB.
        
        Parameters
        ----------
        exploration_constant : float
            Exploration constant.
        
        Returns
        -------
        int
            Selected action ID.
        """
        scores = self.ucb_score(exploration_constant)
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def add_child(self, action: int, child: "MCTSNode") -> None:
        """
        Add child node.
        
        Parameters
        ----------
        action : int
            Action leading to child.
        child : MCTSNode
            Child node.
        """
        child.parent = self
        self.children[action] = child
    
    def backpropagate(self, value: float, risk_value: float) -> None:
        """
        Backpropagate value up the tree.
        
        Parameters
        ----------
        value : float
            Value to propagate.
        risk_value : float
            Risk-adjusted value to propagate.
        """
        self.visit_count += 1
        self.value_sum += value
        self.risk_value_sum += risk_value
        
        if self.parent is not None:
            self.parent.backpropagate(value, risk_value)
```

### 5. MCTS Search

```python
# eucher/players/computer/euchre_zero/mcts/search.py

from typing import Dict, List, Tuple
import torch
import numpy as np
from eucher.players.computer.euchre_zero.mcts.tree import MCTSNode
from eucher.players.computer.euchre_zero.networks.representation import RepresentationNetwork
from eucher.players.computer.euchre_zero.networks.dynamics import DynamicsNetwork
from eucher.players.computer.euchre_zero.networks.prediction import PredictionNetwork

class MCTSSearch:
    """Monte Carlo Tree Search with belief sampling."""
    
    def __init__(
        self,
        representation_net: RepresentationNetwork,
        dynamics_net: DynamicsNetwork,
        prediction_net: PredictionNetwork,
        num_simulations: int = 100,
        exploration_constant: float = 1.0,
        num_belief_samples: int = 10,
    ) -> None:
        """
        Initialize MCTS search.
        
        Parameters
        ----------
        representation_net : RepresentationNetwork
            Representation network.
        dynamics_net : DynamicsNetwork
            Dynamics network.
        prediction_net : PredictionNetwork
            Prediction network.
        num_simulations : int
            Number of MCTS simulations.
        exploration_constant : float
            UCB exploration constant.
        num_belief_samples : int
            Number of belief state samples.
        """
        self.representation_net = representation_net
        self.dynamics_net = dynamics_net
        self.prediction_net = prediction_net
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.num_belief_samples = num_belief_samples
    
    def search(
        self,
        game_state: Dict[str, torch.Tensor],
        deduction_map: np.ndarray,
        risk_factor: float = 0.0,
    ) -> np.ndarray:
        """
        Run MCTS search.
        
        Parameters
        ----------
        game_state : Dict[str, torch.Tensor]
            Current game state.
        deduction_map : np.ndarray
            Deduction probability map.
        risk_factor : float
            Risk factor for value scaling.
        
        Returns
        -------
        np.ndarray
            Improved policy distribution over actions.
        """
        # Encode state
        with torch.no_grad():
            latent = self.representation_net(game_state)
        
        # Create root node
        root = MCTSNode(latent)
        
        # Sample belief states
        belief_states = self._sample_belief_states(deduction_map)
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Sample a belief state
            hidden_state = np.random.choice(belief_states)
            
            # Run one simulation
            self._simulate(root, hidden_state, risk_factor)
        
        # Extract improved policy
        policy = self._extract_policy(root)
        return policy
    
    def _simulate(
        self,
        root: MCTSNode,
        hidden_state: Dict,
        risk_factor: float,
    ) -> None:
        """Run one MCTS simulation."""
        node = root
        
        # Selection
        while not node.is_leaf():
            action = node.select_action(self.exploration_constant)
            if action in node.children:
                node = node.children[action]
            else:
                break
        
        # Expansion and evaluation
        if node.visit_count > 0 or node == root:
            # Evaluate with prediction network
            with torch.no_grad():
                policy_logits, value, risk_value = self.prediction_net(node.state)
                policy = torch.softmax(policy_logits, dim=-1)
            
            # Expand children
            for action in range(18):  # ACTION_SPACE_SIZE
                prior = policy[action].item()
                child = MCTSNode(node.state, prior=prior, hidden_state=hidden_state)
                node.add_child(action, child)
            
            # Backpropagate
            node.backpropagate(value.item(), risk_value.item())
        else:
            # Rollout to terminal state
            value, risk_value = self._rollout(node, hidden_state, risk_factor)
            node.backpropagate(value, risk_value)
    
    def _rollout(
        self,
        node: MCTSNode,
        hidden_state: Dict,
        risk_factor: float,
    ) -> Tuple[float, float]:
        """Rollout to terminal state (simplified)."""
        # Would simulate game to end
        # For now, use prediction network value
        with torch.no_grad():
            _, value, risk_value = self.prediction_net(node.state)
        return value.item(), risk_value.item()
    
    def _sample_belief_states(
        self,
        deduction_map: np.ndarray,
    ) -> List[Dict]:
        """Sample belief states from deduction map."""
        # Would sample hidden card assignments
        return [{}]  # Placeholder
    
    def _extract_policy(self, root: MCTSNode) -> np.ndarray:
        """Extract improved policy from root node."""
        policy = np.zeros(18)  # ACTION_SPACE_SIZE
        total_visits = sum(child.visit_count for child in root.children.values())
        
        if total_visits == 0:
            return np.ones(18) / 18  # Uniform
        
        for action, child in root.children.items():
            policy[action] = child.visit_count / total_visits
        
        return policy
```

### 6. Reward Calculator

```python
# eucher/players/computer/euchre_zero/rewards/reward_calculator.py

from typing import Optional
from eucher.cards import Card

class RewardCalculator:
    """Calculate rewards according to EuchreZero specification."""
    
    # Immediate rewards
    TRICK_WON = 1.0
    TRICK_LOST = -1.0
    DRAW_TRUMP_EFFECTIVE = 1.0
    BURN_TRUMP_USELESS = -1.0
    
    # Bidding rewards
    SUCCESSFUL_CALL = 4.0
    SUCCESSFUL_SWEEP = 5.0
    FAILED_CALL = -4.0
    CORRECT_PASS = 0.0
    INCORRECT_PASS = -1.0
    
    # Discard rewards
    STRONG_DISCARD = 1.0
    HARMFUL_DISCARD = -1.0
    
    # Illegal play penalties
    RENEGE_PENALTY = -5.0
    RENEGE_TRICK_LOST = -2.0
    RENEGE_PARTNER_HARM = -3.0
    RENEGE_HAND_PENALTY = -6.0
    
    # Hand outcome rewards
    CALLING_TEAM_SUCCESS = 3.0
    CALLING_TEAM_SWEEP = 4.0
    CALLING_TEAM_SET = -4.0
    DEFENDERS_SET_CALLER = 3.0
    DEFENDERS_ALLOW_SWEEP = -1.0
    
    def calculate_immediate_reward(
        self,
        action: str,
        outcome: str,
        risk_factor: float = 0.0,
    ) -> float:
        """
        Calculate immediate reward for action.
        
        Parameters
        ----------
        action : str
            Action taken.
        outcome : str
            Outcome description.
        risk_factor : float
            Risk factor for scaling.
        
        Returns
        -------
        float
            Immediate reward.
        """
        reward = 0.0
        
        if outcome == "trick_won":
            reward = self.TRICK_WON
        elif outcome == "trick_lost":
            reward = self.TRICK_LOST
        elif outcome == "draw_trump_effective":
            reward = self.DRAW_TRUMP_EFFECTIVE
        elif outcome == "burn_trump_useless":
            reward = self.BURN_TRUMP_USELESS
        elif outcome == "renege":
            reward = self.RENEGE_PENALTY
        
        return self.apply_risk_scaling(reward, risk_factor, is_penalty=(reward < 0))
    
    def calculate_hand_reward(
        self,
        tricks_won: int,
        calling_team: int,
        player_team: int,
        risk_factor: float = 0.0,
    ) -> float:
        """
        Calculate hand outcome reward.
        
        Parameters
        ----------
        tricks_won : int
            Tricks won by calling team.
        calling_team : int
            Team that called trump.
        player_team : int
            Team of the player.
        risk_factor : float
            Risk factor for scaling.
        
        Returns
        -------
        float
            Hand reward.
        """
        is_caller = (calling_team == player_team)
        
        if is_caller:
            if tricks_won >= 5:
                reward = self.CALLING_TEAM_SWEEP
            elif tricks_won >= 3:
                reward = self.CALLING_TEAM_SUCCESS
            else:
                reward = self.CALLING_TEAM_SET
        else:
            if tricks_won < 3:
                reward = self.DEFENDERS_SET_CALLER
            elif tricks_won >= 5:
                reward = self.DEFENDERS_ALLOW_SWEEP
            else:
                reward = 0.0
        
        return self.apply_risk_scaling(reward, risk_factor, is_penalty=(reward < 0))
    
    def apply_risk_scaling(
        self,
        reward: float,
        risk_factor: float,
        is_penalty: bool = False,
    ) -> float:
        """
        Apply risk factor scaling.
        
        Parameters
        ----------
        reward : float
            Base reward.
        risk_factor : float
            Risk factor (0.0-1.0).
        is_penalty : bool
            Whether this is a penalty.
        
        Returns
        -------
        float
            Scaled reward.
        """
        if is_penalty:
            return reward * (1.0 - 0.4 * risk_factor)
        else:
            return reward * (1.0 + 0.5 * risk_factor)
```

## Training Loop Structure

```python
# scripts/eucher_zero/train_eucher_zero.py (simplified)

def train_iteration(
    model: EuchreZeroModel,
    replay_buffer: ReplayBuffer,
    config: EuchreZeroConfig,
) -> Dict[str, float]:
    """One training iteration."""
    # 1. Generate self-play games
    games = generate_self_play_games(model, config.num_games_per_iteration)
    
    # 2. Add to replay buffer
    for game in games:
        replay_buffer.add_game(game)
    
    # 3. Sample batch
    batch = replay_buffer.sample_batch(config.batch_size)
    
    # 4. Train on batch
    losses = train_step(model, batch, config)
    
    return losses

def train_step(
    model: EuchreZeroModel,
    batch: Batch,
    config: EuchreZeroConfig,
) -> Dict[str, float]:
    """Single training step."""
    # Forward passes
    latent = model.representation_net(batch.states)
    next_latent, reward_pred = model.dynamics_net(latent, batch.actions)
    policy_logits, value, risk_value = model.prediction_net(latent)
    
    # Losses
    policy_loss = kl_divergence(policy_logits, batch.mcts_policies)
    value_loss = mse_loss(value, batch.final_values)
    risk_value_loss = mse_loss(risk_value, batch.risk_values)
    dynamics_loss = mse_loss(next_latent, batch.next_latents)
    reward_loss = mse_loss(reward_pred, batch.immediate_rewards)
    
    # Total loss
    total_loss = (
        config.loss_weights.policy * policy_loss +
        config.loss_weights.value * value_loss +
        config.loss_weights.risk_value * risk_value_loss +
        config.loss_weights.dynamics * dynamics_loss +
        config.loss_weights.reward * reward_loss
    )
    
    # Backward
    total_loss.backward()
    optimizer.step()
    
    return {
        "policy": policy_loss.item(),
        "value": value_loss.item(),
        "risk_value": risk_value_loss.item(),
        "dynamics": dynamics_loss.item(),
        "reward": reward_loss.item(),
        "total": total_loss.item(),
    }
```

## Integration Example

```python
# eucher/players/computer/euchre_zero/player.py

from eucher.players.base import PlayerProfile
from eucher.cards import Card, Suit
from typing import List, Optional
from eucher.players.computer.euchre_zero.mcts.search import MCTSSearch

class EuchreZeroPlayer(PlayerProfile):
    """EuchreZero player profile."""
    
    def __init__(
        self,
        model: EuchreZeroModel,
        num_simulations: int = 100,
        risk_factor: float = 0.0,
    ) -> None:
        """
        Initialize EuchreZero player.
        
        Parameters
        ----------
        model : EuchreZeroModel
            Trained EuchreZero model.
        num_simulations : int
            MCTS simulations per decision.
        risk_factor : float
            Risk factor for decision making.
        """
        self.model = model
        self.mcts = MCTSSearch(
            model.representation_net,
            model.dynamics_net,
            model.prediction_net,
            num_simulations=num_simulations,
        )
        self.risk_factor = risk_factor
    
    def decide_order_up(
        self,
        player,
        turned_card: Card,
        dealer_id: int,
        trump_suit: Optional[Suit],
    ) -> bool:
        """Decide whether to order up."""
        # Encode state
        state = self._encode_state(player, turned_card, dealer_id, trump_suit)
        deduction_map = self._get_deduction_map(player)
        
        # Run MCTS
        policy = self.mcts.search(state, deduction_map, self.risk_factor)
        
        # Select action (ORDER_UP = 1)
        return policy[1] > 0.5  # Simplified
    
    def play_card(
        self,
        player,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """Choose card to play."""
        # Encode state
        state = self._encode_state(player, led_suit, trump_suit, trick_cards)
        deduction_map = self._get_deduction_map(player)
        
        # Run MCTS
        policy = self.mcts.search(state, deduction_map, self.risk_factor)
        
        # Select card from play actions (PLAY_0 to PLAY_4)
        play_probs = policy[13:18]  # PLAY_0 to PLAY_4
        card_idx = play_probs.argmax()
        
        return player.hand[card_idx]
```

## Key Implementation Notes

1. **No Legal Move Masking**: Network learns legality via negative rewards. All 18 actions are always available.

2. **Belief Sampling**: MCTS samples multiple hidden card states and averages results.

3. **Risk Scaling**: All rewards/penalties are scaled by risk factor during training and inference.

4. **Batched Operations**: Use PyTorch batching for efficient GPU utilization.

5. **Checkpointing**: Save model checkpoints regularly for resuming training.

6. **Evaluation**: Run periodic evaluations against baseline players.

7. **Curriculum**: Start with simple stages and gradually increase complexity.

