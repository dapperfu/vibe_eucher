"""Actor-Critic RL agent with PPO-style training for Euchre.

This module implements an actor-critic agent using the transformer network
with PPO (Proximal Policy Optimization) for stable policy learning.
"""

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.ai_players.transformer_rl.euchre_transformer import EuchreTransformer, create_transformer_network

if TYPE_CHECKING:
    from src.ai_players.transformer_rl.checkpoint_manager import CumulativeCheckpointManager


class ExperienceBuffer:
    """Buffer for storing experience tuples for on-policy learning.

    Parameters
    ----------
    capacity : int
        Maximum number of experiences to store.
    """

    def __init__(self, capacity: int = 10000) -> None:
        """Initialize experience buffer.

        Parameters
        ----------
        capacity : int
            Buffer capacity.
        """
        self.capacity = capacity
        self.buffer: Deque[Dict] = deque(maxlen=capacity)

    def push(self, experience: Dict) -> None:
        """Add experience to buffer.

        Parameters
        ----------
        experience : Dict
            Experience dictionary with keys: state, action, reward, next_state, done, log_prob, value.
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Dict]:
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        List[Dict]
            List of sampled experiences.
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """Get buffer size.

        Returns
        -------
        int
            Number of experiences in buffer.
        """
        return len(self.buffer)


class ActorCriticAgent:
    """Actor-Critic agent with PPO training for Euchre RL.

    Uses the transformer network as both actor (policy) and critic (value).
    Implements PPO with clipped objectives for stable learning.

    Parameters
    ----------
    device : torch.device
        Device to run on.
    learning_rate : float
        Learning rate for optimizer.
    gamma : float
        Discount factor for future rewards.
    gae_lambda : float
        GAE lambda parameter.
    clip_epsilon : float
        PPO clip epsilon.
    value_coef : float
        Value loss coefficient.
    entropy_coef : float
        Entropy bonus coefficient.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        **transformer_kwargs,
    ) -> None:
        """Initialize actor-critic agent.

        Parameters
        ----------
        device : Optional[torch.device]
            Device to run on. If None, auto-detects.
        learning_rate : float
            Learning rate.
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE lambda.
        clip_epsilon : float
            PPO clip epsilon.
        value_coef : float
            Value loss coefficient.
        entropy_coef : float
            Entropy coefficient.
        max_grad_norm : float
            Max gradient norm.
        **transformer_kwargs
            Additional arguments for transformer network.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Create transformer network (shared actor-critic)
        self.network = create_transformer_network(device=self.device, **transformer_kwargs)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Experience buffer
        self.buffer = ExperienceBuffer()

        # Training stats
        self.training_stats: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
        }

    def select_action(
        self,
        state: Dict[str, torch.Tensor],
        action_type: str,
        valid_actions: List[int],
        training: bool = True,
    ) -> Tuple[int, float, float]:
        """Select an action using the policy network.

        Parameters
        ----------
        state : Dict[str, torch.Tensor]
            State dictionary with encoded features.
        action_type : str
            Type of action: 'bidding', 'discard', or 'play'.
        valid_actions : List[int]
            List of valid action indices.
        training : bool
            Whether in training mode (affects exploration).

        Returns
        -------
        Tuple[int, float, float]
            (action, log_prob, value)
        """
        self.network.eval()
        with torch.no_grad():
            # Forward pass
            # Move optional tensors to device if they exist
            deduction_maps = state.get("deduction_maps", None)
            if deduction_maps is not None:
                deduction_maps = deduction_maps.to(self.device) if isinstance(deduction_maps, torch.Tensor) else None
            
            current_trick = state.get("current_trick", None)
            if current_trick is not None:
                current_trick = current_trick.to(self.device) if isinstance(current_trick, torch.Tensor) else None
            
            outputs = self.network(
                hand=state["hand"].unsqueeze(0).to(self.device),
                trick_history=state["trick_history"].unsqueeze(0).to(self.device),
                game_context=state["game_context"].unsqueeze(0).to(self.device),
                deduction_maps=deduction_maps,
                current_trick=current_trick,
            )

            # Get appropriate head
            if action_type == "bidding":
                # Use order_up or call_trump based on context
                action_logits = outputs["order_up"].squeeze(0)
            elif action_type == "discard":
                action_logits = outputs["discard"].squeeze(0)
            elif action_type == "play":
                action_logits = outputs["play_card"].squeeze(0)
            else:
                raise ValueError(f"Unknown action type: {action_type}")

            # Get value estimate
            value = outputs["value"].squeeze(0).item()

            # Mask invalid actions
            masked_logits = torch.full_like(action_logits, float("-inf"))
            for action_idx in valid_actions:
                if 0 <= action_idx < len(action_logits):
                    masked_logits[action_idx] = action_logits[action_idx]

            # Sample action
            action_probs = F.softmax(masked_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()

            # Get log probability - ensure tensor is on same device as distribution
            log_prob = action_dist.log_prob(torch.tensor(action, device=self.device)).item()

        return action, log_prob, value

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
        dones: List[bool],
    ) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation (GAE).

        Parameters
        ----------
        rewards : List[float]
            List of rewards.
        values : List[float]
            List of value estimates.
        next_value : float
            Value of next state.
        dones : List[bool]
            List of done flags.

        Returns
        -------
        Tuple[List[float], List[float]]
            (advantages, returns)
        """
        advantages: List[float] = []
        returns: List[float] = []

        gae = 0.0
        next_value_use = next_value

        # Compute advantages backwards
        for step in reversed(range(len(rewards))):
            if dones[step]:
                gae = 0.0
                next_value_use = 0.0

            delta = rewards[step] + self.gamma * next_value_use - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

            next_value_use = values[step]

        return advantages, returns

    def update(self, batch_size: int = 64, num_epochs: int = 4) -> Dict[str, float]:
        """Update the policy using PPO.

        Parameters
        ----------
        batch_size : int
            Batch size for training.
        num_epochs : int
            Number of training epochs.

        Returns
        -------
        Dict[str, float]
            Training statistics.
        """
        if len(self.buffer) < batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        # Sample batch
        batch = self.buffer.sample(batch_size)

        # Extract data
        states = [exp["state"] for exp in batch]
        actions = [exp["action"] for exp in batch]
        old_log_probs = torch.tensor([exp["log_prob"] for exp in batch], device=self.device)
        rewards = [exp["reward"] for exp in batch]
        values = [exp["value"] for exp in batch]
        next_values = [exp.get("next_value", 0.0) for exp in batch]
        dones = [exp.get("done", False) for exp in batch]

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, next_values[-1] if next_values else 0.0, dones)
        advantages_tensor = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, device=self.device, dtype=torch.float32)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Training loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(num_epochs):
            # Forward pass
            self.network.train()

            # Process states in batches
            # For simplicity, process one at a time (can be optimized)
            new_log_probs: List[torch.Tensor] = []
            new_values: List[torch.Tensor] = []
            entropies: List[torch.Tensor] = []

            for i, state in enumerate(states):
                action = actions[i]
                action_type = batch[i].get("action_type", "play")

                # Prepare deduction_maps with batch dimension if provided
                deduction_maps = state.get("deduction_maps", None)
                if deduction_maps is not None and deduction_maps.dim() == 2:
                    # Add batch dimension: (24, 4) -> (1, 24, 4)
                    deduction_maps = deduction_maps.unsqueeze(0).to(self.device)
                elif deduction_maps is not None:
                    deduction_maps = deduction_maps.to(self.device)

                # Prepare current_trick with batch dimension if provided
                current_trick = state.get("current_trick", None)
                if current_trick is not None and current_trick.dim() == 2:
                    # Add batch dimension if needed
                    current_trick = current_trick.unsqueeze(0).to(self.device)
                elif current_trick is not None:
                    current_trick = current_trick.to(self.device)

                outputs = self.network(
                    hand=state["hand"].unsqueeze(0).to(self.device),
                    trick_history=state["trick_history"].unsqueeze(0).to(self.device),
                    game_context=state["game_context"].unsqueeze(0).to(self.device),
                    deduction_maps=deduction_maps,
                    current_trick=current_trick,
                )

                # Get appropriate head
                if action_type == "bidding":
                    action_logits = outputs["order_up"].squeeze(0)
                elif action_type == "discard":
                    action_logits = outputs["discard"].squeeze(0)
                elif action_type == "play":
                    action_logits = outputs["play_card"].squeeze(0)
                else:
                    action_logits = outputs["play_card"].squeeze(0)

                # Get probabilities
                action_probs = F.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)

                # Get log prob and value
                # Ensure value is always a 1D tensor with shape (1,) for consistent stacking
                value = outputs["value"]
                # Squeeze all dimensions except batch dimension
                while value.dim() > 1:
                    value = value.squeeze(-1)
                # If scalar, make it (1,)
                if value.dim() == 0:
                    value = value.unsqueeze(0)
                # Ensure it's exactly (1,) shape
                if value.shape[0] != 1:
                    value = value[:1]  # Take first element if needed
                
                new_log_probs.append(action_dist.log_prob(torch.tensor(action, device=self.device)))
                new_values.append(value)
                entropies.append(action_dist.entropy())

            # Stack tensors
            # All tensors should now have consistent shapes
            new_log_probs_tensor = torch.stack(new_log_probs)
            new_values_tensor = torch.stack(new_values)
            # Squeeze the last dimension if values are (batch, 1) -> (batch,)
            if new_values_tensor.dim() == 2 and new_values_tensor.shape[1] == 1:
                new_values_tensor = new_values_tensor.squeeze(-1)
            
            entropies_tensor = torch.stack(entropies)

            # Compute policy loss (PPO clipped)
            ratio = torch.exp(new_log_probs_tensor - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            # Compute value loss
            value_loss = F.mse_loss(new_values_tensor, returns_tensor)

            # Compute entropy bonus
            entropy = entropies_tensor.mean()

            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Accumulate stats
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        # Average stats
        avg_policy_loss = total_policy_loss / num_epochs
        avg_value_loss = total_value_loss / num_epochs
        avg_entropy = total_entropy / num_epochs
        avg_total_loss = avg_policy_loss + self.value_coef * avg_value_loss - self.entropy_coef * avg_entropy

        # Store stats
        self.training_stats["policy_loss"].append(avg_policy_loss)
        self.training_stats["value_loss"].append(avg_value_loss)
        self.training_stats["entropy"].append(avg_entropy)
        self.training_stats["total_loss"].append(avg_total_loss)

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "total_loss": avg_total_loss,
        }

    def save(self, filepath: Optional[Path] = None, training_stats: Optional[Dict] = None, checkpoint_manager: Optional["CumulativeCheckpointManager"] = None) -> Optional[str]:  # type: ignore
        """Save the agent to disk.

        Parameters
        ----------
        filepath : Optional[Path]
            Path to save to (legacy format). If None and checkpoint_manager provided, uses UUID format.
        training_stats : Optional[Dict]
            Additional training statistics to save.
        checkpoint_manager : Optional[CumulativeCheckpointManager]
            Checkpoint manager for UUID-based saving. If provided, uses .npz format.

        Returns
        -------
        Optional[str]
            UUID of saved checkpoint if using checkpoint_manager, None otherwise.
        """
        if checkpoint_manager is not None:
            # Use UUID-based cumulative checkpoint
            from src.ai_players.transformer_rl.checkpoint_manager import CumulativeCheckpointManager
            
            agent_state = {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            
            merged_stats = self.training_stats.copy()
            if training_stats is not None:
                merged_stats.update(training_stats)
            
            # Save experience buffer if available
            experience_buffer = None
            if hasattr(self, 'buffer') and len(self.buffer) > 0:
                experience_buffer = list(self.buffer.buffer)
            
            # Ensure optimizer state dict is properly structured
            optimizer_state = agent_state["optimizer_state_dict"]
            if not isinstance(optimizer_state, dict) or "param_groups" not in optimizer_state:
                # If optimizer state is not properly structured, get it fresh
                agent_state["optimizer_state_dict"] = self.optimizer.state_dict()
            
            uuid_str = checkpoint_manager.save_checkpoint(
                agent_state=agent_state,
                training_stats=merged_stats,
                experience_buffer=experience_buffer,
                metadata=training_stats,
            )
            return uuid_str
        else:
            # Legacy format
            if filepath is None:
                raise ValueError("filepath required when checkpoint_manager not provided")
            save_dict = {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_stats": self.training_stats,
            }
            if training_stats is not None:
                save_dict["additional_stats"] = training_stats
            torch.save(save_dict, filepath)
            return None

    def load(self, filepath: Optional[Path] = None, checkpoint_uuid: Optional[str] = None, checkpoint_manager: Optional["CumulativeCheckpointManager"] = None) -> Optional[Dict]:  # type: ignore
        """Load the agent from disk.

        Parameters
        ----------
        filepath : Optional[Path]
            Path to load from (legacy format).
        checkpoint_uuid : Optional[str]
            UUID of checkpoint to load (if using checkpoint_manager).
        checkpoint_manager : Optional[CumulativeCheckpointManager]
            Checkpoint manager for UUID-based loading.

        Returns
        -------
        Optional[Dict]
            Training statistics if available.
        """
        if checkpoint_manager is not None:
            # Use UUID-based cumulative checkpoint
            from src.ai_players.transformer_rl.checkpoint_manager import CumulativeCheckpointManager
            
            checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_uuid, device=self.device)
            if checkpoint_data is None:
                return None
            
            agent_state = checkpoint_data["agent_state"]
            
            # Load network state dict with strict=False to handle missing/new keys
            network_state = agent_state.get("network_state_dict", {})
            try:
                self.network.load_state_dict(network_state, strict=False)
                # Check if any keys were missing
                model_keys = set(self.network.state_dict().keys())
                checkpoint_keys = set(network_state.keys())
                missing_keys = model_keys - checkpoint_keys
                unexpected_keys = checkpoint_keys - model_keys
                if missing_keys:
                    print(f"Warning: Missing keys in checkpoint (using defaults): {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in checkpoint (ignored): {unexpected_keys}")
            except Exception as e:
                print(f"Warning: Could not load network state: {e}. Continuing with random initialization.")
            
            # Load optimizer state if it's properly structured
            optimizer_state = agent_state.get("optimizer_state_dict", {})
            if optimizer_state and "param_groups" in optimizer_state:
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except Exception as e:
                    print(f"Warning: Could not load optimizer state: {e}. Continuing with fresh optimizer state.")
            else:
                print("Warning: Optimizer state not found or invalid. Using fresh optimizer state.")
            self.training_stats = checkpoint_data.get("training_stats", {"policy_loss": [], "value_loss": [], "entropy": [], "total_loss": []})
            
            # Load experience buffer if available
            if checkpoint_data.get("experience_buffer") and hasattr(self, 'buffer'):
                self.buffer.buffer.clear()
                self.buffer.buffer.extend(checkpoint_data["experience_buffer"])
            
            return checkpoint_data.get("metadata")
        else:
            # Legacy format
            if filepath is None:
                raise ValueError("filepath required when checkpoint_manager not provided")
            if filepath.exists():
                checkpoint = torch.load(filepath, map_location=self.device)
                # Load network with strict=False for backward compatibility
                self.network.load_state_dict(checkpoint.get("network_state_dict", {}), strict=False)
                # Load optimizer if available
                optimizer_state = checkpoint.get("optimizer_state_dict", {})
                if optimizer_state and "param_groups" in optimizer_state:
                    try:
                        self.optimizer.load_state_dict(optimizer_state)
                    except Exception as e:
                        print(f"Warning: Could not load optimizer state: {e}")
                self.training_stats = checkpoint.get("training_stats", {"policy_loss": [], "value_loss": [], "entropy": [], "total_loss": []})
                return checkpoint.get("additional_stats")
            return None

