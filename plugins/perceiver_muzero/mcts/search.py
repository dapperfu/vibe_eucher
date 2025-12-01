"""MuZero-style MCTS search algorithm for EucherPerceiverMuZero."""

from typing import Optional

import numpy as np
import torch

from ..action_space import ACTION_SPACE_SIZE
from ..config import PerceiverMuZeroConfig
from ..networks.model import PerceiverMuZeroModel
from .tree import MCTSNode


class MCTSSearch:
    """MuZero-style Monte Carlo Tree Search using learned dynamics."""

    def __init__(
        self,
        model: PerceiverMuZeroModel,
        config: PerceiverMuZeroConfig,
    ) -> None:
        """
        Initialize MCTS search.

        Parameters
        ----------
        model : PerceiverMuZeroModel
            PerceiverMuZero model.
        config : PerceiverMuZeroConfig
            Configuration object.
        """
        self.model = model
        self.config = config

    def search(
        self,
        input_tokens: torch.Tensor,
        risk_factor: float = 0.0,
        num_simulations: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run MuZero-style MCTS search.

        Parameters
        ----------
        input_tokens : torch.Tensor
            Current game state tokens [num_tokens, token_dim] or [batch, num_tokens, token_dim].
        risk_factor : float
            Risk factor for value scaling.
        num_simulations : Optional[int]
            Number of simulations to run. If None, uses config.num_simulations.

        Returns
        -------
        np.ndarray
            Improved policy distribution over actions [action_space_size].
        """
        # Use provided num_simulations or default
        sims_to_run = num_simulations if num_simulations is not None else self.config.num_simulations
        
        # Set model to eval mode
        was_training = self.model.training
        self.model.eval_mode()

        try:
            # Move tokens to device
            input_tokens = input_tokens.to(self.model.device)

            # Encode tokens through Perceiver-IO and representation network
            with torch.no_grad():
                perceiver_output = self.model.perceiver_encoder(input_tokens)
                latent = self.model.representation_net(perceiver_output)

            # Create root node
            root = MCTSNode(latent)

            # Run simulations
            for _ in range(sims_to_run):
                self._simulate(root, risk_factor, depth=0)

            # Extract improved policy
            policy = self._extract_policy(root)
            return policy
        finally:
            # Restore original training mode
            if was_training:
                self.model.train_mode()

    def _simulate(
        self, node: MCTSNode, risk_factor: float, depth: int
    ) -> float:
        """
        Run one MCTS simulation using learned dynamics.

        Parameters
        ----------
        node : MCTSNode
            Current node.
        risk_factor : float
            Risk factor for value scaling.
        depth : int
            Current search depth.

        Returns
        -------
        float
            Value estimate.
        """
        # Check depth limit
        if depth >= self.config.max_depth:
            # Use prediction network value at depth limit
            with torch.no_grad():
                _, value, risk_value = self.model.prediction_net(node.state)
            final_value = (
                risk_value.item() if risk_factor > 0 else value.item()
            )
            node.backpropagate(final_value)
            return final_value

        # If leaf node, expand and evaluate
        if node.is_leaf():
            # Evaluate with prediction network
            with torch.no_grad():
                policy_logits, value, risk_value = self.model.prediction_net(
                    node.state
                )
                policy = torch.softmax(policy_logits, dim=-1)

            # Expand children
            for action in range(ACTION_SPACE_SIZE):
                prior = policy[action].item()
                # Child state will be computed on-demand using dynamics
                child = MCTSNode(node.state, prior=prior, parent=node)
                node.add_child(action, child)

            # Use risk-adjusted value if risk_factor > 0
            final_value = (
                risk_value.item() if risk_factor > 0 else value.item()
            )
            node.backpropagate(final_value)
            return final_value

        # Selection: choose action using PUCT
        action = node.select_action(self.config.c_puct)

        # Get or create child
        if action not in node.children:
            # Should not happen if expansion worked correctly
            with torch.no_grad():
                policy_logits, _, _ = self.model.prediction_net(node.state)
                policy = torch.softmax(policy_logits, dim=-1)
                prior = policy[action].item()
            child = MCTSNode(node.state, prior=prior, parent=node)
            node.add_child(action, child)
        else:
            child = node.children[action]

        # Use learned dynamics to predict next state and reward
        with torch.no_grad():
            # Encode action as one-hot
            action_tensor = torch.zeros(
                ACTION_SPACE_SIZE, device=self.model.device
            )
            action_tensor[action] = 1.0

            # Predict next latent state and reward
            next_latent, reward = self.model.dynamics_net(
                node.state, action_tensor
            )

        # Update child state with predicted next state
        child.state = next_latent

        # Recursively simulate from child
        child_value = self._simulate(child, risk_factor, depth + 1)

        # Combine immediate reward with future value
        total_value = reward.item() + child_value

        # Backpropagate
        node.backpropagate(total_value)

        return total_value

    def _extract_policy(self, root: MCTSNode) -> np.ndarray:
        """
        Extract improved policy from root node.

        Parameters
        ----------
        root : MCTSNode
            Root node.

        Returns
        -------
        np.ndarray
            Policy distribution [action_space_size].
        """
        policy = np.zeros(ACTION_SPACE_SIZE)
        total_visits = sum(child.visit_count for child in root.children.values())

        if total_visits == 0:
            return np.ones(ACTION_SPACE_SIZE) / ACTION_SPACE_SIZE

        for action, child in root.children.items():
            policy[action] = child.visit_count / total_visits

        return policy

