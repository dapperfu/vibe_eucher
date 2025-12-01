"""MCTS search algorithm for EucherZero."""

from typing import Dict, Optional

import numpy as np
import torch

from ..action_space import ACTION_SPACE_SIZE
from ..config import EucherZeroConfig
from .tree import MCTSNode
from ..networks.model import EucherZeroModel


class MCTSSearch:
    """Monte Carlo Tree Search with belief sampling."""

    def __init__(
        self,
        model: EucherZeroModel,
        config: EucherZeroConfig,
    ) -> None:
        """
        Initialize MCTS search.

        Parameters
        ----------
        model : EucherZeroModel
            EucherZero model.
        config : EucherZeroConfig
            Configuration object.
        """
        self.model = model
        self.config = config

    def search(
        self,
        state_tensor: torch.Tensor,
        risk_factor: float = 0.0,
        num_simulations: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run MCTS search.

        Parameters
        ----------
        state_tensor : torch.Tensor
            Current game state tensor.
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
        
        # Set model to eval mode for inference (required for BatchNorm with batch size 1)
        was_training = self.model.training
        self.model.eval_mode()

        try:
            # Move state to device
            state_tensor = state_tensor.to(self.model.device)

            # Encode state to latent
            with torch.no_grad():
                latent = self.model.representation_net(state_tensor)

            # Create root node
            root = MCTSNode(latent)

            # Run simulations
            for _ in range(sims_to_run):
                self._simulate(root, risk_factor)

            # Extract improved policy
            policy = self._extract_policy(root)
            return policy
        finally:
            # Restore original training mode
            if was_training:
                self.model.train_mode()

    def _simulate(self, root: MCTSNode, risk_factor: float) -> None:
        """
        Run one MCTS simulation.

        Parameters
        ----------
        root : MCTSNode
            Root node.
        risk_factor : float
            Risk factor for value scaling.
        """
        node = root

        # Selection: traverse to leaf
        while not node.is_leaf():
            action = node.select_action(self.config.exploration_constant)
            if action in node.children:
                node = node.children[action]
            else:
                break

        # Expansion and evaluation
        if node.visit_count > 0 or node == root:
            # Evaluate with prediction network
            with torch.no_grad():
                policy_logits, value, risk_value = self.model.prediction_net(node.state)
                policy = torch.softmax(policy_logits, dim=-1)

            # Expand children
            for action in range(ACTION_SPACE_SIZE):
                prior = policy[action].item()
                child = MCTSNode(node.state, prior=prior, parent=node)
                node.add_child(action, child)

            # Use risk-adjusted value if risk_factor > 0
            final_value = (
                risk_value.item() if risk_factor > 0 else value.item()
            )

            # Backpropagate
            node.backpropagate(value.item(), final_value)
        else:
            # Leaf node: use network value directly
            with torch.no_grad():
                _, value, risk_value = self.model.prediction_net(node.state)

            final_value = (
                risk_value.item() if risk_factor > 0 else value.item()
            )
            node.backpropagate(value.item(), final_value)

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

