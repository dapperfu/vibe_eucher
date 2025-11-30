"""MCTS search algorithm for EucherGo using PUCT."""

from typing import Optional

import numpy as np
import torch

from eucher.players.computer.euchergo.action_space import ActionEncoder
from eucher.players.computer.euchergo.mcts.tree import EucherGoMCTSNode
from eucher.players.computer.euchergo.networks.model import EucherGoModel


class EucherGoMCTSSearch:
    """Monte Carlo Tree Search with PUCT for EucherGo.

    Uses value network for rollouts (no random rollouts).
    Supports batch GPU evaluation.
    """

    def __init__(
        self,
        model: EucherGoModel,
        num_simulations: int = 100,
        exploration_constant: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize MCTS search.

        Parameters
        ----------
        model : EucherGoModel
            EucherGo model.
        num_simulations : int
            Number of MCTS simulations per move.
        exploration_constant : float
            Exploration constant for PUCT.
        device : Optional[torch.device]
            Device to run on.
        """
        self.model = model
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.device = device if device is not None else torch.device("cpu")

    def search(
        self,
        state_tensor: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Run MCTS search.

        Parameters
        ----------
        state_tensor : torch.Tensor
            Current game state tensor [batch_size, state_size] or [state_size].
        action_mask : Optional[torch.Tensor]
            Action mask [batch_size, action_space_size] or [action_space_size].
        temperature : float
            Temperature for final policy extraction.

        Returns
        -------
        np.ndarray
            Improved policy distribution over actions [action_space_size].
        """
        # Handle batch dimension
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            single_batch = True
        else:
            single_batch = False

        if action_mask is not None and action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        # Move to device
        state_tensor = state_tensor.to(self.device)
        if action_mask is not None:
            action_mask = action_mask.to(self.device)

        # Set model to eval mode
        was_training = self.model.training
        self.model.eval()

        try:
            # For now, process first batch item (can be extended for batch MCTS)
            state = state_tensor[0]
            mask = action_mask[0] if action_mask is not None else None

            # Create root node
            root = EucherGoMCTSNode(state)

            # Run simulations
            for _ in range(self.num_simulations):
                self._simulate(root, mask)

            # Extract improved policy
            policy = self._extract_policy(root, temperature)

            return policy
        finally:
            # Restore training mode
            if was_training:
                self.model.train()

    def _simulate(self, root: EucherGoMCTSNode, action_mask: Optional[torch.Tensor]) -> None:
        """
        Run one MCTS simulation.

        Parameters
        ----------
        root : EucherGoMCTSNode
            Root node.
        action_mask : Optional[torch.Tensor]
            Action mask for legal moves.
        """
        node = root

        # Selection: traverse to leaf using PUCT
        while not node.is_leaf():
            action = node.select_action(self.exploration_constant)
            if action in node.children:
                node = node.children[action]
            else:
                break

        # Expansion and evaluation
        if node.visit_count > 0 or node == root:
            # Evaluate with model
            with torch.no_grad():
                state_batch = node.state.unsqueeze(0)  # Add batch dimension
                policy_logits, value = self.model(state_batch, action_mask.unsqueeze(0) if action_mask is not None else None)

                policy_dist = torch.softmax(policy_logits[0], dim=-1)
                value_scalar = value[0].item()

            # Expand children (only legal actions if mask provided)
            legal_actions = range(ActionEncoder.ACTION_SPACE_SIZE)
            if action_mask is not None:
                legal_actions = [a for a in range(ActionEncoder.ACTION_SPACE_SIZE) if action_mask[a].item() > 0.5]

            for action in legal_actions:
                prior = policy_dist[action].item()
                child = EucherGoMCTSNode(node.state, prior=prior, parent=node)
                node.add_child(action, child)

            # Backpropagate
            node.backpropagate(value_scalar)
        else:
            # Leaf node: evaluate and backpropagate
            with torch.no_grad():
                state_batch = node.state.unsqueeze(0)
                _, value = self.model(state_batch, action_mask.unsqueeze(0) if action_mask is not None else None)
                value_scalar = value[0].item()

            node.backpropagate(value_scalar)

    def _extract_policy(self, root: EucherGoMCTSNode, temperature: float = 1.0) -> np.ndarray:
        """
        Extract improved policy from root node.

        Parameters
        ----------
        root : EucherGoMCTSNode
            Root node.
        temperature : float
            Temperature for policy extraction.

        Returns
        -------
        np.ndarray
            Policy distribution [action_space_size].
        """
        policy = np.zeros(ActionEncoder.ACTION_SPACE_SIZE)
        total_visits = sum(child.visit_count for child in root.children.values())

        if total_visits == 0:
            # No visits: return uniform
            return np.ones(ActionEncoder.ACTION_SPACE_SIZE) / ActionEncoder.ACTION_SPACE_SIZE

        # Extract visit counts
        visit_counts = np.zeros(ActionEncoder.ACTION_SPACE_SIZE)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        # Apply temperature
        if temperature > 0:
            visit_counts = visit_counts ** (1.0 / temperature)
            total_visits = visit_counts.sum()

        # Normalize
        if total_visits > 0:
            policy = visit_counts / total_visits

        return policy


