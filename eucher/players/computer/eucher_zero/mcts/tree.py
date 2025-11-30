"""MCTS tree structure."""

from typing import Dict, Optional

import numpy as np
import torch


class MCTSNode:
    """Node in MCTS tree."""

    def __init__(
        self,
        state: torch.Tensor,
        prior: float = 0.0,
        parent: Optional["MCTSNode"] = None,
    ) -> None:
        """
        Initialize MCTS node.

        Parameters
        ----------
        state : torch.Tensor
            Latent state representation.
        prior : float
            Prior probability from policy network.
        parent : Optional[MCTSNode]
            Parent node.
        """
        self.state = state
        self.prior = prior
        self.parent = parent

        self.visit_count = 0
        self.value_sum = 0.0
        self.risk_value_sum = 0.0

        self.children: Dict[int, "MCTSNode"] = {}

    def is_leaf(self) -> bool:
        """
        Check if node is a leaf.

        Returns
        -------
        bool
            True if node is a leaf (no children).
        """
        return len(self.children) == 0

    def get_value(self) -> float:
        """
        Get average value.

        Returns
        -------
        float
            Average value.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def get_risk_value(self) -> float:
        """
        Get average risk-adjusted value.

        Returns
        -------
        float
            Average risk-adjusted value.
        """
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
                scores[action] = float("inf")
            else:
                exploitation = child.get_value()
                exploration = (
                    exploration_constant
                    * child.prior
                    * np.sqrt(self.visit_count)
                    / (1 + child.visit_count)
                )
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

