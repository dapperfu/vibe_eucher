"""MCTS tree structure for MuZero-style search."""

from typing import Dict, Optional

import numpy as np
import torch


class MCTSNode:
    """Node in MuZero-style MCTS tree."""

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

    def ucb_score(self, c_puct: float = 1.5) -> Dict[int, float]:
        """
        Calculate UCB (PUCT) scores for all children.

        Parameters
        ----------
        c_puct : float
            Exploration constant for PUCT (default: 1.5).

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
                    c_puct
                    * child.prior
                    * np.sqrt(self.visit_count)
                    / (1 + child.visit_count)
                )
                scores[action] = exploitation + exploration
        return scores

    def select_action(self, c_puct: float = 1.5) -> int:
        """
        Select action using PUCT.

        Parameters
        ----------
        c_puct : float
            Exploration constant (default: 1.5).

        Returns
        -------
        int
            Selected action ID.
        """
        scores = self.ucb_score(c_puct)
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

    def backpropagate(self, value: float) -> None:
        """
        Backpropagate value up the tree.

        Parameters
        ----------
        value : float
            Value to propagate.
        """
        self.visit_count += 1
        self.value_sum += value

        if self.parent is not None:
            self.parent.backpropagate(value)

