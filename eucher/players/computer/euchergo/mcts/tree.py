"""MCTS tree structure for EucherGo."""

from typing import Dict, Optional

import numpy as np
import torch


class EucherGoMCTSNode:
    """Node in MCTS tree for EucherGo.

    Stores:
    - Prior probability from policy network
    - Visit count
    - Cumulative value
    - Children nodes
    """

    def __init__(
        self,
        state: torch.Tensor,
        prior: float = 0.0,
        parent: Optional["EucherGoMCTSNode"] = None,
    ) -> None:
        """
        Initialize MCTS node.

        Parameters
        ----------
        state : torch.Tensor
            State tensor representation.
        prior : float
            Prior probability from policy network.
        parent : Optional[EucherGoMCTSNode]
            Parent node.
        """
        self.state = state
        self.prior = prior
        self.parent = parent

        self.visit_count = 0
        self.value_sum = 0.0

        self.children: Dict[int, "EucherGoMCTSNode"] = {}

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

    def puct_score(self, exploration_constant: float = 1.0) -> Dict[int, float]:
        """
        Calculate PUCT scores for all children.

        PUCT formula: Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Parameters
        ----------
        exploration_constant : float
            Exploration constant for PUCT.

        Returns
        -------
        Dict[int, float]
            Action -> PUCT score mapping.
        """
        scores = {}
        parent_visits = self.visit_count if self.parent is None else self.parent.visit_count

        for action, child in self.children.items():
            if child.visit_count == 0:
                # Unexplored: use prior only
                scores[action] = exploration_constant * child.prior * np.sqrt(parent_visits + 1)
            else:
                # Exploitation term
                exploitation = child.get_value()
                # Exploration term
                exploration = (
                    exploration_constant
                    * child.prior
                    * np.sqrt(parent_visits)
                    / (1 + child.visit_count)
                )
                scores[action] = exploitation + exploration
        return scores

    def select_action(self, exploration_constant: float = 1.0) -> int:
        """
        Select action using PUCT.

        Parameters
        ----------
        exploration_constant : float
            Exploration constant.

        Returns
        -------
        int
            Selected action ID.
        """
        scores = self.puct_score(exploration_constant)
        return max(scores.items(), key=lambda x: x[1])[0]

    def add_child(self, action: int, child: "EucherGoMCTSNode") -> None:
        """
        Add child node.

        Parameters
        ----------
        action : int
            Action leading to child.
        child : EucherGoMCTSNode
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


