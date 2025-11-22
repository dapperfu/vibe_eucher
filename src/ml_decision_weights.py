"""Utilities for handling decision weights and temperature thresholds."""

from typing import List, Tuple

import numpy as np
import torch


def apply_temperature_threshold(
    weights: np.ndarray, temperature: float, valid_indices: List[int]
) -> Tuple[List[int], np.ndarray]:
    """
    Apply temperature threshold to filter actions by decision weights.

    Temperature acts as threshold: actions with weight >= temperature are considered valid.
    - Low temperature (0.0-0.3): Conservative (only high-confidence actions)
    - Medium temperature (0.3-0.7): Balanced
    - High temperature (0.7-1.0): Risky (accepts lower-confidence actions)

    Parameters
    ----------
    weights : np.ndarray
        Decision weights (confidence scores) for each action.
    temperature : float
        Temperature threshold (0.0-1.0). Actions with weight >= temperature are valid.
    valid_indices : List[int]
        List of initially valid action indices.

    Returns
    -------
    Tuple[List[int], np.ndarray]
        Tuple of (filtered_valid_indices, filtered_weights).
        filtered_valid_indices: List of action indices that pass the temperature threshold.
        filtered_weights: Filtered weights array (invalid actions set to -inf).
    """
    # Normalize weights to [0, 1] range if needed
    if weights.max() > 1.0 or weights.min() < 0.0:
        # Normalize to [0, 1] using softmax-like normalization
        weights_normalized = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    else:
        weights_normalized = weights.copy()

    # Create filtered weights array
    filtered_weights = np.full_like(weights_normalized, float("-inf"))
    filtered_valid_indices: List[int] = []

    # Only include actions that are both valid and pass temperature threshold
    for idx in valid_indices:
        if 0 <= idx < len(weights_normalized):
            weight = weights_normalized[idx]
            if weight >= temperature:
                filtered_weights[idx] = weight
                filtered_valid_indices.append(idx)

    # If no actions pass threshold, fall back to all valid actions (use best one)
    if not filtered_valid_indices:
        for idx in valid_indices:
            if 0 <= idx < len(weights_normalized):
                filtered_weights[idx] = weights_normalized[idx]
                filtered_valid_indices.append(idx)

    return filtered_valid_indices, filtered_weights


def get_decision_weights_from_probs(probs: np.ndarray) -> np.ndarray:
    """
    Convert probability distribution to decision weights.

    Decision weights represent confidence scores for each action.
    For probabilities, weights are simply the probabilities themselves.

    Parameters
    ----------
    probs : np.ndarray
        Probability distribution of shape (n_actions,) or (1, n_actions).

    Returns
    -------
    np.ndarray
        Decision weights of shape (n_actions,).
    """
    if probs.ndim > 1:
        probs = probs.squeeze()
    return probs.astype(np.float32)


def get_decision_weights_from_logits(logits: torch.Tensor) -> np.ndarray:
    """
    Convert logits to decision weights using softmax.

    Parameters
    ----------
    logits : torch.Tensor
        Logits tensor of shape (batch_size, n_actions) or (n_actions,).

    Returns
    -------
    np.ndarray
        Decision weights (probabilities) of shape (n_actions,).
    """
    if logits.dim() > 1:
        logits = logits.squeeze()
    probs = torch.softmax(logits, dim=0)
    return probs.detach().cpu().numpy().astype(np.float32)


def select_action_from_weights(
    weights: np.ndarray, valid_indices: List[int], temperature: float
) -> Tuple[int, float]:
    """
    Select action from decision weights with temperature threshold.

    Parameters
    ----------
    weights : np.ndarray
        Decision weights for each action.
    valid_indices : List[int]
        List of valid action indices.
    temperature : float
        Temperature threshold (0.0-1.0).

    Returns
    -------
    Tuple[int, float]
        Tuple of (selected_action_index, selected_action_weight).
    """
    filtered_indices, filtered_weights = apply_temperature_threshold(
        weights, temperature, valid_indices
    )

    if not filtered_indices:
        # Fallback: select from all valid indices
        best_idx = max(valid_indices, key=lambda i: weights[i] if 0 <= i < len(weights) else -np.inf)
        return best_idx, float(weights[best_idx] if 0 <= best_idx < len(weights) else 0.0)

    # Select action with highest weight among filtered
    best_idx = max(filtered_indices, key=lambda i: filtered_weights[i])
    return best_idx, float(filtered_weights[best_idx])

