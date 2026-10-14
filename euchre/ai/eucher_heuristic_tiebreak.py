"""EucherHeuristicTiebreak, a one-hidden-layer net for close trump calls.

The net answers a single question: given a gray-area hand, is calling this
suit better than passing? Clear hands never reach it. Training uses binary
cross-entropy on labels from simulated deals, and checkpoints keep the
weights, the optimizer, and the random state so a run can resume.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, cast

import numpy
import torch
from torch import nn

from .gray_area import FEATURE_COUNT, FEATURE_NAMES

MODEL_NAME = "EucherHeuristicTiebreak"


class EucherHeuristicTiebreak(nn.Module):
    """Tiny classifier that breaks a heuristic tie on one trump suit."""

    def __init__(self, n_features: int = FEATURE_COUNT, hidden_size: int = 16) -> None:
        """Build the micro network.

        Parameters
        ----------
        n_features : int
            Width of the gray-area feature vector.
        hidden_size : int
            Width of the single hidden layer.
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return call-versus-pass logits.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``(batch, n_features)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)``.
        """
        return cast(torch.Tensor, self.net(features).squeeze(-1))


class TiebreakPredictor:
    """Frozen net that returns the probability that calling is better."""

    def __init__(self, module: EucherHeuristicTiebreak) -> None:
        """Wrap a trained module for play.

        Parameters
        ----------
        module : EucherHeuristicTiebreak
            A module whose weights are already loaded.
        """
        self.module = module
        self.module.eval()

    def call_probability(self, features: Sequence[float]) -> float:
        """Return the probability that calling beats passing.

        Parameters
        ----------
        features : Sequence[float]
            One gray-area feature vector.

        Returns
        -------
        float
            A probability in ``[0, 1]``.
        """
        self.module.eval()
        with torch.no_grad():
            row = torch.tensor([list(features)], dtype=torch.float32)
            logit = self.module(row)
            return float(torch.sigmoid(logit)[0].item())


def default_checkpoint_path() -> Path:
    """Return the checkpoint a player loads when no path is given.

    Returns
    -------
    Path
        ``checkpoints/EucherHeuristicTiebreak/checkpoint_latest.pth``.
    """
    return Path("checkpoints") / MODEL_NAME / "checkpoint_latest.pth"


def save_checkpoint(
    path: Path,
    module: EucherHeuristicTiebreak,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    history: List[Dict[str, float]],
    config: Mapping[str, Any],
) -> None:
    """Write a resumable checkpoint.

    Parameters
    ----------
    path : Path
        Destination file. Parent directories are created.
    module : EucherHeuristicTiebreak
        The network to store.
    optimizer : torch.optim.Optimizer
        Optimizer, stored so training can resume.
    epoch : int
        Number of epochs already completed.
    best_val_loss : float
        Best validation loss seen so far.
    history : List[Dict[str, float]]
        Per-epoch metrics.
    config : Mapping[str, Any]
        Training configuration stored beside the weights.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": MODEL_NAME,
            "model_state": module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "history": history,
            "hidden_size": module.hidden_size,
            "n_features": module.n_features,
            "feature_names": list(FEATURE_NAMES),
            "config": dict(config),
            "rng_state": {
                "python": random.getstate(),
                "numpy": numpy.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        },
        path,
    )


def _payload(path: Path) -> Dict[str, Any]:
    """Load a checkpoint dictionary and check that it is this model.

    Parameters
    ----------
    path : Path
        Checkpoint file.

    Returns
    -------
    Dict[str, Any]
        The stored payload.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is a different model or a different feature layout.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No tie-break checkpoint at {path}")
    payload = cast(Dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
    if payload.get("model_name") != MODEL_NAME:
        raise ValueError(f"{path} is not an {MODEL_NAME} checkpoint")
    names = list(payload.get("feature_names", []))
    if names and names != list(FEATURE_NAMES):
        raise ValueError(f"{path} was trained with a different feature layout")
    return payload


def load_module(path: Path) -> EucherHeuristicTiebreak:
    """Rebuild the network from a checkpoint.

    Parameters
    ----------
    path : Path
        Checkpoint file.

    Returns
    -------
    EucherHeuristicTiebreak
        The module in evaluation mode.
    """
    payload = _payload(path)
    module = EucherHeuristicTiebreak(
        n_features=int(payload.get("n_features", FEATURE_COUNT)),
        hidden_size=int(payload.get("hidden_size", 16)),
    )
    module.load_state_dict(payload["model_state"])
    module.eval()
    return module


def load_tiebreak_predictor(path: Path) -> TiebreakPredictor:
    """Load a checkpoint as a player-facing predictor.

    Parameters
    ----------
    path : Path
        Checkpoint file.

    Returns
    -------
    TiebreakPredictor
        A predictor that returns call probabilities.
    """
    return TiebreakPredictor(load_module(path))


def _restore_rng(payload: Mapping[str, Any]) -> None:
    """Restore the random state stored in a checkpoint.

    Parameters
    ----------
    payload : Mapping[str, Any]
        A checkpoint dictionary that includes ``rng_state``.
    """
    state = payload.get("rng_state")
    if not isinstance(state, dict):
        return
    random.setstate(state["python"])
    numpy.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def _epoch_loss(
    module: EucherHeuristicTiebreak,
    features: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> float:
    """Mean loss of one split, without updating weights.

    Parameters
    ----------
    module : EucherHeuristicTiebreak
        The network being scored.
    features : torch.Tensor
        Feature rows.
    labels : torch.Tensor
        Binary labels.
    criterion : nn.Module
        Loss module.

    Returns
    -------
    float
        Mean loss. An empty split returns 0.
    """
    if len(labels) == 0:
        return 0.0
    module.eval()
    with torch.no_grad():
        logits = module(features)
        return float(criterion(logits, labels).item())


def _accuracy(
    module: EucherHeuristicTiebreak, features: torch.Tensor, labels: torch.Tensor
) -> float:
    """Accuracy of one split at probability 0.5.

    Parameters
    ----------
    module : EucherHeuristicTiebreak
        The network being scored.
    features : torch.Tensor
        Feature rows.
    labels : torch.Tensor
        Binary labels.

    Returns
    -------
    float
        Fraction of rows predicted correctly. An empty split returns 0.
    """
    if len(labels) == 0:
        return 0.0
    module.eval()
    with torch.no_grad():
        predicted = (torch.sigmoid(module(features)) >= 0.5).float()
    return float((predicted == labels).float().mean().item())


def fit_tiebreak(
    features: torch.Tensor,
    labels: torch.Tensor,
    split: torch.Tensor,
    config: Mapping[str, Any],
    checkpoint_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    hidden_size: int,
    checkpoint_interval: int,
    seed: int,
    resume: Optional[Path] = None,
    no_resume: bool = False,
) -> Dict[str, Any]:
    """Train on gray-area rows and write checkpoints.

    Parameters
    ----------
    features : torch.Tensor
        Shape ``(n, n_features)``.
    labels : torch.Tensor
        Shape ``(n,)``, 1 when calling beat passing.
    split : torch.Tensor
        0 train, 1 validation, 2 test. The test rows are not used.
    config : Mapping[str, Any]
        Configuration stored in each checkpoint.
    checkpoint_dir : Path
        Directory for ``checkpoint_latest.pth`` and ``checkpoint_best.pth``.
    epochs : int
        Total epochs to complete, including epochs already in a resume file.
    learning_rate : float
        Adam learning rate.
    weight_decay : float
        Adam weight decay.
    batch_size : int
        Mini-batch size.
    hidden_size : int
        Hidden layer width. A resumed run keeps the checkpoint width.
    checkpoint_interval : int
        Write a numbered checkpoint every this many epochs.
    seed : int
        Seed used when a run starts from scratch.
    resume : Path, optional
        Checkpoint to resume. When omitted, the latest file in
        ``checkpoint_dir`` is used if it exists.
    no_resume : bool
        Start from scratch even if a latest checkpoint exists.

    Returns
    -------
    Dict[str, Any]
        ``best_path``, ``latest_path``, ``history``, and ``best_val_loss``.

    Raises
    ------
    ValueError
        If there are no training rows.
    """
    train_mask = split == 0
    val_mask = split == 1
    train_x = features[train_mask]
    train_y = labels[train_mask]
    val_x = features[val_mask]
    val_y = labels[val_mask]
    if len(train_y) == 0:
        raise ValueError("The gray-area dataset has no training rows")
    if int(features.shape[1]) != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} features, got {int(features.shape[1])}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_dir / "checkpoint_latest.pth"
    best_path = checkpoint_dir / "checkpoint_best.pth"
    resume_path = resume
    if resume_path is None and not no_resume and latest.is_file():
        resume_path = latest

    if resume_path is not None:
        payload = _payload(resume_path)
        module = load_module(resume_path)
        start_epoch = int(payload["epoch"])
        best_val_loss = float(payload["best_val_loss"])
        history = list(payload.get("history", []))
        _restore_rng(payload)
    else:
        torch.manual_seed(seed)
        module = EucherHeuristicTiebreak(features.shape[1], hidden_size)
        start_epoch = 0
        best_val_loss = float("inf")
        history = []

    positives = float(train_y.sum().item())
    negatives = float(len(train_y) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if resume_path is not None:
        optimizer.load_state_dict(_payload(resume_path)["optimizer_state"])

    for epoch in range(start_epoch, epochs):
        module.train()
        order = torch.randperm(len(train_y))
        total_loss = 0.0
        seen = 0
        for start in range(0, len(train_y), max(1, batch_size)):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(module(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            seen += len(batch)
        train_loss = total_loss / max(seen, 1)
        score_loss = criterion
        val_loss = _epoch_loss(module, val_x, val_y, score_loss) if len(val_y) else train_loss
        record = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": _accuracy(module, train_x, train_y),
            "val_accuracy": _accuracy(module, val_x, val_y)
            if len(val_y)
            else _accuracy(module, train_x, train_y),
        }
        history.append(record)
        completed = epoch + 1
        save_checkpoint(latest, module, optimizer, completed, best_val_loss, history, config)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_path, module, optimizer, completed, best_val_loss, history, config)
            save_checkpoint(latest, module, optimizer, completed, best_val_loss, history, config)
        if checkpoint_interval > 0 and completed % checkpoint_interval == 0:
            numbered = checkpoint_dir / f"checkpoint_epoch_{completed}.pth"
            save_checkpoint(numbered, module, optimizer, completed, best_val_loss, history, config)

    if not best_path.is_file():
        save_checkpoint(
            best_path, module, optimizer, max(start_epoch, epochs), best_val_loss, history, config
        )
    return {
        "best_path": best_path,
        "latest_path": latest,
        "history": history,
        "best_val_loss": best_val_loss,
    }
