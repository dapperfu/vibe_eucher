"""Checks that machine-learning personalities have a trained checkpoint."""

from pathlib import Path
from typing import Optional

# Counters saved by trainers. A checkpoint that records only zeros has not
# finished any training, even if a file exists.
_PROGRESS_KEYS = (
    "game_count",
    "games_played",
    "games_trained",
    "trick_count",
    "training_steps",
    "step_count",
    "total_games_played",
    "num_games",
)


class UntrainedPersonalityError(RuntimeError):
    """Raised when a machine-learning personality is played before training.

    Parameters
    ----------
    message : str
        Explanation of which personality is untrained and what is missing.
    """


def find_latest_checkpoint(directory: Path) -> Optional[Path]:
    """Return the newest checkpoint file in ``directory``.

    Parameters
    ----------
    directory : Path
        Directory that trainers use for checkpoints.

    Returns
    -------
    Optional[Path]
        Latest checkpoint, or None when the directory has no checkpoint files.
    """
    if not directory.exists():
        return None

    latest_link = directory / "latest_checkpoint.pt"
    if latest_link.exists():
        resolved = latest_link.resolve()
        if resolved.is_file():
            return resolved

    candidates = [
        path
        for pattern in ("*.pt", "*.pth", "*.pkl")
        for path in directory.glob(pattern)
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _progress_values(payload: dict) -> list[float]:
    """Collect numeric training counters from a checkpoint payload.

    Parameters
    ----------
    payload : dict
        Checkpoint dictionary, including optional nested metadata.

    Returns
    -------
    list[float]
        Values found for known training counters.
    """
    blobs = [payload]
    for key in ("metadata", "training_stats", "metrics", "additional_state"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            blobs.append(nested)

    values: list[float] = []
    for blob in blobs:
        for key in _PROGRESS_KEYS:
            if key not in blob or blob[key] is None:
                continue
            try:
                values.append(float(blob[key]))
            except (TypeError, ValueError):
                continue
    return values


def assert_checkpoint_adequate(personality: str, payload: dict, source: Path) -> None:
    """Raise when a checkpoint explicitly records zero completed training.

    A checkpoint with no training counters is accepted: saving the file is
    the trainer's signal that weights were written. A checkpoint that records
    games, tricks, or steps, all of them zero, has not been trained.

    Parameters
    ----------
    personality : str
        Personality type name, such as ``eucher_zero``.
    payload : dict
        Loaded checkpoint dictionary.
    source : Path
        Path the checkpoint was loaded from.

    Raises
    ------
    UntrainedPersonalityError
        When every recorded training counter is zero.
    """
    values = _progress_values(payload)
    if values and max(values) <= 0:
        raise UntrainedPersonalityError(
            f"{personality} has not been adequately trained. "
            f"Checkpoint {source} records no completed games, tricks, or steps."
        )


def resolve_checkpoint_path(
    personality: str,
    model_path: Optional[str],
    checkpoint_dir: Path,
    require_trained: bool,
) -> Optional[Path]:
    """Find the checkpoint a personality should load.

    Parameters
    ----------
    personality : str
        Personality type name.
    model_path : Optional[str]
        Explicit checkpoint path. None searches ``checkpoint_dir``.
    checkpoint_dir : Path
        Default directory written by that personality's trainer.
    require_trained : bool
        When True, a missing checkpoint is an error.

    Returns
    -------
    Optional[Path]
        Checkpoint to load, or None when an untrained network is allowed.

    Raises
    ------
    UntrainedPersonalityError
        When ``require_trained`` is True and no checkpoint exists.
    """
    if model_path is not None:
        path = Path(model_path)
        if not path.exists():
            raise UntrainedPersonalityError(missing_training_message(personality, path))
        return path

    found = find_latest_checkpoint(checkpoint_dir)
    if found is not None:
        return found
    if require_trained:
        raise UntrainedPersonalityError(missing_training_message(personality, checkpoint_dir))
    return None


def missing_training_message(personality: str, looked_in: Path) -> str:
    """Build the error text for a personality with no trained artifact.

    Parameters
    ----------
    personality : str
        Personality type name.
    looked_in : Path
        File or directory that was expected to hold trained weights.

    Returns
    -------
    str
        Message suitable for ``UntrainedPersonalityError``.
    """
    return (
        f"{personality} has not been adequately trained. "
        f"No trained model was found at {looked_in}. "
        "Train this personality and save a checkpoint before playing it."
    )
