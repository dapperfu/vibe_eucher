"""Checks that machine-learning personalities have a trained checkpoint."""

import json
import pickle
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


# Default locations used when a personality is constructed for play.
_CHECKPOINT_DIRECTORIES = {
    "eucher_zero": "eucher_zero",
    "perceiver_muzero": "perceiver_muzero",
    "euchergo": "euchrego",
    "reinforcement_eucher": "reinforcement_eucher",
}
_PYTORCH_NAMES = frozenset({"ml_pytorch", "pytorch_ai", "pytorch_strategic"})
_SKLEARN_FILES = ("order_up.pkl", "call_trump.pkl", "play_card.pkl", "discard.pkl")


def personality_is_trained(name: str, models_dir: Optional[Path] = None) -> bool:
    """Return whether a personality has the training required to play.

    The check uses the same artifacts play loads: a checkpoint whose recorded
    games, tricks, or steps are not all zero, a saved weight file, or four
    fitted sklearn models. Rule-based personalities are not trained models.

    Parameters
    ----------
    name : str
        Personality type name, such as ``ml_pytorch`` or ``eucher_zero``.
    models_dir : Optional[Path]
        Directory that holds ``models`` artifacts. None uses ``models`` in
        the working directory, which is where play looks.

    Returns
    -------
    bool
        True when the personality can be seated without
        ``UntrainedPersonalityError``.
    """
    root = Path("models") if models_dir is None else models_dir
    lowered = name.strip().lower()
    if lowered in _CHECKPOINT_DIRECTORIES:
        directory = root / "checkpoints" / _CHECKPOINT_DIRECTORIES[lowered]
        return _checkpoint_directory_is_trained(lowered, directory)
    if lowered in _PYTORCH_NAMES:
        return _pytorch_is_trained(root)
    if lowered == "ml_gan":
        return _weight_file_loads(root / "gan_model.pth")
    if lowered == "ml_rl":
        return _rl_is_trained(root / "rl_model.pth")
    if lowered == "ml_sklearn":
        return _sklearn_is_trained(root)
    return False


def _checkpoint_directory_is_trained(personality: str, directory: Path) -> bool:
    """Return whether the checkpoint play would load is adequate.

    Parameters
    ----------
    personality : str
        Personality type name.
    directory : Path
        Checkpoint directory for that personality.

    Returns
    -------
    bool
        True when the newest checkpoint exists and is not zero-progress.
    """
    found = find_latest_checkpoint(directory)
    if found is None:
        return False
    payload = _load_torch_payload(found)
    if payload is None:
        return False
    if not isinstance(payload, dict):
        return True
    try:
        assert_checkpoint_adequate(personality, payload, found)
    except UntrainedPersonalityError:
        return False
    return True


def _pytorch_is_trained(models_dir: Path) -> bool:
    """Return whether PyTorch Strategic has a saved weight file.

    Parameters
    ----------
    models_dir : Path
        Directory that holds ``pytorch_ai_player.pth``.

    Returns
    -------
    bool
        True when a weight file exists and its metadata is not zero-progress.
    """
    primary = models_dir / "pytorch_ai_player.pth"
    if primary.is_file():
        path = primary
    else:
        matches = [item for item in models_dir.glob("pytorch_ai_player*.pth") if item.is_file()]
        if not matches:
            return False
        path = max(matches, key=lambda item: item.stat().st_mtime)

    metadata = _pytorch_metadata(models_dir, path)
    try:
        assert_checkpoint_adequate("ml_pytorch", metadata, path)
    except UntrainedPersonalityError:
        return False
    return True


def _pytorch_metadata(models_dir: Path, model_path: Path) -> dict:
    """Load the metadata dictionary play checks for PyTorch Strategic.

    Parameters
    ----------
    models_dir : Path
        Directory that holds ``model_metadata.json``.
    model_path : Path
        Weight file whose metadata entry should be used.

    Returns
    -------
    dict
        Metadata for ``model_path``, or an empty dictionary when none exists.
    """
    metadata_path = models_dir / "model_metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        loaded = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    models = loaded.get("models")
    if isinstance(models, list):
        for entry in models:
            if isinstance(entry, dict) and entry.get("model_path") == str(model_path):
                return entry
    return loaded


def _rl_is_trained(path: Path) -> bool:
    """Return whether the RL weight file records completed training.

    Parameters
    ----------
    path : Path
        ``rl_model.pth``.

    Returns
    -------
    bool
        True when the file loads and its counters are not all zero.
    """
    payload = _load_torch_payload(path)
    if payload is None:
        return False
    if not isinstance(payload, dict):
        return True
    stats = payload.get("training_stats")
    checked = stats if isinstance(stats, dict) else payload
    try:
        assert_checkpoint_adequate("ml_rl", checked, path)
    except UntrainedPersonalityError:
        return False
    return True


def _weight_file_loads(path: Path) -> bool:
    """Return whether a weight file exists and can be read.

    Parameters
    ----------
    path : Path
        Saved model file.

    Returns
    -------
    bool
        True when ``torch.load`` succeeds.
    """
    return _load_torch_payload(path) is not None


def _sklearn_is_trained(models_dir: Path) -> bool:
    """Return whether all four supervised classifiers are fitted.

    Parameters
    ----------
    models_dir : Path
        Directory that holds the ``.pkl`` classifiers.

    Returns
    -------
    bool
        True when order-up, call-trump, discard, and card play are fitted.
    """
    for filename in _SKLEARN_FILES:
        path = models_dir / filename
        if not path.is_file() or not _sklearn_file_is_fitted(path):
            return False
    return True


def _sklearn_file_is_fitted(path: Path) -> bool:
    """Return whether a pickled classifier has been fitted.

    Parameters
    ----------
    path : Path
        Pickled sklearn model.

    Returns
    -------
    bool
        True when the loaded object has a fitted-model attribute.
    """
    try:
        with path.open("rb") as handle:
            model = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, AttributeError, ModuleNotFoundError):
        return False
    return (
        hasattr(model, "n_features_in_")
        or hasattr(model, "classes_")
        or hasattr(model, "_label_binarizer")
        or hasattr(model, "out_activation_")
    )


def _load_torch_payload(path: Path) -> Optional[object]:
    """Load a torch checkpoint, or return None when it is missing or unreadable.

    Parameters
    ----------
    path : Path
        Checkpoint or weight file.

    Returns
    -------
    Optional[object]
        Loaded object, or None when the file cannot be used.
    """
    if not path.is_file():
        return None
    try:
        import torch
    except ImportError:
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, pickle.UnpicklingError, AttributeError, EOFError, ValueError):
        return None
