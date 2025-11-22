"""Model management for loading and saving PyTorch AI player models."""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from eucher.players.computer.ml.pytorch.pytorch_networks import HybridNetwork, create_network


class ModelManager:
    """Handles model loading, saving, and versioning.

    Parameters
    ----------
    model_dir : str
        Directory for storing models.
    model_name : str
        Base name for model files.
    """

    MODEL_VERSION = "1.0"
    METADATA_FILE = "model_metadata.json"

    def __init__(self, model_dir: str = "models", model_name: str = "pytorch_ai_player") -> None:
        """Initialize model manager.

        Parameters
        ----------
        model_dir : str
            Directory for models.
        model_name : str
            Base model name.
        """
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, suffix: str = "") -> Path:
        """Get path to model file.

        Parameters
        ----------
        suffix : str
            Optional suffix for model filename.

        Returns
        -------
        Path
            Path to model file.
        """
        filename = f"{self.model_name}{suffix}.pth"
        return self.model_dir / filename

    def get_metadata_path(self) -> Path:
        """Get path to metadata file.

        Returns
        -------
        Path
            Path to metadata file.
        """
        return self.model_dir / self.METADATA_FILE

    def save_model(
        self,
        model: HybridNetwork,
        metadata: Optional[Dict] = None,
        suffix: str = "",
    ) -> Path:
        """Save model weights and metadata.

        Parameters
        ----------
        model : HybridNetwork
            Model to save.
        metadata : Optional[Dict]
            Additional metadata to save.
        suffix : str
            Optional suffix for filename.

        Returns
        -------
        Path
            Path to saved model file.
        """
        model_path = self.get_model_path(suffix)
        metadata_path = self.get_metadata_path()

        # Save model state dict
        torch.save(model.state_dict(), model_path)

        # Save metadata
        model_metadata = {
            "version": self.MODEL_VERSION,
            "model_path": str(model_path),
            "parameter_count": model.get_parameter_count(),
            "architecture": {
                "card_embedding_dim": getattr(model.card_embedding.embedding, "out_features", 32),
                "cnn_channels": 64,  # Default, could be extracted from model
                "lstm_hidden_dim": model.lstm_branch.hidden_size,
                "attention_heads": 4,  # Default
                "hidden_dim": 256,  # Default
            },
        }

        if metadata:
            model_metadata.update(metadata)

        # Load existing metadata if present
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    existing_metadata = json.load(f)
                # Update with new model info
                if "models" not in existing_metadata:
                    existing_metadata["models"] = []
                existing_metadata["models"].append(model_metadata)
                model_metadata = existing_metadata
            except (json.JSONDecodeError, KeyError):
                model_metadata = {"models": [model_metadata]}

        with open(metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=2)

        return model_path

    def load_model(
        self,
        model_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> Tuple[HybridNetwork, Dict]:
        """Load model from file.

        Parameters
        ----------
        model_path : Optional[Path]
            Path to model file. If None, loads latest model.
        device : Optional[torch.device]
            Device to load model on.
        strict : bool
            Whether to strictly enforce that keys match.

        Returns
        -------
        Tuple[HybridNetwork, Dict]
            Loaded model and metadata.
        """
        if model_path is None:
            model_path = self._find_latest_model()

        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load metadata
        metadata = self._load_metadata(model_path)

        # Create model with architecture from metadata
        arch = metadata.get("architecture", {})
        model = create_network(
            card_embedding_dim=arch.get("card_embedding_dim", 32),
            cnn_channels=arch.get("cnn_channels", 64),
            lstm_hidden_dim=arch.get("lstm_hidden_dim", 128),
            attention_heads=arch.get("attention_heads", 4),
            hidden_dim=arch.get("hidden_dim", 256),
            device=device,
        )

        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=strict)

        return model, metadata

    def _find_latest_model(self) -> Optional[Path]:
        """Find the latest model file.

        Returns
        -------
        Optional[Path]
            Path to latest model, or None if not found.
        """
        model_path = self.get_model_path()
        if model_path.exists():
            return model_path

        # Try to find any model file with the base name
        pattern = f"{self.model_name}*.pth"
        matches = list(self.model_dir.glob(pattern))
        if matches:
            # Return most recently modified
            return max(matches, key=lambda p: p.stat().st_mtime)

        return None

    def _load_metadata(self, model_path: Path) -> Dict:
        """Load metadata for a model.

        Parameters
        ----------
        model_path : Path
            Path to model file.

        Returns
        -------
        Dict
            Model metadata.
        """
        metadata_path = self.get_metadata_path()
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    all_metadata = json.load(f)
                # Find metadata for this model
                if "models" in all_metadata:
                    for model_meta in all_metadata["models"]:
                        if model_meta.get("model_path") == str(model_path):
                            return model_meta
                return all_metadata
            except (json.JSONDecodeError, KeyError):
                pass

        # Return default metadata
        return {
            "version": self.MODEL_VERSION,
            "model_path": str(model_path),
            "architecture": {
                "card_embedding_dim": 32,
                "cnn_channels": 64,
                "lstm_hidden_dim": 128,
                "attention_heads": 4,
                "hidden_dim": 256,
            },
        }

    def check_compatibility(self, model_path: Path, expected_architecture: Dict) -> bool:
        """Check if a model is compatible with expected architecture.

        Parameters
        ----------
        model_path : Path
            Path to model file.
        expected_architecture : Dict
            Expected architecture parameters.

        Returns
        -------
        bool
            True if compatible, False otherwise.
        """
        try:
            metadata = self._load_metadata(model_path)
            arch = metadata.get("architecture", {})

            for key, expected_value in expected_architecture.items():
                if arch.get(key) != expected_value:
                    return False

            return True
        except Exception:
            return False

    def list_models(self) -> list[Dict]:
        """List all available models.

        Returns
        -------
        list[Dict]
            List of model metadata dictionaries.
        """
        metadata_path = self.get_metadata_path()
        if not metadata_path.exists():
            return []

        try:
            with open(metadata_path, "r") as f:
                all_metadata = json.load(f)
            if "models" in all_metadata:
                return all_metadata["models"]
            return [all_metadata] if all_metadata else []
        except (json.JSONDecodeError, KeyError):
            return []

