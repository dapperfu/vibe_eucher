"""Checkpoint management for training state persistence."""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


class CheckpointManager:
    """Manages training checkpoints with full state persistence.

    Parameters
    ----------
    checkpoint_dir : str
        Directory for storing checkpoints.
    """

    def __init__(self, checkpoint_dir: str = "models/checkpoints/pytorch_ai") -> None:
        """Initialize checkpoint manager.

        Parameters
        ----------
        checkpoint_dir : str
            Checkpoint directory.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict,
        training_duration: float,
        suffix: Optional[str] = None,
        additional_state: Optional[Dict] = None,
    ) -> Path:
        """Save training checkpoint.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        model : torch.nn.Module
            Model to save.
        optimizer : torch.optim.Optimizer
            Optimizer state.
        metrics : Dict
            Training metrics.
        training_duration : float
            Training duration in seconds.
        suffix : Optional[str]
            Optional suffix for checkpoint filename.
        additional_state : Optional[Dict]
            Additional state to save.

        Returns
        -------
        Path
            Path to saved checkpoint.
        """
        timestamp = int(time.time())
        suffix_str = f"_{suffix}" if suffix else ""
        checkpoint_name = f"checkpoint_epoch_{epoch}{suffix_str}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "training_duration": training_duration,
            "timestamp": timestamp,
        }

        if additional_state:
            checkpoint["additional_state"] = additional_state

        torch.save(checkpoint, checkpoint_path)

        # Save metadata
        metadata_path = checkpoint_path.with_suffix(".json")
        metadata = {
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path),
            "timestamp": timestamp,
            "training_duration": training_duration,
            "metrics": metrics,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update latest checkpoint symlink/file
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(checkpoint_path.name)

        return checkpoint_path

    def load_checkpoint(
        self, checkpoint_path: Optional[Path] = None, device: Optional[torch.device] = None
    ) -> Tuple[Dict, Path]:
        """Load training checkpoint.

        Parameters
        ----------
        checkpoint_path : Optional[Path]
            Path to checkpoint. If None, loads latest.
        device : Optional[torch.device]
            Device to load checkpoint on.

        Returns
        -------
        Tuple[Dict, Path]
            Checkpoint dictionary and path to loaded checkpoint.
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        return checkpoint, checkpoint_path

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint.

        Returns
        -------
        Optional[Path]
            Path to latest checkpoint, or None if not found.
        """
        # Try latest symlink
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        if latest_path.exists() and latest_path.is_symlink():
            resolved = latest_path.resolve()
            if resolved.exists():
                return resolved

        # Find most recent checkpoint file
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoint_files:
            return max(checkpoint_files, key=lambda p: p.stat().st_mtime)

        return None

    def list_checkpoints(self) -> list[Dict]:
        """List all available checkpoints.

        Returns
        -------
        list[Dict]
            List of checkpoint metadata dictionaries.
        """
        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pt"):
            metadata_file = checkpoint_file.with_suffix(".json")
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    checkpoints.append(metadata)
                except (json.JSONDecodeError, KeyError):
                    # Fallback to file stats
                    stat = checkpoint_file.stat()
                    checkpoints.append(
                        {
                            "checkpoint_path": str(checkpoint_file),
                            "timestamp": int(stat.st_mtime),
                            "epoch": 0,  # Unknown
                        }
                    )

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return checkpoints

    def resume_from_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[int, Dict, float]:
        """Resume training from checkpoint.

        Parameters
        ----------
        model : torch.nn.Module
            Model to load state into.
        optimizer : torch.optim.Optimizer
            Optimizer to load state into.
        checkpoint_path : Optional[Path]
            Path to checkpoint. If None, loads latest.
        device : Optional[torch.device]
            Device to load on.

        Returns
        -------
        Tuple[int, Dict, float]
            (epoch, metrics, training_duration) from checkpoint.
        """
        checkpoint, path = self.load_checkpoint(checkpoint_path, device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})
        training_duration = checkpoint.get("training_duration", 0.0)

        return epoch, metrics, training_duration

    def get_checkpoint_info(self, checkpoint_path: Path) -> Dict:
        """Get information about a checkpoint.

        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint.

        Returns
        -------
        Dict
            Checkpoint information.
        """
        if not checkpoint_path.exists():
            return {"exists": False}

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            metadata_file = checkpoint_path.with_suffix(".json")
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

            return {
                "exists": True,
                "epoch": checkpoint.get("epoch", 0),
                "timestamp": checkpoint.get("timestamp", 0),
                "training_duration": checkpoint.get("training_duration", 0.0),
                "metrics": checkpoint.get("metrics", {}),
                "metadata": metadata,
            }
        except Exception as e:
            return {"exists": True, "error": str(e)}

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        """Remove old checkpoints, keeping only the most recent N.

        Parameters
        ----------
        keep_last_n : int
            Number of checkpoints to keep.

        Returns
        -------
        int
            Number of checkpoints removed.
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep_last_n:
            return 0

        # Sort by timestamp (oldest first)
        checkpoints.sort(key=lambda x: x.get("timestamp", 0))

        removed = 0
        for checkpoint_info in checkpoints[:-keep_last_n]:
            checkpoint_path = Path(checkpoint_info["checkpoint_path"])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                # Also remove metadata
                metadata_path = checkpoint_path.with_suffix(".json")
                if metadata_path.exists():
                    metadata_path.unlink()
                removed += 1

        return removed

