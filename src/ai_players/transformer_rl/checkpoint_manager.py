"""UUID-based cumulative checkpoint manager for parallel training and syncing.

Saves checkpoints in .npz format with UUID filenames to support:
- Parallel training (multiple processes can train simultaneously)
- Syncing training data between machines
- Cumulative checkpoint merging
"""

import gzip
import json
import pickle
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


class CumulativeCheckpointManager:
    """Manages cumulative checkpoints with UUID-based naming and .npz format.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory for storing checkpoints.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        """Initialize checkpoint manager.

        Parameters
        ----------
        checkpoint_dir : Path
            Checkpoint directory.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track checkpoint UUIDs
        self.checkpoint_index_file = self.checkpoint_dir / "checkpoint_index.json"
        self.checkpoint_index: Dict[str, Dict] = self._load_index()

    def _load_index(self) -> Dict[str, Dict]:
        """Load checkpoint index.

        Returns
        -------
        Dict[str, Dict]
            Checkpoint index mapping UUID to metadata.
        """
        if self.checkpoint_index_file.exists():
            try:
                with open(self.checkpoint_index_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        """Save checkpoint index."""
        with open(self.checkpoint_index_file, "w") as f:
            json.dump(self.checkpoint_index, f, indent=2)

    def save_checkpoint(
        self,
        agent_state: Dict,
        training_stats: Optional[Dict] = None,
        experience_buffer: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Save cumulative checkpoint with UUID.

        Parameters
        ----------
        agent_state : Dict
            Agent state (network, optimizer, etc.).
        training_stats : Optional[Dict]
            Training statistics.
        experience_buffer : Optional[List[Dict]]
            Experience buffer data.
        metadata : Optional[Dict]
            Additional metadata.

        Returns
        -------
        str
            UUID of saved checkpoint.
        """
        checkpoint_uuid = str(uuid.uuid4())
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_uuid}.npz"

        # Convert PyTorch tensors to numpy arrays for .npz format
        npz_data: Dict[str, np.ndarray] = {}

        # Save network state (convert tensors to numpy)
        network_state = agent_state.get("network_state_dict", {})
        for key, value in network_state.items():
            if isinstance(value, torch.Tensor):
                npz_data[f"network_{key}"] = value.cpu().detach().numpy()
            else:
                # Store as pickle for non-tensor data
                npz_data[f"network_{key}"] = np.array([pickle.dumps(value)], dtype=object)

        # Save optimizer state (pickle the entire state dict as it has complex nested structure)
        optimizer_state = agent_state.get("optimizer_state_dict", {})
        if optimizer_state:
            # Convert tensors to CPU numpy arrays for serialization
            optimizer_state_serializable = {}
            for key, value in optimizer_state.items():
                if key == "state":
                    # Handle state dict with parameter IDs
                    optimizer_state_serializable[key] = {}
                    for param_id, param_state in value.items():
                        optimizer_state_serializable[key][param_id] = {}
                        for param_key, param_value in param_state.items():
                            if isinstance(param_value, torch.Tensor):
                                optimizer_state_serializable[key][param_id][param_key] = param_value.cpu().detach().numpy()
                            else:
                                optimizer_state_serializable[key][param_id][param_key] = param_value
                elif key == "param_groups":
                    # Param groups can be pickled directly
                    optimizer_state_serializable[key] = value
                else:
                    optimizer_state_serializable[key] = value
            
            npz_data["optimizer_state_dict"] = np.array([pickle.dumps(optimizer_state_serializable)], dtype=object)

        # Save training stats as JSON string
        if training_stats:
            npz_data["training_stats"] = np.array([json.dumps(training_stats)], dtype=object)

        # Save experience buffer (compressed)
        if experience_buffer:
            buffer_data = pickle.dumps(experience_buffer)
            npz_data["experience_buffer"] = np.array([buffer_data], dtype=object)

        # Save metadata
        checkpoint_metadata = {
            "uuid": checkpoint_uuid,
            "timestamp": time.time(),
            "hands_played": metadata.get("hands_played", 0) if metadata else 0,
            "stage": metadata.get("stage", "unknown") if metadata else "unknown",
        }
        if metadata:
            checkpoint_metadata.update(metadata)

        npz_data["metadata"] = np.array([json.dumps(checkpoint_metadata)], dtype=object)

        # Save as compressed .npz
        # Use gzip compression for .npz files
        np.savez_compressed(checkpoint_path, **npz_data)

        # Update index
        self.checkpoint_index[checkpoint_uuid] = checkpoint_metadata
        self._save_index()

        return checkpoint_uuid

    def load_checkpoint(self, checkpoint_uuid: Optional[str] = None, device: Optional[torch.device] = None) -> Optional[Dict]:
        """Load checkpoint by UUID.

        Parameters
        ----------
        checkpoint_uuid : Optional[str]
            UUID of checkpoint to load. If None, loads latest.
        device : Optional[torch.device]
            Device to load tensors to.

        Returns
        -------
        Optional[Dict]
            Loaded checkpoint data or None if not found.
        """
        if checkpoint_uuid is None:
            # Load latest checkpoint
            checkpoint_uuid = self.get_latest_checkpoint_uuid()
            if checkpoint_uuid is None:
                return None

        checkpoint_path = self.checkpoint_dir / f"{checkpoint_uuid}.npz"
        if not checkpoint_path.exists():
            return None

        # Load .npz file
        npz_data = np.load(checkpoint_path, allow_pickle=True)

        # Reconstruct agent state
        agent_state: Dict = {"network_state_dict": {}, "optimizer_state_dict": {}}

        # Reconstruct network state
        for key in npz_data.files:
            if key.startswith("network_"):
                param_key = key[len("network_") :]
                value = npz_data[key]
                if value.dtype == object:
                    # Unpickle non-tensor data
                    agent_state["network_state_dict"][param_key] = pickle.loads(value.item())
                else:
                    # Convert numpy to tensor
                    tensor = torch.from_numpy(value)
                    if device is not None:
                        tensor = tensor.to(device)
                    agent_state["network_state_dict"][param_key] = tensor

        # Reconstruct optimizer state
        optimizer_state = {}
        if "optimizer_state_dict" in npz_data.files:
            optimizer_state_serializable = pickle.loads(npz_data["optimizer_state_dict"].item())
            
            # Reconstruct with tensors on correct device
            optimizer_state = {}
            for key, value in optimizer_state_serializable.items():
                if key == "state":
                    # Reconstruct state dict with parameter IDs
                    optimizer_state[key] = {}
                    for param_id, param_state in value.items():
                        optimizer_state[key][param_id] = {}
                        for param_key, param_value in param_state.items():
                            if isinstance(param_value, np.ndarray):
                                tensor = torch.from_numpy(param_value)
                                if device is not None:
                                    tensor = tensor.to(device)
                                optimizer_state[key][param_id][param_key] = tensor
                            else:
                                optimizer_state[key][param_id][param_key] = param_value
                elif key == "param_groups":
                    # Param groups can be used directly
                    optimizer_state[key] = value
                else:
                    optimizer_state[key] = value

        agent_state["optimizer_state_dict"] = optimizer_state

        # Load training stats
        training_stats = None
        if "training_stats" in npz_data.files:
            training_stats = json.loads(npz_data["training_stats"].item())

        # Load experience buffer
        experience_buffer = None
        if "experience_buffer" in npz_data.files:
            experience_buffer = pickle.loads(npz_data["experience_buffer"].item())

        # Load metadata
        metadata = None
        if "metadata" in npz_data.files:
            metadata = json.loads(npz_data["metadata"].item())

        return {
            "agent_state": agent_state,
            "training_stats": training_stats,
            "experience_buffer": experience_buffer,
            "metadata": metadata,
            "uuid": checkpoint_uuid,
        }

    def get_latest_checkpoint_uuid(self) -> Optional[str]:
        """Get UUID of latest checkpoint.

        Returns
        -------
        Optional[str]
            UUID of latest checkpoint or None if no checkpoints exist.
        """
        if not self.checkpoint_index:
            return None

        # Find checkpoint with latest timestamp
        latest_uuid = max(
            self.checkpoint_index.keys(),
            key=lambda u: self.checkpoint_index[u].get("timestamp", 0),
        )
        return latest_uuid

    def list_checkpoints(self) -> List[Dict]:
        """List all checkpoints with metadata.

        Returns
        -------
        List[Dict]
            List of checkpoint metadata.
        """
        return list(self.checkpoint_index.values())

    def merge_checkpoints(
        self, checkpoint_uuids: List[str], output_uuid: Optional[str] = None
    ) -> str:
        """Merge multiple checkpoints into one cumulative checkpoint.

        Parameters
        ----------
        checkpoint_uuids : List[str]
            List of checkpoint UUIDs to merge.
        output_uuid : Optional[str]
            UUID for output checkpoint. If None, generates new UUID.

        Returns
        -------
        str
            UUID of merged checkpoint.
        """
        if not checkpoint_uuids:
            raise ValueError("Must provide at least one checkpoint UUID to merge")

        # Load all checkpoints
        checkpoints = []
        for uuid_str in checkpoint_uuids:
            checkpoint = self.load_checkpoint(uuid_str)
            if checkpoint:
                checkpoints.append(checkpoint)

        if not checkpoints:
            raise ValueError("No valid checkpoints found to merge")

        # Merge agent states (average model parameters)
        merged_network_state: Dict = {}
        merged_optimizer_state: Dict = {}

        # Get all parameter keys
        all_network_keys = set()
        all_optimizer_keys: Dict[str, Set] = {}
        for checkpoint in checkpoints:
            network_state = checkpoint["agent_state"]["network_state_dict"]
            optimizer_state = checkpoint["agent_state"]["optimizer_state_dict"]
            all_network_keys.update(network_state.keys())
            for opt_key, opt_dict in optimizer_state.items():
                if opt_key not in all_optimizer_keys:
                    all_optimizer_keys[opt_key] = set()
                all_optimizer_keys[opt_key].update(opt_dict.keys())

        # Average network parameters
        for key in all_network_keys:
            params = []
            for checkpoint in checkpoints:
                network_state = checkpoint["agent_state"]["network_state_dict"]
                if key in network_state:
                    param = network_state[key]
                    if isinstance(param, torch.Tensor):
                        params.append(param)
            if params:
                # Average tensors
                merged_param = torch.stack(params).mean(dim=0)
                merged_network_state[key] = merged_param

        # Merge optimizer states (use latest)
        latest_checkpoint = checkpoints[-1]
        merged_optimizer_state = latest_checkpoint["agent_state"]["optimizer_state_dict"].copy()

        # Merge training stats (accumulate)
        merged_training_stats: Dict = {}
        for checkpoint in checkpoints:
            stats = checkpoint.get("training_stats", {})
            for key, value in stats.items():
                if key not in merged_training_stats:
                    merged_training_stats[key] = []
                if isinstance(value, list):
                    merged_training_stats[key].extend(value)
                else:
                    merged_training_stats[key].append(value)

        # Merge experience buffers
        merged_experience_buffer: List[Dict] = []
        for checkpoint in checkpoints:
            buffer = checkpoint.get("experience_buffer")
            if buffer:
                merged_experience_buffer.extend(buffer)

        # Create merged agent state
        merged_agent_state = {
            "network_state_dict": merged_network_state,
            "optimizer_state_dict": merged_optimizer_state,
        }

        # Save merged checkpoint
        merged_metadata = {
            "hands_played": sum(
                c.get("metadata", {}).get("hands_played", 0) for c in checkpoints
            ),
            "merged_from": checkpoint_uuids,
        }

        if output_uuid is None:
            output_uuid = str(uuid.uuid4())

        return self.save_checkpoint(
            agent_state=merged_agent_state,
            training_stats=merged_training_stats,
            experience_buffer=merged_experience_buffer if merged_experience_buffer else None,
            metadata=merged_metadata,
        )

    def sync_checkpoints(self, remote_checkpoint_dir: Path) -> List[str]:
        """Sync checkpoints from remote directory.

        Parameters
        ----------
        remote_checkpoint_dir : Path
            Remote checkpoint directory to sync from.

        Returns
        -------
        List[str]
            List of UUIDs of synced checkpoints.
        """
        remote_dir = Path(remote_checkpoint_dir)
        if not remote_dir.exists():
            return []

        synced_uuids: List[str] = []

        # Load remote index
        remote_index_file = remote_dir / "checkpoint_index.json"
        if remote_index_file.exists():
            with open(remote_index_file, "r") as f:
                remote_index = json.load(f)
        else:
            # Scan for .npz files
            remote_index = {}
            for npz_file in remote_dir.glob("*.npz"):
                uuid_str = npz_file.stem
                remote_index[uuid_str] = {"uuid": uuid_str}

        # Copy missing checkpoints
        for uuid_str, metadata in remote_index.items():
            if uuid_str not in self.checkpoint_index:
                remote_path = remote_dir / f"{uuid_str}.npz"
                local_path = self.checkpoint_dir / f"{uuid_str}.npz"
                if remote_path.exists():
                    # Copy file
                    import shutil

                    shutil.copy2(remote_path, local_path)
                    self.checkpoint_index[uuid_str] = metadata
                    synced_uuids.append(uuid_str)

        self._save_index()
        return synced_uuids

