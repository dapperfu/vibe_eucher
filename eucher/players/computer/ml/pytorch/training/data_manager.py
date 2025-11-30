"""Training data management for cumulative data collection."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class TrainingDataManager:
    """Manages cumulative training data collection and storage.

    Parameters
    ----------
    data_dir : str
        Directory for training data.
    """

    def __init__(self, data_dir: str = "training_data") -> None:
        """Initialize training data manager.

        Parameters
        ----------
        data_dir : str
            Data directory.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, dataset_name: str, format: str = "auto") -> pd.DataFrame:
        """Load existing training dataset.

        Parameters
        ----------
        dataset_name : str
            Name of dataset (e.g., "play_card", "order_up").
        format : str
            Format to load ("csv", "json", "parquet", "npz", "auto").

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        # Try different formats
        if format == "auto":
            formats = ["npz", "parquet", "csv", "json"]
        else:
            formats = [format]

        for fmt in formats:
            file_path = self.data_dir / f"training_{dataset_name}.{fmt}"
            if file_path.exists():
                try:
                    df = self._load_file(file_path, fmt)
                    # Validate that loaded DataFrame is not empty
                    if df.empty:
                        print(f"Warning: Loaded dataset '{dataset_name}' from {file_path} is empty")
                    return df
                except ImportError:
                    # If parquet fails due to missing pyarrow, try CSV
                    if fmt == "parquet":
                        csv_path = file_path.with_suffix(".csv")
                        if csv_path.exists():
                            try:
                                df = self._load_file(csv_path, "csv")
                                if df.empty:
                                    print(f"Warning: Loaded dataset '{dataset_name}' from {csv_path} is empty")
                                return df
                            except Exception as e:
                                print(f"Error loading CSV fallback for {file_path}: {e}")
                    continue
                except Exception as e:
                    # Log error but continue trying other formats
                    print(f"Error loading {file_path} ({fmt} format): {e}")
                    continue

        # Return empty DataFrame if not found
        return pd.DataFrame()

    def _load_file(self, file_path: Path, format: str) -> pd.DataFrame:
        """Load file in specified format.

        Parameters
        ----------
        file_path : Path
            Path to file.
        format : str
            File format.

        Returns
        -------
        pd.DataFrame
            Loaded data.

        Raises
        ------
        KeyError
            If required keys are missing in npz file.
        ValueError
            If arrays are empty or have mismatched lengths.
        """
        if format == "npz":
            # Load .npz file with X (features) and y (decisions) arrays
            data = np.load(file_path)
            
            # Validate that required keys exist
            if "X" not in data:
                raise KeyError(f"Missing 'X' key in npz file: {file_path}")
            if "y" not in data:
                raise KeyError(f"Missing 'y' key in npz file: {file_path}")
            
            X = data["X"]
            y = data["y"]
            
            # Validate arrays have non-zero length
            if len(X) == 0:
                raise ValueError(f"Array 'X' in npz file is empty: {file_path}")
            if len(y) == 0:
                raise ValueError(f"Array 'y' in npz file is empty: {file_path}")
            
            # Validate arrays have matching lengths
            if len(X) != len(y):
                raise ValueError(
                    f"Array length mismatch in npz file {file_path}: "
                    f"X has {len(X)} samples, y has {len(y)} samples"
                )
            
            # Convert to DataFrame format expected by EuchreDataset
            # Each row should have "features" (as list) and "decision" fields
            records = []
            for i in range(len(X)):
                records.append({
                    "features": X[i].tolist(),  # Convert numpy array to list
                    "decision": int(y[i]),  # Convert numpy scalar to Python int
                })
            
            return pd.DataFrame(records)
        elif format == "parquet":
            try:
                return pd.read_parquet(file_path)
            except ImportError:
                # Fallback to CSV if pyarrow not available
                csv_path = file_path.with_suffix(".csv")
                if csv_path.exists():
                    return pd.read_csv(csv_path)
                raise
        elif format == "csv":
            return pd.read_csv(file_path)
        elif format == "json":
            return pd.read_json(file_path, orient="records")
        else:
            raise ValueError(f"Unsupported format: {format}")

    def append_data(
        self, dataset_name: str, new_data: List[Dict], format: str = "parquet"
    ) -> Path:
        """Append new data to existing dataset.

        Parameters
        ----------
        dataset_name : str
            Name of dataset.
        new_data : List[Dict]
            New data records to append.
        format : str
            Storage format ("parquet", "csv", "json").

        Returns
        -------
        Path
            Path to saved dataset.
        """
        # Load existing data
        existing_df = self.load_dataset(dataset_name)

        # Convert new data to DataFrame
        new_df = pd.DataFrame(new_data)

        # Append
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # Remove duplicates if any
        combined_df = combined_df.drop_duplicates()

        # Save
        return self.save_dataset(dataset_name, combined_df, format)

    def save_dataset(
        self, dataset_name: str, data: pd.DataFrame, format: str = "parquet"
    ) -> Path:
        """Save dataset to file.

        Parameters
        ----------
        dataset_name : str
            Name of dataset.
        data : pd.DataFrame
            Data to save.
        format : str
            Storage format.

        Returns
        -------
        Path
            Path to saved file.
        """
        file_path = self.data_dir / f"training_{dataset_name}.{format}"

        if format == "parquet":
            try:
                data.to_parquet(file_path, index=False)
            except ImportError:
                # Fallback to CSV if pyarrow not available
                data.to_csv(file_path.with_suffix(".csv"), index=False)
                return file_path.with_suffix(".csv")
        elif format == "csv":
            data.to_csv(file_path, index=False)
        elif format == "json":
            data.to_json(file_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return file_path

    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a dataset.

        Parameters
        ----------
        dataset_name : str
            Name of dataset.

        Returns
        -------
        Dict
            Dataset information.
        """
        df = self.load_dataset(dataset_name)
        if df.empty:
            return {"exists": False, "rows": 0}

        return {
            "exists": True,
            "rows": len(df),
            "columns": list(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024**2),
        }

    def list_datasets(self) -> List[str]:
        """List all available datasets.

        Returns
        -------
        List[str]
            List of dataset names.
        """
        datasets = set()
        for file_path in self.data_dir.glob("training_*.npz"):
            name = file_path.stem.replace("training_", "")
            datasets.add(name)
        for file_path in self.data_dir.glob("training_*.parquet"):
            name = file_path.stem.replace("training_", "")
            datasets.add(name)
        for file_path in self.data_dir.glob("training_*.csv"):
            name = file_path.stem.replace("training_", "")
            datasets.add(name)
        for file_path in self.data_dir.glob("training_*.json"):
            name = file_path.stem.replace("training_", "")
            datasets.add(name)

        return sorted(datasets)

    def convert_format(
        self, dataset_name: str, source_format: str, target_format: str
    ) -> Path:
        """Convert dataset from one format to another.

        Parameters
        ----------
        dataset_name : str
            Name of dataset.
        source_format : str
            Source format.
        target_format : str
            Target format.

        Returns
        -------
        Path
            Path to converted file.
        """
        df = self.load_dataset(dataset_name, format=source_format)
        return self.save_dataset(dataset_name, df, format=target_format)

    def validate_data(self, dataset_name: str) -> Dict:
        """Validate dataset integrity.

        Parameters
        ----------
        dataset_name : str
            Name of dataset.

        Returns
        -------
        Dict
            Validation results.
        """
        df = self.load_dataset(dataset_name)
        if df.empty:
            return {"valid": False, "errors": ["Dataset is empty"]}

        errors = []
        warnings = []

        # Check for required columns (adjust based on your data schema)
        # This is a placeholder - adjust based on actual schema
        if len(df) > 0:
            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.any():
                warnings.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "rows": len(df),
        }

    def load_from_collector(self, collector) -> List[Dict]:
        """Convert GameDataCollector in-memory data to training dataset format.

        Parameters
        ----------
        collector : GameDataCollector
            Data collector with in-memory data.

        Returns
        -------
        List[Dict]
            List of training samples in format compatible with EuchreDataset.
        """
        from eucher.training.data_collector import GameDataCollector

        if not isinstance(collector, GameDataCollector):
            raise TypeError(f"Expected GameDataCollector, got {type(collector)}")

        training_data = []

        # Convert each data type
        # Map collector action types to EuchreDataset action types
        action_type_map = {
            "order_up": "order_up",
            "call_trump": "trump_selection",  # Map call_trump to trump_selection
            "play_card": "play_card",
            "discard": "discard",
        }

        for data_list, collector_action_type in [
            (collector.order_up_data, "order_up"),
            (collector.call_trump_data, "call_trump"),
            (collector.play_card_data, "play_card"),
            (collector.discard_data, "discard"),
        ]:
            action_type = action_type_map.get(collector_action_type, collector_action_type)
            for item in data_list:
                # Convert to format expected by EuchreDataset
                # EuchreDataset expects either:
                # 1. {"features": list, "decision": int, "action_type": str}
                # 2. Raw game state format
                sample = {
                    "features": item.get("features", []),
                    "decision": item.get("decision", 0),
                    "action_type": action_type,
                }
                # Add any additional fields that might be useful
                if "game_id" in item:
                    sample["game_id"] = item["game_id"]
                if "player_id" in item:
                    sample["player_id"] = item["player_id"]
                # Add trump_suit if available (for call_trump decisions)
                if "trump_suit" in item:
                    sample["trump_suit"] = item["trump_suit"]
                training_data.append(sample)

        return training_data

