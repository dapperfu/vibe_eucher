"""Training data management for cumulative data collection."""

import json
from pathlib import Path
from typing import Dict, List, Optional

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

    def load_dataset(self, dataset_name: str, format: str = "npz") -> pd.DataFrame:
        """Load existing training dataset from .npz file.

        Parameters
        ----------
        dataset_name : str
            Name of dataset (e.g., "play_card", "order_up").
        format : str
            Format to load (only "npz" is supported).

        Returns
        -------
        pd.DataFrame
            Loaded dataset with 'features' and 'decision' columns.
        """
        import numpy as np
        
        # Only support .npz format
        file_path = self.data_dir / f"training_{dataset_name}.npz"
        
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            # Load .npz file
            loaded = np.load(file_path)
            X = loaded["X"]  # Features array
            y = loaded["y"]  # Decisions array
            
            # Convert to DataFrame format expected by training script
            # Convert features array to list of lists
            features_list = [X[i].tolist() for i in range(len(X))]
            decisions_list = y.tolist()
            
            # Create DataFrame
            df = pd.DataFrame({
                "features": features_list,
                "decision": decisions_list,
            })
            
            return df
        except (KeyError, ValueError, IOError) as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

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
        """List all available datasets (only .npz files).

        Returns
        -------
        List[str]
            List of dataset names.
        """
        datasets = set()
        for file_path in self.data_dir.glob("training_*.npz"):
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

