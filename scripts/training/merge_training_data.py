"""Merge training data from separate datasets (UUID-based files).

This tool merges training data collected across different machines or sessions
into consolidated datasets. It reads all UUID-based .npz files and their
metadata, groups them by dataset type, and creates merged output files.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from eucher.training.system_info import get_system_info, format_system_info
except ImportError:
    def get_system_info() -> Dict[str, any]:
        return {"timestamp": "", "hostname": "unknown"}
    def format_system_info(info: Dict[str, any]) -> str:
        return f"Timestamp: {info.get('timestamp', 'unknown')}\nHostname: {info.get('hostname', 'unknown')}"


def find_training_data_files(data_dir: Path, file_extension: str = "npz") -> List[Path]:
    """
    Find all training data files with the specified extension.

    Parameters
    ----------
    data_dir : Path
        Directory to search for training data files.
    file_extension : str
        File extension to search for (default: "npz").

    Returns
    -------
    List[Path]
        List of paths to training data files.
    """
    return list(data_dir.glob(f"*.{file_extension}"))


def get_dataset_type_from_metadata(metadata_file: Path) -> Optional[str]:
    """
    Extract dataset type from metadata file.

    Parameters
    ----------
    metadata_file : Path
        Path to metadata file.

    Returns
    -------
    Optional[str]
        Dataset type if found, None otherwise.
    """
    if not metadata_file.exists():
        return None

    try:
        with open(metadata_file, "r") as f:
            for line in f:
                if line.startswith("Dataset:"):
                    return line.split(":", 1)[1].strip()
    except (IOError, OSError):
        pass

    return None


def merge_training_data(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    file_extension: str = "npz",
    dataset_types: Optional[List[str]] = None,
    use_uuid_output: bool = True,
    output_prefix: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Merge training data from UUID-based files.

    Parameters
    ----------
    data_dir : Path
        Directory containing training data files.
    output_dir : Optional[Path]
        Directory to save merged files. If None, uses data_dir.
    file_extension : str
        File extension to process (default: "npz").
    dataset_types : Optional[List[str]]
        List of dataset types to merge. If None, merges all found types.
    use_uuid_output : bool
        If True, saves merged data with new UUID. If False, uses prefix-based naming.
    output_prefix : Optional[str]
        Prefix for output files (used only if use_uuid_output is False).

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping dataset types to output file paths.
    """
    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find all data files
    data_files = find_training_data_files(data_dir, file_extension)

    # Group files by dataset type
    dataset_files: Dict[str, List[Path]] = {}
    dataset_metadata: Dict[str, List[str]] = {}

    for data_file in data_files:
        metadata_file = data_file.with_suffix(".txt")
        dataset_type = get_dataset_type_from_metadata(metadata_file)

        if dataset_type is None:
            # Try to infer from filename patterns (backward compatibility)
            if "order_up" in data_file.stem:
                dataset_type = "order_up"
            elif "call_trump" in data_file.stem:
                dataset_type = "call_trump"
            elif "play_card" in data_file.stem:
                dataset_type = "play_card"
            elif "discard" in data_file.stem:
                dataset_type = "discard"
            else:
                print(f"Warning: Could not determine dataset type for {data_file.name}, skipping")
                continue

        # Filter by dataset_types if specified
        if dataset_types is not None and dataset_type not in dataset_types:
            continue

        if dataset_type not in dataset_files:
            dataset_files[dataset_type] = []
            dataset_metadata[dataset_type] = []

        dataset_files[dataset_type].append(data_file)

        # Collect metadata info
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata_content = f.read()
                    dataset_metadata[dataset_type].append(metadata_content)
            except (IOError, OSError):
                pass

    # Merge data for each dataset type
    merged_files: Dict[str, Path] = {}

    for dataset_type, files in dataset_files.items():
        if not files:
            continue

        print(f"\nMerging {dataset_type}: {len(files)} files")

        # Load and concatenate all data
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        total_records = 0

        for data_file in files:
            try:
                loaded = np.load(data_file)
                X = loaded["X"]
                y = loaded["y"]

                X_list.append(X)
                y_list.append(y)
                total_records += len(X)

                print(f"  Loaded {data_file.name}: {len(X)} records")
            except Exception as e:
                print(f"  Error loading {data_file.name}: {e}")
                continue

        if not X_list:
            print(f"  No valid data found for {dataset_type}, skipping")
            continue

        # Concatenate all arrays
        X_merged = np.concatenate(X_list, axis=0)
        y_merged = np.concatenate(y_list, axis=0)

        print(f"  Merged: {total_records} total records")
        print(f"  Final shape: X={X_merged.shape}, y={y_merged.shape}")

        # Save merged data
        if use_uuid_output:
            import uuid
            merged_uuid = str(uuid.uuid4())
            output_file = output_dir / f"{merged_uuid}.{file_extension}"
            metadata_file = output_dir / f"{merged_uuid}.txt"
        else:
            if output_prefix is None:
                output_prefix = "merged"
            output_file = output_dir / f"{output_prefix}_{dataset_type}.{file_extension}"
            metadata_file = None

        np.savez_compressed(output_file, X=X_merged, y=y_merged)
        merged_files[dataset_type] = output_file
        print(f"  Saved: {output_file}")

        # Create metadata file
        if metadata_file is not None:
            system_info = get_system_info()
            metadata_content = format_system_info(system_info)
            metadata_content += f"\nDataset: {dataset_type}\n"
            metadata_content += f"UUID: {merged_uuid}\n"
            metadata_content += f"File: {merged_uuid}.{file_extension}\n"
            metadata_content += f"Source files: {len(files)}\n"
            metadata_content += f"Total records: {total_records}\n"
            metadata_content += f"Features shape: {X_merged.shape}\n"
            metadata_content += f"Decisions shape: {y_merged.shape}\n"
            metadata_content += "\nSource file information:\n"
            for i, source_file in enumerate(files, 1):
                metadata_content += f"  {i}. {source_file.name}\n"
                # Try to get UUID from source metadata
                source_metadata = source_file.with_suffix(".txt")
                if source_metadata.exists():
                    try:
                        with open(source_metadata, "r") as f:
                            for line in f:
                                if line.startswith("UUID:"):
                                    metadata_content += f"     UUID: {line.split(':', 1)[1].strip()}\n"
                                elif line.startswith("Hostname:"):
                                    metadata_content += f"     Host: {line.split(':', 1)[1].strip()}\n"
                                    break
                    except (IOError, OSError):
                        pass

            with open(metadata_file, "w") as f:
                f.write(metadata_content)
            print(f"  Metadata: {metadata_file}")

    return merged_files


def main() -> None:
    """Main entry point for merge tool."""
    parser = argparse.ArgumentParser(
        description="Merge training data from separate datasets (UUID-based files)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing training data files to merge",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save merged files (default: same as data-dir)",
    )
    parser.add_argument(
        "--file-extension",
        type=str,
        default="npz",
        choices=["npz"],
        help="File extension to process (default: npz)",
    )
    parser.add_argument(
        "--dataset-types",
        type=str,
        nargs="+",
        choices=["order_up", "call_trump", "play_card", "discard"],
        default=None,
        help="Dataset types to merge (default: all)",
    )
    parser.add_argument(
        "--use-uuid-output",
        action="store_true",
        default=True,
        help="Save merged data with UUID naming (default: True)",
    )
    parser.add_argument(
        "--no-uuid-output",
        dest="use_uuid_output",
        action="store_false",
        help="Use prefix-based naming instead of UUID",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="merged",
        help="Prefix for output files when not using UUID (default: merged)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None

    print(f"Data directory: {data_dir}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    else:
        print(f"Output directory: {data_dir} (same as input)")

    merged_files = merge_training_data(
        data_dir=data_dir,
        output_dir=output_dir,
        file_extension=args.file_extension,
        dataset_types=args.dataset_types,
        use_uuid_output=args.use_uuid_output,
        output_prefix=args.output_prefix if not args.use_uuid_output else None,
    )

    print(f"\n{'=' * 50}")
    print("Merge complete!")
    print(f"Merged {len(merged_files)} dataset types:")
    for dataset_type, file_path in merged_files.items():
        print(f"  {dataset_type}: {file_path}")


if __name__ == "__main__":
    main()

