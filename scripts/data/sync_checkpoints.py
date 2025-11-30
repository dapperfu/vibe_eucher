#!/usr/bin/env python3
"""Script to sync checkpoints between machines.

Supports syncing UUID-based cumulative checkpoints in .npz format.
"""

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_players.transformer_rl.checkpoint_manager import CumulativeCheckpointManager


def main() -> None:
    """Main sync entry point."""
    parser = argparse.ArgumentParser(description="Sync transformer RL checkpoints between machines")
    parser.add_argument(
        "--local-dir",
        type=str,
        default="models/checkpoints/transformer_rl",
        help="Local checkpoint directory",
    )
    parser.add_argument(
        "--remote-dir",
        type=str,
        required=True,
        help="Remote checkpoint directory to sync from",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge synced checkpoints into one",
    )

    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    remote_dir = Path(args.remote_dir)

    print(f"Syncing checkpoints from {remote_dir} to {local_dir}...")

    local_manager = CumulativeCheckpointManager(local_dir)
    synced_uuids = local_manager.sync_checkpoints(remote_dir)

    if synced_uuids:
        print(f"Synced {len(synced_uuids)} checkpoints:")
        for uuid_str in synced_uuids:
            print(f"  - {uuid_str}")
    else:
        print("No new checkpoints to sync.")

    if args.merge and synced_uuids:
        print("\nMerging checkpoints...")
        merged_uuid = local_manager.merge_checkpoints(synced_uuids)
        print(f"Merged checkpoint UUID: {merged_uuid}")


if __name__ == "__main__":
    main()




