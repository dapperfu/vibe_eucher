#!/usr/bin/env python3
"""Script to update all src. imports to eucher. imports."""

import re
from pathlib import Path

# Mapping of src modules to eucher modules
IMPORT_MAPPINGS = {
    "src.game": "eucher.game",
    "src.cards": "eucher.cards",
    "src.players": "eucher.players",
    "src.player_profiles": "eucher.players.profiles",
    "src.rules": "eucher.rules",
    "src.trump": "eucher.trump",
    "src.tui": "eucher.tui",
    "src.database": "eucher.database",
    "src.db_queries": "eucher.db_queries",
    "src.db_serializers": "eucher.db_serializers",
    "src.db_utils": "eucher.db_utils",
    "src.models": "eucher.models",
    "src.game_config": "eucher.game_config",
    "src.ai": "eucher.players.computer.ai",
    "src.computer_player": "eucher.players.computer.base",
    "src.ml_config": "eucher.players.computer.ml.ml_config",
    "src.ml_features": "eucher.players.computer.ml.ml_features",
    "src.ml_models_supervised": "eucher.players.computer.ml.models.ml_models_supervised",
    "src.ml_models_gan": "eucher.players.computer.ml.models.ml_models_gan",
    "src.ml_models_rl": "eucher.players.computer.ml.models.ml_models_rl",
    "src.ml_model": "eucher.players.computer.ml.ml_model",
    "src.ml_player": "eucher.players.computer.ml.player",
    "src.ml_decision_weights": "eucher.players.computer.ml.ml_decision_weights",
    "src.training.": "eucher.training.",
    "src.ai_players.": "eucher.ai_players.",
    "src.ml_training.": "eucher.ml_training.",
}


def update_imports_in_file(file_path: Path) -> bool:
    """Update imports in a single file. Returns True if changes were made."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content

        # Update from src.X imports
        for old_import, new_import in IMPORT_MAPPINGS.items():
            # Handle both "from src.X" and "import src.X" patterns
            patterns = [
                (rf"from {re.escape(old_import)}", f"from {new_import}"),
                (rf"import {re.escape(old_import)}", f"import {new_import}"),
            ]
            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content)

        if content != original_content:
            file_path.write_text(content, encoding="utf-8")
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Update all imports in src/ directory."""
    src_dir = Path("src")
    if not src_dir.exists():
        print("src/ directory not found")
        return

    updated_files = []
    for py_file in src_dir.rglob("*.py"):
        if update_imports_in_file(py_file):
            updated_files.append(py_file)
            print(f"Updated: {py_file}")

    print(f"\nUpdated {len(updated_files)} files")


if __name__ == "__main__":
    main()






