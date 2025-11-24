# Root Directory Cleanup Summary

## Files Moved

### Examples → `examples/`
- `train_gan_example.py`
- `train_rl_example.py`
- `review_decisions_example.py`

### Scripts → `scripts/`
- `collect_training_data.py`
- `generate_input_docs.py`
- `review_decisions.py`
- `play_rl_game.py`
- `run_demo_game.py`
- `train_models.py`
- `generate_docs.sh`

### Tests → `tests/`
- `test_heuristic_improvements.py`
- `quick_benchmark_test.py`

## Files Remaining in Root

### Essential Files (Keep)
- `main.py` - Entry point for the application
- `__init__.py` - Package initialization
- `README.md` - Project documentation
- `pyproject.toml` - Python package configuration
- `Makefile` - Build automation

### Documentation Files (Consider Moving)
- `README_SCRIPTS.md` - Could move to `docs/` or `scripts/`
- `euchre_transformer_rl_requirements.md` - Could move to `docs/` or `bot_requirements/`

### Legacy/Duplicate Files (Remove)
These files are duplicates of files in `eucher/` package and should be removed:
- `ai.py` → Use `eucher/players/computer/ai.py`
- `cards.py` → Use `eucher/cards.py`
- `cli.py` → Use `eucher/cli.py`
- `computer_player.py` → Use `eucher/players/computer/`
- `database.py` → Use `eucher/database.py`
- `db_queries.py` → Use `eucher/db_queries.py`
- `db_serializers.py` → Use `eucher/db_serializers.py`
- `db_utils.py` → Use `eucher/db_utils.py`
- `game.py` → Use `eucher/game.py`
- `game_config.py` → Use `eucher/game_config.py`
- `models.py` → Use `eucher/models.py`
- `player_profiles.py` → Use `eucher/players/profiles.py`
- `players.py` → Use `eucher/players/`
- `rules.py` → Use `eucher/rules.py`
- `trump.py` → Use `eucher/trump.py`
- `tui.py` → Use `eucher/tui.py`
- `ml_config.py` → Use `eucher/players/computer/ml/ml_config.py`
- `ml_decision_weights.py` → Use `eucher/players/computer/ml/ml_decision_weights.py`
- `ml_features.py` → Use `eucher/players/computer/ml/ml_features.py`
- `ml_model.py` → Use `eucher/players/computer/ml/ml_model.py`
- `ml_models_gan.py` → Use `eucher/players/computer/ml/models/ml_models_gan.py`
- `ml_models_rl.py` → Use `eucher/players/computer/ml/models/ml_models_rl.py`
- `ml_models_supervised.py` → Use `eucher/players/computer/ml/models/ml_models_supervised.py`
- `ml_player.py` → Use `eucher/players/computer/ml/player.py`

## Import Updates Needed

Scripts that were moved need import path updates:
- ✅ `scripts/collect_training_data.py` - Updated
- ✅ `scripts/train_models.py` - Updated
- ⚠️ Other scripts may still need updates (check for `from src.` imports)

## Next Steps

1. Verify no scripts import from root-level files
2. Remove duplicate root-level files
3. Move documentation files to appropriate locations
4. Update any remaining import references


