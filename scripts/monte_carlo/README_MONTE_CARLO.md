# Monte Carlo Bot Comparison Script

This script runs Monte Carlo simulations to compare different Euchre bot types and determine which performs best.

## Usage

```bash
# Compare all bots with default settings (100 games per matchup)
python scripts/monte_carlo_bot_comparison.py

# Compare specific bots with more games
python scripts/monte_carlo_bot_comparison.py --bots heuristic weighted_heuristic random --num-games 200

# Save results to a JSON file
python scripts/monte_carlo_bot_comparison.py --num-games 100 --output results.json

# Use a specific random seed for reproducibility
python scripts/monte_carlo_bot_comparison.py --num-games 100 --seed 42
```

## Available Bot Types

- `heuristic` - Improved HeuristicPlayer (rule-based)
- `weighted_heuristic` - Weighted heuristic-based player (formerly 'ai')
- `random` - RandomPlayer (baseline)
- `eucher_zero` - EucherZeroPlayer (MCTS-based, if available)

## How It Works

1. **Generates Matchups**: Creates all unique combinations of bot types
   - Tests same bot vs same bot (e.g., heuristic+heuristic vs weighted_heuristic+weighted_heuristic)
   - Tests mixed teams (e.g., heuristic+weighted_heuristic vs random+random)
   - Tests all permutations

2. **Runs Games**: For each matchup, runs the specified number of games
   - Each game is played to completion
   - Tracks winner, scores, and number of hands

3. **Collects Statistics**: 
   - Win rates for each bot type
   - Average scores
   - Per-matchup statistics

4. **Ranks Bots**: Sorts bots by overall win rate across all matchups

## Output

The script prints:
- Bot rankings (sorted by win rate)
- Detailed matchup statistics
- Optionally saves JSON results to a file

## Example Output

```
BOT RANKINGS (by win rate)
================================================================================

1. AI
   Win Rate: 58.3% (350/600)
   Avg Score: 10.2
   Games Played: 600

2. HEURISTIC
   Win Rate: 45.2% (271/600)
   Avg Score: 9.8
   Games Played: 600

3. RANDOM
   Win Rate: 25.1% (151/600)
   Avg Score: 7.5
   Games Played: 600
```

## Notes

- The script requires all dependencies to be installed (including torch if using ML players)
- EucherZero will be automatically skipped if not available
- Games are deterministic when using a seed
- Progress is shown with a progress bar

---

# Generic Model Comparison Script

The `generic_model_comparison.py` script provides a flexible way to compare any two player models with configurable parameters.

## Usage

```bash
# List all available player plugins
python scripts/monte_carlo/generic_model_comparison.py --list-plugins

# Compare two models with default settings
python scripts/monte_carlo/generic_model_comparison.py \
    --model-a euchergo \
    --model-b eucher_zero \
    --num-games 1000

# Compare with custom model paths and parameters
python scripts/monte_carlo/generic_model_comparison.py \
    --model-a euchergo \
    --model-b eucher_zero \
    --num-games 500 \
    --config-a '{"model_path": "path/to/euchergo/model.pt", "num_simulations": 100, "risk_factor": 0.1}' \
    --config-b '{"model_path": "path/to/eucher_zero/model.pt", "num_simulations": 50, "risk_factor": 0.0}'

# Compare PerceiverMuZero variants
python scripts/monte_carlo/generic_model_comparison.py \
    --model-a perceiver_muzero_64 \
    --model-b perceiver_muzero_128 \
    --num-games 200

# Use checkpointing for long runs
python scripts/monte_carlo/generic_model_comparison.py \
    --model-a euchergo \
    --model-b eucher_zero \
    --num-games 10000 \
    --checkpoint results.checkpoint.json \
    --checkpoint-interval 100
```

## Configuration Parameters

Configuration is passed as JSON strings via `--config-a` and `--config-b`. Common parameters include:

- `model_path`: Path to model checkpoint file
- `num_simulations`: Number of MCTS simulations (for MCTS-based players)
- `risk_factor`: Risk factor for decision making (0.0-1.0)
- `config`: Custom configuration object (advanced)

Different player types support different parameters. Check the plugin documentation for details.

## Available Player Types

Use `--list-plugins` to see all available player types. Common ones include:

- `euchergo` - EucherGo player (AlphaZero/MuZero-style)
- `eucher_zero` - EucherZero player (MCTS-based)
- `perceiver_muzero` - PerceiverMuZero player
- `perceiver_muzero_16`, `perceiver_muzero_64`, `perceiver_muzero_128` - PerceiverMuZero variants with fixed simulation counts
- `heuristic` - Rule-based heuristic player
- `weighted_heuristic` - Weighted heuristic-based player using scoring system (formerly 'ai')
- `random` - Random player (baseline)

## Output

The script outputs:
- Win rates and statistics for both models
- Average scores, tricks won, trump makes
- Going alone statistics
- Detailed JSON results file with all game data

Results are saved to `stats/{model_a}_vs_{model_b}_{timestamp}.json` by default, or to the file specified by `--output`.



