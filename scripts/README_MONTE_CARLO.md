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


