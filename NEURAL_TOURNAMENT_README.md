# Neural Network Euchre Tournament System

This system allows you to run thousands of euchre games between AI players using trained neural network models. Perfect for evaluating model performance, running tournaments, and gathering comprehensive statistics.

## Features

- **Neural Network AI Players**: Use trained `.json` model files for AI opponents
- **Mass Game Execution**: Run 10,000+ games in parallel for statistical significance
- **Detailed Statistics**: Track wins, losses, tricks won, team sets, and more
- **Flexible Team Configurations**: Mix and match any models for team play
- **Performance Analysis**: Comprehensive analysis with visualizations
- **Parallel Processing**: Multi-core execution for fast results

## Quick Start

### 1. List Available Models

First, see what neural network models you have available:

```bash
# Using the CLI
make list-neural-models

# Or directly
venv/bin/python -m euchre.cli list-neural-models

# Or specify a custom models directory
venv/bin/python -m euchre.cli list-neural-models -d trained_models
```

### 2. Run a Simple Tournament

Run 1000 games between Alice and Bob:

```bash
# Using the Makefile
make neural-tournament

# Or using the CLI directly
venv/bin/python -m euchre.cli run-neural-games -m1 Alice -m2 Bob -n 1000

# Or with custom options
venv/bin/python -m euchre.cli run-neural-games \
  -m1 Alice \
  -m2 Bob \
  -n 10000 \
  -w 8 \
  -o my_tournament_results
```

### 3. Analyze Results

After running games, analyze the results:

```bash
# Using the Makefile
make neural-analysis

# Or run the analysis script directly
venv/bin/python analyze_neural_results.py
```

## Command Line Interface

### Run Neural Games

```bash
euchre run-neural-games [OPTIONS]

Options:
  -n, --num-games INTEGER     Number of games to run [default: 10000]
  -m1, --model1 TEXT         First model name (plays North/South) [required]
  -m2, --model2 TEXT         Second model name (plays East/West) [required]
  -w, --max-workers INTEGER  Maximum number of parallel workers
  -o, --output-dir TEXT      Output directory for game results [default: neural_games]
```

### Run All Model Combinations

```bash
euchre run-all-neural-combinations [OPTIONS]

Options:
  -n, --num-games INTEGER     Number of games per combination [default: 1000]
  -w, --max-workers INTEGER  Maximum number of parallel workers
  -o, --output-dir TEXT      Output directory for game results [default: neural_games]
```

### List Neural Models

```bash
euchre list-neural-models [OPTIONS]

Options:
  -d, --models-dir TEXT      Directory containing trained models [default: demo_models]
```

## Examples

### Example 1: Alice vs Bob (10,000 games)

```bash
venv/bin/python -m euchre.cli run-neural-games \
  -m1 Alice \
  -m2 Bob \
  -n 10000 \
  -w 8 \
  -o alice_vs_bob_tournament
```

This will:
- Load the `Alice.json` model for North/South positions
- Load the `Bob.json` model for East/West positions
- Run 10,000 games using 8 parallel workers
- Save results to `alice_vs_bob_tournament/` directory

### Example 2: All Model Combinations (1,000 games each)

```bash
venv/bin/python -m euchre.cli run-all-neural-combinations \
  -n 1000 \
  -w 4 \
  -o all_combinations_tournament
```

This will run:
- Alice vs Bob (1,000 games)
- Alice vs Charlie (1,000 games)
- Alice vs David (1,000 games)
- Bob vs Charlie (1,000 games)
- Bob vs David (1,000 games)
- Charlie vs David (1,000 games)

### Example 3: Custom Team Configuration

```python
from euchre.neural_mass_game_runner import NeuralMassGameRunner

# Initialize runner
runner = NeuralMassGameRunner(
    models_dir="demo_models",
    output_dir="custom_tournament",
    max_workers=6
)

# Create custom team configuration
config = runner.create_team_config(
    team1_models=["Alice", "Bob"],      # North/South
    team2_models=["Charlie", "David"],  # East/West
    config_name="AliceBob_vs_CharlieDavid"
)

# Run games
runner.run_neural_games(5000, ["Alice", "Bob"], ["Charlie", "David"])
```

## Output Structure

### Game Results

Each game generates a JSON file with detailed information:

```json
{
  "game_id": "uuid",
  "timestamp": 1234567890,
  "team_config": "Alice_vs_Bob",
  "team1": {
    "players": [["North", "Alice"], ["South", "Alice"]],
    "final_score": 10,
    "total_tricks": 25,
    "tricks_per_round": [5, 5, 5, 5, 5]
  },
  "team2": {
    "players": [["East", "Bob"], ["West", "Bob"]],
    "final_score": 8,
    "total_tricks": 20,
    "tricks_per_round": [0, 0, 0, 0, 0]
  },
  "winner": "team1",
  "total_rounds": 5,
  "rounds": [...],
  "game_duration": 1.23,
  "team_set": false,
  "set_team": null,
  "total_tricks_team1": 25,
  "total_tricks_team2": 20
}
```

### Summary Files

After each tournament, a summary file is generated:

```json
{
  "config_name": "Alice_vs_Bob",
  "total_games": 10000,
  "successful_games": 9998,
  "failed_games": 2,
  "team1_wins": 5123,
  "team2_wins": 4875,
  "ties": 0,
  "team1_total_tricks": 250123,
  "team2_total_tricks": 249877,
  "team_sets": 156,
  "timestamp": 1234567890
}
```

## Analysis and Statistics

### Comprehensive Analysis

The analysis system provides:

- **Win Rates**: By team configuration and individual model
- **Trick Statistics**: Average tricks per game, distribution
- **Team Performance**: Set rates, scoring patterns
- **Model Comparison**: Head-to-head performance metrics

### Visualizations

Automatically generated plots:

1. **Win Rates by Configuration**: Bar chart showing team performance
2. **Model Performance**: Win rates and average tricks comparison
3. **Tricks Distribution**: Visual representation of trick patterns

### Excel Export

All analysis results are exported to Excel with multiple sheets:

- Team Statistics
- Model Performance  
- Tournament Summaries

## Performance Considerations

### Parallel Processing

- **Default**: Uses all available CPU cores
- **Custom**: Specify `--max-workers` for optimal performance
- **Memory**: Each worker loads models independently

### Model Loading

- Models are loaded once per worker process
- Large models may require significant memory
- Consider using fewer workers for memory-constrained systems

### Output Storage

- Each game generates a ~2-5KB JSON file
- 10,000 games ≈ 20-50MB of data
- Results are compressed and organized by tournament

## Troubleshooting

### Common Issues

1. **"Models not found"**
   - Check the `--models-dir` path
   - Ensure model files have `.json` extension
   - Verify model files are valid JSON

2. **Memory errors**
   - Reduce `--max-workers`
   - Close other applications
   - Use smaller batch sizes

3. **Import errors**
   - Ensure virtual environment is activated
   - Check that all dependencies are installed
   - Verify Python path includes the project

### Performance Tips

- **Optimal workers**: Usually 1-2 workers per CPU core
- **Batch size**: 1000-10000 games per run for best performance
- **Storage**: Use SSD for faster I/O with large tournaments

## Advanced Usage

### Custom Risk Profiles

```python
from euchre.ai_model.euchre_nn import create_risk_profile

# Create custom risk parameters
custom_risk = create_risk_profile("aggressive")
custom_risk.aggression = 0.9
custom_risk.conservatism = 0.1

# Use in model player
player = ModelPlayer(name="Custom", model=model, device="cpu", custom_risk=custom_risk)
```

### Integration with Training Pipeline

```python
from euchre.ai_model.self_play_trainer import SelfPlayTrainer

# Train new models
trainer = SelfPlayTrainer(model_config, "models_dir")
trainer.train_self_play(num_games=50000)

# Evaluate with tournament
runner = NeuralMassGameRunner("models_dir")
runner.run_model_vs_model("NewModel", "BaselineModel", 1000)
```

## File Structure

```
project/
├── demo_models/                    # Pre-trained models
│   ├── Alice.json
│   ├── Bob.json
│   ├── Charlie.json
│   └── David.json
├── neural_games/                   # Tournament results
│   ├── game_uuid1.json
│   ├── game_uuid2.json
│   └── summary_Alice_vs_Bob_1234567890.json
├── analysis_plots/                 # Generated visualizations
│   ├── win_rates_by_config.png
│   ├── model_performance.png
│   └── tricks_distribution.png
├── euchre/
│   ├── neural_mass_game_runner.py # Core tournament runner
│   └── cli.py                     # Command line interface
├── run_neural_tournament.py       # Example tournament script
├── analyze_neural_results.py      # Analysis script
└── Makefile                       # Build commands
```

## Contributing

To extend the tournament system:

1. **Add new statistics**: Modify `NeuralMassGameRunner._run_single_neural_game()`
2. **Custom visualizations**: Extend `NeuralTournamentAnalyzer.generate_visualizations()`
3. **New analysis metrics**: Add methods to the analyzer class
4. **CLI commands**: Add new Click commands to `cli.py`

## License

This project is open source. See the LICENSE file for details. 