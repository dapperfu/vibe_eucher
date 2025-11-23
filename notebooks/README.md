# Statistical Analysis Notebooks

This directory contains Jupyter notebooks for analyzing PyTorch AI player performance.

## Notebook Organization

### 0x_* - Introduction and Data Manipulation
- **00_introduction.ipynb**: Setup, imports, and directory structure
- **01_single_game_analysis.ipynb**: Run and analyze a single game

### 1x_* - Statistical Analysis
- **10_data_loading.ipynb**: Load Monte Carlo simulation data
- **11_basic_statistics.ipynb**: Basic statistical analysis
- **12_bell_curve_analysis.ipynb**: Normality tests and bell curve analysis
- **13_card_distribution.ipynb**: Card distribution analysis

### 2x_* - Visualization and Plotting
- **20_visualizations.ipynb**: Comprehensive visualizations
- **21_risk_factor_analysis.ipynb**: Risk factor performance analysis

### 5x_* - Interactive Gameplay
- **50_interactive_trick.ipynb**: Play a single trick interactively
- **51_interactive_hand.ipynb**: Play a full hand interactively
- **52_interactive_game.ipynb**: Play a full game interactively

### 6x_* - AI Assistant
- **60_ai_assistant_setup.ipynb**: Setup and benchmark AI Assistant
- **61_ai_assistant_order_up.ipynb**: Analyze order up decisions with win probabilities
- **62_ai_assistant_card_play.ipynb**: Analyze card play decisions with win percentages
- **63_ai_assistant_full_analysis.ipynb**: Comprehensive decision analysis

## Usage

1. Run Monte Carlo simulation first:
   ```bash
   ./scripts/statistical_analysis.py --num-games 1000
   ```

2. Open notebooks in order (00, 01, 10, 11, etc.)

3. Execute cells to see analysis results

## Requirements

- Jupyter Notebook or JupyterLab
- Python packages: numpy, pandas, matplotlib, seaborn, scipy, ipywidgets
- Project dependencies (see requirements.txt)

## Benchmarking

Before using the AI Assistant (6x_* notebooks), run a benchmark to configure thinking time:

```bash
python scripts/benchmark_simulations.py
```

This will:
- Measure simulation performance on your machine
- Save benchmarks to `benchmarks/performance.json`
- Configure thinking time presets (fast, quick, normal, thorough, deep)

The AI Assistant uses these benchmarks to determine how many simulations to run for each thinking time preset.
