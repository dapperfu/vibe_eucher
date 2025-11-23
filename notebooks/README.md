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

## Usage

1. Run Monte Carlo simulation first:
   ```bash
   ./scripts/statistical_analysis.py --num-games 1000
   ```

2. Open notebooks in order (00, 01, 10, 11, etc.)

3. Execute cells to see analysis results

## Requirements

- Jupyter Notebook or JupyterLab
- Python packages: numpy, pandas, matplotlib, seaborn, scipy
- Project dependencies (see requirements.txt)
