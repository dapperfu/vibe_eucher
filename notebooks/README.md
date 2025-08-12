# 📚 Euchre Jupyter Notebooks

This directory contains interactive Jupyter notebooks for exploring and analyzing the euchre game system.

## 📖 Available Notebooks

### 1. 🎮 `euchre_game_playground.ipynb`
**Interactive Game Playground**
- Play individual euchre games with different AI profiles
- Experiment with risk ratios and team compositions
- Watch games unfold round by round
- Compare different AI strategies
- Analyze individual game results

**Best for**: Learning about AI behavior, testing strategies, understanding game dynamics

### 2. 📊 `euchre_data_analysis.ipynb`
**Mass Game Data Analysis**
- Load and analyze thousands of game results
- Statistical analysis of team performance
- Visualization of game patterns
- Comparative analysis of AI strategies
- Export insights to Excel and plots

**Best for**: Research, statistical analysis, understanding patterns across many games

### 3. 🚀 `run_mass_games.ipynb`
**Mass Game Execution**
- Run thousands of games in parallel
- Monitor progress in real-time
- Configure parallel processing
- Generate sample datasets for analysis
- Clean up generated files

**Best for**: Generating data, running experiments, scaling up analysis

## 🚀 Getting Started

### Prerequisites
1. **Install Dependencies**: Make sure you have all required packages
   ```bash
   make install
   ```

2. **Start Jupyter**: Launch Jupyter Lab or Jupyter Notebook
   ```bash
   cd notebooks
   jupyter lab
   # or
   jupyter notebook
   ```

3. **Kernel**: Use the Python 3 kernel with the virtual environment

### Recommended Workflow

1. **Start with Game Playground** (`euchre_game_playground.ipynb`)
   - Understand how AI profiles work
   - See games in action
   - Experiment with different configurations

2. **Generate Data** (`run_mass_games.ipynb`)
   - Run demo games (100-1000)
   - Generate production datasets (10,000+)
   - Monitor progress and performance

3. **Analyze Results** (`euchre_data_analysis.ipynb`)
   - Load generated game data
   - Perform statistical analysis
   - Create visualizations
   - Export insights

## 🎯 Use Cases

### Academic Research
- Game theory analysis
- AI behavior studies
- Strategy effectiveness research
- Statistical validation

### Strategy Development
- Testing AI approaches
- Optimizing risk ratios
- Team composition analysis
- Performance benchmarking

### Educational Use
- Understanding game dynamics
- Learning about AI decision-making
- Statistical analysis practice
- Data visualization skills

### Tournament Design
- Large-scale competition analysis
- Fairness assessment
- Performance metrics
- Ranking systems

## 🔧 Configuration Options

### AI Profiles
- **Aggressive**: High risk tolerance, leads with high cards
- **Conservative**: Low risk tolerance, leads with low cards
- **Balanced**: Moderate approach, adaptive strategy
- **Opportunistic**: Game state aware, dynamic strategy

### Risk Ratios
- **0.0**: Very conservative (requires 4+ trump cards)
- **0.5**: Balanced approach (requires 2-3 trump cards)
- **1.0**: Very aggressive (requires 1+ trump cards)

### Team Configurations
- `aggressive_vs_conservative`: Classic high vs low risk
- `aggressive_vs_balanced`: Aggressive vs moderate
- `conservative_vs_balanced`: Conservative vs moderate
- `opportunistic_vs_aggressive`: Adaptive vs high risk
- `mixed_vs_mixed`: Balanced team compositions

## 📊 Output Formats

### Game Results
- **JSON files**: Individual game data with UUIDs
- **Structured data**: Rounds, scores, player decisions
- **Metadata**: Timestamps, configurations, performance

### Analysis Outputs
- **Excel spreadsheets**: Multiple analysis sheets
- **Visualization plots**: Professional charts and graphs
- **Statistical summaries**: Comprehensive game insights
- **Round-by-round data**: Game progression analysis

## 🚀 Performance Tips

### Parallel Processing
- Use `max_workers=None` for all CPU cores
- Monitor system resources during large runs
- Consider running overnight for massive datasets

### Storage Management
- Monitor disk space for large runs
- Use `cleanup_games` to remove old data
- Organize output directories by experiment

### Analysis Optimization
- Load games in batches for memory efficiency
- Use `max_games` parameter to limit dataset size
- Export results to avoid recomputing analysis

## 🔍 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure you're using the correct Python environment
2. **Memory Issues**: Limit `max_games` parameter for large datasets
3. **Performance**: Check CPU usage and adjust `max_workers`
4. **Storage**: Monitor disk space for large game runs

### Getting Help
- Check the main project README
- Review error messages in notebook output
- Verify all dependencies are installed
- Ensure proper file paths and permissions

## 📈 Scaling Up

### Small Scale (100-1,000 games)
- Use for testing and development
- Quick feedback on configurations
- Basic statistical analysis

### Medium Scale (1,000-10,000 games)
- Good for research projects
- Reliable statistical significance
- Comprehensive analysis possible

### Large Scale (10,000+ games)
- Research-grade datasets
- High statistical confidence
- Complex pattern analysis
- Publication-quality results

## 🎉 Happy Analyzing!

These notebooks provide a powerful platform for exploring euchre AI behavior, running experiments, and generating insights. Start small, experiment freely, and scale up as needed!

For questions or issues, refer to the main project documentation or create an issue in the repository. 