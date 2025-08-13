# Euchre AI Notebooks

This folder contains Jupyter notebooks for exploring and demonstrating the Euchre AI systems.

## Available Notebooks

### `m_series_ai_demo.ipynb`

A comprehensive demonstration of the M-Series AI's decision-making process, including:

- **M-Series AI Architecture**: Understanding the neural network models and risk profiles
- **Deck and Card System**: Exploring the OOP design of the game components
- **Trump Calling Decisions**: Testing scenarios where the AI should make good decisions
- **Card Playing Logic**: Analyzing how the AI chooses which card to play
- **Risk Profile System**: Understanding how different AI personalities make decisions
- **Game State Encoding**: How the neural network processes game information

## Key Features Demonstrated

1. **Excellent Hand Scenarios**: Hands that should always order up (e.g., Left Bower + 3 trump cards)
2. **Marginal Hand Scenarios**: Hands where the decision depends on AI risk profile
3. **Weak Hand Scenarios**: Hands that should always pass
4. **Partner Coordination**: How dealer position affects decisions
5. **Game State Awareness**: How scores and round numbers influence AI behavior

## Running the Notebooks

1. Ensure you have Jupyter installed: `pip install jupyter`
2. Navigate to this folder: `cd notebooks`
3. Start Jupyter: `jupyter notebook`
4. Open `m_series_ai_demo.ipynb`

## Requirements

- Python 3.8+
- PyTorch (for M-Series models)
- Jupyter notebook
- The euchre package and its dependencies

## Expected Outputs

The notebook will demonstrate:
- Deck creation and card dealing
- Hand analysis and strength evaluation
- AI decision-making processes
- Risk profile effects on decisions
- Neural network input encoding

## Notes

- Some M-Series model functionality may not be fully implemented yet
- The notebook includes fallback explanations when models aren't available
- All scenarios are designed to show clear, logical decision-making
- The OOP design makes it easy to create and test different scenarios 