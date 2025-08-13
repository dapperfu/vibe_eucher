# Euchre AI Notebooks

This folder contains Jupyter notebooks for exploring and demonstrating the Euchre AI systems.

## Available Notebooks

### `m_series_ai_demo.ipynb`

A comprehensive demonstration of the M-Series AI's decision-making process, including:

- **M-Series AI Architecture**: Understanding the neural network models (Magnus, Maverick, Mentor, Mystic) and risk profiles
- **Trained Model Loading**: Loading pre-trained weights from the trained_models/ directory
- **Deck and Card System**: Exploring the OOP design of the game components with Unicode card display
- **Trump Calling Decisions**: Testing scenarios where the AI should make good decisions
- **Risk Profile System**: Understanding how different AI personalities make decisions
- **Game State Encoding**: How the neural network processes game information into 256-feature tensors

## Key Features Demonstrated

1. **Excellent Hand Scenarios**: Hands that should always order up (e.g., J♦ + Q♥ K♥ A♥ with J♥ top card = 4 trump cards)
2. **Trained Model Integration**: Loading and using pre-trained M-Series neural network models
3. **Risk Profile Analysis**: Comparing Magnus (strategic) vs Maverick (aggressive) decision-making
4. **OOP Deck System**: Demonstrating the object-oriented design of cards, suits, and deck management
5. **Neural Network Input**: Understanding how game state gets encoded into 256-feature tensors

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
- Deck creation and card dealing with Unicode card display
- Hand analysis and strength evaluation (e.g., 4 trump cards = excellent hand)
- M-Series AI model creation and trained weight loading
- Risk profile effects on decisions (Magnus vs Maverick strategies)
- Neural network input encoding (256-feature game state representation)
- Successful execution of all cells with proper error handling

## Notes

- **Fully Functional**: All M-Series model functionality is implemented and working
- **Trained Models**: Pre-trained weights are loaded from trained_models/ directory
- **Comprehensive Testing**: All scenarios are tested and execute successfully
- **OOP Design**: The object-oriented design makes it easy to create and test different scenarios
- **Error Handling**: Robust error handling ensures the notebook runs smoothly 