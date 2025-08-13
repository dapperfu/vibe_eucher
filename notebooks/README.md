# Euchre AI Notebooks

This folder contains Jupyter notebooks for exploring and demonstrating the Euchre AI systems.

## Notebook Generation System

**IMPORTANT**: All notebooks are generated programmatically from Python scripts with the same name. These scripts are the "golden master" for each notebook.

- `level1_ai_demo.py` → `level1_ai_demo.ipynb`
- `level2_ai_demo.py` → `level2_ai_demo.ipynb`

To regenerate a notebook, run: `python notebook_name.py`

## Available Notebooks

### `level1_ai_demo.ipynb`

A comprehensive demonstration of the Level 1 AI's decision-making process, including:

- **Level 1 AI Architecture**: Understanding the rule-based AI models and risk profiles
- **Traditional Logic**: Demonstrating rule-based decision-making with configurable risk
- **Deck and Card System**: Exploring the OOP design of the game components with Unicode card display
- **Trump Calling Decisions**: Testing scenarios where the AI should make good decisions
- **Risk Profile System**: Understanding how different AI personalities make decisions
- **Rule-Based Strategy**: How traditional AI evaluates hands and makes choices

### `level2_ai_demo.ipynb`

A comprehensive demonstration of the Level 2 AI's decision-making process, including:

- **Level 2 AI Architecture**: Understanding the neural network models and risk profiles
- **Trained Model Loading**: Loading pre-trained weights from the trained_models/ directory
- **Deck and Card System**: Exploring the OOP design of the game components with Unicode card display
- **Trump Calling Decisions**: Testing scenarios where the AI should make good decisions
- **Risk Profile System**: Understanding how different AI personalities make decisions
- **Game State Encoding**: How the neural network processes game information into 256-feature tensors

## Key Features Demonstrated

### Level 1 AI
1. **Rule-Based Logic**: Traditional AI using hard-coded rules and heuristics
2. **Risk Profiles**: Configurable risk tolerance (0.0 = conservative, 1.0 = aggressive)
3. **Fast Decision Making**: Quick rule-based evaluation without neural network overhead
4. **Explainable Decisions**: Clear, understandable decision-making process
5. **OOP Deck System**: Demonstrating the object-oriented design of cards, suits, and deck management

### Level 2 AI
1. **Excellent Hand Scenarios**: Hands that should always order up (e.g., J♦ + Q♥ K♥ A♥ with J♥ top card = 4 trump cards)
2. **Trained Model Integration**: Loading and using pre-trained Level 2 neural network models
3. **Risk Profile Analysis**: Comparing Strategic (strategic) vs Aggressive (aggressive) decision-making
4. **Neural Network Input**: Understanding how game state gets encoded into 256-feature tensors

## Running the Notebooks

1. Ensure you have Jupyter installed: `pip install jupyter`
2. Navigate to this folder: `cd notebooks`
3. Start Jupyter: `jupyter notebook`
4. Open either `level1_ai_demo.ipynb` or `level2_ai_demo.ipynb`

## Requirements

- Python 3.8+
- PyTorch (for Level 2 models)
- Jupyter notebook
- The euchre package and its dependencies

## Expected Outputs

### Level 1 Notebook
- Deck creation and card dealing with Unicode card display
- Hand analysis and strength evaluation (e.g., 4 trump cards = excellent hand)
- Level 1 AI model creation and personality demonstration
- Risk profile effects on decisions (Aggressive vs Conservative strategies)
- Rule-based decision-making demonstration

### Level 2 Notebook
- Deck creation and card dealing with Unicode card display
- Hand analysis and strength evaluation (e.g., 4 trump cards = excellent hand)
- Level 2 AI model creation and trained weight loading
- Risk profile effects on decisions (Strategic vs Aggressive strategies)
- Neural network input encoding (256-feature game state representation)
- Successful execution of all cells with proper error handling

## Notes

- **Fully Functional**: All Level 1 and Level 2 model functionality is implemented and working
- **Trained Models**: Pre-trained weights are loaded from trained_models/ directory (Level 2)
- **Comprehensive Testing**: All scenarios are tested and execute successfully
- **OOP Design**: The object-oriented design makes it easy to create and test different scenarios
- **Error Handling**: Robust error handling ensures the notebook runs smoothly
- **Programmatic Generation**: All notebooks can be regenerated from their Python scripts

## AI Level System

The Euchre AI system is organized into levels:

- **Level 1**: Traditional rule-based AI with randomness and risk (level1_aggressive, level1_conservative, level1_balanced, level1_opportunistic)
- **Level 2**: Heavily trained neural network models (level2_strategic, level2_aggressive, level2_balanced, level2_intuitive)
- **Level 3**: Future advanced models with different architecture (TBD)

### Level 1 AI Types
- **level1_aggressive**: Bold, takes chances, orders up marginal hands
- **level1_conservative**: Cautious, only orders up strong hands
- **level1_balanced**: Moderate risk tolerance, balanced approach
- **level1_opportunistic**: Adapts to game situation, takes calculated risks

### Level 2 AI Types
- **level2_strategic**: Deep strategic thinking and partner coordination
- **level2_aggressive**: Bold and unpredictable, taking calculated risks
- **level2_balanced**: Most balanced and adaptable player
- **level2_intuitive**: Deep intuition about game patterns

These notebooks demonstrate the capabilities of both AI levels, showing how traditional rule-based systems compare to neural network-based approaches. 