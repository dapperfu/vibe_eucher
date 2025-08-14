# Euchre CLI Game

A command-line implementation of the classic euchre card game with three levels of AI opponents. Play euchre against intelligent AI players in a fully-featured CLI interface with a unified AI system.

## Features

- **Full Euchre Rules**: Implements standard euchre gameplay with proper scoring
- **Three AI Levels**: 
  - **Level 1**: Traditional rule-based AI with configurable risk profiles
  - **Level 2**: Neural network-based AI with pre-trained models
  - **Level 3**: Advanced neural network AI with comprehensive game modeling
- **Unified AI Interface**: All AI levels share the same outward interface for seamless integration
- **CLI Interface**: Clean, intuitive command-line interface using Click
- **Type Safety**: Full mypy typing support for robust code
- **Testing**: Comprehensive test suite with pytest (100% pass rate)
- **Modern Python**: Uses Python 3.8+ features and best practices

## Game Rules

Euchre is a trick-taking card game played with a deck of 24 cards (9, 10, J, Q, K, A of each suit):

1. **Objective**: Be the first team to score 10 points
2. **Players**: 4 players in teams of 2 (sitting across from each other)
3. **Cards**: 9, 10, J, Q, K, A of each suit
4. **Dealing**: 5 cards to each player, 4 cards left in deck
5. **Trump**: The top card can be "ordered up" to set the trump suit
6. **Gameplay**: Players take turns playing cards, following suit if possible
7. **Scoring**: 
   - Win 3-4 tricks: 1 point
   - Win all 5 tricks: 2 points
   - Win 5 tricks after being "euchred": 4 points

## AI System

### Unified Interface
All three AI levels implement the same `BaseAIInterface`, ensuring they can be used interchangeably:

- **Input**: `GameContext` object containing complete game state
- **Output**: `DecisionResult` with decision, confidence, and reasoning
- **Methods**: `should_order_up()`, `should_call_trump()`, `select_trump_suit()`, `play_card()`, `discard_card()`

### AI Levels

#### Level 1: Traditional AI
- **Types**: `level1_aggressive`, `level1_conservative`, `level1_balanced`, `level1_opportunistic`
- **Strategy**: Rule-based decision making with configurable risk profiles
- **Performance**: Fast, explainable, consistent behavior
- **Best For**: Learning, testing, and predictable gameplay

#### Level 2: Neural Network AI
- **Types**: `level2_strategic`, `level2_aggressive`, `level2_balanced`, `level2_intuitive`
- **Strategy**: Pre-trained neural networks with risk profile integration
- **Performance**: Sophisticated decision making, learns from data
- **Best For**: Advanced gameplay and research

#### Level 3: Advanced Neural AI
- **Types**: `level3_strategic`, `level3_aggressive`, `level3_balanced`, `level3_conservative`, `level3_opportunistic`
- **Strategy**: Advanced neural networks with comprehensive game state modeling
- **Performance**: Highest level of sophistication and adaptability
- **Best For**: Research, advanced AI development, and cutting-edge gameplay

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- For Level 2/3 AI: PyTorch (automatically installed with GPU support if available)

### Setup

#### Method 1: Traditional Installation (Recommended for Development)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd vibe_eucher
   ```

2. **Create virtual environment and install dependencies**:
   ```bash
   make install
   ```

   Or manually:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

#### Method 2: Pip Installation (Recommended for End Users)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd vibe_eucher
   ```

2. **Install the package with pip**:
   ```bash
   make install-pip
   ```

   Or manually:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

   This method installs the `euchre` command directly to your PATH, making it available as a system command.

## Usage

### Quick Start

#### With Traditional Installation
Start a new game with default settings:
```bash
make run
```

Or manually:
```bash
venv/bin/python -m euchre.cli play
```

#### With Pip Installation
Start a new game directly:
```bash
euchre play
```

### AI Game Examples

#### Level 1 AI Game
```bash
# Start game with Level 1 AI players
euchre play --ai-types level1_aggressive level1_conservative level1_balanced level1_opportunistic
```

#### Level 2 AI Game
```bash
# Start game with Level 2 AI players (requires trained models)
euchre play --ai-types level2_strategic level2_aggressive level2_balanced level2_intuitive
```

#### Level 3 AI Game
```bash
# Start game with Level 3 AI players
euchre play --ai-types level3_strategic level3_aggressive level3_balanced level3_conservative
```

#### Mixed AI Levels
```bash
# Mix different AI levels
euchre play --ai-types level1_aggressive level2_strategic level3_balanced level1_conservative
```

### Available Commands

#### Play a Game
```bash
# Start a new game with default settings
euchre play

# Customize player name and AI types
euchre play --player-name "YourName" --ai-types level1_aggressive level1_conservative level1_balanced level1_opportunistic

# Use short options
euchre play -n "YourName" -t level1_aggressive -t level1_conservative -t level1_balanced -t level1_opportunistic
```

#### AI Tournaments
```bash
# Run AI vs AI tournament
euchre ai-tournament

# Run comprehensive tournament with all AI levels
euchre comprehensive-tournament

# Run Level 3 specific tournament
euchre run-level3-tournament
```

#### View Rules
```bash
euchre rules
```

#### Run Demo
```bash
euchre demo
```

#### Get Help
```bash
euchre --help
euchre play --help
```

### Game Controls

During gameplay:
- **Card Selection**: Choose cards by number (1-5)
- **Trump Decisions**: Choose whether to order up or pass
- **Game Flow**: Follow the prompts to play your turn

## Development

### Project Structure

```
vibe_eucher/
├── euchre/                 # Main package
│   ├── __init__.py        # Package initialization
│   ├── models.py          # Game models (Player, Card, etc.)
│   ├── game.py            # Game logic and AI integration
│   ├── cli.py             # Command-line interface
│   ├── ai/                # AI system
│   │   ├── __init__.py
│   │   ├── base_ai_interface.py # Unified AI interface
│   │   ├── traditional_ai_impl.py # Level 1 AI
│   │   ├── level2_ai_impl.py # Level 2 AI
│   │   ├── level3_ai_impl.py # Level 3 AI
│   │   ├── ai_factory.py  # AI player factory
│   │   └── ai_adapter.py  # Legacy API adapter
│   ├── core/              # Core game engine
│   │   ├── __init__.py
│   │   ├── deck.py        # Deck management
│   │   ├── game_state.py  # Game state tracking
│   │   ├── trick_manager.py # Trick management
│   │   └── scoring.py     # Scoring system
│   ├── game_logic/        # Game rules and logic
│   │   ├── __init__.py
│   │   └── trump_selection.py # Trump selection logic
│   ├── utils/             # Utility functions
│   │   ├── __init__.py
│   │   └── logging_config.py # Logging configuration
│   └── ai_model/          # Neural network models
│       ├── __init__.py
│       ├── level2_models.py # Level 2 neural models
│       └── level3_models.py # Level 3 neural models
├── tests/                  # Test suite (100% pass rate)
│   ├── __init__.py
│   ├── test_models.py     # Tests for models
│   ├── test_game.py       # Tests for game logic
│   ├── test_ai_game.py    # Tests for AI gameplay
│   └── test_ai_profiles.py # Tests for AI profiles
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks (generated from Python scripts)
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
├── Makefile               # Build and development commands
└── README.md              # This file
```

### Development Commands

```bash
# Set up development environment (traditional method)
make install

# Set up development environment (pip method)
make install-pip

# Run tests (all tests pass)
make test

# Run the game (traditional method)
make run

# Run the game (pip method)
euchre play

# Clean up
make clean

# Show all available commands
make help
```

### Code Quality

The project follows strict coding standards:

- **Type Hints**: Full mypy typing support
- **Documentation**: NumPy-style docstrings
- **Formatting**: Black code formatter
- **Linting**: Flake8 for style checking
- **Testing**: pytest for unit tests (100% pass rate)

### Running Tests

```bash
# Run all tests (all pass)
venv/bin/pytest

# Run with verbose output
venv/bin/pytest -v

# Run specific test file
venv/bin/pytest tests/test_models.py

# Run with coverage (if pytest-cov installed)
venv/bin/pytest --cov=euchre
```

## Configuration

### Customizing AI Types

You can customize the AI opponent types when starting a game:

```bash
# Level 1 AI with different personalities
euchre play --ai-types level1_aggressive level1_conservative level1_balanced level1_opportunistic

# Level 2 AI with trained models
euchre play --ai-types level2_strategic level2_aggressive level2_balanced level2_intuitive

# Level 3 AI with advanced models
euchre play --ai-types level3_strategic level3_aggressive level3_balanced level3_conservative level3_opportunistic
```

### Environment Variables

No environment variables are required for basic operation.

## Troubleshooting

### Common Issues

1. **"Euchre requires exactly 4 players"**
   - This is expected behavior - euchre is a 4-player game

2. **Import errors**
   - Ensure you're using the virtual environment: `source venv/bin/activate`

3. **Permission errors**
   - Make sure you have write permissions in the project directory

4. **"euchre command not found"**
   - If using pip installation, ensure the virtual environment is activated
   - Try running `which euchre` to verify the command location

5. **AI model loading errors**
   - Level 2/3 AI require PyTorch: `pip install torch torchvision torchaudio`
   - Ensure trained models are in the correct directory

### Getting Help

- Run `euchre --help` for command-line help
- Check the test files for usage examples
- Review the source code for implementation details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `make test`
6. Submit a pull request

## License

This project is open source. See the LICENSE file for details.

## Acknowledgments

- Built with [Click](https://click.palletsprojects.com/) for the CLI interface
- Uses [pytest](https://pytest.org/) for testing
- Follows modern Python packaging standards
- AI system built with [PyTorch](https://pytorch.org/) for neural networks

## Version History

- **0.2.0**: Unified AI Interface and Three AI Levels
  - Unified `BaseAIInterface` for all AI levels
  - Level 1: Traditional rule-based AI with risk profiles
  - Level 2: Neural network AI with pre-trained models
  - Level 3: Advanced neural network AI with comprehensive modeling
  - 100% test pass rate
  - Comprehensive code cleanup and API simplification
- **0.1.0**: Initial release with basic game functionality
  - Player class implementation
  - Basic AI opponents
  - CLI interface
  - Core game mechanics
  - Pip installation support
  - Command-line tool in PATH 