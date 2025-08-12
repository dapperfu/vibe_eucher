# Euchre CLI Game

A command-line implementation of the classic euchre card game with AI opponents. Play euchre against three AI players in a fully-featured CLI interface.

## Features

- **Full Euchre Rules**: Implements standard euchre gameplay with proper scoring
- **AI Opponents**: Three AI players with intelligent card-playing strategies
- **CLI Interface**: Clean, intuitive command-line interface using Click
- **Type Safety**: Full mypy typing support for robust code
- **Testing**: Comprehensive test suite with pytest
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

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

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

## Usage

### Quick Start

Start a new game with default settings:
```bash
make run
```

Or manually:
```bash
venv/bin/python -m euchre.cli play
```

### Available Commands

#### Play a Game
```bash
# Start a new game with default settings
euchre play

# Customize player name and AI names
euchre play --player-name "YourName" --ai-names Alice Bob Charlie

# Use short options
euchre play -n "YourName" -a Alice -a Bob -a Charlie
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
│   ├── game.py            # Game logic and AI
│   └── cli.py             # Command-line interface
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_models.py     # Tests for models
│   └── test_game.py       # Tests for game logic
├── Makefile               # Build and development commands
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

### Development Commands

```bash
# Set up development environment
make install

# Run tests
make test

# Run the game
make run

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
- **Testing**: pytest for unit tests

### Running Tests

```bash
# Run all tests
venv/bin/pytest

# Run with verbose output
venv/bin/pytest -v

# Run specific test file
venv/bin/pytest tests/test_models.py

# Run with coverage (if pytest-cov installed)
venv/bin/pytest --cov=euchre
```

## Configuration

### Customizing AI Names

You can customize the AI opponent names when starting a game:

```bash
euchre play --ai-names "Sherlock" "Watson" "Moriarty"
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

## Version History

- **0.1.0**: Initial release with basic game functionality
  - Player class implementation
  - Basic AI opponents
  - CLI interface
  - Core game mechanics 