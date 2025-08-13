# Development Setup

This guide helps developers set up the Euchre project for development, testing, and contribution.

## Prerequisites

### Required Software
- **Python**: Version 3.8 or higher
- **Git**: For version control
- **Make**: For build automation (optional but recommended)
- **pip**: Python package manager

### Optional Software
- **Docker**: For containerized development
- **PyCharm/VS Code**: For IDE development
- **Jupyter**: For interactive development and analysis

## Development Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/dapperfu/vibe_eucher.git
cd vibe_eucher
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### 3. Install Development Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 4. Verify Setup

```bash
# Test basic functionality
python -m euchre.cli_main --help

# Run tests
python -m pytest tests/ -v

# Check code quality
python -m flake8 euchre/
python -m black --check euchre/
python -m mypy euchre/
```

## Using Makefile (Recommended)

The project includes a comprehensive Makefile for common development tasks:

```bash
# Install dependencies
make install

# Run tests
make test

# Run the game
make run

# Code quality checks
make lint

# Format code
make format

# Clean up
make clean

# Show all available commands
make help
```

## Project Structure

```
vibe_eucher/
├── euchre/                    # Main package
│   ├── __init__.py           # Package initialization
│   ├── models.py             # Game models (Player, Card, etc.)
│   ├── game.py               # Main game controller
│   ├── cli.py                # CLI implementation
│   ├── cli_main.py           # CLI entry point
│   ├── ai/                   # AI system
│   │   ├── __init__.py
│   │   ├── base_ai.py        # Base AI class
│   │   ├── ai_factory.py     # AI player factory
│   │   └── ai_profiles.py    # AI personality profiles
│   ├── core/                 # Core game engine
│   │   ├── __init__.py
│   │   ├── deck.py           # Deck management
│   │   ├── game_state.py     # Game state tracking
│   │   ├── trick_manager.py  # Trick management
│   │   └── scoring.py        # Scoring system
│   ├── game_logic/           # Game rules and logic
│   │   ├── __init__.py
│   │   └── trump_selection.py # Trump selection logic
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   └── logging_config.py # Logging configuration
│   └── ai_model/             # Neural network models
│       ├── __init__.py
│       └── ...               # AI model implementations
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_models.py        # Model tests
│   ├── test_game.py          # Game logic tests
│   └── test_ai.py            # AI system tests
├── docs/                     # Documentation
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
├── Makefile                  # Build automation
└── README.md                 # Project overview
```

## Development Workflow

### 1. Code Development

```bash
# Activate virtual environment
source venv/bin/activate

# Make code changes
# ... edit files ...

# Run tests to ensure nothing broke
make test

# Check code quality
make lint

# Format code
make format
```

### 2. Testing

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_models.py -v

# Run with coverage
python -m pytest --cov=euchre tests/

# Run specific test function
python -m pytest tests/test_game.py::test_game_initialization -v
```

### 3. Code Quality

```bash
# Check code style
make lint

# Auto-format code
make format

# Type checking
make type-check

# Security checks
make security-check
```

## Development Tools

### Code Quality Tools

#### Black (Code Formatter)
```bash
# Format code
black euchre/

# Check formatting
black --check euchre/
```

#### Flake8 (Linting)
```bash
# Run linter
flake8 euchre/

# Specific checks
flake8 --select=E,W,F euchre/  # Errors, warnings, pyflakes
```

#### MyPy (Type Checking)
```bash
# Type checking
mypy euchre/

# Strict mode
mypy --strict euchre/
```

### Testing Tools

#### Pytest
```bash
# Basic test run
pytest

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run tests in parallel
pytest -n auto
```

#### Coverage
```bash
# Run with coverage
pytest --cov=euchre tests/

# Generate HTML report
pytest --cov=euchre --cov-report=html tests/
# Open htmlcov/index.html in browser
```

### Development Tools

#### Jupyter
```bash
# Start Jupyter Notebook
jupyter notebook

# Start Jupyter Lab
jupyter lab
```

#### Debugging
```bash
# Run with debugger
python -m pdb -m euchre.cli_main play

# Add breakpoints in code
import pdb; pdb.set_trace()
```

## Configuration

### Environment Variables

```bash
# Development mode
export EUCHRE_DEV_MODE=1

# Logging level
export EUCHRE_LOG_LEVEL=DEBUG

# AI model path
export EUCHRE_MODEL_PATH=./models/
```

### Configuration Files

#### pyproject.toml
```toml
[tool.black]
line-length = 88
target-version = ['py38']

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
```

#### .flake8
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist,venv
```

## Common Development Tasks

### Adding New Features

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-feature-name
   ```

2. **Implement Feature**
   - Write code following project standards
   - Add appropriate tests
   - Update documentation

3. **Test Changes**
   ```bash
   make test
   make lint
   make type-check
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add new feature: description"
   ```

### Debugging Issues

1. **Enable Debug Logging**
   ```bash
   python -m euchre.cli_main --very-verbose play
   ```

2. **Use Interactive Debugger**
   ```python
   import pdb; pdb.set_trace()
   ```

3. **Check Logs**
   ```bash
   # Look for log files
   find . -name "*.log"
   ```

### Performance Profiling

```bash
# Profile specific functions
python -m cProfile -o profile.stats -m euchre.cli_main ai-vs-ai

# Analyze results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

## Contributing Guidelines

### Code Standards

1. **Python Style**: Follow PEP 8 (enforced by Black)
2. **Type Hints**: Use mypy-compatible type annotations
3. **Documentation**: NumPy-style docstrings
4. **Testing**: Maintain >90% test coverage

### Commit Messages

Follow the project's commit message format:
```
Brief description of changes

/**
 * Technical attribution comment
 * Generated via Cursor IDE with AI assistance
 * Model: Anthropic Claude 3.5 Sonnet
 * Generation timestamp: YYYY-MM-DD HH:MM:SS
 * Context: Brief description of what this code does
 * 
 * Technical details:
 * - LLM: Claude 3.5 Sonnet (YYYY-MM-DD)
 * - IDE: Cursor (cursor.sh)
 * - Generation method: AI-assisted pair programming
 * - Code style: Python with numpy docstring style
 * - Dependencies: Key dependencies
 */
```

### Pull Request Process

1. **Fork Repository**: Create your own fork
2. **Feature Branch**: Work on feature branches
3. **Tests**: Ensure all tests pass
4. **Quality Checks**: Pass linting and type checking
5. **Documentation**: Update relevant documentation
6. **Submit PR**: Create pull request with clear description

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall package
pip install -e .
```

#### Test Failures
```bash
# Check test dependencies
pip install -r requirements.txt

# Run tests individually
python -m pytest tests/test_specific.py -v
```

#### Code Quality Issues
```bash
# Auto-format code
make format

# Fix common linting issues
make lint-fix
```

#### Performance Issues
```bash
# Profile the code
python -m cProfile -o profile.stats -m euchre.cli_main ai-vs-ai

# Check memory usage
python -m memory_profiler euchre/game.py
```

### Getting Help

1. **Check Documentation**: Review this guide and other docs
2. **Run Tests**: Ensure your environment is working
3. **Check Issues**: Look for similar problems on GitHub
4. **Ask Questions**: Use GitHub Discussions or Issues

## Advanced Development

### Docker Development

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "-m", "euchre.cli_main", "--help"]
```

### CI/CD Integration

The project includes GitHub Actions for automated testing:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python -m pytest tests/ -v
```

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

---

*For testing details, see [Testing Guide](testing.md)*
*For code standards, see [Code Standards](standards.md)*
*For project structure, see [Project Structure](structure.md)* 