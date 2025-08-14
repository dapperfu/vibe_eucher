# Testing Guide

This document explains how to run tests, write new tests, and maintain test quality for the Euchre project. The test suite now achieves 100% pass rate with comprehensive coverage of the unified AI interface.

## Quick Start

### Run All Tests
```bash
# Basic test run (all tests pass)
make test

# Or manually
venv/bin/pytest

# With verbose output
venv/bin/pytest -v
```

### Run Specific Tests
```bash
# Test specific file
venv/bin/pytest tests/test_models.py

# Test specific function
venv/bin/pytest tests/test_game.py::test_game_initialization

# Test with pattern matching
venv/bin/pytest -k "test_trump"
```

## Test Structure

### Directory Organization
```
tests/
├── __init__.py
├── test_models.py          # Tests for game models
├── test_game.py            # Tests for game logic and AI integration
├── test_ai_game.py         # Tests for AI gameplay functionality
├── test_ai_profiles.py     # Tests for AI profile system
├── conftest.py             # Test configuration and fixtures
└── __init__.py
```

### Test File Naming
- **Files**: `test_*.py`
- **Functions**: `test_*`
- **Classes**: `Test*`

## Running Tests

### Basic Commands

#### Run All Tests
```bash
# Using make (recommended)
make test

# Using pytest directly
pytest

# Using Python module
python -m pytest
```

#### Verbose Output
```bash
# Show test names and results
pytest -v

# Show more details
pytest -vv

# Show local variables on failure
pytest -l
```

#### Stop on First Failure
```bash
# Stop immediately on first failure
pytest -x

# Stop after N failures
pytest --maxfail=3
```

### Test Selection

#### By File
```bash
# Test specific file
pytest tests/test_models.py

# Test multiple files
pytest tests/test_models.py tests/test_game.py
```

#### By Function
```bash
# Test specific function
pytest tests/test_game.py::test_game_initialization

# Test multiple functions
pytest tests/test_game.py::test_game_initialization tests/test_game.py::test_card_dealing
```

#### By Pattern
```bash
# Test functions matching pattern
pytest -k "test_trump"

# Test functions excluding pattern
pytest -k "not test_trump"

# Test with regex
pytest -k "test_.*_trump"
```

#### By Markers
```bash
# Run only fast tests
pytest -m fast

# Run only slow tests
pytest -m slow

# Run integration tests
pytest -m integration

# Skip marked tests
pytest -m "not slow"
```

### Test Execution

#### Parallel Execution
```bash
# Run tests in parallel
pytest -n auto

# Specify number of workers
pytest -n 4

# Install pytest-xdist for parallel execution
pip install pytest-xdist
```

#### Coverage
```bash
# Run with coverage
pytest --cov=euchre

# Generate HTML report
pytest --cov=euchre --cov-report=html

# Generate XML report (for CI)
pytest --cov=euchre --cov-report=xml

# Install pytest-cov for coverage
pip install pytest-cov
```

## Writing Tests

### Basic Test Structure

#### Simple Test
```python
def test_card_creation():
    """Test that cards are created correctly."""
    card = Card(rank=Rank.ACE, suit=Suit.HEARTS)
    assert card.rank == Rank.ACE
    assert card.suit == Suit.HEARTS
    assert str(card) == "Ace of Hearts"
```

#### Test with Setup
```python
def test_deck_dealing():
    """Test that deck deals correct number of cards."""
    deck = Deck()
    hands = deck.deal_cards(4)
    
    assert len(hands) == 4
    for hand in hands:
        assert len(hand) == 5
```

#### Test with Teardown
```python
def test_game_state_reset():
    """Test that game state resets correctly."""
    game_state = GameStateManager()
    game_state.set_dealer(Player("Alice"))
    game_state.set_trump_suit(Suit.HEARTS, Player("Bob"))
    
    # Verify state is set
    assert game_state.get_dealer().name == "Alice"
    assert game_state.get_trump_suit() == Suit.HEARTS
    
    # Reset and verify
    game_state.reset()
    assert game_state.get_dealer() is None
    assert game_state.get_trump_suit() is None
```

### Test Fixtures

#### Using Fixtures
```python
import pytest
from euchre.models import Player, Card, Suit, Rank

@pytest.fixture
def sample_players():
    """Create sample players for testing."""
    return [
        Player("Alice", PlayerType.AI),
        Player("Bob", PlayerType.AI),
        Player("Charlie", PlayerType.AI),
        Player("David", PlayerType.AI)
    ]

@pytest.fixture
def sample_hand():
    """Create a sample hand for testing."""
    return [
        Card(rank=Rank.ACE, suit=Suit.HEARTS),
        Card(rank=Rank.KING, suit=Suit.DIAMONDS),
        Card(rank=Rank.QUEEN, suit=Suit.CLUBS),
        Card(rank=Rank.JACK, suit=Suit.SPADES),
        Card(rank=Rank.TEN, suit=Suit.HEARTS)
    ]

def test_player_with_hand(sample_players, sample_hand):
    """Test player with a hand of cards."""
    player = sample_players[0]
    player.hand = sample_hand
    
    assert len(player.hand) == 5
    assert player.hand[0].suit == Suit.HEARTS
    assert player.hand[0].rank == Rank.ACE
```

#### Fixture Scope
```python
@pytest.fixture(scope="session")
def shared_data():
    """Data shared across all tests in session."""
    return load_large_dataset()

@pytest.fixture(scope="module")
def module_data():
    """Data shared across tests in module."""
    return setup_module_data()

@pytest.fixture(scope="class")
def class_data():
    """Data shared across tests in class."""
    return setup_class_data()

@pytest.fixture(scope="function")
def function_data():
    """Data created for each test function."""
    return setup_function_data()
```

### Test Classes

#### Class-Based Tests
```python
class TestDeck:
    """Test deck functionality."""
    
    def setup_method(self):
        """Set up before each test method."""
        self.deck = Deck()
    
    def test_deck_creation(self):
        """Test deck is created with correct number of cards."""
        assert len(self.deck.cards) == 24
    
    def test_deck_reset(self):
        """Test deck resets to original state."""
        # Deal some cards
        hands = self.deck.deal_cards(4)
        assert len(self.deck.cards) == 4  # 4 cards left
        
        # Reset deck
        self.deck.reset()
        assert len(self.deck.cards) == 24
    
    def test_card_dealing(self):
        """Test dealing cards to players."""
        hands = self.deck.deal_cards(4)
        
        assert len(hands) == 4
        for hand in hands:
            assert len(hand) == 5
            # Verify no duplicate cards
            assert len(set(hand)) == 5
```

### Parameterized Tests

#### Multiple Test Cases
```python
import pytest

@pytest.mark.parametrize("suit,rank,expected_string", [
    (Suit.HEARTS, Rank.ACE, "Ace of Hearts"),
    (Suit.DIAMONDS, Rank.KING, "King of Diamonds"),
    (Suit.CLUBS, Rank.QUEEN, "Queen of Clubs"),
    (Suit.SPADES, Rank.JACK, "Jack of Spades"),
])
def test_card_string_representation(suit, rank, expected_string):
    """Test card string representation."""
    card = Card(suit, rank)
    assert str(card) == expected_string

@pytest.mark.parametrize("num_players,expected_hands", [
    (2, 2),
    (3, 3),
    (4, 4),
])
def test_deck_dealing_players(num_players, expected_hands):
    """Test dealing to different numbers of players."""
    deck = Deck()
    hands = deck.deal_cards(num_players)
    assert len(hands) == expected_hands
```

## AI System Testing

### Testing Unified AI Interface

#### Test BaseAIInterface Implementation
```python
def test_ai_interface_implementation():
    """Test that AI classes implement the unified interface."""
    from euchre.ai.base_ai_interface import BaseAIInterface
    from euchre.ai.traditional_ai_impl import TraditionalAI
    from euchre.ai.level2_ai_impl import Level2AI
    from euchre.ai.level3_ai_impl import Level3AI
    
    # Test that all AI classes implement the interface
    assert issubclass(TraditionalAI, BaseAIInterface)
    assert issubclass(Level2AI, BaseAIInterface)
    assert issubclass(Level3AI, BaseAIInterface)
    
    # Test that required methods exist
    ai_classes = [TraditionalAI, Level2AI, Level3AI]
    required_methods = [
        'should_order_up', 'should_call_trump', 'select_trump_suit',
        'play_card', 'discard_card'
    ]
    
    for ai_class in ai_classes:
        for method_name in required_methods:
            assert hasattr(ai_class, method_name)
            method = getattr(ai_class, method_name)
            assert callable(method)
```

#### Test GameContext Usage
```python
def test_ai_game_context_usage():
    """Test that AI methods use GameContext correctly."""
    from euchre.ai.traditional_ai_impl import TraditionalAI
    from euchre.ai.game_context import GameContext
    from euchre.models import Card, Suit, Rank
    
    ai = TraditionalAI("Test", "balanced")
    
    # Create game context
    context = GameContext(
        hand=[Card(rank=Rank.ACE, suit=Suit.HEARTS)],
        position=0,
        is_dealer=False,
        flipped_card=Card(rank=Rank.JACK, suit=Suit.HEARTS),
        lead_suit=None,
        trump_suit=None,
        trick_history=[],
        team_scores=(0, 0),
        round_number=1
    )
    
    # Test that AI methods accept GameContext
    result = ai.should_order_up(context)
    assert hasattr(result, 'decision_type')
    assert hasattr(result, 'confidence')
    assert hasattr(result, 'reasoning')
```

#### Test DecisionResult Structure
```python
def test_ai_decision_result_structure():
    """Test that AI methods return DecisionResult objects."""
    from euchre.ai.traditional_ai_impl import TraditionalAI
    from euchre.ai.decision_result import DecisionResult, DecisionType
    from euchre.ai.game_context import GameContext
    from euchre.models import Card, Suit, Rank
    
    ai = TraditionalAI("Test", "balanced")
    
    # Create minimal game context
    context = GameContext(
        hand=[Card(rank=Rank.ACE, suit=Suit.HEARTS)],
        position=0,
        is_dealer=False,
        flipped_card=None,
        lead_suit=None,
        trump_suit=None,
        trick_history=[],
        team_scores=(0, 0),
        round_number=1
    )
    
    # Test all AI methods return DecisionResult
    methods_to_test = [
        'should_order_up', 'should_call_trump', 'select_trump_suit',
        'play_card', 'discard_card'
    ]
    
    for method_name in methods_to_test:
        method = getattr(ai, method_name)
        result = method(context)
        assert isinstance(result, DecisionResult)
        assert hasattr(result, 'decision_type')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reasoning')
        assert hasattr(result, 'metadata')
```

### Testing AI Levels

#### Test Level 1 AI (Traditional)
```python
def test_level1_ai_creation():
    """Test Level 1 AI creation and basic functionality."""
    from euchre.ai.traditional_ai_impl import TraditionalAI
    
    # Test different AI styles
    styles = ["aggressive", "conservative", "balanced", "opportunistic"]
    
    for style in styles:
        ai = TraditionalAI("Test", style)
        assert ai.name == "Test"
        assert ai.ai_style == style
        assert 0.0 <= ai.risk_ratio <= 1.0
        assert hasattr(ai, 'risk_profile')
```

#### Test Level 2 AI (Neural Network)
```python
def test_level2_ai_creation():
    """Test Level 2 AI creation and basic functionality."""
    from euchre.ai.level2_ai_impl import Level2AI
    
    # Test different AI types
    ai_types = ["strategic", "aggressive", "balanced", "intuitive"]
    
    for ai_type in ai_types:
        ai = Level2AI("Test", ai_type, 0.5)
        assert ai.name == "Test"
        assert ai.ai_type == ai_type
        assert 0.0 <= ai.risk_profile <= 1.0
```

#### Test Level 3 AI (Advanced Neural)
```python
def test_level3_ai_creation():
    """Test Level 3 AI creation and basic functionality."""
    from euchre.ai.level3_ai_impl import Level3AI
    
    # Test different AI types
    ai_types = ["strategic", "aggressive", "balanced", "conservative", "opportunistic"]
    
    for ai_type in ai_types:
        ai = Level3AI("Test", ai_type, 0.5)
        assert ai.name == "Test"
        assert ai.ai_type == ai_type
        assert 0.0 <= ai.risk_profile <= 1.0
```

### Testing AI Factory

#### Test AI Creation
```python
def test_ai_factory_creation():
    """Test AI factory creates correct AI types."""
    from euchre.ai.ai_factory import AIFactory
    
    # Test Level 1 AI creation
    level1_ai = AIFactory.create_ai_player("Test", "level1_balanced", 0.5)
    assert level1_ai.__class__.__name__ == "TraditionalAI"
    assert level1_ai.ai_style == "balanced"
    
    # Test Level 2 AI creation
    level2_ai = AIFactory.create_ai_player("Test", "level2_strategic", 0.5)
    assert level2_ai.__class__.__name__ == "Level2AI"
    assert level2_ai.ai_type == "strategic"
    
    # Test Level 3 AI creation
    level3_ai = AIFactory.create_ai_player("Test", "level3_balanced", 0.5)
    assert level3_ai.__class__.__name__ == "Level3AI"
    assert level3_ai.ai_type == "balanced"
```

## Test Quality

### Assertions

#### Basic Assertions
```python
def test_basic_assertions():
    """Test basic assertion types."""
    # Equality
    assert 2 + 2 == 4
    
    # Truthiness
    assert True
    assert not False
    
    # Containment
    assert "hello" in "hello world"
    assert 1 in [1, 2, 3]
    
    # Type checking
    assert isinstance("hello", str)
    assert isinstance(42, int)
```

#### Advanced Assertions
```python
def test_advanced_assertions():
    """Test advanced assertion types."""
    # Approximate equality
    assert 3.14159 == pytest.approx(3.14, rel=1e-2)
    
    # Exception testing
    with pytest.raises(ValueError):
        int("not a number")
    
    # Exception message testing
    with pytest.raises(ValueError, match="invalid literal"):
        int("abc")
    
    # Warning testing
    with pytest.warns(UserWarning):
        warnings.warn("test warning", UserWarning)
```

### Test Organization

#### Test Categories
```python
# Unit tests
def test_unit_functionality():
    """Test individual function in isolation."""
    pass

# Integration tests
@pytest.mark.integration
def test_integration_functionality():
    """Test multiple components working together."""
    pass

# Performance tests
@pytest.mark.slow
def test_performance():
    """Test performance characteristics."""
    pass

# Edge case tests
def test_edge_cases():
    """Test boundary conditions and edge cases."""
    pass

# AI-specific tests
@pytest.mark.ai
def test_ai_functionality():
    """Test AI system functionality."""
    pass
```

## Test Configuration

### pytest.ini Configuration
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    performance: marks tests as performance tests
    ai: marks tests as AI system tests
```

### conftest.py Configuration
```python
import pytest
from euchre.models import Player, PlayerType

@pytest.fixture(scope="session")
def sample_players():
    """Create sample players for testing."""
    return [
        Player("Alice", PlayerType.AI),
        Player("Bob", PlayerType.AI),
        Player("Charlie", PlayerType.AI),
        Player("David", PlayerType.AI)
    ]

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "ai: marks tests as AI system tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)
        if "ai" in item.keywords:
            item.add_marker(pytest.mark.ai)
```

## Continuous Integration

### GitHub Actions
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run tests
      run: |
        pytest --cov=euchre --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

### Local CI Simulation
```bash
# Run tests like CI
make test-ci

# Check code quality
make lint

# Run all checks
make check-all
```

## Best Practices

### Test Design
1. **Test One Thing**: Each test should verify one specific behavior
2. **Descriptive Names**: Test names should clearly describe what they test
3. **Arrange-Act-Assert**: Structure tests with setup, action, verification
4. **Independent Tests**: Tests should not depend on each other

### Test Maintenance
1. **Keep Tests Simple**: Avoid complex logic in tests
2. **Use Fixtures**: Reuse common test setup
3. **Update Tests**: Keep tests in sync with code changes
4. **Remove Dead Tests**: Delete tests for removed functionality

### Performance
1. **Fast Tests**: Keep unit tests fast (< 1 second each)
2. **Efficient Setup**: Use appropriate fixture scopes
3. **Parallel Execution**: Use pytest-xdist for parallel testing
4. **Mock External Dependencies**: Don't test external services

### AI Testing Specifics
1. **Test Interface Compliance**: Ensure all AI classes implement the unified interface
2. **Test GameContext Usage**: Verify AI methods use GameContext correctly
3. **Test DecisionResult Structure**: Check that AI methods return proper DecisionResult objects
4. **Test Risk Profile Integration**: Verify risk profiles affect AI behavior correctly

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Check virtual environment
source venv/bin/activate

# Install test dependencies
pip install pytest pytest-cov pytest-xdist

# Check Python path
python -c "import euchre; print(euchre.__file__)"
```

#### Test Failures
```bash
# Run with verbose output
pytest -v -s

# Show local variables on failure
pytest -l

# Run specific failing test
pytest tests/test_specific.py::test_failing_function -v -s
```

#### Coverage Issues
```bash
# Check coverage installation
pip install pytest-cov

# Run coverage with specific source
pytest --cov=euchre --cov-report=html

# Check coverage configuration
pytest --cov-config=.coveragerc
```

#### AI-Specific Issues
```bash
# Check AI module imports
python -c "from euchre.ai import TraditionalAI, Level2AI, Level3AI; print('AI imports OK')"

# Check PyTorch installation (for Level 2/3 AI)
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check CUDA availability (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Getting Help
```bash
# Check pytest help
pytest --help

# Check specific option help
pytest --help | grep -A 10 "coverage"

# Run pytest in verbose mode
pytest -v --tb=long
```

## Next Steps

### Improve Testing
1. **Add More Tests**: Increase test coverage
2. **Test Edge Cases**: Add boundary condition tests
3. **Performance Tests**: Add timing and performance tests
4. **Integration Tests**: Test component interactions

### Advanced Testing
1. **Property-Based Testing**: Use hypothesis for property-based tests
2. **Mutation Testing**: Use mutmut for mutation testing
3. **Fuzzing**: Add fuzz testing for robustness
4. **Load Testing**: Test with large datasets

### AI Testing Enhancements
1. **Model Validation**: Test neural network model outputs
2. **Performance Benchmarking**: Test AI decision-making speed
3. **Strategy Validation**: Test AI strategy effectiveness
4. **Cross-Level Testing**: Test AI level interactions

---

*For development setup, see [Development Setup](setup.md)*
*For code standards, see [Code Standards](standards.md)*
*For project structure, see [Project Structure](structure.md)* 