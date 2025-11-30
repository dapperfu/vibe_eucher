# Euchre Player Plugin Development Guide

This guide explains how to create custom player plugins for the Euchre game. Plugins allow you to create standalone bot implementations that are automatically discovered and available for gameplay.

## Overview

The plugin system supports two types of plugins:

1. **Directory-based plugins**: Python files in the `plugins/` directory
2. **Entry point plugins**: Installable Python packages that register via entry points

Both types are automatically discovered and made available in the game.

## Plugin Interface

All plugins must implement a factory function that creates a `PlayerProfile` instance. The factory function signature is:

```python
def create_player(game: Optional[Game] = None, **kwargs) -> PlayerProfile:
    """
    Create a player profile instance.
    
    Parameters
    ----------
    game : Optional[Game]
        Game instance (required for some plugins that need game state).
    **kwargs
        Additional configuration (e.g., risk factors, model paths).
    
    Returns
    -------
    PlayerProfile
        A PlayerProfile instance that implements all required methods.
    """
    return YourPlayerProfile()
```

## Creating a Directory-Based Plugin

Create a Python file in the `plugins/` directory (e.g., `plugins/my_bot.py`):

```python
"""My custom Euchre bot."""

from typing import Optional

from eucher.players.base import PlayerProfile
from eucher.cards import Card, Suit
from eucher.plugins import register_plugin


class MyBotPlayer(PlayerProfile):
    """My custom bot implementation."""
    
    def decide_order_up(self, player, turned_card, dealer_id, trump_suit):
        # Your logic here
        return False
    
    def decide_call_trump(self, player, turned_card, trump_suit, must_choose=False):
        # Your logic here
        return None
    
    def choose_card_to_discard(self, player, turned_card=None, ordered_up_by=None):
        # Your logic here
        return player.hand[0]
    
    def play_card(self, player, led_suit, trump_suit, trick_cards, trick_player_ids):
        # Your logic here
        return player.hand[0]
    
    def decide_going_alone(self, player, trump_suit):
        # Your logic here
        return False
    
    def decide_trade_in(self, player, eligible_cards):
        # Your logic here
        return False


def create_player(game=None, **kwargs):
    """Factory function for MyBotPlayer."""
    return MyBotPlayer()


# Register the plugin
register_plugin(
    name="my_bot",
    factory=create_player,
    display_name="My Bot",
    description="A custom Euchre bot",
    requires_game=False,  # Set to True if you need game instance
    supports_kwargs=False,  # Set to True if you accept kwargs
)
```

## Creating an Entry Point Plugin

For a standalone Python package, register your plugin via entry points in `setup.py` or `pyproject.toml`:

### Using setup.py

```python
from setuptools import setup

setup(
    name="my-euchre-bot",
    version="1.0.0",
    # ... other metadata ...
    entry_points={
        "eucher.plugins": [
            "my_bot = my_bot.plugin:create_player",
        ],
    },
)
```

### Using pyproject.toml

```toml
[project.entry-points."eucher.plugins"]
my_bot = "my_bot.plugin:create_player"
```

The entry point should point to a factory function that matches the signature above.

## Plugin Registration Options

When registering a plugin, you can specify:

- **name**: Unique identifier for the plugin (used in player config)
- **factory**: Function that creates the player instance
- **display_name**: Human-readable name
- **description**: Description of the plugin
- **requires_game**: Whether the factory needs a game instance
- **supports_kwargs**: Whether the plugin accepts additional keyword arguments

## Using Plugins in Games

Once registered, plugins can be used in player configurations:

```python
from eucher.game import Game

# Create a game with your plugin
player_config = [
    ("Player 1", "my_bot"),
    ("Player 2", "heuristic"),
    ("Player 3", "random"),
    ("Player 4", "ai"),
]

game = Game(player_config)
```

## Plugin Requirements

All plugins must implement the `PlayerProfile` interface, which requires:

1. `decide_order_up(player, turned_card, dealer_id, trump_suit) -> bool`
2. `decide_call_trump(player, turned_card, trump_suit, must_choose=False) -> Optional[Suit]`
3. `choose_card_to_discard(player, turned_card=None, ordered_up_by=None) -> Card`
4. `play_card(player, led_suit, trump_suit, trick_cards, trick_player_ids) -> Card`
5. `decide_going_alone(player, trump_suit) -> bool`
6. `decide_trade_in(player, eligible_cards) -> bool`

See `eucher/players/base.py` for the full interface definition.

## Examples

### Simple Random Bot

```python
import random
from eucher.players.base import PlayerProfile
from eucher.cards import Card, Suit
from eucher.plugins import register_plugin


class RandomBot(PlayerProfile):
    def decide_order_up(self, player, turned_card, dealer_id, trump_suit):
        return random.choice([True, False])
    
    def decide_call_trump(self, player, turned_card, trump_suit, must_choose=False):
        if must_choose:
            suits = [s for s in Suit if s != turned_card.suit]
            return random.choice(suits)
        return random.choice([None] + [s for s in Suit if s != turned_card.suit])
    
    def choose_card_to_discard(self, player, turned_card=None, ordered_up_by=None):
        return random.choice(player.hand)
    
    def play_card(self, player, led_suit, trump_suit, trick_cards, trick_player_ids):
        # Get valid plays
        from eucher.rules import RulesEngine
        rules = RulesEngine()
        valid = rules.get_valid_plays(player.hand, led_suit, trump_suit)
        return random.choice(valid) if valid else player.hand[0]
    
    def decide_going_alone(self, player, trump_suit):
        return random.choice([True, False])
    
    def decide_trade_in(self, player, eligible_cards):
        return random.choice([True, False])


def create_player(game=None, **kwargs):
    return RandomBot()


register_plugin(
    name="random_bot",
    factory=create_player,
    display_name="Random Bot",
    description="A simple random decision bot",
)
```

### ML-Based Bot (Requires Game Instance)

```python
from typing import Optional
from eucher.players.base import PlayerProfile
from eucher.plugins import register_plugin


class MLBot(PlayerProfile):
    def __init__(self, game, model_path=None):
        self.game = game
        # Load your ML model here
        # ...
    
    # Implement all PlayerProfile methods using your ML model
    # ...


def create_player(game=None, **kwargs):
    if game is None:
        raise ValueError("MLBot requires game instance")
    return MLBot(game, model_path=kwargs.get("model_path"))


register_plugin(
    name="ml_bot",
    factory=create_player,
    display_name="ML Bot",
    description="Machine learning-based bot",
    requires_game=True,
    supports_kwargs=True,
)
```

## Testing Your Plugin

You can test your plugin by creating a simple game:

```python
from eucher.game import Game

# Test your plugin
player_config = [
    ("Test Player", "my_bot"),
    ("Opponent 1", "random"),
    ("Opponent 2", "random"),
    ("Opponent 3", "random"),
]

game = Game(player_config, seed=42)

# Play a hand
game.play_hand()
```

## Best Practices

1. **Error Handling**: Handle edge cases gracefully (empty hands, invalid states, etc.)
2. **Documentation**: Document your plugin's strategy and behavior
3. **Testing**: Test your plugin with various game scenarios
4. **Performance**: Consider performance implications for real-time gameplay
5. **Compatibility**: Ensure your plugin works with the current game version

## Troubleshooting

### Plugin Not Found

- Ensure your plugin file is in the `plugins/` directory
- Check that `register_plugin()` is called during module import
- Verify the plugin name matches what you're using in player config

### Import Errors

- Ensure all dependencies are installed
- Check that your plugin imports work correctly
- Verify that `PlayerProfile` interface is fully implemented

### Game Instance Required

- If your plugin needs game state, set `requires_game=True`
- The game instance will be passed to your factory function automatically

## Further Reading

- `eucher/players/base.py` - PlayerProfile interface definition
- `eucher/plugins/builtin/` - Examples of built-in plugins
- `eucher/game.py` - How plugins are used in game creation

