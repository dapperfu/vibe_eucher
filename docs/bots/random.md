# Random Bot

## Overview

The Random Bot is the simplest possible player. It makes every decision by randomly choosing from the available legal options.

## Architecture

- **Decision Making**: Random selection from legal moves
- **No Strategy**: No evaluation of game state
- **No Learning**: Decisions are purely random

## Decision Making

### Ordering Up
- Randomly chooses True or False

### Calling Trump
- Randomly chooses a suit (excluding turned card suit) or passes
- If must choose, randomly picks from available suits

### Playing Cards
- Randomly selects from valid plays (respects suit following rules)

### Discarding
- Randomly selects a card from hand

### Going Alone
- Randomly chooses True or False

## Configuration

No configuration options available.

## Usage

```python
from eucher.game import Game

player_config = [
    ("Random1", "random"),
    ("Random2", "random"),
    ("Random3", "random"),
    ("Random4", "random"),
]

game = Game(player_config)
```

## Strengths

- Completely unpredictable (can confuse opponents)
- Very fast decisions
- No computational overhead

## Weaknesses

- No strategy at all
- Makes terrible plays as often as good ones
- Easy to beat once understood

## Best For

- Testing game mechanics
- Creating chaos in games
- Baseline comparison for other bots




