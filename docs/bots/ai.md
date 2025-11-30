# AI Bot

## Overview

The AI Bot is like the Heuristic Bot, but much smarter. Instead of simple yes/no rules, it uses a scoring system to evaluate situations.

## Architecture

- **Weighted Heuristics**: Uses scoring system for decisions
- **Position Awareness**: Considers dealer position and team dynamics
- **Strategic Play**: Adapts based on game situation

## Decision Making

### Ordering Up
- Calculates hand strength score for potential trump
- Considers position (dealer's partner gets bonus)
- Orders up if score exceeds threshold

### Calling Trump
- Evaluates hand strength for each potential trump suit
- Considers card power values (bowers worth 100/90, trump aces 80, etc.)
- Selects suit with highest score

### Playing Cards
- **Card Power Values**: Each card gets a power score
- **Team Play**: Plays low if teammate is winning, tries harder if opponent winning
- **Position Awareness**: Considers whether last player in trick

## Configuration

No configuration options available.

## Usage

```python
from eucher.game import Game

player_config = [
    ("AI1", "ai"),
    ("AI2", "ai"),
    ("AI3", "ai"),
    ("AI4", "ai"),
]

game = Game(player_config)
```

## Strengths

- Much more strategic than simple bot
- Considers team play and positioning
- Uses weighted decision-making
- Adapts based on game situation

## Weaknesses

- Still rule-based, not learning from experience
- Doesn't use machine learning
- May not adapt to opponent patterns

## Best For

- A challenging but predictable opponent
- Games where you want strategic play without randomness
- Testing intermediate-level strategies


