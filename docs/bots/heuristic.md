# Heuristic Bot

## Overview

The Heuristic Bot follows basic rules that a beginner Euchre player might use. It's like having a checklist of simple decision rules.

## Architecture

- **Rule-Based**: Uses simple if-then rules
- **No Learning**: Decisions are deterministic based on hand evaluation
- **Basic Strategy**: Considers hand strength and basic Euchre principles

## Decision Making

### Ordering Up
- Orders up if has 2+ trump cards OR has a bower (Right/Left Bower)

### Calling Trump
- Picks suit with most cards (at least 2) OR suit with a bower

### Discarding
- Throws away lowest-value card

### Playing Cards
- **Leading**: Plays highest card
- **Following**: Tries to win with lowest card that can win, or plays lowest if can't win

## Configuration

No configuration options available.

## Usage

```python
from eucher.game import Game

player_config = [
    ("Heuristic1", "heuristic"),
    ("Heuristic2", "heuristic"),
    ("Heuristic3", "heuristic"),
    ("Heuristic4", "heuristic"),
]

game = Game(player_config)
```

## Strengths

- Makes logical, predictable decisions
- Won't make obviously stupid plays
- Fast and consistent

## Weaknesses

- Too simple - doesn't consider game context
- Doesn't think about what opponents might have
- Doesn't adapt strategy based on score or game situation

## Best For

- Learning the basic rules of Euchre
- Playing against a predictable opponent
- Testing your own strategies


