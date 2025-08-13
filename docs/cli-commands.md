# CLI Commands Reference

This document provides a complete reference for all available command-line interface commands in the Euchre game.

## Command Structure

All commands follow this pattern:
```bash
euchre [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

## Global Options

These options apply to all commands:

```bash
-v, --verbose        Enable verbose logging (INFO level)
-vv, --very-verbose  Enable very verbose logging (DEBUG level)
--version            Show version information
--help               Show help for the command
```

## Core Commands

### Play Command

Start a new euchre game.

```bash
euchre play [OPTIONS]
```

#### Options
- `-n, --player-name TEXT`: Your player name (omit for AI-only game)
- `-a, --ai-names TEXT`: Names for AI opponents (can specify multiple)

#### Examples

```bash
# Start a human vs AI game
euchre play --player-name "YourName"

# Use short options
euchre play -n "YourName"

# Customize AI opponent names
euchre play -n "Player1" -a "Sherlock" -a "Watson" -a "Moriarty"

# AI-only game (no human player)
euchre play
```

### AI vs AI Command

Run an AI vs AI euchre game.

```bash
euchre ai-vs-ai [OPTIONS]
```

#### Examples

```bash
# Basic AI vs AI game
euchre ai-vs-ai

# With verbose logging
euchre --verbose ai-vs-ai

# With very verbose logging
euchre --very-verbose ai-vs-ai
```

### AI Profiles Command

Run a game with different AI profiles and risk ratios.

```bash
euchre ai-profiles [OPTIONS]
```

#### Options
- `-p, --ai-profiles TEXT`: AI profiles (aggressive, conservative, balanced, opportunistic)
- `-r, --risk-ratios TEXT`: Risk ratios (0.0-1.0) for each AI player

#### Examples

```bash
# Default balanced profiles
euchre ai-profiles

# Custom profiles
euchre ai-profiles -p aggressive -p conservative -p balanced -p opportunistic

# Custom risk ratios
euchre ai-profiles -r 0.8 -r 0.3 -r 0.5 -r 0.7

# Mix of profiles and risk ratios
euchre ai-profiles -p aggressive -r 0.9 -p conservative -r 0.2
```

### Tournament Command

Run a tournament between trained players.

```bash
euchre tournament [OPTIONS]
```

#### Options
- `-n, --num-games INTEGER`: Number of games to play (default: 100)

#### Examples

```bash
# Default 100-game tournament
euchre tournament

# Custom number of games
euchre tournament --num-games 500

# Short form
euchre tournament -n 250
```

## Advanced Commands

### Logged Game Command

Run AI vs AI euchre game with logging (no ncurses interface).

```bash
euchre logged-game [OPTIONS]
```

#### Examples

```bash
# Basic logged game
euchre logged-game

# With verbose logging
euchre --verbose logged-game
```

### Ncurses Command

Run AI vs AI euchre game with ncurses interface.

```bash
euchre ncurses [OPTIONS]
```

#### Examples

```bash
# Start ncurses interface
euchre ncurses
```

## Training Commands

### Train Self-Play Command

Train AI models using self-play.

```bash
euchre train-self-play [OPTIONS]
```

#### Options
- `-n, --num-games INTEGER`: Number of games to play (default: 100000)
- `-p, --players TEXT`: Player names for training (can specify multiple)

#### Examples

```bash
# Default training
euchre train-self-play

# Custom number of games
euchre train-self-play --num-games 50000

# Custom player names
euchre train-self-play -p Alice -p Bob -p Charlie -p David

# Combined options
euchre train-self-play -n 75000 -p Alice -p Bob -p Charlie -p David
```

### Train Integer vs Float Command

Train integer models against float models.

```bash
euchre train-integer-vs-float [OPTIONS]
```

#### Options
- `-n, --num-games INTEGER`: Number of games to play (default: 200000)

#### Examples

```bash
# Default training
euchre train-integer-vs-float

# Custom number of games
euchre train-integer-vs-float --num-games 150000
```

### Generate Player Profiles Command

Generate AI player profiles with different characteristics.

```bash
euchre generate-player-profiles [OPTIONS]
```

#### Options
- `-n, --num-games INTEGER`: Number of games to play (default: 15000)
- `-o, --output-dir TEXT`: Output directory for profiles (default: trained_models)
- `-d, --device TEXT`: Device to use (cpu, cuda, auto) (default: auto)
- `--enable-amp`: Enable automatic mixed precision
- `--gpu-memory-fraction FLOAT`: GPU memory fraction to use (default: 0.8)

#### Examples

```bash
# Default generation
euchre generate-player-profiles

# Custom output directory
euchre generate-player-profiles --output-dir my_models

# CPU-only generation
euchre generate-player-profiles --device cpu

# GPU with custom settings
euchre generate-player-profiles --device cuda --enable-amp --gpu-memory-fraction 0.9

# Custom number of games
euchre generate-player-profiles -n 20000
```

## Analysis Commands

### Analyze Games Command

Analyze game results and generate statistics.

```bash
euchre analyze-games [OPTIONS]
```

#### Examples

```bash
# Basic analysis
euchre analyze-games
```

### Benchmark Command

Run performance benchmark comparing float vs integer models.

```bash
euchre benchmark [OPTIONS]
```

#### Options
- `-d, --device TEXT`: Device to use (cpu, cuda) (default: cpu)
- `--save-results`: Save benchmark results to file

#### Examples

```bash
# CPU benchmark
euchre benchmark --device cpu

# GPU benchmark with results saved
euchre benchmark --device cuda --save-results
```

### Run Mass Games Command

Run thousands of games in parallel for analysis.

```bash
euchre run-mass-games [OPTIONS]
```

#### Options
- `-n, --num-games INTEGER`: Number of games to run (default: 10000)

#### Examples

```bash
# Default mass games
euchre run-mass-games

# Custom number of games
euchre run-mass-games --num-games 25000
```

### Run Neural Games Command

Run neural network tournament between models.

```bash
euchre run-neural-games [OPTIONS]
```

#### Options
- `-m1, --model1 TEXT`: First model name (default: Alice)
- `-m2, --model2 TEXT`: Second model name (default: Bob)
- `-n, --num-games INTEGER`: Number of games to run (default: 1000)

#### Examples

```bash
# Default neural tournament
euchre run-neural-games

# Custom models
euchre run-neural-games -m1 Sherlock -m2 Watson

# Custom number of games
euchre run-neural-games -n 500

# Combined options
euchre run-neural-games -m1 Alice -m2 Bob -n 2000
```

## Utility Commands

### List Players Command

List available trained AI players.

```bash
euchre list-players [OPTIONS]
```

#### Examples

```bash
# List all players
euchre list-players
```

### List Neural Models Command

List available neural network models.

```bash
euchre list-neural-models [OPTIONS]
```

#### Examples

```bash
# List all models
euchre list-neural-models
```

### Play Trained Players Command

Play a game with trained AI players.

```bash
euchre play-trained-players [OPTIONS]
```

#### Options
- `-p1, --player1 TEXT`: First player name (default: Alice)
- `-p2, --player2 TEXT`: Second player name (default: Bob)
- `-p3, --player3 TEXT`: Third player name (default: Charlie)
- `-p4, --player4 TEXT`: Fourth player name (default: David)

#### Examples

```bash
# Default trained players
euchre play-trained-players

# Custom player selection
euchre play-trained-players -p1 Sherlock -p2 Watson -p3 Moriarty -p4 Irene
```

### Cleanup Games Command

Clean up old game result files.

```bash
euchre cleanup-games [OPTIONS]
```

#### Examples

```bash
# Clean up old games
euchre cleanup-games
```

## Development Commands

### Jupyter Command

Start Jupyter Notebook for development and analysis.

```bash
euchre jupyter [OPTIONS]
```

#### Examples

```bash
# Start Jupyter
euchre jupyter
```

### Jupyter Lab Command

Start Jupyter Lab for development and analysis.

```bash
euchre jupyter-lab [OPTIONS]
```

#### Examples

```bash
# Start Jupyter Lab
euchre jupyter-lab
```

## Command Categories

### For Players
- `play`: Start a new game
- `ai-vs-ai`: Watch AI games
- `ai-profiles`: Try different AI personalities

### For Developers
- `jupyter`: Interactive development
- `jupyter-lab`: Advanced development environment
- `list-players`: View available AI players

### For Researchers
- `train-self-play`: AI training
- `generate-player-profiles`: Create AI profiles
- `benchmark`: Performance testing
- `analyze-games`: Game analysis

### For Analysis
- `run-mass-games`: Large-scale simulation
- `run-neural-games`: Neural network tournaments
- `tournament`: Player competitions

## Help and Documentation

### Getting Help

```bash
# General help
euchre --help

# Command-specific help
euchre play --help
euchre ai-profiles --help
euchre train-self-play --help
```

### Version Information

```bash
# Show version
euchre --version
```

## Command Examples by Use Case

### Quick Start
```bash
# Play against AI
euchre play -n "YourName"

# Watch AI vs AI
euchre ai-vs-ai
```

### Training and Research
```bash
# Train AI models
euchre train-self-play -n 100000

# Generate profiles
euchre generate-player-profiles -n 20000 --device cuda

# Run benchmarks
euchre benchmark --device cpu --save-results
```

### Analysis and Testing
```bash
# Run tournaments
euchre tournament -n 500

# Mass game analysis
euchre run-mass-games -n 50000

# Neural network comparison
euchre run-neural-games -m1 Alice -m2 Bob -n 1000
```

### Development
```bash
# Interactive development
euchre jupyter

# Advanced development
euchre jupyter-lab

# List available resources
euchre list-players
euchre list-neural-models
```

## Troubleshooting Commands

### Verbose Logging
```bash
# Get detailed information
euchre --verbose play -n "YourName"

# Get debug information
euchre --very-verbose ai-vs-ai
```

### Validation Commands
```bash
# Test basic functionality
euchre --help

# Test command availability
euchre play --help
euchre ai-vs-ai --help
```

---

*For installation instructions, see [Installation Guide](installation.md)*
*For game rules, see [Game Rules](game-rules.md)*
*For AI documentation, see [AI Overview](ai/overview.md)* 