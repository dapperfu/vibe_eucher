# ML GAN Bot

## Overview

The ML GAN Bot uses an advanced machine learning approach where two neural networks compete against each other - a generator that creates card plays and a discriminator that judges them.

## Architecture

- **Generative Adversarial Network**: Two competing neural networks
- **Generator**: Creates card play decisions
- **Discriminator**: Judges whether plays are good or bad
- **Adversarial Training**: Networks improve each other through competition

## Decision Making

- Generator creates card play decisions
- Discriminator evaluates decision quality
- Networks compete and improve over time
- Eventually learns to make realistic, strategic plays

## Training

**⚠️ IMPORTANT: The GAN backend REQUIRES training before use.**

```bash
python scripts/training/train_models.py --mode train_gan
```

Or use the example script:
```bash
python examples/train_gan_example.py
```

## Configuration

- `backend`: "gan"
- `trump_selection_risk`: Risk factor for trump decisions
- `gameplay_risk`: Risk factor for gameplay decisions

## Usage

```python
from eucher.game import Game

player_config = [
    ("GAN1", "ml_gan"),
    ("GAN2", "ml_gan"),
    ("GAN3", "ml_gan"),
    ("GAN4", "ml_gan"),
]

game = Game(player_config)
```

## Strengths

- Can learn complex patterns
- Generates creative strategies
- Good at mimicking expert play styles

## Weaknesses

- Very complex to train
- Requires significant computational resources
- May be unstable during training
- Primarily used for card play decisions, not trump selection

## Best For

- Advanced research
- Generating diverse play styles
- Complex strategic situations

