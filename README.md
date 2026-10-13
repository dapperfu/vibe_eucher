# Euchre Game

A Python implementation of the classic Euchre card game with AI opponents.

## Features

- 4-player Euchre game (Human vs AI or AI vs AI)
- Full rules enforcement
- Scoring to 10 points
- Text-based user interface

## Setup

```bash
make venv
make install
```

## Running

```bash
make run
```

`make run` asks for a profile at each seat. To pick personalities from the command line, use `eucher play` (installed by `make install`).

## Playing

You are seat 0. The other three seats are, in order, the opponent on your left, your partner, and the opponent on your right.

`--opponent-type` sets the opponents only. One type is used for both opponents. Two comma-separated types are the left-hand opponent and the right-hand opponent. That list does not set your partner.

When `--partner-type` is omitted, your partner is a machine-learning personality chosen at random (`euchergo`, `eucher_zero`, `perceiver_muzero`, `reinforcement_eucher`, `ml_pytorch`, and the other learned models; aliases of the same model count once). Pass `--partner-type` to choose the partner, including a rule-based personality.

Both opponents Heuristic, partner a random machine-learning personality:

```bash
venv_vibe_eucher/bin/eucher play --opponent-type heuristic
```

Left-hand opponent EucherGo, right-hand opponent EucherZero, partner a random machine-learning personality:

```bash
venv_vibe_eucher/bin/eucher play --opponent-type euchergo,eucher_zero
```

Play with Heuristic2 against Heuristic:

```bash
venv_vibe_eucher/bin/eucher play --opponent-type heuristic --partner-type heuristic2
```

### Personality vs personality

`scripts/tournaments/ai_tournament.py` plays two personalities as teams. The first name sits in seats 0 and 2. The second name sits in seats 1 and 3. `--num-games` is the number of games in that matchup (the default is 200).

Heuristic against Heuristic2, 1000 games:

```bash
venv_vibe_eucher/bin/python scripts/tournaments/ai_tournament.py \
  --types heuristic heuristic2 \
  --num-games 1000
```

While the games run, the script prints games completed, wins and win rate for each personality, total hands, and average hands per game. At the end it prints rankings (win rate, average score, average score against, score difference, games played) and the matchup record.

Any registered type name can replace `heuristic` or `heuristic2`. Pass more than two names to run every pairing.

### Untrained machine-learning personalities

`ml_sklearn`, `ml_gan`, `ml_rl`, `ml_pytorch` (and its aliases), `eucher_zero`, `perceiver_muzero`, `euchergo`, and `reinforcement_eucher` raise `UntrainedPersonalityError` when they are played without adequate training.

Adequate training means:

- `ml_sklearn`: fitted models for order-up, call-trump, discard, and card play
- `ml_gan`, `ml_rl`, and `ml_pytorch`: a saved weight file
- `eucher_zero`, `perceiver_muzero`, `euchergo`, and `reinforcement_eucher`: a checkpoint; if that file records games, tricks, or steps, the count has to be greater than zero

Heuristic personalities are rule-based and can play immediately. Train a machine-learning personality with the command in its section below before selecting it for a game or a matchup. A tournament prints the same error and skips that personality when the checkpoint is missing.

## Testing

```bash
make test
```

## Linting and Type Checking

```bash
make lint
make type-check
```

## AI/ML Personalities

Computer opponents are personalities. Each one is a player plugin, selected by its type name:

```bash
eucher play --opponent-type heuristic
```

That seats Heuristic as both opponents. Your partner is a random machine-learning personality unless you pass `--partner-type`.

Two names are the left-hand and right-hand opponents:

```bash
eucher play --opponent-type euchergo,eucher_zero
```

Names that share one implementation are listed together. Longer write-ups live under `docs/bots/`.

### Random

**Type name:** `random`

Random chooses uniformly among legal moves. Ordering up, calling trump, discarding, and playing a card are all untrained. It is the baseline for checking rules and comparing stronger personalities.

See [docs/bots/random.md](docs/bots/random.md).

### Heuristic

**Type name:** `heuristic`

Heuristic scores the hand with fixed weights: bowers and trump rank set card power, and order-up and call-trump decisions must clear a threshold. Leads prefer trump and high off-suit cards. Every choice comes from those weights.

See [docs/bots/heuristic.md](docs/bots/heuristic.md).

### Risk-Tuned Heuristic

**Type name:** `heuristic2`

The same rule family as Heuristic, with a `risk_factor` from 0.0 (conservative) to 1.0 (aggressive). Higher risk lowers the thresholds for ordering up and calling trump, and raises the chance of going alone. The default risk is 0.5.

### Advanced Heuristic

**Type name:** `heuristic3`

Advanced Heuristic keeps a card tracker of everything it has seen and estimates what remains. Trump calls use finer bonuses for bowers, trump length, and off-suit aces. Play shifts with the score: it loosens up when behind or near 10, tightens when ahead, and adds defensive value for stopping a march.

### Weighted Heuristic

**Type names:** `weighted_heuristic`, `ai`

Weighted Heuristic scores candidate cards with situation weights. It ducks when its partner is winning the trick, plays up when an opponent is winning, and treats the last seat differently from earlier seats. Trump selection uses the same power scale as the simpler heuristics (right bower highest, then left bower, then trump ace downward). `ai` is the older name for this personality.

See [docs/bots/weighted_heuristic.md](docs/bots/weighted_heuristic.md).

### Supervised ML

**Type name:** `ml_sklearn`

Supervised ML trains scikit-learn classifiers on recorded games. Separate models cover order-up, call-trump, discard, and card play. Supported model types are random forest, gradient boosting, and a neural network. At decision time the game state is encoded to features, the model emits a distribution, and a risk threshold picks the action. Playing it before all four models are fitted raises `UntrainedPersonalityError`.

See [docs/bots/ml_supervised.md](docs/bots/ml_supervised.md).

### DQN Reinforcement Learning

**Type name:** `ml_rl`

This personality learns with a Deep Q-Network and an experience replay buffer. It samples actions from Q-values, explores while `training_mode` is on, and updates from trick and game rewards. Trump and card-play risk factors gate how readily it commits to a predicted action. Playing it before `rl_model.pth` is saved raises `UntrainedPersonalityError`. A checkpoint that records zero games, tricks, or steps is treated as untrained.

See [docs/bots/ml_rl.md](docs/bots/ml_rl.md).

### GAN

**Type name:** `ml_gan`

GAN card play uses a generator that proposes a card distribution and a discriminator that scores whether that play looks sound. Both networks take the 260-dimensional game-state encoding. Playing it before `gan_model.pth` is saved raises `UntrainedPersonalityError`. Trump decisions still go through the shared ML player risk settings.

See [docs/bots/ml_gan.md](docs/bots/ml_gan.md).

### PyTorch Strategic

**Type names:** `ml_pytorch`, `pytorch_ai`, `pytorch_strategic`

PyTorch Strategic combines a deep network with explicit strategy. A feature encoder and a hybrid network score actions, a game-state tracker remembers cards and estimates bower locations, and strategic overrides can force bower-drawing plays. `exploration_epsilon` mixes in random legal moves. `pytorch_ai` and `pytorch_strategic` are aliases of `ml_pytorch`. Playing any of those names before a saved model exists raises `UntrainedPersonalityError`.

See [docs/bots/pytorch_strategic.md](docs/bots/pytorch_strategic.md).

### EucherZero

**Type name:** `eucher_zero`

EucherZero is an AlphaZero-style agent: a representation network, a learned dynamics model, and a prediction network for policy and value, searched with Monte Carlo tree search. Search samples beliefs about hidden cards. A risk factor scales values inside the search. It trains by self-play. Playing it with no checkpoint, or with a checkpoint that records zero games, raises `UntrainedPersonalityError`.

```bash
python scripts/eucher_zero/train_eucher_zero.py --duration 1h
```

See [docs/bots/eucher_zero.md](docs/bots/eucher_zero.md).

### Perceiver MuZero

**Type name:** `perceiver_muzero`

Perceiver MuZero encodes the position as tokens with a Perceiver-IO encoder, then plans with MuZero-style learned dynamics and MCTS. Simulations default to 128 and clamp between 1 and 800. `fast_mode` skips search and plays from the network alone. Risk is applied by temperature scaling, and training is multi-agent self-play. Playing it with no checkpoint raises `UntrainedPersonalityError`.

```bash
python scripts/perceiver_muzero/train_perceiver_muzero.py --duration 1h
```

See [docs/bots/perceiver_muzero.md](docs/bots/perceiver_muzero.md).

### EucherGo

**Type name:** `euchergo`

EucherGo pairs a shared trunk (policy head and value head) with PUCT Monte Carlo tree search. The value network supplies the rollouts. The state tensor includes the hand, trump, dealer and leader, the current trick, trick history, and a belief map over unseen cards. Action masking keeps the search on legal moves. It trains by self-play. Playing it with no checkpoint raises `UntrainedPersonalityError`.

```bash
python scripts/euchergo/train_euchergo.py --duration 1h
```

See [docs/bots/euchergo.md](docs/bots/euchergo.md).

### ReinforcementEucher

**Type name:** `reinforcement_eucher`

ReinforcementEucher learns from reward with a policy network and a value network (an MLP, or a Transformer when configured), trained with PPO. Training starts on isolated tricks, then moves to full hands so trump, going alone, and scoring come from the same reward signal. Temperature controls how sharply it samples the policy. Rewards cover tricks, hands, successful loners, and euchres. Playing it with no checkpoint, or with a checkpoint that records zero games and zero tricks, raises `UntrainedPersonalityError`.

```bash
python scripts/reinforcement_eucher/train_reinforcement_eucher.py --stage auto
```

See [docs/ReinforcementEucher_Requirements.md](docs/ReinforcementEucher_Requirements.md).

