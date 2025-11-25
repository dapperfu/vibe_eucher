# ReinforcementEucher Requirements Document

A Pure Reinforcement Learning Euchre Engine\
Python 3.11+ and PyTorch Target

## 1. Overview

ReinforcementEucher is a Euchre playing agent trained entirely through
reinforcement learning. Every decision the agent makes is learned from
reward feedback without search trees or supervised data. The model
begins by training on isolated trick scenarios to rapidly develop card
selection skill, then transitions to full hand play to learn trump
calling, going alone, partner coordination, and scoring strategy.

Goals:

-   Fully emergent strategy without explicit rules programming.
-   Stable training on both micro decisions and macro decisions.
-   Modular architecture that can train trick policy and hand policy
    separately or jointly.
-   High performance GPU accelerated training.

## 2. Functional Requirements

### 2.1 Game Engine

The simulator must support:

-   Standard 24 card deck.
-   Shuffling, dealing, trick order, leader rotation.
-   Trump selection sequence: order up, turn down, naming a suit.
-   Lone hand logic.
-   Scoring: 1, 2, and 4 point hands.
-   Regional rule toggles.

The simulator must expose state transitions and reward hooks for both
trick level and hand level reinforcement.

## 3. Reinforcement Learning Design

ReinforcementEucher uses RL exclusively.

### 3.1 Per Trick RL

Each played trick produces:

-   State: hand, trick lead, legal moves, trump suit, unseen card
    estimates.
-   Action: selected card.
-   Reward: positive for winning the trick, negative for losing.
-   Optional shaping: reward proportional to trick importance under
    trump context.

### 3.2 Per Hand RL

Each completed hand produces:

-   State: full pre hand state, trump round state, trick sequences.
-   Actions: pass, order up, call suit, go alone, play cards.
-   Reward: game score change normalized to plus 1 or minus 1 per hand.

### 3.3 Combined RL

During full training:

-   Trick rewards accumulate into the hand reward.
-   The RL algorithm learns both short term and long term behavior.

## 4. Training Stages

### Stage 1 Trick Only Training

-   Create random trick scenarios with legal move options.
-   Train the card play policy to maximize immediate trick rewards.
-   Provides fast early learning.

### Stage 2 Full Hand Training

-   Enable the complete game simulator.
-   Reward at hand completion gives long term credit for strategic
    actions.
-   Final model integrates both stages.

## 5. RL Algorithm Requirements

ReinforcementEucher supports these RL modes.

### 5.1 Policy Gradient (recommended)

-   PPO or A2C.
-   Policy network outputs probability distribution over legal moves.
-   Value network predicts expected return.
-   Entropy bonus for exploration.

### 5.2 Q Learning Mode (optional)

-   DQN or Rainbow variant.
-   Q values for each legal action.
-   Legal move masking required.

Policy gradient is preferred for variable action spaces.

## 6. State Representation

The state tensor must include:

-   Player hand (binary 24).
-   Trump suit one hot.
-   Dealer index.
-   Leader index.
-   Played cards in current trick.
-   Trick history.
-   Estimated distribution of unseen cards for each opponent.
-   Flags: can follow suit, can order up, can go alone.

## 7. Neural Network Architecture

Two networks:

### 7.1 Policy Network

-   MLP or small Transformer.
-   Takes encoded state.
-   Outputs logits for all possible actions.
-   Legal mask applied before softmax.

### 7.2 Value Network

-   Predicts expected reward from state.
-   Shared trunk or separate weights.

Architecture components:

-   Residual MLP or Transformer blocks.
-   Layer norm.
-   GELU activation.

## 8. Reward Structure

### Trick level rewards

-   Win trick: +0.1
-   Lose trick: -0.1
-   Optional shaping penalties or bonuses.

### Hand level rewards

-   Win hand: +1
-   Lose hand: -1
-   Successful lone hand: +1 extra
-   Getting euchred: -1 extra

Rewards are discounted by gamma.

## 9. Training Pipeline

### 9.1 Self Play

-   All data generated via agent self competition.
-   Multiprocess environment workers.
-   Randomized seat and dealer.

### 9.2 Experience Buffer

-   Stores tuples: state, action, reward, advantage, legal mask.
-   Stores full hand returns for credit assignment.

### 9.3 Optimization

-   PPO updates every fixed batch size.
-   Gradient clipping and LR scheduling.
-   Optional curriculum versus stronger opponents.

## 10. Evaluation Tools

ReinforcementEucher must provide:

-   Win rate tracking.
-   Trick win rate.
-   Trump decision analysis.
-   Loner success percentage.
-   Baseline bot comparisons.

## 11. Performance Requirements

-   GPU based network training.
-   Vectorized state encoding.
-   Multiprocessing for rollout.
-   Trick only training mode for acceleration.

## 12. API Requirements

### Python API

    from reinforcementeucher import ReinforcementEucher
    bot = ReinforcementEucher.load("weights.pt")
    action = bot.act(state)

### CLI

-   reinforcementeucher train
-   reinforcementeucher selfplay
-   reinforcementeucher eval
-   reinforcementeucher play-human

## 13. Configuration

Config file options:

-   Network depth.
-   RL algorithm choice.
-   Reward shaping.
-   Trick only mode.
-   Full hand mode.
-   Logging detail.

## 14. Roadmap

Phase 1: Trick simulator and policy.\
Phase 2: Trick level PPO training.\
Phase 3: Full Euchre simulator.\
Phase 4: Hand level PPO training.\
Phase 5: Integrated model.\
Phase 6: Evaluation and release.
