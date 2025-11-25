# EucherGo Technical Requirements Document

A Hybrid Monte Carlo Tree Search and Deep Neural Network Engine for
Euchre Target Framework: PyTorch Programming Language: Python 3.11+
Author: System Specification Draft

## 1. Overview

EucherGo is a competitive Euchre engine that learns optimal play using
reinforcement learning, self play, a policy network, a value network,
and a Monte Carlo Tree Search loop, patterned after the Go engine
architecture used by DeepMind. The system simulates thousands of games,
evaluates states, and learns both the probability distribution of strong
actions and the expected value of a position.

Objectives:

-   Strong tactical and strategic Euchre play.
-   Ability to reason about unknown cards and partner tendencies.
-   State evaluation that balances trick taking, trump dynamics, and
    risk.
-   Support for the full Euchre rule set including dealer ordering,
    turning down trump, going alone, and scoring.

## 2. Functional Requirements

### 2.1 Game Model

The simulation engine must support:

-   Full deck logic (24 card deck).
-   Shuffling, dealing, turn order, trick resolution.
-   Trump selection: order up, turn down, naming a suit.
-   Special actions: going alone, stick the dealer rule guard.
-   Scoring mechanics: 1 point, 2 points, 4 point loner potential.
-   Optional regional rules configurable by flags but default must match
    common Midwest rules.

## 3. System Architecture

### 3.1 High Level Flow

1.  Generate self play matches.
2.  For each move, perform a Monte Carlo Tree Search guided by the
    policy and value networks.
3.  Store state, action probabilities, game result tuples.
4.  Train networks with cross entropy for policy and MSE for value.
5.  Repeat for thousands of iterations until convergence.

## 4. State Representation

The model must encode:

-   Player hand (binary 24 vector).
-   Trump suit (one hot).
-   Dealer index and leader index.
-   Played cards in current trick.
-   History of tricks taken.
-   Probabilities of unknown cards using a belief map (24 vector per
    player).
-   Flags: can follow suit, must call trump, can declare lone hand, etc.

The combined tensor should be fixed length and optimized for GPU
batching.

## 5. Network Architecture

### 5.1 Policy and Value Networks

EucherGo uses a shared trunk with two heads.

Trunk:

-   Residual Conv1D or MLP based stack.
-   Layer norm.
-   Nonlinear activations.
-   Depth chosen by configuration.

Policy head:

-   Outputs action logits for:
    -   Calling or passing in trump selection.
    -   Declaring lone hand.
    -   Playing cards from hand.
-   Softmax over legal moves.

Value head:

-   Outputs a scalar that predicts the chance of victory.

### 5.2 Action Masking

The policy head must accept per state legal move masks to zero out
illegal actions before softmax.

## 6. Monte Carlo Tree Search

MCTS uses:

-   Upper Confidence Bound formula using PUCT.
-   Expanded nodes store prior probability, visit count, cumulative
    value.
-   Value network for rollouts.
-   Temperature control for exploration.

Requirements:

-   Configurable simulations per move.
-   Parallel batch GPU evaluation.

## 7. Training Pipeline

### 7.1 Self Play Generator

-   Runs many GPU enabled games.
-   Saves training samples after each MCTS decision.

### 7.2 Replay Buffer

-   Stores the most recent training games.
-   Uniform sampling.

### 7.3 Optimization Loop

-   Policy loss: cross entropy.
-   Value loss: MSE.
-   Regularization.
-   Evaluation against older snapshots.

## 8. Belief Modeling of Unknown Cards

-   Track probability distribution of unseen cards in opponents hands.
-   Update after each trick using card constraints.
-   Feed belief map into state tensor.

## 9. Strategy Modules

-   Loner estimation.
-   Trick expectation.
-   Probability of euchre.
-   Bluff potential.
-   Partner inference.

## 10. Evaluation Suite

-   Mirror matches.
-   Static rule based bots.
-   Checkpoint matches.

Metrics:

-   Win rate.
-   Points per hand.
-   Euchre frequency.
-   Loner success rate.

## 11. Performance Requirements

-   Full GPU acceleration.
-   Vectorized neural calls.
-   Multiprocessing for self play.

Target:

-   10000 games of self play per training hour on a modern GPU.

## 12. Implementation Details

-   Ruff lint.
-   Python 3.11.
-   PyTorch 2.x.

Directory structure:

    euchergo/
      game/
      mcts/
      model/
      training/
      evaluation/
      utils/

## 13. Interfaces and API

### 13.1 Python Interface

    from euchergo import EucherGoEngine
    engine = EucherGoEngine.load("model.pt")
    move = engine.choose_action(state)

### 13.2 CLI

-   euchergo selfplay
-   euchergo train
-   euchergo eval
-   euchergo play-human

## 14. Configuration

A simple TOML or YAML file defines training parameters and rule
variants.

## 15. Roadmap

Phase 1. Rule engine. Phase 2. Networks. Phase 3. MCTS. Phase 4.
Training loop. Phase 5. Evaluation tools. Phase 6. Public release.
