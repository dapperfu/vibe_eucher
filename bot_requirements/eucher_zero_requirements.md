# EucherZero Requirements Document

## Overview
EucherZero is a search enhanced reinforcement learning architecture inspired by AlphaZero and MuZero, adapted for Euchre. It combines representation learning, learned dynamics, policy and value prediction, risk tuned decision making, and Monte Carlo Tree Search. It supports perfect memory, hidden card inference, tactical look ahead, and self play training on GPU.

## Core Design Goals
- Learn optimal Euchre strategy through self play.
- Support hidden information via belief based dynamics.
- Perform forward search using MCTS.
- Support tunable risk profiles.
- Avoid forcing legal move masks; instead learn legality via negative rewards.
- Learn partner inference and opponent deduction.
- Train efficiently on GPU with batched MCTS evaluations.

## Architecture Components
### 1. Representation Network
Encodes full game state into a latent vector.
Inputs include:
- Player hand as one hot card vectors.
- Upcard representation.
- Discard status if dealer.
- Player seat and partner seat.
- Trump context and bidding stage.
- Score state.
- Trick history.
- Perfect memory card state.
- Deduction probability maps for unseen cards.
- Suit void inference.
- Trump count inference.
- Risk factor.
- No legal move masking.
Outputs:
- Latent state vector (size 256 to 512).

### 2. Dynamics Network
Predicts next latent state after an action and the immediate reward.
Inputs:
- Current latent state.
- Chosen action.
Outputs:
- Next latent state.
- Predicted immediate reward.

### 3. Prediction Network
Outputs policy and value signals for MCTS.
Inputs:
- Latent state.
Outputs:
- Policy over all possible actions (bids, discards, plays) without masking.
- Value estimate for expected outcome.
- Risk adjusted value estimate.

## Action Space
Actions are unified:
- Bidding phase: pass, order up, call one of four suits, stick the dealer.
- Discard phase: choose one of six cards.
- Trick phase: choose any of five cards (illegal options allowed).

## Reward Model
### Immediate Rewards
- Trick won by agent team: +1
- Trick lost: -1
- Drawing trump effectively: +1
- Burning trump uselessly: -1

### Bidding Rewards
- Successful call with 3 or 4 tricks: +4
- Successful sweep: +5
- Failed call: -4
- Correct pass: 0
- Incorrect pass: -1

### Discard Rewards
- Strong discard: +1
- Harmful discard: -1

### Illegal Play Penalties
- Renege (holding suit but not following): -5 immediate
- Trick lost due to renege: -2
- Partner harmed by renege: -3
- Hand penalty due to renege: -6

### Hand Outcome Rewards
- Calling team success: +3
- Calling team sweep: +4
- Calling team set: -4
- Defenders set caller: +3
- Defenders allow sweep: -1

### Risk Factor Scaling
Rewards and penalties are modified by:
- reward_scaled = reward * (1 + 0.5 * risk)
- penalty_scaled = penalty * (1 - 0.4 * risk)

## MCTS Algorithm
- Uses policy from Prediction Network as prior.
- Uses value and risk adjusted value during backpropagation.
- Samples hidden card assignments from deduction distributions.
- Averages rollouts across sampled hidden states.
- Expands branches based on combined policy and exploration constant.
- No legal move masking; illegal moves naturally suppressed via value.

## Deduction and Hidden Information
- Deduction maps provide probability of each unseen card being held by partner, left opponent, or right opponent.
- Dynamics Network learns transitions for hidden information.
- MCTS samples hidden card states consistent with deduction probabilities.
- Repeated sampling produces belief weighted search.

## Training Method
### Self Play
- All four players controlled by EucherZero.
- Each action guided by MCTS.
- Store (state, improved policy, final outcome) for training.

### Loss Function
- Policy loss: match network policy to MCTS improved policy.
- Value loss: predict final hand outcome.
- Risk value loss: predict risk modified value.
- Dynamics loss: predict next latent state.
- Reward loss: predict immediate reward.

### Curriculum
1. Basic legality training via penalties.
2. Trick strategy training with partial MCTS.
3. Add deduction learning.
4. Introduce bidding.
5. Full hand MCTS.
6. Multi agent self play at scale.

## GPU and Parallel Training Support
- Use PyTorch with GPU acceleration.
- Batched inference for representation, dynamics, and prediction networks.
- Parallel MCTS workers on CPU.
- Replay buffer with prioritized sampling.
- Distributed training optional.

## Evaluation Metrics
- Tricks per hand.
- Win rate vs previous versions.
- Sweep and set rates.
- Bidding accuracy.
- Legality violation frequency.
- Value prediction error.
- MCTS branching efficiency.
- Tournament win rate.

## Additional Enhancements
- Partner style embeddings.
- Opponent modeling head.
- Endgame exact solver integration.
- Risk personality training for multiple play styles.
- Optional GAN module for stylistic imitation.

## Summary
EucherZero is a full AlphaZero style implementation adapted for Euchre. Hidden information is handled through belief sampling inside MCTS. Legality is learned via harsh penalties. Risk tuning allows adjustable play styles. Self play produces superhuman strategy once trained at scale.

