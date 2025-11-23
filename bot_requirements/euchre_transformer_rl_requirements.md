# EuchreTransformerRL Requirements Document

## Overview
EuchreTransformerRL is a PyTorch based multi head reinforcement learning and supervised hybrid model for full stack Euchre gameplay. It performs bidding, trump selection, discard decisions, trick play, deduction, and risk tuned decision making with perfect memory.

## Architecture Goals
- Combine policy and value learning.
- Use full game state input including hand, deduction probabilities, trick history, trump context, and risk factor.
- Output actions for bidding, discarding, and playing cards.
- Support auxiliary predictions for training stability.
- Learn human like deductive reasoning about unseen cards.

## Inputs
- Hand representation as 5 one hot cards.
- Upcard features including card identity and potential bower information.
- Trump context and bidding stage.
- Player seat, partner position, and score.
- Perfect memory card states.
- Deduction probability maps for each player.
- Trick history for all previous tricks.
- Current trick context including legal play mask.
- Risk factor from 0.0 to 1.0.
- Auxiliary features such as suit void probabilities, trump counts, and dominance ranking.

## Outputs
- Bidding action distribution.
- Discard action distribution.
- Play action distribution masked for legality.
- Auxiliary heads including expected tricks, win probability, and partner trump probability.

## Reward Model
- Bidding rewards for successful calls, sweeps, sets, and correct passes.
- Discard rewards for optimal or harmful choices.
- Trick play rewards for winning tricks, strategic losses, drawing trump, or wasting trump.
- Trick outcome rewards per trick won or lost.
- Hand outcome rewards for team success or failure.
- Risk based reward scaling.
- Auxiliary intuition shaping rewards.
- Total hand rewards capped to a fixed range.

## Training Data Structure
- State, action, reward, next state, done.
- Sequence based episodes for each hand.
- Multi phase action labeling for bidding, discard, and play.

## Training Curriculum
- Stage 1: Learn card legality and basic play.
- Stage 2: Learn trick strategy using self play.
- Stage 3: Add full deduction logic and perfect memory.
- Stage 4: Introduce bidding and risk tuning.
- Stage 5: Full hand reinforcement learning.
- Stage 6: Multi agent self play with varying risk.
- Stage 7: Evaluation against fixed bots.

## Simulation Requirements
- Euchre deck representation.
- Dealer, shuffle, order up round, calling round, stick the dealer.
- Trick play rules including follow suit and bower mapping.
- Team scoring and full hand resolution.
- Accurate memory and deduction state.

## Evaluation Metrics
- Average tricks per hand.
- Hands won per thousand.
- Sweep and set rates.
- Bidding accuracy.
- Play accuracy.
- Stability of loss curves.
- Tournament win rate.

## Additional Enhancements
- Opponent style embeddings.
- Partner consistency modeling.
- Monte Carlo auxiliary prediction.
- Endgame deterministic solver for last two tricks.

