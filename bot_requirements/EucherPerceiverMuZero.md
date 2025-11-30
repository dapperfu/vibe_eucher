# EucherPerceiverMuZero Requirements Specification
Version 1.1

## 1. Overview
EucherPerceiverMuZero is an AI engine designed to play advanced Midwest-style Euchre variants using a MuZero-style planning architecture with a Perceiver-IO encoder.

Supported variants:
- Standard Euchre
- Screw the dealer
- 9–10 trade-in
- Going alone
- Renege behavior learned (not hardcoded)
- Tunable risk profiles
- GPU-accelerated training
- Multi-agent self-play

The engine learns the game model implicitly rather than relying on explicit rule logic.

## 2. High-Level Architecture

```
State Encoder:        Perceiver-IO
Representation Net:   f(s) -> latent
Dynamics Net:         g(latent, action) -> latent', reward
Prediction Net:       h(latent) -> policy, value
Planner:              MCTS (MuZero-style)
Trainer:              Self-play RL pipeline
```

## 3. Game State Representation

### 3.1 Variant Enablement Flags
Only three gameplay variant flags:

```
enable_screw_the_dealer: bool
enable_nine_ten_tradein: bool
enable_go_alone: bool
```

### 3.2 Phase (Non-Redundant)
```
phase: enum[
    DEAL,
    PICKUP,
    CALL_TRUMP,
    PLAY,
    SCORE
]
```

### 3.3 Mode (Only Two)
```
mode: enum[
    NORMAL,
    ALONE
]
```

### 3.4 Information Tokens
Embedded tokens:
- Player hand
- Played cards
- Trick index
- Kitty (revealed)
- Dealer position
- Score
- Trump suit
- Partner/teammate alignment
- Belief models for unseen cards

Belief state includes:
- Suit-void inference
- Probability distribution over unseen cards
- Opponent modeling

## 4. Action Space

### 4.1 Bidding
- Pass
- Order up
- Call trump
- Go alone (if enabled)
- 9–10 trade-in (if enabled)
- Pickup (dealer)

### 4.2 Card Play
- Play any card in hand
- Illegal plays allowed; environment penalizes renege when caught

## 5. Reward Design

### 5.1 Scoring
- Standard Euchre scoring:
  - +1 point for normal win
  - +2 sweep
  - +4 lone sweep
  - +1 euchre

### 5.2 Variation Effects
Screw dealer and 9–10 trades modify expected values naturally through dynamics learning.

### 5.3 Renege Penalties
- -4 penalty if caught
- Small + reward if it succeeds and improves outcome

## 6. Perceiver-IO Encoder

### 6.1 Inputs
Token groups:
- Cards
- Trick history
- Bidding sequence
- Phase
- Mode
- Dealer
- Variant enablement flags
- Legal action mask

### 6.2 Latent Array
```
latent_size: 128
latent_slots: 32
```

### 6.3 Output
Fixed-size latent vector for MuZero representation.

## 7. MuZero Networks

### 7.1 Representation Network
```
latent = MLP(PerceiverOutput)
```

### 7.2 Dynamics Network
```
(next_latent, reward) = g(latent, action)
```

### 7.3 Prediction Network
Outputs:
- policy logits
- value estimate

## 8. MCTS Planner

### Parameters
- 128–800 simulations per move
- c_puct = 1.5
- Depth: 4–12
- Lone mode reduces branching factor

## 9. Training Pipeline

### 9.1 Self-Play Workers
- Parallel GPU workers generate full games

### 9.2 Replay Buffer
- Size 50k–200k
- Prioritized optional

### 9.3 Loss
```
L = value_loss + policy_loss + reward_loss + entropy_bonus
```

### 9.4 Optimizer
- AdamW
- lr = 1e-4
- weight_decay = 1e-3

## 10. Risk Modulation
```
risk_score = sigmoid(MLP(latent))
temp = base_temp * (1 + risk_score * RISK_FACTOR)
```

## 11. Evaluation
- Self-play Elo
- Variant stress tests
- Adversarial renege detection tests

## 12. Deployment
- Torch model export
- Lightweight API:
  ```
  /get_action
  /update_state
  /reset
  ```

