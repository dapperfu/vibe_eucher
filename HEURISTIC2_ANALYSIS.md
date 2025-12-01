# Heuristic2 vs Heuristic: Analysis of Performance Issues

## Executive Summary

Heuristic2 loses 48.4% of games vs Heuristic (51.6% win rate). The primary issues stem from:

1. **Missing context-aware card evaluation when leading** (CRITICAL)
2. **Overly aggressive trump selection thresholds** (MODERATE)
3. **No strategic bonuses for leading with trump or strong off-suit cards** (CRITICAL)

## Detailed Analysis

### 1. Leading Strategy - Missing Context Multipliers (CRITICAL ISSUE)

**Heuristic Implementation:**
```python
# In _decide_lead()
card_powers = [
    (card, self._calculate_card_power(card, trump_suit, context="leading"))
    for card in valid_cards
]
```

**Heuristic2 Implementation:**
```python
# In _decide_lead()
card_powers = [
    (card, self._calculate_card_power(card, trump_suit))  # NO context parameter!
    for card in valid_cards
]
```

**Impact:**
- Heuristic applies multipliers when leading:
  - `LEAD_TRUMP_PREFERENCE = 1.2` (20% bonus for trump cards)
  - `LEAD_OFFSUIT_ACE = 1.1` (10% bonus for off-suit Aces)
  - `LEAD_OFFSUIT_KING = 1.05` (5% bonus for off-suit Kings)
- Heuristic2 does NOT apply these multipliers, undervaluing:
  - Trump cards when leading (should get 20% bonus)
  - Off-suit Aces when leading (should get 10% bonus)
  - Off-suit Kings when leading (should get 5% bonus)

**Example:**
- Leading with Right Bower (trump): Heuristic values at 120.0 (100 * 1.2), Heuristic2 at 100.0
- Leading with off-suit Ace: Heuristic values at 38.5 (35 * 1.1), Heuristic2 at 35.0
- This causes Heuristic2 to make suboptimal lead choices, potentially leading with weaker cards

### 2. Overly Aggressive Trump Selection (MODERATE ISSUE)

**Order Up Threshold:**
- **Heuristic**: Fixed 150.0 (130.0 if dealer's partner)
- **Heuristic2**: 127.5 (107.5 if dealer's partner) at risk_factor=0.5
  - Calculation: `150.0 - (0.5 * 0.30 * 150.0) = 127.5`
  - 15% reduction makes Heuristic2 order up with weaker hands

**Call Trump Threshold:**
- **Heuristic**: Fixed 140.0 (80.0 if must choose)
- **Heuristic2**: 119.0 (70.0 if must choose) at risk_factor=0.5
  - Calculation: `140.0 - (0.5 * 0.30 * 140.0) = 119.0`
  - Further reduced by `0.5 * 20.0 = 10.0` when must_choose=True

**Impact:**
- Heuristic2 orders up/calls trump with weaker hands more frequently
- This leads to more games where Heuristic2 is the maker with a marginal hand
- Opponents can more easily euchre (defeat) Heuristic2's weaker trump calls

### 3. Card Power Calculation Differences

**Heuristic:**
```python
def _calculate_card_power(
    self, card: Card, trump_suit: Optional[Suit], context: str = "general"
) -> float:
    # ... base power calculation ...
    
    # Apply context-based adjustments
    if context == "leading":
        if card.suit == trump_suit or is_right_bower or is_left_bower:
            power *= HeuristicWeights.LEAD_TRUMP_PREFERENCE  # 1.2x
        elif card.rank == Rank.ACE:
            power *= HeuristicWeights.LEAD_OFFSUIT_ACE  # 1.1x
        elif card.rank == Rank.KING:
            power *= HeuristicWeights.LEAD_OFFSUIT_KING  # 1.05x
```

**Heuristic2:**
```python
def _calculate_card_power(self, card: Card, trump_suit: Optional[Suit]) -> float:
    # ... base power calculation ...
    # NO context parameter, NO multipliers applied
```

### 4. Leading Trump Strategy

**Heuristic:**
- Leads with trump if has 2+ trump cards
- Applies 1.2x multiplier to trump cards when evaluating

**Heuristic2:**
- Leads with trump if has 2+ trump cards (at risk_factor < 0.6) OR 1+ trump (at risk_factor >= 0.6)
- At default risk_factor=0.5, requires 2+ trump (same as Heuristic)
- BUT: Does not apply trump preference multiplier, so trump cards are undervalued

### 5. Following/Last Play Strategy

**Both implementations:**
- Duck when teammate is winning (at default risk_factor=0.5)
- Try to win with lowest winning card when opponent is winning
- No significant difference here

## Root Cause Summary

The primary issue is **Heuristic2's failure to apply context-aware multipliers when leading**. This causes:

1. **Suboptimal lead selection**: Heuristic2 undervalues trump cards and strong off-suit cards when deciding what to lead
2. **Weaker opening plays**: Leading with weaker cards gives opponents better opportunities
3. **Lost trick control**: Not leading with trump when appropriate reduces control over trick outcomes

The secondary issue is **overly aggressive trump selection**, which leads to:
1. More marginal trump calls
2. Higher euchre rate against Heuristic2
3. Lower average scores when Heuristic2 is the maker

## Recommendations

1. **Fix leading strategy** (HIGH PRIORITY):
   - Add `context` parameter to `_calculate_card_power()` in Heuristic2
   - Apply same multipliers as Heuristic when `context="leading"`
   - Update `_decide_lead()` to pass `context="leading"`

2. **Adjust risk factor or thresholds** (MEDIUM PRIORITY):
   - Consider lowering default risk_factor from 0.5 to 0.3-0.4
   - OR increase base thresholds to compensate for risk adjustments
   - OR make risk adjustments smaller (reduce THRESHOLD_REDUCTION_MAX)

3. **Test improvements**:
   - Run tournament with fixed Heuristic2
   - Compare win rates and average scores
   - Verify improvements in leading strategy effectiveness

