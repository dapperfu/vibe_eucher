"""Comprehensive reward model for Euchre RL training.

This module implements a reward system with:
- Bidding rewards (successful calls, sweeps, sets, correct passes)
- Discard rewards (optimal/harmful choices)
- Trick play rewards (winning tricks, strategic losses, drawing trump, wasting trump)
- Trick outcome rewards
- Hand outcome rewards
- Risk-based reward scaling
- Auxiliary intuition shaping rewards
- Total hand rewards capped to fixed range
"""

from typing import Dict, List, Optional, Tuple

from eucher.cards import Card, Suit


class RewardCalculator:
    """Calculates comprehensive rewards for RL training.

    Parameters
    ----------
    reward_cap : Tuple[float, float]
        Minimum and maximum reward cap for total hand rewards.
    risk_scaling : bool
        Whether to apply risk-based reward scaling.
    """

    def __init__(
        self, reward_cap: Tuple[float, float] = (-10.0, 10.0), risk_scaling: bool = True
    ) -> None:
        """Initialize reward calculator.

        Parameters
        ----------
        reward_cap : Tuple[float, float]
            Reward cap range (min, max).
        risk_scaling : bool
            Enable risk-based scaling.
        """
        self.reward_cap = reward_cap
        self.risk_scaling = risk_scaling

        # Reward weights
        self.bidding_reward_weight = 1.0
        self.discard_reward_weight = 0.5
        self.trick_play_reward_weight = 1.0
        self.trick_outcome_reward_weight = 1.0
        self.hand_outcome_reward_weight = 2.0
        self.auxiliary_reward_weight = 0.3

    def calculate_bidding_reward(
        self,
        action_type: str,
        action_value: bool,
        hand_won: bool,
        tricks_won: int,
        was_sweep: bool,
        was_set: bool,
        risk_factor: float = 0.5,
    ) -> float:
        """Calculate reward for bidding actions.

        Parameters
        ----------
        action_type : str
            Type of action: 'order_up', 'call_trump', or 'pass'.
        action_value : bool
            Whether action was taken (True) or passed (False).
        hand_won : bool
            Whether the hand was won.
        tricks_won : int
            Number of tricks won (0-5).
        was_sweep : bool
            Whether it was a sweep (won all 5 tricks).
        was_set : bool
            Whether opponents were set (won 0 tricks).
        risk_factor : float
            Risk factor for scaling (0.0-1.0).

        Returns
        -------
        float
            Bidding reward.
        """
        reward = 0.0

        if action_type == "order_up" or action_type == "call_trump":
            if action_value:  # Made a call
                if hand_won:
                    reward += 3.0  # Successful call
                    if was_sweep:
                        reward += 5.0  # Sweep bonus
                    if was_set:
                        reward += 3.0  # Set bonus
                else:
                    reward -= 2.0  # Failed call
            else:  # Passed
                # Reward correct passes (when hand would have been lost)
                if not hand_won:
                    reward += 1.0  # Correct pass
                else:
                    reward -= 0.5  # Missed opportunity

        elif action_type == "pass":
            if not action_value:  # Passed
                # Reward correct passes
                if not hand_won:
                    reward += 1.0
                else:
                    reward -= 0.5

        # Apply risk scaling
        if self.risk_scaling:
            reward *= (1.0 + risk_factor * 0.5)  # Scale by 1.0 to 1.5

        return reward * self.bidding_reward_weight

    def calculate_discard_reward(
        self,
        discarded_card: Card,
        hand: List[Card],
        trump_suit: Optional[Suit],
        hand_won: bool,
        tricks_won: int,
        risk_factor: float = 0.5,
    ) -> float:
        """Calculate reward for discard decisions.

        Parameters
        ----------
        discarded_card : Card
            Card that was discarded.
        hand : List[Card]
            Player's hand after discard.
        trump_suit : Optional[Suit]
            Current trump suit.
        hand_won : bool
            Whether the hand was won.
        tricks_won : int
            Number of tricks won.
        risk_factor : float
            Risk factor for scaling.

        Returns
        -------
        float
            Discard reward.
        """
        reward = 0.0

        # Evaluate discard quality
        # Good discards: low-value non-trump cards
        # Bad discards: high-value cards, trump cards

        if trump_suit is not None:
            is_trump = discarded_card._is_trump(trump_suit)
            if is_trump:
                # Discarding trump is usually bad
                reward -= 1.0
            else:
                # Discarding non-trump is usually good
                reward += 0.5

        # Consider card value
        card_value = discarded_card.rank.value
        if card_value >= 13:  # King or Ace
            reward -= 0.5  # Discarding high cards is usually bad
        elif card_value <= 10:  # 9 or 10
            reward += 0.3  # Discarding low cards is usually good

        # Outcome-based reward
        if hand_won:
            reward += 1.0  # Good discard if hand won
        else:
            reward -= 0.5  # Bad discard if hand lost

        # Apply risk scaling
        if self.risk_scaling:
            reward *= (1.0 + risk_factor * 0.3)

        return reward * self.discard_reward_weight

    def calculate_trick_play_reward(
        self,
        card_played: Card,
        trick_won: bool,
        was_leading: bool,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_number: int,
        was_strategic_loss: bool = False,
        drew_trump: bool = False,
        wasted_trump: bool = False,
        risk_factor: float = 0.5,
    ) -> float:
        """Calculate reward for trick play actions.

        Parameters
        ----------
        card_played : Card
            Card that was played.
        trick_won : bool
            Whether the trick was won.
        was_leading : bool
            Whether player was leading the trick.
        led_suit : Optional[Suit]
            Suit that was led.
        trump_suit : Optional[Suit]
            Current trump suit.
        trick_number : int
            Current trick number (0-4).
        was_strategic_loss : bool
            Whether this was a strategic loss (intentional).
        drew_trump : bool
            Whether this drew out opponent's trump.
        wasted_trump : bool
            Whether this wasted trump unnecessarily.
        risk_factor : float
            Risk factor for scaling.

        Returns
        -------
        float
            Trick play reward.
        """
        reward = 0.0

        # Base reward for winning trick
        if trick_won:
            reward += 1.0
        else:
            reward -= 0.3

        # Strategic loss bonus
        if was_strategic_loss:
            reward += 0.5

        # Trump-related rewards
        if trump_suit is not None:
            is_trump = card_played._is_trump(trump_suit)
            if drew_trump:
                reward += 1.0  # Drawing trump is good
            if wasted_trump:
                reward -= 1.0  # Wasting trump is bad

            # Leading with trump
            if was_leading and is_trump:
                if trick_number < 3:  # Early in hand
                    reward += 0.5  # Drawing trump early is good
                else:  # Late in hand
                    reward -= 0.3  # Wasting trump late is bad

        # Following suit
        if led_suit is not None and not was_leading:
            if card_played.suit == led_suit:
                reward += 0.2  # Following suit is good
            else:
                reward -= 0.3  # Not following suit when possible is bad

        # Apply risk scaling
        if self.risk_scaling:
            reward *= (1.0 + risk_factor * 0.4)

        return reward * self.trick_play_reward_weight

    def calculate_trick_outcome_reward(
        self, tricks_won: int, tricks_lost: int, target_tricks: int = 3
    ) -> float:
        """Calculate reward based on trick outcomes.

        Parameters
        ----------
        tricks_won : int
            Number of tricks won.
        tricks_lost : int
            Number of tricks lost.
        target_tricks : int
            Target number of tricks (default 3 for maker).

        Returns
        -------
        float
            Trick outcome reward.
        """
        reward = 0.0

        # Reward per trick won
        reward += tricks_won * 0.5

        # Penalty per trick lost
        reward -= tricks_lost * 0.3

        # Bonus for meeting target
        if tricks_won >= target_tricks:
            reward += 2.0

        return reward * self.trick_outcome_reward_weight

    def calculate_hand_outcome_reward(
        self,
        hand_won: bool,
        was_sweep: bool,
        was_set: bool,
        was_euchred: bool,
        team_score: int,
        opponent_score: int,
    ) -> float:
        """Calculate reward based on hand outcome.

        Parameters
        ----------
        hand_won : bool
            Whether the hand was won.
        was_sweep : bool
            Whether it was a sweep.
        was_set : bool
            Whether opponents were set.
        was_euchred : bool
            Whether team was euchred (lost hand they called).
        team_score : int
            Team's current score.
        opponent_score : int
            Opponent's current score.

        Returns
        -------
        float
            Hand outcome reward.
        """
        reward = 0.0

        if hand_won:
            reward += 5.0  # Base win reward
            if was_sweep:
                reward += 3.0  # Sweep bonus
            if was_set:
                reward += 2.0  # Set bonus
        else:
            reward -= 3.0  # Base loss penalty
            if was_euchred:
                reward -= 2.0  # Euchred penalty

        # Score differential bonus
        score_diff = team_score - opponent_score
        reward += score_diff * 0.1

        return reward * self.hand_outcome_reward_weight

    def calculate_auxiliary_reward(
        self,
        predicted_tricks: float,
        actual_tricks: int,
        predicted_win_prob: float,
        actual_won: bool,
        predicted_partner_trump: float,
        actual_partner_trump: Optional[int] = None,
    ) -> float:
        """Calculate reward for auxiliary predictions (intuition shaping).

        Parameters
        ----------
        predicted_tricks : float
            Predicted number of tricks.
        actual_tricks : int
            Actual number of tricks won.
        predicted_win_prob : float
            Predicted win probability.
        actual_won : bool
            Whether hand was actually won.
        predicted_partner_trump : float
            Predicted partner trump probability.
        actual_partner_trump : Optional[int]
            Actual partner trump count, if known.

        Returns
        -------
        float
            Auxiliary reward.
        """
        reward = 0.0

        # Reward accurate predictions
        trick_error = abs(predicted_tricks - actual_tricks)
        reward -= trick_error * 0.2  # Penalty for prediction error

        win_error = abs(predicted_win_prob - (1.0 if actual_won else 0.0))
        reward -= win_error * 0.3  # Penalty for win probability error

        if actual_partner_trump is not None:
            partner_error = abs(predicted_partner_trump - (actual_partner_trump / 6.0))
            reward -= partner_error * 0.2  # Penalty for partner trump error

        return reward * self.auxiliary_reward_weight

    def calculate_total_hand_reward(
        self,
        bidding_reward: float,
        discard_reward: float,
        trick_play_rewards: List[float],
        trick_outcome_reward: float,
        hand_outcome_reward: float,
        auxiliary_reward: float,
    ) -> float:
        """Calculate total hand reward with capping.

        Parameters
        ----------
        bidding_reward : float
            Bidding reward.
        discard_reward : float
            Discard reward.
        trick_play_rewards : List[float]
            List of trick play rewards.
        trick_outcome_reward : float
            Trick outcome reward.
        hand_outcome_reward : float
            Hand outcome reward.
        auxiliary_reward : float
            Auxiliary reward.

        Returns
        -------
        float
            Total hand reward (capped).
        """
        total_reward = (
            bidding_reward
            + discard_reward
            + sum(trick_play_rewards)
            + trick_outcome_reward
            + hand_outcome_reward
            + auxiliary_reward
        )

        # Apply cap
        total_reward = max(self.reward_cap[0], min(self.reward_cap[1], total_reward))

        return total_reward

    def calculate_step_reward(
        self,
        action_type: str,
        action_data: Dict,
        game_state: Dict,
        risk_factor: float = 0.5,
    ) -> float:
        """Calculate reward for a single step.

        Parameters
        ----------
        action_type : str
            Type of action: 'bidding', 'discard', 'play', 'trick_outcome', 'hand_outcome'.
        action_data : Dict
            Action-specific data.
        game_state : Dict
            Current game state.
        risk_factor : float
            Risk factor for scaling.

        Returns
        -------
        float
            Step reward.
        """
        if action_type == "bidding":
            return self.calculate_bidding_reward(
                action_type=action_data.get("action_type", "pass"),
                action_value=action_data.get("action_value", False),
                hand_won=action_data.get("hand_won", False),
                tricks_won=action_data.get("tricks_won", 0),
                was_sweep=action_data.get("was_sweep", False),
                was_set=action_data.get("was_set", False),
                risk_factor=risk_factor,
            )

        elif action_type == "discard":
            return self.calculate_discard_reward(
                discarded_card=action_data.get("discarded_card"),
                hand=action_data.get("hand", []),
                trump_suit=action_data.get("trump_suit"),
                hand_won=action_data.get("hand_won", False),
                tricks_won=action_data.get("tricks_won", 0),
                risk_factor=risk_factor,
            )

        elif action_type == "play":
            return self.calculate_trick_play_reward(
                card_played=action_data.get("card_played"),
                trick_won=action_data.get("trick_won", False),
                was_leading=action_data.get("was_leading", False),
                led_suit=action_data.get("led_suit"),
                trump_suit=action_data.get("trump_suit"),
                trick_number=action_data.get("trick_number", 0),
                was_strategic_loss=action_data.get("was_strategic_loss", False),
                drew_trump=action_data.get("drew_trump", False),
                wasted_trump=action_data.get("wasted_trump", False),
                risk_factor=risk_factor,
            )

        elif action_type == "trick_outcome":
            return self.calculate_trick_outcome_reward(
                tricks_won=action_data.get("tricks_won", 0),
                tricks_lost=action_data.get("tricks_lost", 0),
                target_tricks=action_data.get("target_tricks", 3),
            )

        elif action_type == "hand_outcome":
            return self.calculate_hand_outcome_reward(
                hand_won=action_data.get("hand_won", False),
                was_sweep=action_data.get("was_sweep", False),
                was_set=action_data.get("was_set", False),
                was_euchred=action_data.get("was_euchred", False),
                team_score=action_data.get("team_score", 0),
                opponent_score=action_data.get("opponent_score", 0),
            )

        elif action_type == "auxiliary":
            return self.calculate_auxiliary_reward(
                predicted_tricks=action_data.get("predicted_tricks", 0.0),
                actual_tricks=action_data.get("actual_tricks", 0),
                predicted_win_prob=action_data.get("predicted_win_prob", 0.0),
                actual_won=action_data.get("actual_won", False),
                predicted_partner_trump=action_data.get("predicted_partner_trump", 0.0),
                actual_partner_trump=action_data.get("actual_partner_trump"),
            )

        return 0.0


