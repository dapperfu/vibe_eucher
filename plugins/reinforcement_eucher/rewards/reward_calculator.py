"""Reward calculator for ReinforcementEucher.

Enhanced with individual player performance tracking and decision-based penalties.
"""

from typing import Dict, Optional

from ..config import ReinforcementEucherConfig


class RewardCalculator:
    """Calculate rewards according to ReinforcementEucher specification.

    Parameters
    ----------
    config : ReinforcementEucherConfig
        Configuration with reward values.
    """

    # Individual player trick rewards
    TRICK_WON_REWARD = 0.2  # Reward per trick won by this player

    # Decision-based penalties (only for trump maker, context-aware)
    BAD_CALL_PENALTY = -1.5  # Penalty for ordering up/selecting trump but getting euchred
    POOR_CALL_PENALTY = -0.8  # Penalty for ordering up but winning 0-1 tricks (partner bailed out)
    POOR_TRUMP_SELECTION_PENALTY = -0.6  # Penalty for selecting trump but winning 0-1 tricks

    # Context-aware penalty weights
    SCREW_DEALER_PENALTY_WEIGHT = 0.5  # Lower penalty for screw the dealer (dealer had to choose)
    GOING_ALONE_PENALTY_WEIGHT = 2.0  # Higher penalty for going alone failures (harder, should be penalized more)
    NORMAL_CALL_PENALTY_WEIGHT = 1.0  # Standard penalty weight

    def __init__(self, config: ReinforcementEucherConfig) -> None:
        """Initialize reward calculator.

        Parameters
        ----------
        config : ReinforcementEucherConfig
            Configuration with reward values.
        """
        self.config = config

    def calculate_trick_reward(self, won_trick: bool) -> float:
        """Calculate reward for trick outcome.

        Parameters
        ----------
        won_trick : bool
            Whether the trick was won.

        Returns
        -------
        float
            Trick reward.
        """
        if won_trick:
            return self.config.trick_reward_win
        else:
            return self.config.trick_reward_lose

    def calculate_hand_reward(
        self,
        hand_won: bool,
        was_lone_hand: bool = False,
        was_euchred: bool = False,
        # Enhanced parameters
        player_id: Optional[int] = None,
        tricks_won_per_player: Optional[Dict[int, int]] = None,
        trump_maker_id: Optional[int] = None,
        calling_team: Optional[int] = None,
        player_team: Optional[int] = None,
        tricks_won: Optional[int] = None,
        screw_the_dealer: bool = False,
    ) -> float:
        """Calculate reward for hand outcome with enhanced individual player tracking.

        Parameters
        ----------
        hand_won : bool
            Whether the hand was won.
        was_lone_hand : bool
            Whether this was a lone hand.
        was_euchred : bool
            Whether the team was euchred.
        player_id : Optional[int]
            ID of the player (0-3) for individual tracking.
        tricks_won_per_player : Optional[Dict[int, int]]
            Dictionary mapping player_id to tricks won by that player.
        trump_maker_id : Optional[int]
            ID of the player who made trump (ordered up or selected).
        calling_team : Optional[int]
            Team that called trump (0 or 1).
        player_team : Optional[int]
            Team of the player (0 or 1).
        tricks_won : Optional[int]
            Tricks won by calling team (0-5).
        screw_the_dealer : bool
            Whether this was a "screw the dealer" situation.

        Returns
        -------
        float
            Hand reward with individual performance adjustments.
        """
        # Base hand reward
        if hand_won:
            base_reward = self.config.hand_reward_win
            if was_lone_hand:
                base_reward += self.config.hand_reward_lone_success
        else:
            base_reward = self.config.hand_reward_lose
            if was_euchred:
                base_reward += self.config.hand_reward_euchred

        # Add individual player performance rewards (applies to all players)
        # Each player gets reward based on their own tricks won
        individual_reward = 0.0
        if player_id is not None and tricks_won_per_player is not None:
            player_tricks = tricks_won_per_player.get(player_id, 0)
            # Reward for individual trick performance (everyone gets this)
            individual_reward += player_tricks * self.TRICK_WON_REWARD

        # Decision-based penalties (ONLY for trump maker, not partners)
        decision_penalty = 0.0
        if (
            player_id is not None
            and trump_maker_id is not None
            and player_id == trump_maker_id
            and calling_team is not None
            and player_team is not None
            and tricks_won is not None
        ):
            # This player made the trump decision - they get penalties for bad calls
            is_caller = calling_team == player_team
            penalty_weight = self.NORMAL_CALL_PENALTY_WEIGHT

            # Adjust penalty weight based on context
            if screw_the_dealer:
                penalty_weight = self.SCREW_DEALER_PENALTY_WEIGHT
            elif was_lone_hand:
                penalty_weight = self.GOING_ALONE_PENALTY_WEIGHT

            if is_caller:
                if tricks_won < 3:
                    # Got euchred - penalty
                    decision_penalty = self.BAD_CALL_PENALTY * penalty_weight
                elif tricks_won >= 3:
                    # Additional penalty for poor individual performance despite team win
                    if tricks_won_per_player and tricks_won_per_player.get(player_id, 0) == 0:
                        # Ordered up/selected trump but won 0 tricks (partner bailed out)
                        decision_penalty = self.POOR_CALL_PENALTY * penalty_weight
                    elif tricks_won_per_player and tricks_won_per_player.get(player_id, 0) <= 1:
                        # Ordered up/selected trump but won very few tricks
                        decision_penalty = self.POOR_TRUMP_SELECTION_PENALTY * penalty_weight * 0.5

        # Combine all rewards
        total_reward = base_reward + individual_reward + decision_penalty

        return total_reward

    def calculate_discounted_return(
        self,
        rewards: list[float],
        gamma: Optional[float] = None,
    ) -> list[float]:
        """Calculate discounted returns from rewards.

        Parameters
        ----------
        rewards : list[float]
            List of rewards (in reverse order: most recent first).
        gamma : Optional[float]
            Discount factor. If None, uses config value.

        Returns
        -------
        list[float]
            Discounted returns (same order as input).
        """
        if gamma is None:
            gamma = self.config.gamma

        returns = []
        cumulative = 0.0
        for reward in reversed(rewards):
            cumulative = reward + gamma * cumulative
            returns.insert(0, cumulative)

        return returns

    def calculate_advantage(
        self,
        returns: list[float],
        values: list[float],
    ) -> list[float]:
        """Calculate advantages from returns and values.

        Parameters
        ----------
        returns : list[float]
            Discounted returns.
        values : list[float]
            Value estimates.

        Returns
        -------
        list[float]
            Advantages.
        """
        advantages = []
        for ret, val in zip(returns, values):
            advantages.append(ret - val)
        return advantages


