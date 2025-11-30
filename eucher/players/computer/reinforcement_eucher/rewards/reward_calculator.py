"""Reward calculator for ReinforcementEucher."""

from typing import Optional

from eucher.players.computer.reinforcement_eucher.config import ReinforcementEucherConfig


class RewardCalculator:
    """Calculate rewards according to ReinforcementEucher specification.

    Parameters
    ----------
    config : ReinforcementEucherConfig
        Configuration with reward values.
    """

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
    ) -> float:
        """Calculate reward for hand outcome.

        Parameters
        ----------
        hand_won : bool
            Whether the hand was won.
        was_lone_hand : bool
            Whether this was a lone hand.
        was_euchred : bool
            Whether the team was euchred.

        Returns
        -------
        float
            Hand reward.
        """
        if hand_won:
            reward = self.config.hand_reward_win
            if was_lone_hand:
                reward += self.config.hand_reward_lone_success
        else:
            reward = self.config.hand_reward_lose
            if was_euchred:
                reward += self.config.hand_reward_euchred

        return reward

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


