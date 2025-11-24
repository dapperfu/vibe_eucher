"""Reward calculator for EuchreZero."""

from typing import Optional


class RewardCalculator:
    """Calculate rewards according to EuchreZero specification."""

    # Immediate rewards
    TRICK_WON = 1.0
    TRICK_LOST = -1.0
    DRAW_TRUMP_EFFECTIVE = 1.0
    BURN_TRUMP_USELESS = -1.0

    # Bidding rewards
    SUCCESSFUL_CALL = 4.0
    SUCCESSFUL_SWEEP = 5.0
    FAILED_CALL = -4.0
    CORRECT_PASS = 0.0
    INCORRECT_PASS = -1.0

    # Discard rewards
    STRONG_DISCARD = 1.0
    HARMFUL_DISCARD = -1.0

    # Illegal play penalties
    RENEGE_PENALTY = -5.0
    RENEGE_TRICK_LOST = -2.0
    RENEGE_PARTNER_HARM = -3.0
    RENEGE_HAND_PENALTY = -6.0

    # Hand outcome rewards
    CALLING_TEAM_SUCCESS = 3.0
    CALLING_TEAM_SWEEP = 4.0
    CALLING_TEAM_SET = -4.0
    DEFENDERS_SET_CALLER = 3.0
    DEFENDERS_ALLOW_SWEEP = -1.0

    def calculate_immediate_reward(
        self,
        action: str,
        outcome: str,
        risk_factor: float = 0.0,
    ) -> float:
        """
        Calculate immediate reward for action.

        Parameters
        ----------
        action : str
            Action taken.
        outcome : str
            Outcome description.
        risk_factor : float
            Risk factor for scaling.

        Returns
        -------
        float
            Immediate reward.
        """
        reward = 0.0

        if outcome == "trick_won":
            reward = self.TRICK_WON
        elif outcome == "trick_lost":
            reward = self.TRICK_LOST
        elif outcome == "draw_trump_effective":
            reward = self.DRAW_TRUMP_EFFECTIVE
        elif outcome == "burn_trump_useless":
            reward = self.BURN_TRUMP_USELESS
        elif outcome == "renege":
            reward = self.RENEGE_PENALTY

        return self.apply_risk_scaling(reward, risk_factor, is_penalty=(reward < 0))

    def calculate_hand_reward(
        self,
        tricks_won: int,
        calling_team: int,
        player_team: int,
        risk_factor: float = 0.0,
        renege_occurred: bool = False,
        renege_team: Optional[int] = None,
    ) -> float:
        """
        Calculate hand outcome reward.

        Parameters
        ----------
        tricks_won : int
            Tricks won by calling team.
        calling_team : int
            Team that called trump.
        player_team : int
            Team of the player.
        risk_factor : float
            Risk factor for scaling.
        renege_occurred : bool
            Whether a renege occurred in this hand.
        renege_team : Optional[int]
            Team that committed the renege, if any.

        Returns
        -------
        float
            Hand reward.
        """
        # Apply heavy penalty if this player's team reneged
        if renege_occurred and renege_team == player_team:
            # Massive penalty for reneging - worse than losing the hand
            reward = self.RENEGE_HAND_PENALTY
            return self.apply_risk_scaling(reward, risk_factor, is_penalty=True)
        
        # Apply penalty if opponent team reneged (but less severe)
        if renege_occurred and renege_team is not None and renege_team != player_team:
            # Opponent reneged - we benefit, but still apply some penalty to discourage reneges
            # The calling team loses if they renege, so defenders win
            if calling_team == renege_team:
                reward = self.DEFENDERS_SET_CALLER  # We set them due to their renege
            else:
                reward = 0.0  # Neutral - their renege doesn't directly help us
            return self.apply_risk_scaling(reward, risk_factor, is_penalty=(reward < 0))

        is_caller = calling_team == player_team

        if is_caller:
            if tricks_won >= 5:
                reward = self.CALLING_TEAM_SWEEP
            elif tricks_won >= 3:
                reward = self.CALLING_TEAM_SUCCESS
            else:
                reward = self.CALLING_TEAM_SET
        else:
            if tricks_won < 3:
                reward = self.DEFENDERS_SET_CALLER
            elif tricks_won >= 5:
                reward = self.DEFENDERS_ALLOW_SWEEP
            else:
                reward = 0.0

        return self.apply_risk_scaling(reward, risk_factor, is_penalty=(reward < 0))

    def apply_risk_scaling(
        self,
        reward: float,
        risk_factor: float,
        is_penalty: bool = False,
    ) -> float:
        """
        Apply risk factor scaling.

        Parameters
        ----------
        reward : float
            Base reward.
        risk_factor : float
            Risk factor (0.0-1.0).
        is_penalty : bool
            Whether this is a penalty.

        Returns
        -------
        float
            Scaled reward.
        """
        if is_penalty:
            return reward * (1.0 - 0.4 * risk_factor)
        else:
            return reward * (1.0 + 0.5 * risk_factor)

