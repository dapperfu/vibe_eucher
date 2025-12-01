"""Reward calculator for EucherZero.

Enhanced with individual player performance tracking and decision-based penalties.
"""

from typing import Dict, Optional


class RewardCalculator:
    """Calculate rewards according to EucherZero specification."""

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

    # Illegal play penalties - must be severe to prevent reneges early in training
    RENEGE_PENALTY = -10.0  # Strong penalty to ensure reneges fall out of training early
    RENEGE_TRICK_LOST = -2.0
    RENEGE_PARTNER_HARM = -3.0
    RENEGE_HAND_PENALTY = -10.0  # Strong penalty to ensure reneges fall out of training early

    # Hand outcome rewards
    CALLING_TEAM_SUCCESS = 3.0
    CALLING_TEAM_SWEEP = 4.0
    CALLING_TEAM_SET = -4.0
    DEFENDERS_SET_CALLER = 3.0
    DEFENDERS_ALLOW_SWEEP = -1.0

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
        # Enhanced parameters
        player_id: Optional[int] = None,
        tricks_won_per_player: Optional[Dict[int, int]] = None,
        trump_maker_id: Optional[int] = None,
        is_alone: bool = False,
        screw_the_dealer: bool = False,
    ) -> float:
        """
        Calculate hand outcome reward with enhanced individual player tracking.

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
        player_id : Optional[int]
            ID of the player (0-3) for individual tracking.
        tricks_won_per_player : Optional[Dict[int, int]]
            Dictionary mapping player_id to tricks won by that player.
        trump_maker_id : Optional[int]
            ID of the player who made trump (ordered up or selected).
        is_alone : bool
            Whether the calling team went alone.
        screw_the_dealer : bool
            Whether this was a "screw the dealer" situation.

        Returns
        -------
        float
            Hand reward with individual performance adjustments.
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

        # Start with base team reward
        base_reward = 0.0
        if is_caller:
            if tricks_won >= 5:
                base_reward = self.CALLING_TEAM_SWEEP
            elif tricks_won >= 3:
                base_reward = self.CALLING_TEAM_SUCCESS
            else:
                base_reward = self.CALLING_TEAM_SET
        else:
            if tricks_won < 3:
                base_reward = self.DEFENDERS_SET_CALLER
            elif tricks_won >= 5:
                base_reward = self.DEFENDERS_ALLOW_SWEEP
            else:
                base_reward = 0.0

        # Add individual player performance rewards (applies to all players)
        # Each player gets reward based on their own tricks won
        individual_reward = 0.0
        if player_id is not None and tricks_won_per_player is not None:
            player_tricks = tricks_won_per_player.get(player_id, 0)
            # Reward for individual trick performance (everyone gets this)
            individual_reward += player_tricks * self.TRICK_WON_REWARD

        # Decision-based penalties (ONLY for trump maker, not partners)
        decision_penalty = 0.0
        is_trump_maker = (player_id is not None and trump_maker_id is not None and player_id == trump_maker_id)
        
        if is_trump_maker and is_caller:
            # This player made the trump decision - they get penalties for bad calls
            penalty_weight = self.NORMAL_CALL_PENALTY_WEIGHT
            
            # Adjust penalty weight based on context
            if screw_the_dealer:
                penalty_weight = self.SCREW_DEALER_PENALTY_WEIGHT
            elif is_alone:
                penalty_weight = self.GOING_ALONE_PENALTY_WEIGHT
            
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

        # Apply risk scaling to the total reward
        return self.apply_risk_scaling(total_reward, risk_factor, is_penalty=(total_reward < 0))

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

