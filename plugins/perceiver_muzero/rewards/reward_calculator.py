"""Reward calculator for EucherPerceiverMuZero.

Enhanced with individual player performance tracking and decision-based penalties.
"""

from typing import Dict, Optional


class RewardCalculator:
    """Calculate rewards according to EucherPerceiverMuZero specification.

    Reward system that:
    - Tracks individual player trick performance (all players get reward per trick won)
    - Base team rewards for hand outcomes
    - Decision-based penalties ONLY for trump maker (not partners) when they make bad calls
    - Partners receive base team reward + individual trick rewards only
    - Trump maker receives base team reward + individual tricks + decision penalties (if bad call)
    """

    # Standard Euchre scoring rewards (base team rewards)
    NORMAL_WIN = 1.0
    SWEEP = 2.0
    LONE_SWEEP = 4.0
    EUCHRE = 1.0

    # Renege penalties - must be severe to prevent reneges early in training
    RENEGE_PENALTY = -10.0  # Strong penalty to ensure reneges fall out of training early

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

    def calculate_hand_reward(
        self,
        tricks_won: int,
        calling_team: int,
        player_team: int,
        is_alone: bool = False,
        renege_occurred: bool = False,
        renege_team: Optional[int] = None,
        renege_successful: bool = False,
        # Enhanced parameters
        player_id: Optional[int] = None,
        tricks_won_per_player: Optional[Dict[int, int]] = None,
        trump_maker_id: Optional[int] = None,
        screw_the_dealer: bool = False,
    ) -> float:
        """
        Calculate hand outcome reward with enhanced individual player tracking.

        Parameters
        ----------
        tricks_won : int
            Tricks won by calling team (0-5).
        calling_team : int
            Team that called trump (0 or 1).
        player_team : int
            Team of the player (0 or 1).
        is_alone : bool
            Whether the calling team went alone.
        renege_occurred : bool
            Whether a renege occurred in this hand.
        renege_team : Optional[int]
            Team that committed the renege, if any.
        renege_successful : bool
            Whether the renege was successful (deprecated - reneges are always penalized).
        player_id : Optional[int]
            ID of the player (0-3) for individual tracking.
        tricks_won_per_player : Optional[Dict[int, int]]
            Dictionary mapping player_id to tricks won by that player.
        trump_maker_id : Optional[int]
            ID of the player who made trump (ordered up or selected).
        screw_the_dealer : bool
            Whether this was a "screw the dealer" situation.

        Returns
        -------
        float
            Hand reward with individual performance adjustments.
        """
        is_caller = calling_team == player_team

        # Handle renege penalties - reneges should never be rewarded
        if renege_occurred:
            if renege_team == player_team:
                # This player's team reneged - apply severe penalty
                # Reneges should always be penalized, regardless of outcome
                return self.RENEGE_PENALTY
            else:
                # Opponent team reneged - we benefit from their penalty
                # The game rules handle this, but we don't get extra reward
                pass

        # Start with base team reward
        base_reward = 0.0

        # Calculate standard team rewards
        if is_caller:
            # Calling team
            if tricks_won >= 5:
                # Sweep
                if is_alone:
                    base_reward = self.LONE_SWEEP
                else:
                    base_reward = self.SWEEP
            elif tricks_won >= 3:
                # Normal win
                base_reward = self.NORMAL_WIN
            else:
                # Set (euchred) - negative reward
                base_reward = -self.EUCHRE
        else:
            # Defending team
            if tricks_won < 3:
                # Set the calling team (euchre)
                base_reward = self.EUCHRE
            elif tricks_won >= 5:
                # Allowed sweep - negative reward
                base_reward = -self.SWEEP
            else:
                # Normal loss
                base_reward = -self.NORMAL_WIN

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
        # All players get: base_reward + individual_reward (based on their tricks won)
        # Trump maker also gets: decision_penalty (if bad call)
        total_reward = base_reward + individual_reward + decision_penalty

        return total_reward

    def calculate_immediate_reward(
        self,
        action: str,
        outcome: str,
        renege_occurred: bool = False,
        renege_successful: bool = False,
    ) -> float:
        """
        Calculate immediate reward for action.

        Parameters
        ----------
        action : str
            Action taken.
        outcome : str
            Outcome description.
        renege_occurred : bool
            Whether a renege occurred.
        renege_successful : bool
            Whether the renege was successful (deprecated - reneges are always penalized).

        Returns
        -------
        float
            Immediate reward.
        """
        if renege_occurred:
            # Always penalize reneges - they should never be rewarded
            return self.RENEGE_PENALTY

        # Default immediate reward (can be extended for trick-level rewards)
        return 0.0

