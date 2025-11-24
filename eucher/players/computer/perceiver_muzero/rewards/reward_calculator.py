"""Reward calculator for EuchrePerceiverMuZero."""

from typing import Optional


class RewardCalculator:
    """Calculate rewards according to EuchrePerceiverMuZero specification."""

    # Standard Euchre scoring rewards
    NORMAL_WIN = 1.0
    SWEEP = 2.0
    LONE_SWEEP = 4.0
    EUCHRE = 1.0

    # Renege penalties
    RENEGE_PENALTY = -4.0
    SUCCESSFUL_RENEGE_BONUS = 0.5  # Small positive reward for successful renege

    def calculate_hand_reward(
        self,
        tricks_won: int,
        calling_team: int,
        player_team: int,
        is_alone: bool = False,
        renege_occurred: bool = False,
        renege_team: Optional[int] = None,
        renege_successful: bool = False,
    ) -> float:
        """
        Calculate hand outcome reward according to EuchrePerceiverMuZero spec.

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
            Whether the renege was successful (not caught and improved outcome).

        Returns
        -------
        float
            Hand reward.
        """
        is_caller = calling_team == player_team

        # Handle renege penalties and rewards
        if renege_occurred:
            if renege_team == player_team:
                # This player's team reneged
                if renege_successful:
                    # Successful renege that improved outcome
                    return self.SUCCESSFUL_RENEGE_BONUS
                else:
                    # Caught renege - apply penalty
                    return self.RENEGE_PENALTY
            else:
                # Opponent team reneged - we benefit from their penalty
                # The game rules handle this, but we don't get extra reward
                pass

        # Calculate standard rewards
        if is_caller:
            # Calling team
            if tricks_won >= 5:
                # Sweep
                if is_alone:
                    return self.LONE_SWEEP
                else:
                    return self.SWEEP
            elif tricks_won >= 3:
                # Normal win
                return self.NORMAL_WIN
            else:
                # Set (euchred) - negative reward
                return -self.EUCHRE
        else:
            # Defending team
            if tricks_won < 3:
                # Set the calling team (euchre)
                return self.EUCHRE
            elif tricks_won >= 5:
                # Allowed sweep - negative reward
                return -self.SWEEP
            else:
                # Normal loss
                return -self.NORMAL_WIN

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
            Whether the renege was successful.

        Returns
        -------
        float
            Immediate reward.
        """
        if renege_occurred:
            if renege_successful:
                return self.SUCCESSFUL_RENEGE_BONUS
            else:
                return self.RENEGE_PENALTY

        # Default immediate reward (can be extended for trick-level rewards)
        return 0.0

