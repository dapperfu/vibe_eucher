"""Self-play game generation for EucherPerceiverMuZero training."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from eucher.game import Game

from ..action_space import ActionEncoder
from ..config import PerceiverMuZeroConfig
from ..mcts.search import MCTSSearch
from ..networks.model import PerceiverMuZeroModel
from ..rewards.reward_calculator import RewardCalculator
from ..state_encoder import StateEncoder


class GameStep:
    """Training example from a game step."""

    def __init__(
        self,
        input_tokens: torch.Tensor,
        action: int,
        policy: np.ndarray,
        value: float,
        reward: float,
        next_input_tokens: Optional[torch.Tensor],
        player_id: int,
    ) -> None:
        """
        Initialize game step.

        Parameters
        ----------
        input_tokens : torch.Tensor
            Input token sequence.
        action : int
            Action taken.
        policy : np.ndarray
            Improved policy from MCTS.
        value : float
            Final value (hand outcome).
        reward : float
            Immediate reward.
        next_input_tokens : Optional[torch.Tensor]
            Next state tokens.
        player_id : int
            Player ID.
        """
        self.input_tokens = input_tokens
        self.action = action
        self.policy = policy
        self.value = value
        self.reward = reward
        self.next_input_tokens = next_input_tokens
        self.player_id = player_id


def generate_self_play_game(
    model: PerceiverMuZeroModel,
    config: PerceiverMuZeroConfig,
    game: Optional[Game] = None,
    risk_factor: float = 0.0,
) -> List[GameStep]:
    """
    Generate training examples from a self-play game.

    Parameters
    ----------
    model : PerceiverMuZeroModel
        PerceiverMuZero model.
    config : PerceiverMuZeroConfig
        Configuration.
    game : Optional[Game]
        Game instance (if None, creates new game).
    risk_factor : float
        Risk factor for decisions.

    Returns
    -------
    List[GameStep]
        List of training examples.
    """
    # Create state encoder and MCTS
    state_encoder = StateEncoder(config)
    mcts = MCTSSearch(model, config)
    reward_calc = RewardCalculator()

    training_examples: List[GameStep] = []

    try:
        # Use provided game or create new one
        if game is None:
            player_config = [
                ("PerceiverMuZero_0", "perceiver_muzero"),
                ("PerceiverMuZero_1", "perceiver_muzero"),
                ("PerceiverMuZero_2", "perceiver_muzero"),
                ("PerceiverMuZero_3", "perceiver_muzero"),
            ]
            game = Game(player_config)

            # Set game reference for all players
            for player in game.players:
                if hasattr(player.profile, "set_game"):
                    player.profile.set_game(game)

            # Play one hand
            game.play_hand()

        # Collect training examples from the hand
        # Get tricks won from game state
        tricks_won = getattr(game, "_current_tricks_won", [0, 0])

        # Get renege information
        renege_occurred = getattr(game, "_renege_occurred", False)
        renege_team = getattr(game, "_renege_team", None)
        # Note: renege_successful is deprecated - reneges are always penalized
        renege_successful = False

        # Find which team called trump
        calling_team = 0
        is_alone = getattr(game, "alone_mode", False)
        if game.trump_selector and hasattr(game.trump_selector, "trump_maker_name"):
            for i, player in enumerate(game.players):
                if player.name == game.trump_selector.trump_maker_name:
                    calling_team = player.team
                    break
        elif game.trump_selector and hasattr(game.trump_selector, "trump_maker_id"):
            calling_team = game.players[game.trump_selector.trump_maker_id].team

        # Get individual player tricks won from game statistics
        tricks_won_per_player: Dict[int, int] = {}
        if hasattr(game, "stats") and hasattr(game.stats, "tricks_won_per_player"):
            tricks_won_per_player = game.stats.tricks_won_per_player.copy()
        else:
            # Fallback: initialize with zeros
            tricks_won_per_player = {i: 0 for i in range(4)}

        # Get trump maker ID
        trump_maker_id: Optional[int] = None
        if game.trump_selector:
            if hasattr(game.trump_selector, "trump_maker_id"):
                trump_maker_id = game.trump_selector.trump_maker_id
            elif hasattr(game.trump_selector, "trump_maker_name"):
                # Find player ID by name
                for i, player in enumerate(game.players):
                    if player.name == game.trump_selector.trump_maker_name:
                        trump_maker_id = i
                        break

        # Get screw the dealer flag
        screw_the_dealer = getattr(game.trump_selector, "screw_the_dealer_occurred", False) if game.trump_selector else False

        # Collect examples for each player
        for player_id in range(4):
            player = game.players[player_id]
            team = player.team

            # Encode final state as tokens
            input_tokens = state_encoder.encode_tokens(game, player_id)

            # Run MCTS to get improved policy
            policy = mcts.search(input_tokens, risk_factor)

            # Calculate hand reward with enhanced individual tracking
            hand_reward = reward_calc.calculate_hand_reward(
                tricks_won[calling_team] if calling_team < len(tricks_won) else 0,
                calling_team,
                team,
                is_alone=is_alone,
                renege_occurred=renege_occurred,
                renege_team=renege_team,
                renege_successful=renege_successful,
                player_id=player_id,
                tricks_won_per_player=tricks_won_per_player,
                trump_maker_id=trump_maker_id,
                screw_the_dealer=screw_the_dealer,
            )

            # Sample action from policy for training
            action = np.random.choice(len(policy), p=policy)

            # Create training example
            step = GameStep(
                input_tokens=input_tokens,
                action=action,
                policy=policy,
                value=hand_reward,
                reward=0.0,  # Immediate rewards tracked during play
                next_input_tokens=None,  # Would be set during actual play
                player_id=player_id,
            )
            training_examples.append(step)

    except Exception as e:
        # Log error but return what we have
        print(f"Error in self-play game generation: {e}")

    return training_examples

