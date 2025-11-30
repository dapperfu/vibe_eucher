"""Self-play game generation for EucherZero training."""

from typing import List, Optional

import numpy as np
import torch

from eucher.game import Game
from eucher.players.computer.eucher_zero.action_space import ActionEncoder, ACTION_SPACE_SIZE
from eucher.players.computer.eucher_zero.config import EucherZeroConfig
from eucher.players.computer.eucher_zero.mcts.search import MCTSSearch
from eucher.players.computer.eucher_zero.networks.model import EucherZeroModel
from eucher.players.computer.eucher_zero.rewards.reward_calculator import RewardCalculator
from eucher.players.computer.eucher_zero.state_encoder import StateEncoder


class GameStep:
    """Training example from a game step."""

    def __init__(
        self,
        state: torch.Tensor,
        policy: np.ndarray,
        value: float,
        reward: float,
        player_id: int,
    ) -> None:
        """
        Initialize game step.

        Parameters
        ----------
        state : torch.Tensor
            Game state tensor.
        policy : np.ndarray
            Improved policy from MCTS.
        value : float
            Final value (hand outcome).
        reward : float
            Immediate reward.
        player_id : int
            Player ID.
        """
        self.state = state
        self.policy = policy
        self.value = value
        self.reward = reward
        self.player_id = player_id


def generate_self_play_game(
    model: EucherZeroModel,
    config: EucherZeroConfig,
    game: Optional[Game] = None,
    risk_factor: float = 0.0,
) -> List[GameStep]:
    """
    Generate training examples from a self-play game.

    Parameters
    ----------
    model : EucherZeroModel
        EucherZero model.
    config : EucherZeroConfig
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
    state_encoder = StateEncoder()
    mcts = MCTSSearch(model, config)
    reward_calc = RewardCalculator()

    training_examples: List[GameStep] = []

    try:
        # Use provided game or create new one
        if game is None:
            player_config = [
                ("EucherZero_0", "eucher_zero"),
                ("EucherZero_1", "eucher_zero"),
                ("EucherZero_2", "eucher_zero"),
                ("EucherZero_3", "eucher_zero"),
            ]
            game = Game(player_config)

            # Set game reference for all EucherZero players
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

        # Find which team called trump
        calling_team = 0
        if game.trump_selector and hasattr(game.trump_selector, "trump_maker_name"):
            # Find which team called trump
            for i, player in enumerate(game.players):
                if player.name == game.trump_selector.trump_maker_name:
                    calling_team = player.team
                    break
        elif game.trump_selector and hasattr(game.trump_selector, "trump_maker_id"):
            calling_team = game.players[game.trump_selector.trump_maker_id].team

        for player_id in range(4):
            player = game.players[player_id]
            team = player.team

            # Encode final state
            state_dict = state_encoder.encode_full_state(game, player_id)
            state_dict["risk_factor"] = torch.tensor([risk_factor])
            state_tensor = state_encoder.encode_state_dict_to_tensor(state_dict)

            # Run MCTS to get improved policy
            policy = mcts.search(state_tensor, risk_factor)

            # Calculate hand reward (with renege penalties)
            hand_reward = reward_calc.calculate_hand_reward(
                tricks_won[calling_team],
                calling_team,
                team,
                risk_factor,
                renege_occurred=renege_occurred,
                renege_team=renege_team,
            )

            # Create training example
            step = GameStep(
                state=state_tensor,
                policy=policy,
                value=hand_reward,
                reward=0.0,  # Immediate rewards would be tracked during play
                player_id=player_id,
            )
            training_examples.append(step)

    except Exception as e:
        # If game fails, return empty list
        print(f"Self-play game generation failed: {e}")
        import traceback

        traceback.print_exc()
        return []

    return training_examples

