"""Self-play game generation for EucherGo training."""

from typing import Dict, List, Optional

import numpy as np
import torch

from eucher.game import Game
from eucher.players.computer.euchergo.action_space import ActionEncoder
from eucher.players.computer.euchergo.config import EucherGoConfig
from eucher.players.computer.euchergo.mcts.search import EucherGoMCTSSearch
from eucher.players.computer.euchergo.networks.model import EucherGoModel
from eucher.players.computer.euchergo.state_encoder import EucherGoStateEncoder


class GameStep:
    """Training example from a game step."""

    def __init__(
        self,
        state: torch.Tensor,
        policy: np.ndarray,
        value: float,
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
        player_id : int
            Player ID.
        """
        self.state = state
        self.policy = policy
        self.value = value
        self.player_id = player_id


def generate_self_play_game(
    model: EucherGoModel,
    config: EucherGoConfig,
    game: Optional[Game] = None,
    risk_factor: float = 0.0,
) -> List[GameStep]:
    """
    Generate training examples from a self-play game.

    Parameters
    ----------
    model : EucherGoModel
        EucherGo model.
    config : EucherGoConfig
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
    state_encoder = EucherGoStateEncoder()
    mcts = EucherGoMCTSSearch(model, config.num_simulations, config.exploration_constant, config.device)

    training_examples: List[GameStep] = []

    try:
        # Use provided game or create new one
        if game is None:
            from eucher.players.computer.euchergo.player import EucherGoPlayer

            player_config = [
                ("EucherGo_0", "euchergo"),
                ("EucherGo_1", "euchergo"),
                ("EucherGo_2", "euchergo"),
                ("EucherGo_3", "euchergo"),
            ]
            game = Game(player_config)

            # Set game reference and model for all EucherGo players
            for player in game.players:
                if hasattr(player.profile, "set_game"):
                    player.profile.set_game(game)
                if isinstance(player.profile, EucherGoPlayer):
                    player.profile.model = model
                    player.profile.mcts = mcts
                    player.profile.state_encoder = state_encoder

            # Play one hand
            game.play_hand()

        # Collect training examples from the hand
        # Get tricks won from game state
        tricks_won = getattr(game, "_current_tricks_won", [0, 0])

        # Find which team called trump
        calling_team = 0
        is_alone = getattr(game, "alone_mode", False)
        if game.trump_selector and hasattr(game.trump_selector, "trump_maker_name"):
            # Find which team called trump
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

        # Calculate hand outcome value
        # Simple value: +1 if calling team won (3+ tricks), -1 if euchred (< 3 tricks)
        calling_team_tricks = tricks_won[calling_team] if calling_team < len(tricks_won) else 0
        
        # Base hand value
        if calling_team_tricks >= 3:
            hand_value = 1.0
        else:
            hand_value = -1.0

        # Heavily penalize going alone failures
        if is_alone:
            if calling_team_tricks < 3:
                # Got euchred when going alone - severe penalty (2x normal euchre penalty)
                hand_value = -2.0
            elif calling_team_tricks < 5:
                # Failed to sweep when going alone - significant penalty
                # Still won but didn't get the bonus, so penalize relative to expectation
                hand_value = 0.0  # Neutral/negative outcome since going alone was risky
            elif calling_team_tricks == 5:
                # Successful sweep when going alone - bonus
                hand_value = 2.0

        for player_id in range(4):
            player = game.players[player_id]
            team = player.team

            # Encode final state
            # Need to get current game state for encoding
            state_tensor = state_encoder.encode_state(
                player_hand=player.hand,
                trump_suit=game.trump_suit,
                dealer_id=game.dealer_id if hasattr(game, "dealer_id") else 0,
                leader_id=0,  # Simplified
                current_trick_cards=[],
                trick_history=[],
                belief_map={},  # Simplified - would need belief tracker
                can_follow_suit=False,
                must_call_trump=False,
                can_go_alone=False,
                player_id=player_id,
            )

            # Run MCTS to get improved policy
            action_mask = None  # Would need proper action mask
            policy = mcts.search(state_tensor, action_mask)

            # Adjust value based on team
            # If player is on calling team, use hand_value; otherwise negate
            if team == calling_team:
                player_value = hand_value
            else:
                player_value = -hand_value

            # Create training example
            step = GameStep(
                state=state_tensor,
                policy=policy,
                value=player_value,
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

