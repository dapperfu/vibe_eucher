"""Self-play generator for ReinforcementEucher training."""

from typing import Dict, List, Optional, Tuple

import torch

from eucher.game import Game
from eucher.players.computer.reinforcement_eucher.action_space import ActionEncoder, ACTION_SPACE_SIZE
from eucher.players.computer.reinforcement_eucher.config import ReinforcementEucherConfig
from eucher.players.computer.reinforcement_eucher.networks.model import ReinforcementEucherModel
from eucher.players.computer.reinforcement_eucher.rewards.reward_calculator import RewardCalculator
from eucher.players.computer.reinforcement_eucher.state_encoder import StateEncoder


class GameExperience:
    """Experience from a game step.

    Parameters
    ----------
    state : torch.Tensor
        Game state.
    action : int
        Action taken.
    reward : float
        Reward received.
    value : float
        Value estimate.
    log_prob : float
        Log probability of action.
    legal_mask : torch.Tensor
        Legal action mask.
    player_id : int
        Player ID.
    done : bool
        Whether episode is done.
    """

    def __init__(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        legal_mask: torch.Tensor,
        player_id: int,
        done: bool = False,
    ) -> None:
        """Initialize game experience.

        Parameters
        ----------
        state : torch.Tensor
            Game state.
        action : int
            Action taken.
        reward : float
            Reward received.
        value : float
            Value estimate.
        log_prob : float
            Log probability of action.
        legal_mask : torch.Tensor
            Legal action mask.
        player_id : int
            Player ID.
        done : bool
            Whether episode is done.
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.value = value
        self.log_prob = log_prob
        self.legal_mask = legal_mask
        self.player_id = player_id
        self.done = done


class SelfPlayGenerator:
    """Generate self-play games for training.

    Parameters
    ----------
    model : ReinforcementEucherModel
        Model to use for self-play.
    config : ReinforcementEucherConfig
        Configuration.
    """

    def __init__(
        self,
        model: ReinforcementEucherModel,
        config: ReinforcementEucherConfig,
    ) -> None:
        """Initialize self-play generator.

        Parameters
        ----------
        model : ReinforcementEucherModel
            Model to use for self-play.
        config : ReinforcementEucherConfig
            Configuration.
        """
        self.model = model
        self.config = config
        self.device = config.torch_device
        self.state_encoder = StateEncoder()
        self.reward_calculator = RewardCalculator(config)
        self.model.eval()

    def generate_game(
        self,
        game: Optional[Game] = None,
    ) -> List[GameExperience]:
        """Generate a self-play game and collect experiences.

        Parameters
        ----------
        game : Optional[Game]
            Game instance (if None, creates new game).

        Returns
        -------
        List[GameExperience]
            List of game experiences.
        """
        experiences: List[GameExperience] = []

        # Create game if not provided
        if game is None:
            # For self-play, we'll need to create a custom game
            # that uses the model for decisions
            # For now, create a placeholder that will be filled by trainer
            pass

        # This will be implemented to work with the actual game engine
        # For now, return empty list - will be completed when integrating with trainer
        return experiences

    def collect_experiences_from_game(
        self,
        game: Game,
        player_experiences: Dict[int, List[Dict]],
    ) -> List[GameExperience]:
        """Collect experiences from a completed game.

        Parameters
        ----------
        game : Game
            Completed game.
        player_experiences : Dict[int, List[Dict]]
            Experiences collected during game play (from player callbacks).

        Returns
        -------
        List[GameExperience]
            List of game experiences.
        """
        experiences: List[GameExperience] = []

        # Process experiences for each player
        for player_id, exp_list in player_experiences.items():
            for exp in exp_list:
                experience = GameExperience(
                    state=exp["state"],
                    action=exp["action"],
                    reward=exp["reward"],
                    value=exp.get("value", 0.0),
                    log_prob=exp.get("log_prob", 0.0),
                    legal_mask=exp["legal_mask"],
                    player_id=player_id,
                    done=exp.get("done", False),
                )
                experiences.append(experience)

        return experiences

    def get_legal_actions_for_state(
        self,
        game: Game,
        player_id: int,
    ) -> List[int]:
        """Get legal actions for current game state.

        Parameters
        ----------
        game : Game
            Game instance.
        player_id : int
            Player ID.

        Returns
        -------
        List[int]
            List of legal action IDs.
        """
        player = game.players[player_id]
        legal_actions = []

        # Determine game phase and get legal actions
        if game.trump_suit is None:
            # Bidding phase
            if game.turned_card:
                # Can order up or pass
                legal_actions.append(ActionEncoder.encode_bid_action("pass"))
                legal_actions.append(ActionEncoder.encode_bid_action("order_up"))
            else:
                # Can call suit or pass
                legal_actions.append(ActionEncoder.encode_bid_action("pass"))
                for suit in [s for s in game.turned_card.suit.__class__ if s != game.turned_card.suit]:
                    legal_actions.append(ActionEncoder.encode_bid_action("call", suit))
        else:
            # Playing phase
            if len(player.hand) == 6:
                # Discard phase
                for i in range(len(player.hand)):
                    legal_actions.append(ActionEncoder.encode_discard_action(i))
            else:
                # Card play phase
                if game.current_trick and game.current_trick.cards:
                    # Must follow suit if possible
                    lead_card = game.current_trick.cards[0]
                    lead_suit = lead_card.suit

                    # Check if can follow suit
                    can_follow = any(
                        card.suit == lead_suit
                        or (game.trump_suit and card.suit == game.trump_suit)
                        for card in player.hand
                    )

                    if can_follow:
                        # Only legal cards
                        for i, card in enumerate(player.hand):
                            if (
                                card.suit == lead_suit
                                or (game.trump_suit and card.suit == game.trump_suit)
                            ):
                                legal_actions.append(ActionEncoder.encode_play_action(i))
                    else:
                        # Can play any card
                        for i in range(len(player.hand)):
                            legal_actions.append(ActionEncoder.encode_play_action(i))
                else:
                    # Leading trick - can play any card
                    for i in range(len(player.hand)):
                        legal_actions.append(ActionEncoder.encode_play_action(i))

        return legal_actions

    def encode_state(
        self,
        game: Game,
        player_id: int,
        legal_actions: List[int],
    ) -> torch.Tensor:
        """Encode game state for player.

        Parameters
        ----------
        game : Game
            Game instance.
        player_id : int
            Player ID.
        legal_actions : List[int]
            Legal actions.

        Returns
        -------
        torch.Tensor
            Encoded state.
        """
        return self.state_encoder.encode_full_state(game, player_id, legal_actions)


