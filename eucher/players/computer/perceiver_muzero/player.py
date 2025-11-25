"""EuchrePerceiverMuZero player profile implementation."""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from eucher.cards import Card, Rank, Suit
from eucher.players.base import PlayerProfile

from .action_space import ActionEncoder, ACTION_SPACE_SIZE
from .config import PerceiverMuZeroConfig
from .mcts.search import MCTSSearch
from .networks.model import PerceiverMuZeroModel
from .state_encoder import StateEncoder

if TYPE_CHECKING:
    from eucher.players import Player


class EuchrePerceiverMuZeroPlayer(PlayerProfile):
    """EuchrePerceiverMuZero player profile using MuZero-style MCTS for decisions."""

    def __init__(
        self,
        model: Optional[PerceiverMuZeroModel] = None,
        model_path: Optional[str] = None,
        config: Optional[PerceiverMuZeroConfig] = None,
        num_simulations: Optional[int] = None,
        risk_factor: float = 0.0,
        game: Optional[object] = None,
        fast_mode: bool = False,
    ) -> None:
        """
        Initialize EuchrePerceiverMuZero player.

        Parameters
        ----------
        model : Optional[PerceiverMuZeroModel]
            Pre-initialized model. If None, creates new model.
        model_path : Optional[str]
            Path to model checkpoint. If None and model is None, creates new model.
        config : Optional[PerceiverMuZeroConfig]
            Configuration. If None, uses default.
        num_simulations : Optional[int]
            Number of MCTS simulations. If None, uses config default.
        risk_factor : float
            Risk factor for decision making (0.0-1.0).
        game : Optional[object]
            Game instance (will be set by game integration).
        fast_mode : bool
            If True, use feed-forward only (no MCTS). Much faster for tournaments.
        """
        if config is None:
            config = PerceiverMuZeroConfig()

        if num_simulations is not None:
            config.num_simulations = num_simulations

        self.config = config
        self.risk_factor = risk_factor
        self.game = game
        self.fast_mode = fast_mode

        # Initialize model
        if model is not None:
            self.model = model
        elif model_path is not None and Path(model_path).exists():
            self.model = PerceiverMuZeroModel(config)
            self.model.load_checkpoint(Path(model_path))
        else:
            self.model = PerceiverMuZeroModel(config)

        self.model.eval_mode()

        # Initialize MCTS and state encoder (only if not in fast mode)
        if not fast_mode:
            self.mcts = MCTSSearch(self.model, config)
        self.state_encoder = StateEncoder(config)

    def set_game(self, game: object) -> None:
        """
        Set game reference.

        Parameters
        ----------
        game : object
            Game instance.
        """
        self.game = game

    def _get_policy(self, input_tokens: torch.Tensor) -> np.ndarray:
        """
        Get policy distribution using either fast mode (feed-forward) or MCTS.

        Parameters
        ----------
        input_tokens : torch.Tensor
            Encoded game state tokens.

        Returns
        -------
        np.ndarray
            Policy distribution over actions [action_space_size].
        """
        if self.fast_mode:
            # Fast mode: direct feed-forward without MCTS
            input_tokens = input_tokens.to(self.model.device)
            with torch.no_grad():
                perceiver_output = self.model.perceiver_encoder(input_tokens)
                latent = self.model.representation_net(perceiver_output)
                policy_logits, _, _ = self.model.prediction_net(latent)
                policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()
            return policy
        else:
            # Run MCTS
            return self.mcts.search(input_tokens, self.risk_factor)

    def decide_order_up(
        self,
        player: "Player",
        turned_card: Card,
        dealer_id: int,
        trump_suit: Optional[Suit],
    ) -> bool:
        """
        Decide whether to order up the turned card.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up.
        dealer_id : int
            The ID of the dealer.
        trump_suit : Optional[Suit]
            Current trump suit if already determined.

        Returns
        -------
        bool
            True to order up, False to pass.
        """
        if trump_suit is not None:
            return False

        # Encode state as tokens
        input_tokens = self.state_encoder.encode_tokens(
            self._get_game(player), player.player_id
        )

        # Get policy (fast mode or MCTS)
        policy = self._get_policy(input_tokens)

        # Select action (ORDER_UP = 1)
        action_id = np.argmax(policy)
        action_type, _, _ = ActionEncoder.decode_action(action_id)

        return action_type == "order_up"

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Decide which suit to call as trump (or pass).

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up (cannot be chosen).
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit (cannot pass).

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        if trump_suit is not None:
            return None

        # Encode state as tokens
        input_tokens = self.state_encoder.encode_tokens(
            self._get_game(player), player.player_id
        )

        # Get policy (fast mode or MCTS)
        policy = self._get_policy(input_tokens)

        # Get call actions (CALL_HEARTS=2, CALL_DIAMONDS=3, CALL_CLUBS=4, CALL_SPADES=5)
        call_probs = policy[2:6]
        call_action_id = 2 + np.argmax(call_probs)

        # Decode action
        action_type, _, suit = ActionEncoder.decode_action(call_action_id)

        if action_type == "call" and suit is not None and suit != turned_card.suit:
            return suit

        # If must_choose, pick first available suit
        if must_choose:
            suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
            for s in suits:
                if s != turned_card.suit:
                    return s

        return None

    def choose_card_to_discard(
        self,
        player: "Player",
        turned_card: Optional[Card] = None,
        ordered_up_by: Optional[str] = None,
    ) -> Card:
        """
        Choose a card to discard (dealer only, after ordering up).

        Parameters
        ----------
        player : Player
            The dealer player.
        turned_card : Optional[Card]
            The card that was ordered up, if available.
        ordered_up_by : Optional[str]
            Name of the player who ordered up, if available.

        Returns
        -------
        Card
            The card to discard.
        """
        # Encode state as tokens
        input_tokens = self.state_encoder.encode_tokens(
            self._get_game(player), player.player_id
        )

        # Get policy (fast mode or MCTS)
        policy = self._get_policy(input_tokens)

        # Get discard actions (DISCARD_0=9 to DISCARD_5=14)
        discard_probs = policy[9:15]
        discard_idx = np.argmax(discard_probs)

        # Ensure index is valid
        if discard_idx >= len(player.hand):
            discard_idx = len(player.hand) - 1

        return player.hand[discard_idx]

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Choose a card to play in a trick.

        Parameters
        ----------
        player : Player
            The player making the decision.
        led_suit : Optional[Suit]
            The suit that was led, if any.
        trump_suit : Optional[Suit]
            The current trump suit, if any.
        trick_cards : List[Card]
            Cards already played in the trick.
        trick_player_ids : List[int]
            Player IDs who played each card in trick_cards.

        Returns
        -------
        Card
            The card to play.
        """
        # Encode state as tokens
        input_tokens = self.state_encoder.encode_tokens(
            self._get_game(player), player.player_id
        )

        # Get policy (fast mode or MCTS)
        policy = self._get_policy(input_tokens)

        # Get play actions (PLAY_0=15 to PLAY_4=19)
        play_probs = policy[15:20]
        play_idx = np.argmax(play_probs)

        # Ensure index is valid
        if play_idx >= len(player.hand):
            play_idx = len(player.hand) - 1

        return player.hand[play_idx]

    def decide_going_alone(self, player: "Player", trump_suit: Suit) -> bool:
        """
        Decide whether to go alone after making trump.

        Parameters
        ----------
        player : Player
            The player making the decision.
        trump_suit : Suit
            The trump suit that was selected.

        Returns
        -------
        bool
            True to go alone, False to play with partner.
        """
        if not self.config.enable_go_alone:
            return False

        # Encode state as tokens
        input_tokens = self.state_encoder.encode_tokens(
            self._get_game(player), player.player_id
        )

        # Get policy (fast mode or MCTS)
        policy = self._get_policy(input_tokens)

        # Check GO_ALONE action (action_id = 6)
        go_alone_prob = policy[6]

        # Go alone if probability is high
        return go_alone_prob > 0.5

    def decide_trade_in(self, player: "Player", eligible_cards: List[Card]) -> bool:
        """
        Decide whether to trade-in eligible cards for kitty cards.

        Parameters
        ----------
        player : Player
            The player making the decision.
        eligible_cards : List[Card]
            The three cards that are eligible for trade-in (same suit, all 9s or 10s).

        Returns
        -------
        bool
            True to trade-in, False to pass.
        """
        if not self.config.enable_nine_ten_tradein:
            return False

        # Encode state as tokens
        input_tokens = self.state_encoder.encode_tokens(
            self._get_game(player), player.player_id
        )

        # Get policy (fast mode or MCTS)
        policy = self._get_policy(input_tokens)

        # Check TRADE_IN action (action_id = 7)
        trade_in_prob = policy[7]

        return trade_in_prob > 0.5

    def _get_game(self, player: "Player"):
        """
        Get game instance.

        Parameters
        ----------
        player : Player
            Player instance (unused, kept for interface compatibility).

        Returns
        -------
        Game
            Game instance.
        """
        if self.game is None:
            raise RuntimeError("Game not set for EuchrePerceiverMuZero player")
        return self.game

