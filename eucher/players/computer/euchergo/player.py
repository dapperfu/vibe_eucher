"""EucherGo player profile implementation."""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from eucher.cards import Card, Suit
from eucher.players.base import PlayerProfile
from eucher.players.computer.euchergo.action_space import ActionEncoder
from eucher.players.computer.euchergo.config import EucherGoConfig
from eucher.players.computer.euchergo.mcts.search import EucherGoMCTSSearch
from eucher.players.computer.euchergo.networks.model import EucherGoModel
from eucher.players.computer.euchergo.state_encoder import EucherGoStateEncoder
from eucher.players.computer.euchergo.utils.belief_tracker import BeliefTracker
from eucher.rules import RulesEngine

if TYPE_CHECKING:
    from eucher.players import Player


class EucherGoPlayer(PlayerProfile):
    """EucherGo player profile using MCTS for decisions.

    Implements AlphaZero/MuZero-style architecture with:
    - Shared trunk network
    - Policy and value heads
    - MCTS with PUCT
    - Belief tracking for unknown cards
    """

    def __init__(
        self,
        model: Optional[EucherGoModel] = None,
        model_path: Optional[str] = None,
        config: Optional[EucherGoConfig] = None,
        num_simulations: Optional[int] = None,
        risk_factor: float = 0.0,
        game: Optional[object] = None,
    ) -> None:
        """
        Initialize EucherGo player.

        Parameters
        ----------
        model : Optional[EucherGoModel]
            Pre-initialized model. If None, creates new model.
        model_path : Optional[str]
            Path to model checkpoint. If None and model is None, creates new model.
        config : Optional[EucherGoConfig]
            Configuration. If None, uses default.
        num_simulations : Optional[int]
            Number of MCTS simulations. If None, uses config default.
        risk_factor : float
            Risk factor for decision making (0.0-1.0).
        game : Optional[object]
            Game instance (will be set by game integration).
        """
        if config is None:
            config = EucherGoConfig()

        if num_simulations is not None:
            config.num_simulations = num_simulations

        self.config = config
        self.risk_factor = risk_factor
        self.game = game

        # Initialize model
        if model is not None:
            self.model = model
        elif model_path is not None and Path(model_path).exists():
            self.model = EucherGoModel(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                use_conv1d=config.use_conv1d,
                action_space_size=ActionEncoder.ACTION_SPACE_SIZE,
            )
            self.model.load_checkpoint(Path(model_path))
        else:
            self.model = EucherGoModel(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                use_conv1d=config.use_conv1d,
                action_space_size=ActionEncoder.ACTION_SPACE_SIZE,
            )

        self.model.to(config.device)
        self.model.eval_mode()

        # Initialize MCTS and state encoder
        self.mcts = EucherGoMCTSSearch(
            self.model, config.num_simulations, config.exploration_constant, config.device
        )
        self.state_encoder = EucherGoStateEncoder()
        self.belief_tracker = BeliefTracker()
        self.rules = RulesEngine()

        # Game state tracking
        self.trick_history: List[List[Card]] = []
        self.current_trick_cards: List[Card] = []
        self.current_trick_player_ids: List[int] = []

    def set_game(self, game: object) -> None:
        """
        Set game reference.

        Parameters
        ----------
        game : object
            Game instance.
        """
        self.game = game

    def _get_game(self, player: "Player"):
        """
        Get game instance.

        Parameters
        ----------
        player : Player
            Player instance.

        Returns
        -------
        Game
            Game instance.
        """
        if self.game is None:
            raise RuntimeError("Game not set for EucherGo player")
        return self.game

    def _encode_state(self, player: "Player", game) -> torch.Tensor:
        """
        Encode current game state.

        Parameters
        ----------
        player : Player
            Current player.
        game : Game
            Game instance.

        Returns
        -------
        torch.Tensor
            Encoded state tensor.
        """
        # Get current trick info
        current_trick_cards = self.current_trick_cards.copy()
        current_trick_player_ids = self.current_trick_player_ids.copy()

        # Get trick history
        trick_history = self.trick_history.copy()

        # Get belief maps
        belief_maps = self.belief_tracker.get_all_belief_maps()

        # Determine flags
        led_suit = None
        if current_trick_cards:
            led_suit = current_trick_cards[0].suit

        can_follow_suit = False
        if led_suit and game.trump_suit:
            valid_plays = self.rules.get_valid_plays(player.hand, led_suit, game.trump_suit)
            can_follow_suit = len(valid_plays) > 0

        must_call_trump = False  # TODO: Determine from game state
        can_go_alone = game.trump_suit is not None  # Can go alone after trump is called

        # Determine leader
        leader_id = 0
        if current_trick_cards:
            leader_id = current_trick_player_ids[0] if current_trick_player_ids else 0
        else:
            # First trick: player left of dealer
            leader_id = (game.dealer_id + 1) % 4

        # Encode state
        state_tensor = self.state_encoder.encode_state(
            player_hand=player.hand,
            trump_suit=game.trump_suit,
            dealer_id=game.dealer_id,
            leader_id=leader_id,
            current_trick_cards=current_trick_cards,
            trick_history=trick_history,
            belief_map=belief_maps,
            can_follow_suit=can_follow_suit,
            must_call_trump=must_call_trump,
            can_go_alone=can_go_alone,
            player_id=player.player_id,
        )

        return state_tensor

    def _get_action_mask(self, player: "Player", game, action_type: str) -> torch.Tensor:
        """
        Get action mask for legal moves.

        Parameters
        ----------
        player : Player
            Current player.
        game : Game
            Game instance.
        action_type : str
            Type of action: "order_up", "call_trump", "play_card", "discard", "go_alone".

        Returns
        -------
        torch.Tensor
            Action mask [action_space_size] where 1 = legal, 0 = illegal.
        """
        mask = torch.zeros(ActionEncoder.ACTION_SPACE_SIZE, dtype=torch.float32)

        if action_type == "order_up":
            # Can order up or pass
            mask[ActionEncoder.PASS] = 1.0
            mask[ActionEncoder.ORDER_UP] = 1.0
        elif action_type == "call_trump":
            # Can call any suit except turned card suit, or pass
            mask[ActionEncoder.PASS] = 1.0
            if game.turned_card:
                forbidden_suit = game.turned_card.suit
                for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
                    if suit != forbidden_suit:
                        mask[ActionEncoder.encode_call_trump(suit)] = 1.0
            else:
                for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
                    mask[ActionEncoder.encode_call_trump(suit)] = 1.0
        elif action_type == "play_card":
            # Can play any valid card
            led_suit = None
            if self.current_trick_cards:
                led_suit = self.current_trick_cards[0].suit
            valid_cards = self.rules.get_valid_plays(player.hand, led_suit, game.trump_suit)
            for i, card in enumerate(player.hand):
                if card in valid_cards:
                    mask[ActionEncoder.encode_play_card(i)] = 1.0
        elif action_type == "discard":
            # Can discard any card (supports up to 6 cards for dealer after ordering up)
            for i in range(min(len(player.hand), 6)):
                mask[ActionEncoder.encode_discard(i)] = 1.0
        elif action_type == "go_alone":
            # Can go alone or not
            mask[ActionEncoder.GO_ALONE] = 1.0
            # Also allow not going alone (implicit pass)

        return mask

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
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

        game = self._get_game(player)

        # Encode state
        state_tensor = self._encode_state(player, game)
        action_mask = self._get_action_mask(player, game, "order_up")

        # Run MCTS
        policy = self.mcts.search(state_tensor, action_mask)

        # Select action
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

        game = self._get_game(player)

        # Encode state
        state_tensor = self._encode_state(player, game)
        action_mask = self._get_action_mask(player, game, "call_trump")

        # Run MCTS
        policy = self.mcts.search(state_tensor, action_mask)

        # Get call actions
        call_actions = [
            ActionEncoder.CALL_HEARTS,
            ActionEncoder.CALL_DIAMONDS,
            ActionEncoder.CALL_CLUBS,
            ActionEncoder.CALL_SPADES,
        ]
        call_probs = [policy[action] for action in call_actions]
        best_call_idx = np.argmax(call_probs)
        best_call_action = call_actions[best_call_idx]

        # Decode action
        action_type, _, suit = ActionEncoder.decode_action(best_call_action)

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
        self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
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
        game = self._get_game(player)

        # Encode state
        state_tensor = self._encode_state(player, game)
        action_mask = self._get_action_mask(player, game, "discard")

        # Run MCTS
        policy = self.mcts.search(state_tensor, action_mask)

        # Get discard actions (supports up to 6 cards for dealer after ordering up)
        discard_probs = [
            policy[ActionEncoder.encode_discard(i)] for i in range(min(len(player.hand), 6))
        ]
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
        # Update current trick tracking
        self.current_trick_cards = trick_cards.copy()
        self.current_trick_player_ids = trick_player_ids.copy()

        game = self._get_game(player)

        # Encode state
        state_tensor = self._encode_state(player, game)
        action_mask = self._get_action_mask(player, game, "play_card")

        # Run MCTS
        policy = self.mcts.search(state_tensor, action_mask)

        # Get play actions
        valid_cards = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)
        play_probs = []
        play_indices = []
        for i, card in enumerate(player.hand):
            if card in valid_cards:
                play_probs.append(policy[ActionEncoder.encode_play_card(i)])
                play_indices.append(i)

        if not play_probs:
            # Fallback: play first valid card
            return valid_cards[0] if valid_cards else player.hand[0]

        best_idx = play_indices[np.argmax(play_probs)]
        return player.hand[best_idx]

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
        game = self._get_game(player)

        # Encode state
        state_tensor = self._encode_state(player, game)
        action_mask = self._get_action_mask(player, game, "go_alone")

        # Run MCTS
        policy = self.mcts.search(state_tensor, action_mask)

        # Check go alone probability
        go_alone_prob = policy[ActionEncoder.GO_ALONE]
        return go_alone_prob > 0.5

    def decide_trade_in(self, player: "Player", eligible_cards: List[Card]) -> bool:
        """
        Decide whether to trade-in eligible cards for kitty cards.

        Parameters
        ----------
        player : Player
            The player making the decision.
        eligible_cards : List[Card]
            The three cards that are eligible for trade-in.

        Returns
        -------
        bool
            True to trade-in, False to pass.
        """
        # For now, use simple heuristic: don't trade-in by default
        # This can be improved with MCTS/ML model prediction later
        return False

    def record_trick_completion(
        self, played_cards: List[Card], player_ids: List[int], winner_team: int, winner_player_id: int
    ) -> None:
        """
        Record a completed trick (called by game after trick ends).

        Parameters
        ----------
        played_cards : List[Card]
            Cards played in the trick.
        player_ids : List[int]
            Player IDs.
        winner_team : int
            Winning team.
        winner_player_id : int
            Winning player ID.
        """
        # Update trick history
        self.trick_history.append(played_cards.copy())

        # Update belief tracker
        self.belief_tracker.update_after_trick(played_cards, player_ids)

        # Clear current trick
        self.current_trick_cards.clear()
        self.current_trick_player_ids.clear()

    def reset_for_new_hand(self) -> None:
        """Reset state for a new hand."""
        self.trick_history.clear()
        self.current_trick_cards.clear()
        self.current_trick_player_ids.clear()
        # Reinitialize belief tracker when hand starts
        # (will be initialized when we know the hand)

