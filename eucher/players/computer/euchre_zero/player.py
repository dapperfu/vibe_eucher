"""EuchreZero player profile implementation."""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from eucher.cards import Card, Rank, Suit
from eucher.players.base import PlayerProfile
from eucher.players.computer.euchre_zero.action_space import ActionEncoder, ACTION_SPACE_SIZE
from eucher.players.computer.euchre_zero.config import EuchreZeroConfig
from eucher.players.computer.euchre_zero.mcts.search import MCTSSearch
from eucher.players.computer.euchre_zero.networks.model import EuchreZeroModel
from eucher.players.computer.euchre_zero.state_encoder import StateEncoder

if TYPE_CHECKING:
    from eucher.players import Player


class EuchreZeroPlayer(PlayerProfile):
    """EuchreZero player profile using MCTS for decisions."""

    def __init__(
        self,
        model: Optional[EuchreZeroModel] = None,
        model_path: Optional[str] = None,
        config: Optional[EuchreZeroConfig] = None,
        num_simulations: Optional[int] = None,
        risk_factor: float = 0.0,
        game: Optional[object] = None,
    ) -> None:
        """
        Initialize EuchreZero player.

        Parameters
        ----------
        model : Optional[EuchreZeroModel]
            Pre-initialized model. If None, creates new model.
        model_path : Optional[str]
            Path to model checkpoint. If None and model is None, creates new model.
        config : Optional[EuchreZeroConfig]
            Configuration. If None, uses default.
        num_simulations : Optional[int]
            Number of MCTS simulations. If None, uses config default.
        risk_factor : float
            Risk factor for decision making (0.0-1.0).
        game : Optional[object]
            Game instance (will be set by game integration).
        """
        if config is None:
            config = EuchreZeroConfig()

        if num_simulations is not None:
            config.num_simulations = num_simulations

        self.config = config
        self.risk_factor = risk_factor
        self.game = game

        # Initialize model
        if model is not None:
            self.model = model
        elif model_path is not None and Path(model_path).exists():
            self.model = EuchreZeroModel(config)
            self.model.load_checkpoint(Path(model_path))
        else:
            self.model = EuchreZeroModel(config)

        self.model.eval_mode()

        # Initialize MCTS and state encoder
        self.mcts = MCTSSearch(self.model, config)
        self.state_encoder = StateEncoder()

    def set_game(self, game: object) -> None:
        """
        Set game reference.

        Parameters
        ----------
        game : object
            Game instance.
        """
        self.game = game

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

        # Get game reference (simplified - would need proper game access)
        # For now, create minimal state
        # In full implementation, would get actual game state
        state_dict = self.state_encoder.encode_full_state(
            self._get_game(player), player.player_id
        )
        state_dict["risk_factor"] = torch.tensor([self.risk_factor])
        state_tensor = self.state_encoder.encode_state_dict_to_tensor(state_dict)

        # Run MCTS
        policy = self.mcts.search(state_tensor, self.risk_factor)

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

        # Encode state
        state_dict = self.state_encoder.encode_full_state(
            self._get_game(player), player.player_id
        )
        state_dict["risk_factor"] = torch.tensor([self.risk_factor])
        state_tensor = self.state_encoder.encode_state_dict_to_tensor(state_dict)

        # Run MCTS
        policy = self.mcts.search(state_tensor, self.risk_factor)

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
        # Encode state
        state_dict = self.state_encoder.encode_full_state(
            self._get_game(player), player.player_id
        )
        state_dict["risk_factor"] = torch.tensor([self.risk_factor])
        state_tensor = self.state_encoder.encode_state_dict_to_tensor(state_dict)

        # Run MCTS
        policy = self.mcts.search(state_tensor, self.risk_factor)

        # Get discard actions (DISCARD_0=7 to DISCARD_5=12)
        discard_probs = policy[7:13]
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
        # Encode state
        state_dict = self.state_encoder.encode_full_state(
            self._get_game(player), player.player_id
        )
        state_dict["risk_factor"] = torch.tensor([self.risk_factor])
        state_tensor = self.state_encoder.encode_state_dict_to_tensor(state_dict)

        # Run MCTS
        policy = self.mcts.search(state_tensor, self.risk_factor)

        # Get play actions (PLAY_0=13 to PLAY_4=17)
        play_probs = policy[13:18]
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
        # For now, use simple heuristic: check for flush in trump or very strong hand
        # This can be improved with MCTS/ML model prediction later
        trump_count = 0
        for card in player.hand:
            if card.rank == Rank.JACK and card.suit == trump_suit:
                trump_count += 1
            elif card.rank == Rank.JACK:
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    trump_count += 1
            elif card.suit == trump_suit:
                trump_count += 1

        # Go alone if flush in trump (all 5 cards are trump)
        if trump_count >= 5:
            return True

        # Go alone if 4+ trump cards (very strong)
        if trump_count >= 4:
            return True

        return False

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
        # For now, use simple heuristic: trade-in if we have weak eligible cards
        # This can be improved with MCTS/ML model prediction later
        # Generally, trade-in if eligible cards are low value (9s or 10s)
        # and we might get better cards from the kitty
        return False  # Conservative: don't trade-in by default

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
            raise RuntimeError("Game not set for EuchreZero player")
        return self.game

