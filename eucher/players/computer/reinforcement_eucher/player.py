"""Player profile for ReinforcementEucher."""

from typing import List, Optional

import torch

from eucher.cards import Card, Suit
from eucher.players.base import PlayerProfile

if False:  # TYPE_CHECKING
    from eucher.players.base import Player


class ReinforcementEucherPlayer(PlayerProfile):
    """Player profile using ReinforcementEucher model.

    Parameters
    ----------
    model : ReinforcementEucherModel
        Trained model.
    game : Optional[Game]
        Game instance (for state encoding).
    temperature : float
        Temperature for action sampling (1.0 = normal, >1.0 = more random).
    """

    def __init__(
        self,
        model,
        game=None,
        temperature: float = 1.0,
    ) -> None:
        """Initialize ReinforcementEucher player.

        Parameters
        ----------
        model : ReinforcementEucherModel
            Trained model.
        game : Optional[Game]
            Game instance.
        temperature : float
            Temperature for sampling.
        """
        self.model = model
        self.game = game
        self.temperature = temperature
        self.model.eval()

        from eucher.players.computer.reinforcement_eucher.action_space import ActionEncoder
        from eucher.players.computer.reinforcement_eucher.state_encoder import StateEncoder

        self.action_encoder = ActionEncoder
        self.state_encoder = StateEncoder()

    def set_game(self, game) -> None:
        """Set game reference.

        Parameters
        ----------
        game : Game
            Game instance.
        """
        self.game = game

    def decide_order_up(
        self,
        player: "Player",
        turned_card: Card,
        dealer_id: int,
        trump_suit: Optional[Suit],
    ) -> bool:
        """Decide whether to order up.

        Parameters
        ----------
        player : Player
            Player making decision.
        turned_card : Card
            Turned card.
        dealer_id : int
            Dealer ID.
        trump_suit : Optional[Suit]
            Current trump suit (None during bidding).

        Returns
        -------
        bool
            True to order up, False to pass.
        """
        if self.game is None:
            return False

        legal_actions = [self.action_encoder.encode_bid_action("pass"), self.action_encoder.encode_bid_action("order_up")]
        state = self.state_encoder.encode_full_state(self.game, player.player_id, legal_actions)
        legal_mask = torch.tensor([1.0 if i in legal_actions else 0.0 for i in range(18)], device=self.model.config.torch_device)

        with torch.no_grad():
            action, _, _ = self.model.sample_action(
                state.unsqueeze(0).to(self.model.config.torch_device),
                legal_mask.unsqueeze(0),
                self.temperature,
            )
            action_id = action.item()

        action_type, _, _, _ = self.action_encoder.decode_action(action_id)
        return action_type == "order_up"

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """Decide which suit to call as trump.

        Parameters
        ----------
        player : Player
            Player making decision.
        turned_card : Card
            Turned card.
        trump_suit : Optional[Suit]
            Current trump suit (None during bidding).
        must_choose : bool
            Whether player must choose a suit.

        Returns
        -------
        Optional[Suit]
            Suit to call, or None to pass.
        """
        if self.game is None:
            return None

        legal_actions = [self.action_encoder.encode_bid_action("pass")]
        for suit in Suit:
            if suit != turned_card.suit:
                legal_actions.append(self.action_encoder.encode_bid_action("call", suit))

        state = self.state_encoder.encode_full_state(self.game, player.player_id, legal_actions)
        legal_mask = torch.tensor([1.0 if i in legal_actions else 0.0 for i in range(18)], device=self.model.config.torch_device)

        with torch.no_grad():
            action, _, _ = self.model.sample_action(
                state.unsqueeze(0).to(self.model.config.torch_device),
                legal_mask.unsqueeze(0),
                self.temperature,
            )
            action_id = action.item()

        action_type, _, suit, _ = self.action_encoder.decode_action(action_id)
        if action_type == "call" and suit:
            return suit
        return None

    def choose_card_to_discard(
        self,
        player: "Player",
        turned_card: Optional[Card] = None,
        ordered_up_by: Optional[str] = None,
    ) -> Card:
        """Choose card to discard.

        Parameters
        ----------
        player : Player
            Player discarding.
        turned_card : Optional[Card]
            Turned card.
        ordered_up_by : Optional[str]
            Name of player who ordered up.

        Returns
        -------
        Card
            Card to discard.
        """
        if self.game is None or len(player.hand) != 6:
            return player.hand[0]

        legal_actions = [self.action_encoder.encode_discard_action(i) for i in range(len(player.hand))]
        state = self.state_encoder.encode_full_state(self.game, player.player_id, legal_actions)
        legal_mask = torch.tensor([1.0 if i in legal_actions else 0.0 for i in range(18)], device=self.model.config.torch_device)

        with torch.no_grad():
            action, _, _ = self.model.sample_action(
                state.unsqueeze(0).to(self.model.config.torch_device),
                legal_mask.unsqueeze(0),
                self.temperature,
            )
            action_id = action.item()

        action_type, card_idx, _, _ = self.action_encoder.decode_action(action_id)
        if action_type == "discard" and card_idx is not None and card_idx < len(player.hand):
            return player.hand[card_idx]
        return player.hand[0]

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """Play a card.

        Parameters
        ----------
        player : Player
            Player playing.
        led_suit : Optional[Suit]
            Suit that was led.
        trump_suit : Optional[Suit]
            Current trump suit.
        trick_cards : List[Card]
            Cards already played in trick.
        trick_player_ids : List[int]
            IDs of players who played cards.

        Returns
        -------
        Card
            Card to play.
        """
        if self.game is None:
            return player.hand[0]

        # Get legal actions based on game rules
        legal_actions = []
        if led_suit:
            # Must follow suit if possible
            can_follow = any(card.suit == led_suit or (trump_suit and card.suit == trump_suit) for card in player.hand)
            if can_follow:
                for i, card in enumerate(player.hand):
                    if card.suit == led_suit or (trump_suit and card.suit == trump_suit):
                        legal_actions.append(self.action_encoder.encode_play_action(i))
            else:
                # Can play any card
                for i in range(len(player.hand)):
                    legal_actions.append(self.action_encoder.encode_play_action(i))
        else:
            # Leading - can play any card
            for i in range(len(player.hand)):
                legal_actions.append(self.action_encoder.encode_play_action(i))

        if not legal_actions:
            return player.hand[0]

        state = self.state_encoder.encode_full_state(self.game, player.player_id, legal_actions)
        legal_mask = torch.tensor([1.0 if i in legal_actions else 0.0 for i in range(18)], device=self.model.config.torch_device)

        with torch.no_grad():
            action, _, _ = self.model.sample_action(
                state.unsqueeze(0).to(self.model.config.torch_device),
                legal_mask.unsqueeze(0),
                self.temperature,
            )
            action_id = action.item()

        action_type, card_idx, _, _ = self.action_encoder.decode_action(action_id)
        if action_type == "play" and card_idx is not None and card_idx < len(player.hand):
            return player.hand[card_idx]
        return player.hand[0]

    def decide_going_alone(self, player: "Player", trump_suit: Suit) -> bool:
        """Decide whether to go alone.

        Parameters
        ----------
        player : Player
            Player deciding.
        trump_suit : Suit
            Trump suit.

        Returns
        -------
        bool
            True to go alone, False otherwise.
        """
        if self.game is None:
            return False

        legal_actions = [self.action_encoder.encode_bid_action("go_alone")]
        state = self.state_encoder.encode_full_state(self.game, player.player_id, legal_actions)
        legal_mask = torch.tensor([1.0 if i in legal_actions else 0.0 for i in range(18)], device=self.model.config.torch_device)

        with torch.no_grad():
            action, _, _ = self.model.sample_action(
                state.unsqueeze(0).to(self.model.config.torch_device),
                legal_mask.unsqueeze(0),
                self.temperature,
            )
            action_id = action.item()

        action_type, _, _, go_alone = self.action_encoder.decode_action(action_id)
        return go_alone or action_type == "go_alone"


