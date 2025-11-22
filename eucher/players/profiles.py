"""Player profile implementations."""

from typing import TYPE_CHECKING, List, Optional

import torch

from eucher.cards import Card, Suit
from eucher.players.base import PlayerProfile
from eucher.players.computer import AIPlayer, HeuristicPlayer, RandomPlayer
from eucher.rules import RulesEngine

if TYPE_CHECKING:
    from eucher.players.base import Player

# Re-export computer players for convenience
__all__ = ["HumanProfile", "MLBasedProfile", "AIPlayer", "HeuristicPlayer", "RandomPlayer"]

# Import TUI for HumanProfile
try:
    from eucher.tui import TextTUI
except ImportError:
    TextTUI = None  # type: ignore


class HumanProfile(PlayerProfile):
    """Profile for human players using TUI (Terminal User Interface)."""

    def __init__(self, tui: Optional[TextTUI] = None) -> None:
        """
        Initialize a human profile.

        Parameters
        ----------
        tui : Optional[TextTUI]
            The TUI object. If None, will raise error when used.
        """
        self.tui = tui

    def set_tui(self, tui: TextTUI) -> None:
        """
        Set the TUI object.

        Parameters
        ----------
        tui : TextTUI
            The TUI object.
        """
        self.tui = tui

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """Get user input for ordering up."""
        if self.tui is None:
            raise RuntimeError("TUI not set for human profile")
        return self.tui.get_order_up_decision(player, turned_card, dealer_id)

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """Get user input for calling trump."""
        if self.tui is None:
            raise RuntimeError("TUI not set for human profile")
        result = self.tui.get_call_trump_decision(player, turned_card, must_choose)
        if must_choose and result is None:
            raise ValueError("Must choose a suit when must_choose=True")
        return result

    def choose_card_to_discard(self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None) -> Card:
        """Get user input for discarding a card."""
        if self.tui is None:
            raise RuntimeError("TUI not set for human profile")
        return self.tui.get_discard_decision(player, turned_card, ordered_up_by)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """Get user input for playing a card."""
        if self.tui is None:
            raise RuntimeError("TUI not set for human profile")
        return self.tui.get_play_card_decision(player, led_suit, trump_suit, trick_cards)


class MLBasedProfile(PlayerProfile):
    """Profile that uses PyTorch ML models for decision-making."""

    def __init__(
        self,
        ml_model,
        encoder,
        game_state_provider=None,
        trump_selection_risk: Optional[float] = None,
        gameplay_risk: Optional[float] = None,
    ) -> None:
        """
        Initialize an ML-based profile.

        Parameters
        ----------
        ml_model
            The EuchreMLModel instance to use.
        encoder
            The GameStateEncoder instance to use.
        game_state_provider
            Optional callable that provides additional game state (tricks_won, trick_number).
            Should return (trick_number, tricks_won_team0, tricks_won_team1).
        trump_selection_risk : Optional[float]
            Risk factor for trump selection decisions (0.0-1.0). If None, uses config default.
        gameplay_risk : Optional[float]
            Risk factor for gameplay decisions (0.0-1.0). If None, uses config default.
        """
        self.ml_model = ml_model
        self.encoder = encoder
        self.rules = RulesEngine()
        self.game_state_provider = game_state_provider

        # Set risk factors (temperature thresholds)
        from eucher.players.computer.ml.ml_config import MLConfig

        config = MLConfig()
        self.trump_selection_risk = (
            trump_selection_risk if trump_selection_risk is not None else config.trump_selection_risk
        )
        self.gameplay_risk = gameplay_risk if gameplay_risk is not None else config.gameplay_risk

    def _get_game_state(self) -> tuple[int, int, int]:
        """Get current game state information."""
        if self.game_state_provider is not None:
            return self.game_state_provider()
        return (0, 0, 0)

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """Use ML model to decide whether to order up."""
        if trump_suit is not None:
            return False

        trick_num, tricks_won_team0, tricks_won_team1 = self._get_game_state()

        features = self.encoder.encode_game_state(
            hand=player.hand,
            turned_card=turned_card,
            trump_suit=None,
            led_suit=None,
            trick_cards=[],
            player_id=player.player_id,
            dealer_id=dealer_id,
            team=player.team,
            trick_number=trick_num,
            tricks_won_team0=tricks_won_team0,
            tricks_won_team1=tricks_won_team1,
        )

        with torch.no_grad():
            features = features.unsqueeze(0).to(self.ml_model.device)
            order_up_logits, _ = self.ml_model.trump_net(features)
            order_up_prob = torch.sigmoid(order_up_logits).item()

        return order_up_prob >= self.trump_selection_risk

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """Use ML model to decide which suit to call as trump."""
        if trump_suit is not None:
            if must_choose:
                return trump_suit
            return None

        trick_num, tricks_won_team0, tricks_won_team1 = self._get_game_state()

        features = self.encoder.encode_game_state(
            hand=player.hand,
            turned_card=turned_card,
            trump_suit=None,
            led_suit=None,
            trick_cards=[],
            player_id=player.player_id,
            dealer_id=0,
            team=player.team,
            trick_number=trick_num,
            tricks_won_team0=tricks_won_team0,
            tricks_won_team1=tricks_won_team1,
        )

        with torch.no_grad():
            features = features.unsqueeze(0).to(self.ml_model.device)
            _, call_trump_logits = self.ml_model.trump_net(features)
            call_trump_probs = torch.softmax(call_trump_logits, dim=1).squeeze(0)

        forbidden_suit = turned_card.suit
        suits = [None, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]

        from eucher.players.computer.ml.ml_decision_weights import apply_temperature_threshold, get_decision_weights_from_logits

        weights = get_decision_weights_from_logits(call_trump_logits.squeeze(0))
        valid_indices = [i for i, suit in enumerate(suits) if suit != forbidden_suit]

        filtered_indices, filtered_weights = apply_temperature_threshold(
            weights, self.trump_selection_risk, valid_indices
        )

        if filtered_indices:
            best_idx = max(filtered_indices, key=lambda i: filtered_weights[i])
        else:
            best_idx = 0
            best_prob = call_trump_probs[0].item()
            for i, suit in enumerate(suits[1:], start=1):
                if suit != forbidden_suit and call_trump_probs[i].item() > best_prob:
                    best_prob = call_trump_probs[i].item()
                    best_idx = i

        if must_choose and best_idx == 0:
            for i, suit in enumerate(suits[1:], start=1):
                if suit != forbidden_suit:
                    return suit

        if best_idx == 0:
            return None
        return suits[best_idx]

    def choose_card_to_discard(self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None) -> Card:
        """Use ML model to choose a card to discard."""
        if not player.hand:
            raise ValueError("Player has no cards to discard")

        if len(player.hand) != 6:
            return min(player.hand, key=lambda c: c.rank.value)

        trick_num, tricks_won_team0, tricks_won_team1 = self._get_game_state()

        features = self.encoder.encode_game_state(
            hand=player.hand,
            turned_card=None,
            trump_suit=None,
            led_suit=None,
            trick_cards=[],
            player_id=player.player_id,
            dealer_id=player.player_id,
            team=player.team,
            trick_number=trick_num,
            tricks_won_team0=tricks_won_team0,
            tricks_won_team1=tricks_won_team1,
        )

        with torch.no_grad():
            features = features.unsqueeze(0).to(self.ml_model.device)
            discard_logits = self.ml_model.discard_net(features)

        from eucher.players.computer.ml.ml_decision_weights import get_decision_weights_from_logits, select_action_from_weights

        weights = get_decision_weights_from_logits(discard_logits.squeeze(0))
        hand_indices = list(range(len(player.hand)))

        selected_idx, _ = select_action_from_weights(weights, hand_indices, self.gameplay_risk)

        if selected_idx < len(player.hand):
            return player.hand[selected_idx]

        return min(player.hand, key=lambda c: c.rank.value)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """Use ML model to choose a card to play."""
        valid_cards = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)
        if not valid_cards:
            return player.hand[0]

        trick_num, tricks_won_team0, tricks_won_team1 = self._get_game_state()

        features = self.encoder.encode_game_state(
            hand=player.hand,
            turned_card=None,
            trump_suit=trump_suit,
            led_suit=led_suit,
            trick_cards=trick_cards,
            player_id=player.player_id,
            dealer_id=0,
            team=player.team,
            trick_number=trick_num,
            tricks_won_team0=tricks_won_team0,
            tricks_won_team1=tricks_won_team1,
        )

        with torch.no_grad():
            features = features.unsqueeze(0).to(self.ml_model.device)
            card_logits = self.ml_model.card_play_net(features)

        valid_indices = self.encoder.get_valid_card_indices(valid_cards)

        from eucher.players.computer.ml.ml_decision_weights import get_decision_weights_from_logits, select_action_from_weights

        weights = get_decision_weights_from_logits(card_logits.squeeze(0))
        selected_idx, _ = select_action_from_weights(weights, valid_indices, self.gameplay_risk)

        selected_card = self.encoder.decode_card_index(selected_idx)

        if selected_card in valid_cards:
            return selected_card

        return valid_cards[0]
