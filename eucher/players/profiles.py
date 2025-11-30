"""Player profile implementations."""

from typing import TYPE_CHECKING, Dict, List, Optional

# torch is imported lazily only when needed for ML profiles
from eucher.cards import Card, Rank, Suit
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

    def decide_going_alone(self, player: "Player", trump_suit: Suit) -> bool:
        """Get user input for going alone."""
        if self.tui is None:
            raise RuntimeError("TUI not set for human profile")
        # Default to False if TUI doesn't support going alone yet
        if hasattr(self.tui, "get_going_alone_decision"):
            return self.tui.get_going_alone_decision(player, trump_suit)
        return False

    def decide_trade_in(self, player: "Player", eligible_cards: List[Card]) -> bool:
        """Get user input for trade-in decision."""
        if self.tui is None:
            raise RuntimeError("TUI not set for human profile")
        # Default to False if TUI doesn't support trade-in yet
        if hasattr(self.tui, "get_trade_in_decision"):
            return self.tui.get_trade_in_decision(player, eligible_cards)
        return False


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

        import torch  # Lazy import
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

        import torch  # Lazy import
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

        import torch  # Lazy import
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

        import torch  # Lazy import
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

    def decide_going_alone(self, player: "Player", trump_suit: Suit) -> bool:
        """Use ML model to decide whether to go alone."""
        # For now, use a simple heuristic: go alone if hand strength is very high
        # This can be improved with ML model prediction later
        trick_num, tricks_won_team0, tricks_won_team1 = self._get_game_state()

        # Simple heuristic: count trump cards and high cards
        trump_count = 0
        high_trump_count = 0
        for card in player.hand:
            # Check if trump
            if card.rank == Rank.JACK and card.suit == trump_suit:
                trump_count += 1
                high_trump_count += 1
            elif card.rank == Rank.JACK:
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    trump_count += 1
                    high_trump_count += 1
            elif card.suit == trump_suit:
                trump_count += 1
                if card.rank == Rank.ACE:
                    high_trump_count += 1

        # Go alone if we have 4+ trump cards or flush in trump
        if trump_count >= 4:
            return True

        # Check for flush (all 5 cards are trump)
        if trump_count == 5:
            return True

        return False

    def decide_trade_in(self, player: "Player", eligible_cards: List[Card]) -> bool:
        """
        Decide whether to trade-in using ML heuristics.

        Uses simple heuristic evaluation since ML model doesn't have
        trade-in training data yet.

        Parameters
        ----------
        player : Player
            The player making the decision.
        eligible_cards : List[Card]
            The three cards eligible for trade-in.

        Returns
        -------
        bool
            True to trade-in, False to pass.
        """
        # Use heuristic evaluation (similar to HeuristicPlayer)
        # Calculate current hand strength
        current_hand_strength = sum(card.rank.value for card in player.hand)

        # Count high cards (Ace, King, Queen) in current hand
        high_card_count = sum(
            1 for card in player.hand if card.rank in (Rank.ACE, Rank.KING, Rank.QUEEN)
        )

        # Count Jacks (potential bowers)
        jack_count = sum(1 for card in player.hand if card.rank == Rank.JACK)

        # Trade-in if hand is weak
        if high_card_count <= 2 and jack_count == 0:
            return True

        # Trade-in if hand strength is very low
        if current_hand_strength <= 50:
            return True

        return False

    def get_order_up_probabilities(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> Dict[bool, float]:
        """
        Return probability distribution for order up decision from ML model.

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
        Dict[bool, float]
            Dictionary mapping decision (True=order up, False=pass) to probability.
        """
        if trump_suit is not None:
            return {False: 1.0, True: 0.0}

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

        import torch  # Lazy import
        with torch.no_grad():
            features = features.unsqueeze(0).to(self.ml_model.device)
            order_up_logits, _ = self.ml_model.trump_net(features)
            order_up_prob = torch.sigmoid(order_up_logits).item()

        return {True: float(order_up_prob), False: float(1.0 - order_up_prob)}

    def get_call_trump_probabilities(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Dict[Optional[Suit], float]:
        """
        Return probability distribution for call trump decision from ML model.

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
        Dict[Optional[Suit], float]
            Dictionary mapping suit (or None for pass) to probability.
        """
        if trump_suit is not None:
            return {trump_suit: 1.0}

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

        import torch  # Lazy import
        with torch.no_grad():
            features = features.unsqueeze(0).to(self.ml_model.device)
            _, call_trump_logits = self.ml_model.trump_net(features)
            call_trump_probs = torch.softmax(call_trump_logits, dim=1).squeeze(0)

        # Map to suits: [pass, HEARTS, DIAMONDS, CLUBS, SPADES]
        suits = [None, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
        forbidden_suit = turned_card.suit

        result: Dict[Optional[Suit], float] = {}
        for i, suit in enumerate(suits):
            if suit != forbidden_suit:
                if must_choose and suit is None:
                    continue
                result[suit] = float(call_trump_probs[i].item())

        # Normalize if needed
        total = sum(result.values())
        if total > 0:
            for suit in result:
                result[suit] /= total

        return result

    def get_play_card_probabilities(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Dict[Card, float]:
        """
        Return probability distribution for play card decision from ML model.

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
        Dict[Card, float]
            Dictionary mapping card to probability (only valid cards).
        """
        valid_cards = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)
        if not valid_cards:
            return {}

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

        import torch  # Lazy import
        with torch.no_grad():
            features = features.unsqueeze(0).to(self.ml_model.device)
            card_logits = self.ml_model.card_play_net(features)

        valid_indices = self.encoder.get_valid_card_indices(valid_cards)

        from eucher.players.computer.ml.ml_decision_weights import get_decision_weights_from_logits

        weights = get_decision_weights_from_logits(card_logits.squeeze(0))

        result: Dict[Card, float] = {}
        total_weight = 0.0
        for idx in valid_indices:
            if 0 <= idx < len(weights):
                card = self.encoder.decode_card_index(idx)
                if card in valid_cards:
                    weight = float(weights[idx])
                    result[card] = weight
                    total_weight += weight

        # Normalize
        if total_weight > 0:
            for card in result:
                result[card] /= total_weight
        else:
            # Fallback: equal distribution
            prob = 1.0 / len(valid_cards)
            for card in valid_cards:
                result[card] = prob

        return result

    def get_discard_probabilities(
        self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
    ) -> Dict[Card, float]:
        """
        Return probability distribution for discard decision from ML model.

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
        Dict[Card, float]
            Dictionary mapping card to probability.
        """
        if not player.hand:
            return {}

        trick_num, tricks_won_team0, tricks_won_team1 = self._get_game_state()

        features = self.encoder.encode_game_state(
            hand=player.hand,
            turned_card=turned_card,
            trump_suit=turned_card.suit if turned_card else None,
            led_suit=None,
            trick_cards=[],
            player_id=player.player_id,
            dealer_id=player.player_id,
            team=player.team,
            trick_number=trick_num,
            tricks_won_team0=tricks_won_team0,
            tricks_won_team1=tricks_won_team1,
        )

        import torch  # Lazy import
        with torch.no_grad():
            features = features.unsqueeze(0).to(self.ml_model.device)
            discard_logits = self.ml_model.discard_net(features)

        from eucher.players.computer.ml.ml_decision_weights import get_decision_weights_from_logits

        weights = get_decision_weights_from_logits(discard_logits.squeeze(0))

        result: Dict[Card, float] = {}
        for i, card in enumerate(player.hand):
            if i < len(weights):
                result[card] = float(weights[i])
            else:
                result[card] = 0.0

        # Normalize
        total = sum(result.values())
        if total > 0:
            for card in result:
                result[card] /= total

        return result
