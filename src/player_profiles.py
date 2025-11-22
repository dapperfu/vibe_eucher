"""Player profile system for pluggable decision-making backends."""

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import torch

from src.cards import Card, Rank, Suit
from src.rules import RulesEngine

if TYPE_CHECKING:
    from src.players import Player

# Import ComputerPlayer after PlayerProfile is defined to avoid circular import
# This will be imported at the end of the file


# Import TUI for HumanProfile
try:
    from src.tui import TextTUI
except ImportError:
    TextTUI = None  # type: ignore


class PlayerProfile(ABC):
    """Abstract base class for player decision-making profiles."""

    @abstractmethod
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
        pass

    @abstractmethod
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
            If True, player must choose a suit (cannot pass).
            Used for "screw the dealer" rule.

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass (only if must_choose=False).

        Raises
        ------
        ValueError
            If must_choose=True and None is returned.
        """
        pass

    @abstractmethod
    def choose_card_to_discard(self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None) -> Card:
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
        pass

    @abstractmethod
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
            Player IDs who played each card in trick_cards (same order).

        Returns
        -------
        Card
            The card to play.
        """
        pass


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
        """
        Get user input for ordering up.

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
        """
        Get user input for calling trump.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up.
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit (cannot pass).

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        if self.tui is None:
            raise RuntimeError("TUI not set for human profile")
        result = self.tui.get_call_trump_decision(player, turned_card, must_choose)
        if must_choose and result is None:
            raise ValueError("Must choose a suit when must_choose=True")
        return result

    def choose_card_to_discard(self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None) -> Card:
        """
        Get user input for discarding a card.

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
        """
        Get user input for playing a card.

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
            Player IDs who played each card in trick_cards (same order).

        Returns
        -------
        Card
            The card to play.
        """
        if self.tui is None:
            raise RuntimeError("TUI not set for human profile")
        return self.tui.get_play_card_decision(player, led_suit, trump_suit, trick_cards)


class ComputerPlayer(PlayerProfile, ABC):
    """Base class for all computer-controlled players."""

    def __init__(self) -> None:
        """Initialize the computer player."""
        self.rules = RulesEngine()

    def _count_trump_cards(self, hand: List[Card], trump_suit: Suit) -> int:
        """
        Count how many trump cards are in a hand.

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        int
            Number of trump cards.
        """
        count = 0
        for card in hand:
            # Right Bower (Jack of trump suit)
            if card.rank == Rank.JACK and card.suit == trump_suit:
                count += 1
            # Left Bower (Jack of same color as trump)
            elif card.rank == Rank.JACK:
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    count += 1
            # Regular trump suit card
            elif card.suit == trump_suit:
                count += 1
        return count

    def _has_bower(self, hand: List[Card], trump_suit: Suit) -> bool:
        """
        Check if hand has a bower (Right or Left).

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        bool
            True if hand has a bower, False otherwise.
        """
        for card in hand:
            if card.rank == Rank.JACK:
                # Right Bower
                if card.suit == trump_suit:
                    return True
                # Left Bower
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    return True
        return False

    def _has_strong_trump(self, hand: List[Card], trump_suit: Suit) -> bool:
        """
        Check if hand has strong trump cards (Ace or King).

        Parameters
        ----------
        hand : List[Card]
            The hand to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        bool
            True if hand has strong trump, False otherwise.
        """
        for card in hand:
            if card.suit == trump_suit and card.rank in [Rank.ACE, Rank.KING]:
                return True
        return False

    def _card_value(self, card: Card, trump_suit: Optional[Suit], led_suit: Optional[Suit] = None) -> int:
        """
        Get the value of a card for comparison.

        Parameters
        ----------
        card : Card
            The card to value.
        trump_suit : Optional[Suit]
            The trump suit.
        led_suit : Optional[Suit]
            The led suit (unused but kept for compatibility).

        Returns
        -------
        int
            Card value (higher is better).
        """
        if trump_suit is not None:
            # Right Bower (Jack of trump suit) is highest
            if card.rank == Rank.JACK and card.suit == trump_suit:
                return 7
            # Left Bower (Jack of same color as trump) is second
            if card.rank == Rank.JACK:
                trump_card = Card(trump_suit, Rank.ACE)
                if card.is_same_color(trump_card):
                    return 6
            # Regular trump suit cards
            if card.suit == trump_suit:
                rank_map = {
                    Rank.ACE: 5,
                    Rank.KING: 4,
                    Rank.QUEEN: 3,
                    Rank.TEN: 2,
                    Rank.NINE: 1,
                }
                return rank_map.get(card.rank, 0)
        return card.rank.value

    def _can_win_trick(
        self, card: Card, trick_cards: List[Card], led_suit: Suit, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Check if a card can win the current trick.

        Parameters
        ----------
        card : Card
            The card to check.
        trick_cards : List[Card]
            Cards already played.
        led_suit : Suit
            The led suit.
        trump_suit : Optional[Suit]
            The trump suit.

        Returns
        -------
        bool
            True if card can win, False otherwise.
        """
        if not trick_cards:
            return True

        current_winner = trick_cards[0]
        for c in trick_cards[1:]:
            if c.compare_to(current_winner, trump_suit, led_suit) > 0:
                current_winner = c

        return card.compare_to(current_winner, trump_suit, led_suit) > 0

    def _is_trump_card(self, card: Card, trump_suit: Suit) -> bool:
        """
        Check if a card is a trump card.

        Parameters
        ----------
        card : Card
            The card to check.
        trump_suit : Suit
            The trump suit.

        Returns
        -------
        bool
            True if card is trump, False otherwise.
        """
        # Right Bower
        if card.rank == Rank.JACK and card.suit == trump_suit:
            return True
        # Left Bower
        if card.rank == Rank.JACK:
            trump_card = Card(trump_suit, Rank.ACE)
            if card.is_same_color(trump_card):
                return True
        # Regular trump
        return card.suit == trump_suit


class HeuristicPlayer(ComputerPlayer):
    """Heuristic-based player using rule-based logic (formerly SimpleRuleBasedProfile)."""

    def __init__(self) -> None:
        """Initialize a heuristic-based player."""
        super().__init__()

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Decide whether to order up using simple rules.

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

        potential_trump = turned_card.suit
        trump_count = self._count_trump_cards(player.hand, potential_trump)

        # Order up if we have 2+ trump cards
        if trump_count >= 2:
            return True

        # Order up if we have Right or Left Bower
        if self._has_bower(player.hand, potential_trump):
            return True

        return False

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Decide which suit to call as trump using simple rules.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up.
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit.

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        if trump_suit is not None:
            if must_choose:
                # Must return something, but trump already set
                return trump_suit
            return None

        forbidden_suit = turned_card.suit
        suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]

        best_suit = None
        best_count = 0

        for suit in suits:
            if suit == forbidden_suit:
                continue
            count = self._count_trump_cards(player.hand, suit)
            if count > best_count:
                best_count = count
                best_suit = suit

        # Call trump if we have at least 2 cards of that suit
        if best_count >= 2:
            return best_suit

        # Call trump if we have a bower
        if best_suit is not None and self._has_bower(player.hand, best_suit):
            return best_suit

        # If must choose, return best suit even if weak
        if must_choose and best_suit is not None:
            return best_suit

        return None

    def choose_card_to_discard(self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None) -> Card:
        """
        Choose a card to discard using simple rules.

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
        if not player.hand:
            raise ValueError("Player has no cards to discard")
        # Discard the lowest card
        return min(player.hand, key=lambda c: c.rank.value)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Choose a card to play using simple rules.

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
            Player IDs who played each card in trick_cards (same order).

        Returns
        -------
        Card
            The card to play.
        """
        valid_cards = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)
        if not valid_cards:
            return player.hand[0]

        # If leading, play highest card
        if led_suit is None:
            return max(valid_cards, key=lambda c: self._card_value(c, trump_suit))

        # Try to win with lowest winning card, or play lowest
        winning_cards = []
        for card in valid_cards:
            if self._can_win_trick(card, trick_cards, led_suit, trump_suit):
                winning_cards.append(card)

        if winning_cards:
            return min(winning_cards, key=lambda c: self._card_value(c, trump_suit))

        # Can't win, play lowest
        return min(valid_cards, key=lambda c: self._card_value(c, trump_suit))



class AIPlayer(ComputerPlayer):
    """AI-based player using the existing AIDecisionMaker (formerly AIBasedProfile)."""

    def __init__(self, ai_decision_maker) -> None:
        """
        Initialize an AI-based player.

        Parameters
        ----------
        ai_decision_maker
            The AIDecisionMaker instance to use.
        """
        super().__init__()
        self.ai = ai_decision_maker

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Use AI to decide whether to order up.

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
        return self.ai.decide_order_up(player, turned_card, dealer_id, trump_suit)

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Use AI to decide which suit to call as trump.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up.
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit.

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        return self.ai.decide_call_trump(player, turned_card, trump_suit, must_choose)

    def choose_card_to_discard(self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None) -> Card:
        """
        Use AI to choose a card to discard.

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
        return self.ai.choose_card_to_discard(player)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Use AI to choose a card to play.

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
            Player IDs who played each card in trick_cards (same order).

        Returns
        -------
        Card
            The card to play.
        """
        return self.ai.play_card(player, led_suit, trump_suit, trick_cards, trick_player_ids)


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
        from src.ml_config import MLConfig

        config = MLConfig()
        self.trump_selection_risk = (
            trump_selection_risk if trump_selection_risk is not None else config.trump_selection_risk
        )
        self.gameplay_risk = gameplay_risk if gameplay_risk is not None else config.gameplay_risk

    def _get_game_state(self) -> tuple[int, int, int]:
        """
        Get current game state information.

        Returns
        -------
        tuple[int, int, int]
            (trick_number, tricks_won_team0, tricks_won_team1)
        """
        if self.game_state_provider is not None:
            return self.game_state_provider()
        # Default values if no provider
        return (0, 0, 0)

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Use ML model to decide whether to order up.

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

        trick_num, tricks_won_team0, tricks_won_team1 = self._get_game_state()

        # Encode game state
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

        # Run inference
        with torch.no_grad():
            features = features.unsqueeze(0).to(self.ml_model.device)
            order_up_logits, _ = self.ml_model.trump_net(features)
            order_up_prob = torch.sigmoid(order_up_logits).item()

        # Apply temperature threshold: weight >= temperature means order up
        return order_up_prob >= self.trump_selection_risk

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Use ML model to decide which suit to call as trump.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up.
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit.

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        if trump_suit is not None:
            if must_choose:
                return trump_suit
            return None

        trick_num, tricks_won_team0, tricks_won_team1 = self._get_game_state()

        # Encode game state
        features = self.encoder.encode_game_state(
            hand=player.hand,
            turned_card=turned_card,
            trump_suit=None,
            led_suit=None,
            trick_cards=[],
            player_id=player.player_id,
            dealer_id=0,  # Will be set correctly by caller if needed
            team=player.team,
            trick_number=trick_num,
            tricks_won_team0=tricks_won_team0,
            tricks_won_team1=tricks_won_team1,
        )

        # Run inference
        with torch.no_grad():
            features = features.unsqueeze(0).to(self.ml_model.device)
            _, call_trump_logits = self.ml_model.trump_net(features)
            call_trump_probs = torch.softmax(call_trump_logits, dim=1).squeeze(0)

        # Map probabilities to suits: [pass, HEARTS, DIAMONDS, CLUBS, SPADES]
        forbidden_suit = turned_card.suit
        suits = [None, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]

        # Get decision weights and apply temperature threshold
        from src.ml_decision_weights import apply_temperature_threshold, get_decision_weights_from_logits

        weights = get_decision_weights_from_logits(call_trump_logits.squeeze(0))
        valid_indices = [i for i, suit in enumerate(suits) if suit != forbidden_suit]

        filtered_indices, filtered_weights = apply_temperature_threshold(
            weights, self.trump_selection_risk, valid_indices
        )

        if filtered_indices:
            # Select best suit from filtered options
            best_idx = max(filtered_indices, key=lambda i: filtered_weights[i])
        else:
            # Fallback: use original method
            best_idx = 0  # Start with pass
            best_prob = call_trump_probs[0].item()
            for i, suit in enumerate(suits[1:], start=1):
                if suit != forbidden_suit and call_trump_probs[i].item() > best_prob:
                    best_prob = call_trump_probs[i].item()
                    best_idx = i

        if must_choose and best_idx == 0:
            # Must choose, find best non-forbidden suit
            for i, suit in enumerate(suits[1:], start=1):
                if suit != forbidden_suit:
                    return suit

        if best_idx == 0:
            return None
        return suits[best_idx]

    def choose_card_to_discard(self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None) -> Card:
        """
        Use ML model to choose a card to discard.

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
        if not player.hand:
            raise ValueError("Player has no cards to discard")

        if len(player.hand) != 6:
            # Fallback if hand size is unexpected
            return min(player.hand, key=lambda c: c.rank.value)

        trick_num, tricks_won_team0, tricks_won_team1 = self._get_game_state()

        # Encode game state (dealer has 6 cards after picking up)
        features = self.encoder.encode_game_state(
            hand=player.hand,
            turned_card=None,
            trump_suit=None,  # Will be set, but for discard we might not have it yet
            led_suit=None,
            trick_cards=[],
            player_id=player.player_id,
            dealer_id=player.player_id,
            team=player.team,
            trick_number=trick_num,
            tricks_won_team0=tricks_won_team0,
            tricks_won_team1=tricks_won_team1,
        )

        # Run inference
        with torch.no_grad():
            features = features.unsqueeze(0).to(self.ml_model.device)
            discard_logits = self.ml_model.discard_net(features)
            discard_probs = torch.softmax(discard_logits, dim=1).squeeze(0)

        # Get decision weights and apply temperature threshold for gameplay decisions
        from src.ml_decision_weights import get_decision_weights_from_logits, select_action_from_weights

        weights = get_decision_weights_from_logits(discard_logits.squeeze(0))
        hand_indices = list(range(len(player.hand)))

        selected_idx, _ = select_action_from_weights(weights, hand_indices, self.gameplay_risk)

        if selected_idx < len(player.hand):
            return player.hand[selected_idx]

        # Fallback
        return min(player.hand, key=lambda c: c.rank.value)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Use ML model to choose a card to play.

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
        valid_cards = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)
        if not valid_cards:
            return player.hand[0]

        trick_num, tricks_won_team0, tricks_won_team1 = self._get_game_state()

        # Encode game state
        features = self.encoder.encode_game_state(
            hand=player.hand,
            turned_card=None,
            trump_suit=trump_suit,
            led_suit=led_suit,
            trick_cards=trick_cards,
            player_id=player.player_id,
            dealer_id=0,  # Not critical for card play
            team=player.team,
            trick_number=trick_num,
            tricks_won_team0=tricks_won_team0,
            tricks_won_team1=tricks_won_team1,
        )

        # Run inference
        with torch.no_grad():
            features = features.unsqueeze(0).to(self.ml_model.device)
            card_logits = self.ml_model.card_play_net(features)
            card_probs = torch.softmax(card_logits, dim=1).squeeze(0)

        # Get valid card indices
        valid_indices = self.encoder.get_valid_card_indices(valid_cards)

        # Get decision weights and apply temperature threshold for gameplay decisions
        from src.ml_decision_weights import get_decision_weights_from_logits, select_action_from_weights

        weights = get_decision_weights_from_logits(card_logits.squeeze(0))
        selected_idx, _ = select_action_from_weights(weights, valid_indices, self.gameplay_risk)

        selected_card = self.encoder.decode_card_index(selected_idx)

        # Verify selected card is valid
        if selected_card in valid_cards:
            return selected_card

        # Fallback to first valid card
        return valid_cards[0]


class RandomPlayer(ComputerPlayer):
    """Random player that makes all decisions randomly (formerly RandomProfile)."""

    def __init__(self) -> None:
        """Initialize a random player."""
        super().__init__()

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Randomly decide whether to order up the turned card.

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
            Randomly True to order up, False to pass.
        """
        return random.choice([True, False])

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Randomly decide which suit to call as trump (or pass).

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
            Randomly chosen suit, or None to pass (only if must_choose=False).
        """
        if trump_suit is not None:
            return None

        forbidden_suit = turned_card.suit
        available_suits = [suit for suit in Suit if suit != forbidden_suit]

        if must_choose:
            # Must choose a suit
            return random.choice(available_suits)
        else:
            # Can pass or choose a suit
            choices: List[Optional[Suit]] = [None] + available_suits
            return random.choice(choices)

    def choose_card_to_discard(self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None) -> Card:
        """
        Randomly choose a card to discard.

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
            A randomly chosen card from the hand.
        """
        if not player.hand:
            raise ValueError("Player has no cards to discard")
        return random.choice(player.hand)

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Randomly choose a card to play from valid plays.

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
            Player IDs who played each card in trick_cards (same order).

        Returns
        -------
        Card
            A randomly chosen valid card to play.
        """
        valid_cards = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)

        if not valid_cards:
            # Fallback if no valid cards (should not happen)
            return player.hand[0]

        return random.choice(valid_cards)


# Backward compatibility aliases
SimpleRuleBasedProfile = HeuristicPlayer
AIBasedProfile = AIPlayer
RandomProfile = RandomPlayer
