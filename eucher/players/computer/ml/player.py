"""ML-based player using sklearn, GAN, or RL models for decision making."""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from src.cards import Card, Suit
from src.computer_player import ComputerPlayer
from src.ml_config import MLConfig
from src.ml_decision_weights import (
    apply_temperature_threshold,
    get_decision_weights_from_probs,
    select_action_from_weights,
)
from src.ml_features import GameStateEncoder
from src.ml_models_supervised import (
    CallTrumpClassifier,
    DiscardCardClassifier,
    OrderUpClassifier,
    PlayCardClassifier,
)
from src.training.profiling import timed_operation

if TYPE_CHECKING:
    from src.players import Player

# Optional imports for GAN and RL
try:
    from src.ml_models_gan import GANCardPlayModel
except ImportError:
    GANCardPlayModel = None  # type: ignore

try:
    from src.ml_models_rl import RLAgent
except ImportError:
    RLAgent = None  # type: ignore


class MLPlayer(ComputerPlayer):
    """ML-based player using sklearn, GAN, or RL models for decision making."""

    def __init__(
        self,
        backend: str = "supervised",
        model_type: str = "random_forest",
        model_dir: Optional[Path] = None,
        game_state_provider=None,
        training_mode: bool = False,
        trump_selection_risk: Optional[float] = None,
        gameplay_risk: Optional[float] = None,
    ) -> None:
        """
        Initialize an ML-based player.

        Parameters
        ----------
        backend : str
            ML backend: "supervised" (sklearn), "gan", or "rl"
        model_type : str
            Type of sklearn model (only for supervised): "random_forest", "gradient_boosting", or "neural_network"
        model_dir : Optional[Path]
            Directory containing trained models. If None, uses default from MLConfig.
        game_state_provider
            Optional callable that provides additional game state (tricks_won, trick_number).
            Should return (trick_number, tricks_won_team0, tricks_won_team1).
        training_mode : bool
            Whether in training mode (affects RL exploration).
        trump_selection_risk : Optional[float]
            Risk factor for trump selection decisions (0.0-1.0). If None, uses config default.
        gameplay_risk : Optional[float]
            Risk factor for gameplay decisions (0.0-1.0). If None, uses config default.
        """
        super().__init__()
        self.backend = backend
        self.model_type = model_type
        self.config = MLConfig()
        self.encoder = GameStateEncoder()
        self.game_state_provider = game_state_provider
        self.training_mode = training_mode

        # Set risk factors (temperature thresholds)
        self.trump_selection_risk = (
            trump_selection_risk if trump_selection_risk is not None else self.config.trump_selection_risk
        )
        self.gameplay_risk = gameplay_risk if gameplay_risk is not None else self.config.gameplay_risk

        # Determine model directory
        if model_dir is None:
            model_dir = self.config.models_dir
        self.model_dir = Path(model_dir)

        # Initialize based on backend
        if backend == "supervised":
            self._init_supervised_models()
        elif backend == "gan":
            self._init_gan_models()
        elif backend == "rl":
            self._init_rl_models()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_supervised_models(self) -> None:
        """Initialize supervised learning models."""
        # Initialize classifiers
        self.order_up_classifier = OrderUpClassifier(self.model_type)
        self.call_trump_classifier = CallTrumpClassifier(self.model_type)
        self.play_card_classifier = PlayCardClassifier(self.model_type)
        self.discard_card_classifier = DiscardCardClassifier(self.model_type)

        # Try to load trained models
        self._load_models()

    def _init_gan_models(self) -> None:
        """Initialize GAN models."""
        if GANCardPlayModel is None:
            raise ImportError("GAN models not available. Install PyTorch.")
        self.gan_model = GANCardPlayModel(self.config)
        gan_path = self.model_dir / "gan_model.pth"
        if gan_path.exists():
            self.gan_model.load(gan_path)

    def _init_rl_models(self) -> None:
        """Initialize RL models."""
        if RLAgent is None:
            raise ImportError("RL models not available. Install PyTorch.")
        self.rl_agent = RLAgent(self.config)
        rl_path = self.model_dir / "rl_model.pth"
        if rl_path.exists():
            self.rl_agent.load(rl_path)

    def _load_models(self) -> None:
        """Load trained models from disk if they exist."""
        order_up_path = self.model_dir / "order_up.pkl"
        call_trump_path = self.model_dir / "call_trump.pkl"
        play_card_path = self.model_dir / "play_card.pkl"
        discard_path = self.model_dir / "discard.pkl"

        if order_up_path.exists():
            self.order_up_classifier.load(order_up_path)
        if call_trump_path.exists():
            self.call_trump_classifier.load(call_trump_path)
        if play_card_path.exists():
            self.play_card_classifier.load(play_card_path)
        if discard_path.exists():
            self.discard_card_classifier.load(discard_path)

    def _is_model_fitted(self, model) -> bool:
        """
        Check if a sklearn model has been fitted.

        Parameters
        ----------
        model
            The sklearn model to check.

        Returns
        -------
        bool
            True if the model is fitted, False otherwise.
        """
        # Check for common fitted attributes in sklearn models
        # RandomForestClassifier and GradientBoostingClassifier use n_features_in_
        # MLPClassifier uses _label_binarizer or out_activation_
        # Most classifiers have classes_ after fitting
        return (
            hasattr(model, "n_features_in_")
            or hasattr(model, "classes_")
            or hasattr(model, "_label_binarizer")
            or hasattr(model, "out_activation_")
        )

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

        # Use appropriate backend
        if self.backend == "supervised":
            # Check if model is fitted before using it
            if self._is_model_fitted(self.order_up_classifier.model):
                # Convert to numpy array for sklearn
                X = features.numpy().reshape(1, -1)
                # Get decision weights (probabilities)
                probs = self.order_up_classifier.predict_proba(X)[0]
                weights = get_decision_weights_from_probs(probs)
                # Apply temperature threshold: weight >= temperature means order up
                order_up_weight = weights[1]  # Probability of ordering up
                return order_up_weight >= self.trump_selection_risk
            else:
                # Model not fitted, use heuristic fallback
                trump_count = self._count_trump_cards(player.hand, turned_card.suit)
                return trump_count >= 2 or self._has_bower(player.hand, turned_card.suit)
        else:
            # For GAN/RL, use heuristic fallback for now
            # (GAN/RL are primarily for card play decisions)
            trump_count = self._count_trump_cards(player.hand, turned_card.suit)
            return trump_count >= 2 or self._has_bower(player.hand, turned_card.suit)

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
            dealer_id=0,  # Not critical for call trump
            team=player.team,
            trick_number=trick_num,
            tricks_won_team0=tricks_won_team0,
            tricks_won_team1=tricks_won_team1,
        )

        # Use appropriate backend
        if self.backend == "supervised":
            # Check if model is fitted before using it
            if self._is_model_fitted(self.call_trump_classifier.model):
                # Convert to numpy array for sklearn
                X = features.numpy().reshape(1, -1)
                # Get decision weights (probabilities for each suit + pass)
                probs = self.call_trump_classifier.predict_proba(X)[0]
                weights = get_decision_weights_from_probs(probs)

                # Map to suits: [pass, HEARTS, DIAMONDS, CLUBS, SPADES]
                suits = [None, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
                valid_indices = [i for i, suit in enumerate(suits) if suit != turned_card.suit]

                # Apply temperature threshold
                filtered_indices, _ = apply_temperature_threshold(weights, self.trump_selection_risk, valid_indices)

                if filtered_indices:
                    # Select best suit from filtered options
                    best_idx = max(filtered_indices, key=lambda i: weights[i])
                    predicted_suit = suits[best_idx]
                else:
                    # Fallback: use original method
                    predicted_suit = self.call_trump_classifier.predict_suit(X, turned_card.suit)
            else:
                # Model not fitted, use heuristic fallback
                suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
                best_suit = None
                best_count = 0
                for suit in suits:
                    if suit != turned_card.suit:
                        count = self._count_trump_cards(player.hand, suit)
                        if count > best_count:
                            best_count = count
                            best_suit = suit
                predicted_suit = best_suit if best_count >= 2 else None
        else:
            # For GAN/RL, use heuristic fallback
            suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
            best_suit = None
            best_count = 0
            for suit in suits:
                if suit != turned_card.suit:
                    count = self._count_trump_cards(player.hand, suit)
                    if count > best_count:
                        best_count = count
                        best_suit = suit
            predicted_suit = best_suit if best_count >= 2 else None

        if must_choose and predicted_suit is None:
            # Must choose, find best available suit
            suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
            for suit in suits:
                if suit != turned_card.suit:
                    return suit
            # Fallback (shouldn't happen)
            return Suit.HEARTS

        return predicted_suit

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

        trick_num, tricks_won_team0, tricks_won_team1 = self._get_game_state()

        # Encode game state
        features = self.encoder.encode_game_state(
            hand=player.hand,
            turned_card=None,
            trump_suit=None,  # May not be set yet
            led_suit=None,
            trick_cards=[],
            player_id=player.player_id,
            dealer_id=player.player_id,
            team=player.team,
            trick_number=trick_num,
            tricks_won_team0=tricks_won_team0,
            tricks_won_team1=tricks_won_team1,
        )

        # Use appropriate backend
        if self.backend == "supervised":
            # Check if model is fitted before using it
            if self._is_model_fitted(self.discard_card_classifier.model):
                # Convert to numpy array for sklearn
                X = features.numpy().reshape(1, -1)
                hand_indices = self.encoder.get_valid_card_indices(player.hand)
                # Get decision weights
                probs = self.discard_card_classifier.predict_proba(X)[0]
                weights = get_decision_weights_from_probs(probs)

                # Apply temperature threshold for gameplay decisions
                predicted_idx, _ = select_action_from_weights(weights, hand_indices, self.gameplay_risk)

                # Find card in hand
                predicted_card = self.encoder.decode_card_index(predicted_idx)
                if predicted_card is not None and predicted_card in player.hand:
                    return predicted_card
            # Model not fitted or prediction failed, use heuristic fallback
            return min(player.hand, key=lambda c: c.rank.value)
        else:
            # For GAN/RL, use heuristic fallback
            return min(player.hand, key=lambda c: c.rank.value)

        # Find card in hand
        predicted_card = self.encoder.decode_card_index(predicted_idx)
        if predicted_card is not None and predicted_card in player.hand:
            return predicted_card

        # Fallback: discard lowest card
        return min(player.hand, key=lambda c: c.rank.value)

    @timed_operation("MLPlayer.play_card")
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

        # Get valid card indices
        valid_indices = self.encoder.get_valid_card_indices(valid_cards)

        # Use appropriate backend
        if self.backend == "supervised":
            # Check if model is fitted before using it
            if self._is_model_fitted(self.play_card_classifier.model):
                # Convert to numpy array for sklearn
                X = features.numpy().reshape(1, -1)
                # Get decision weights
                probs = self.play_card_classifier.predict_proba(X)[0]
                weights = get_decision_weights_from_probs(probs)

                # Apply temperature threshold for gameplay decisions
                predicted_idx, _ = select_action_from_weights(weights, valid_indices, self.gameplay_risk)

                # Find card in valid cards
                predicted_card = self.encoder.decode_card_index(predicted_idx)
                if predicted_card is not None and predicted_card in valid_cards:
                    return predicted_card
            # Model not fitted or prediction failed, use heuristic fallback
            return valid_cards[0]
        elif self.backend == "gan":
            features_np = features.numpy()
            predicted_idx = self.gan_model.predict(features_np, valid_indices)
            # Find card in valid cards
            predicted_card = self.encoder.decode_card_index(predicted_idx)
            if predicted_card is not None and predicted_card in valid_cards:
                return predicted_card
            return valid_cards[0]
        elif self.backend == "rl":
            features_np = features.numpy()
            # RL agent needs to be updated to support temperature thresholds
            # For now, use existing method but could be enhanced
            predicted_idx = self.rl_agent.select_action(
                features_np, valid_indices, training=self.training_mode, temperature=self.gameplay_risk
            )
            # Find card in valid cards
            predicted_card = self.encoder.decode_card_index(predicted_idx)
            if predicted_card is not None and predicted_card in valid_cards:
                return predicted_card
            return valid_cards[0]
        else:
            # Fallback
            return valid_cards[0]

