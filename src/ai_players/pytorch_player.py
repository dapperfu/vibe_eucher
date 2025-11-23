"""PyTorch-based strategic AI player profile."""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import torch

from src.cards import Card, Suit
from src.player_profiles import PlayerProfile
from src.rules import RulesEngine

from src.ai_players.feature_encoder import EuchreFeatureEncoder
from src.ai_players.game_state_tracker import BowerProbabilityEstimator, TrickHistoryTracker
from src.ai_players.model_manager import ModelManager
from src.ai_players.pytorch_networks import HybridNetwork, create_network
from src.ai_players.strategic_engine import BowerDrawingStrategy, CardEvaluationSystem

if TYPE_CHECKING:
    from src.players import Player


class PyTorchStrategicPlayer(PlayerProfile):
    """PyTorch-based strategic AI player with bower drawing logic.

    Parameters
    ----------
    model_path : Optional[str]
        Path to pre-trained model. If None, uses random initialization.
    device : Optional[str]
        Device to run on ("cpu", "cuda", or None for auto).
    use_strategic_overrides : bool
        Whether to use strategic overrides (bower drawing).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_strategic_overrides: bool = True,
        trump_selection_risk: Optional[float] = None,
        gameplay_risk: Optional[float] = None,
    ) -> None:
        """Initialize PyTorch strategic player.

        Parameters
        ----------
        model_path : Optional[str]
            Path to model file.
        device : Optional[str]
            Device string.
        use_strategic_overrides : bool
            Enable strategic overrides.
        trump_selection_risk : Optional[float]
            Risk factor for trump selection (0.0-1.0). If None, uses default 0.5.
        gameplay_risk : Optional[float]
            Risk factor for gameplay (0.0-1.0). If None, uses default 0.5.
        """
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize components
        self.model_manager = ModelManager()
        self.feature_encoder = EuchreFeatureEncoder()
        self.tracker = TrickHistoryTracker()
        self.bower_estimator = BowerProbabilityEstimator(self.tracker)
        self.rules = RulesEngine()
        self.card_evaluator = CardEvaluationSystem(self.rules)
        self.bower_strategy = BowerDrawingStrategy(
            self.tracker, self.bower_estimator, self.card_evaluator
        )

        self.use_strategic_overrides = use_strategic_overrides

        # Set risk factors (temperature thresholds)
        from src.ml_config import MLConfig

        config = MLConfig()
        self.trump_selection_risk = (
            trump_selection_risk if trump_selection_risk is not None else config.trump_selection_risk
        )
        self.gameplay_risk = (
            gameplay_risk if gameplay_risk is not None else config.gameplay_risk
        )

        # Load or create model
        if model_path:
            try:
                self.model, _ = self.model_manager.load_model(
                    Path(model_path), device=self.device
                )
            except Exception:
                print(f"Failed to load model from {model_path}, using random initialization")
                self.model = create_network(device=self.device)
        else:
            # Try to load existing model
            try:
                self.model, _ = self.model_manager.load_model(device=self.device)
            except FileNotFoundError:
                # Create new model
                self.model = create_network(device=self.device)

        self.model.eval()

        # Game state tracking
        self.current_trick_number = 0
        self.team_scores = [0, 0]
        self.dealer_id = 0
        self.trick_history: List[dict] = []

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """Decide whether to order up the turned card.

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
        self.dealer_id = dealer_id

        # Encode game state
        features = self.feature_encoder.encode_full_state(
            hand=player.hand,
            trick_history=self.trick_history,
            tracker=self.tracker,
            trick_number=0,
            team_score=self.team_scores[player.team],
            opponent_score=self.team_scores[1 - player.team],
            player_position=player.player_id,
            dealer_id=dealer_id,
            trump_suit=turned_card.suit,  # Potential trump
        )

        # Get model prediction
        with torch.no_grad():
            outputs = self.model(
                hand=features["hand"].unsqueeze(0).to(self.device),
                trick_history=features["trick_history"].unsqueeze(0).to(self.device),
                won_tricks_summary=features["won_tricks_summary"].unsqueeze(0).to(self.device),
                game_context=features["game_context"].unsqueeze(0).to(self.device),
            )

            order_up_logit = outputs["order_up"][0, 0].item()
            decision = order_up_logit > 0.0

        return decision

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
            The player making the decision.
        turned_card : Card
            The card that was turned up (cannot be chosen).
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit.

        Returns
        -------
        Optional[Suit]
            The suit to call, or None to pass.
        """
        if trump_suit is not None:
            return trump_suit

        # Encode game state for each possible trump suit
        available_suits = [s for s in Suit if s != turned_card.suit]
        best_suit = None
        best_score = float("-inf")

        for suit in available_suits:
            features = self.feature_encoder.encode_full_state(
                hand=player.hand,
                trick_history=self.trick_history,
                tracker=self.tracker,
                trick_number=0,
                team_score=self.team_scores[player.team],
                opponent_score=self.team_scores[1 - player.team],
                player_position=player.player_id,
                dealer_id=self.dealer_id,
                trump_suit=suit,
            )

            with torch.no_grad():
                outputs = self.model(
                    hand=features["hand"].unsqueeze(0).to(self.device),
                    trick_history=features["trick_history"].unsqueeze(0).to(self.device),
                    won_tricks_summary=features["won_tricks_summary"].unsqueeze(0).to(self.device),
                    game_context=features["game_context"].unsqueeze(0).to(self.device),
                )

                # Get trump selection logits (5 classes: 4 suits + pass)
                trump_logits = outputs["trump_selection"][0]
                suit_idx = list(Suit).index(suit)
                score = trump_logits[suit_idx].item()

                if score > best_score:
                    best_score = score
                    best_suit = suit

        # Check if pass is better
        if not must_choose:
            with torch.no_grad():
                features = self.feature_encoder.encode_full_state(
                    hand=player.hand,
                    trick_history=self.trick_history,
                    tracker=self.tracker,
                    trick_number=0,
                    team_score=self.team_scores[player.team],
                    opponent_score=self.team_scores[1 - player.team],
                    player_position=player.player_id,
                    dealer_id=self.dealer_id,
                    trump_suit=None,
                )

                outputs = self.model(
                    hand=features["hand"].unsqueeze(0).to(self.device),
                    trick_history=features["trick_history"].unsqueeze(0).to(self.device),
                    won_tricks_summary=features["won_tricks_summary"].unsqueeze(0).to(self.device),
                    game_context=features["game_context"].unsqueeze(0).to(self.device),
                )

                pass_score = outputs["trump_selection"][0, 4].item()  # Pass is index 4

                if pass_score > best_score and not must_choose:
                    return None

        return best_suit if best_score > 0 else (available_suits[0] if must_choose else None)

    def choose_card_to_discard(
        self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
    ) -> Card:
        """Choose a card to discard.

        Parameters
        ----------
        player : Player
            The dealer player.
        turned_card : Optional[Card]
            The card that was ordered up.
        ordered_up_by : Optional[str]
            Name of player who ordered up.

        Returns
        -------
        Card
            The card to discard.
        """
        if not player.hand:
            raise ValueError("Player has no cards to discard")

        # Use model to select discard
        features = self.feature_encoder.encode_full_state(
            hand=player.hand,
            trick_history=self.trick_history,
            tracker=self.tracker,
            trick_number=0,
            team_score=self.team_scores[player.team],
            opponent_score=self.team_scores[1 - player.team],
            player_position=player.player_id,
            dealer_id=self.dealer_id,
            trump_suit=turned_card.suit if turned_card else None,
        )

        with torch.no_grad():
            outputs = self.model(
                hand=features["hand"].unsqueeze(0).to(self.device),
                trick_history=features["trick_history"].unsqueeze(0).to(self.device),
                won_tricks_summary=features["won_tricks_summary"].unsqueeze(0).to(self.device),
                game_context=features["game_context"].unsqueeze(0).to(self.device),
            )

            discard_logits = outputs["discard"][0]
            discard_idx = torch.argmax(discard_logits).item()

        # Select card from hand (ensure valid index)
        if discard_idx >= len(player.hand):
            discard_idx = 0

        return player.hand[discard_idx]

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """Choose a card to play in a trick.

        Parameters
        ----------
        player : Player
            The player making the decision.
        led_suit : Optional[Suit]
            The suit that was led.
        trump_suit : Optional[Suit]
            The current trump suit.
        trick_cards : List[Card]
            Cards already played in the trick.
        trick_player_ids : List[int]
            Player IDs who played each card.

        Returns
        -------
        Card
            The card to play.
        """
        # Update trick history if trick was completed
        # (This would be called after trick completion, but we track current trick here)

        # Check for strategic override (bower drawing)
        if (
            self.use_strategic_overrides
            and trump_suit is not None
            and led_suit is None
            and not trick_cards
        ):
            should_draw, draw_card = self.bower_strategy.should_draw_bowers(
                hand=player.hand,
                trump_suit=trump_suit,
                led_suit=None,
                trick_number=self.current_trick_number,
                team_score=self.team_scores[player.team],
                opponent_score=self.team_scores[1 - player.team],
                player_position=player.player_id,
                is_leading=True,
            )

            if should_draw and draw_card and draw_card in player.hand:
                # Validate it's a legal play
                valid_plays = self.rules.get_valid_plays(player.hand, None, trump_suit)
                if draw_card in valid_plays:
                    return draw_card

        # Get valid plays
        valid_plays = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)
        if not valid_plays:
            # Fallback - should not happen
            return player.hand[0]

        # Encode game state
        features = self.feature_encoder.encode_full_state(
            hand=player.hand,
            trick_history=self.trick_history,
            tracker=self.tracker,
            trick_number=self.current_trick_number,
            team_score=self.team_scores[player.team],
            opponent_score=self.team_scores[1 - player.team],
            player_position=player.player_id,
            dealer_id=self.dealer_id,
            trump_suit=trump_suit,
            current_trick=trick_cards,
        )

        # Get model prediction
        with torch.no_grad():
            outputs = self.model(
                hand=features["hand"].unsqueeze(0).to(self.device),
                trick_history=features["trick_history"].unsqueeze(0).to(self.device),
                won_tricks_summary=features["won_tricks_summary"].unsqueeze(0).to(self.device),
                game_context=features["game_context"].unsqueeze(0).to(self.device),
                current_trick=features["current_trick"].unsqueeze(0).to(self.device)
                if trick_cards
                else None,
            )

            card_play_logits = outputs["card_play"][0]

            # Filter to valid plays only
            valid_indices = [i for i, card in enumerate(player.hand) if card in valid_plays]
            if not valid_indices:
                return valid_plays[0]

            # Get scores for valid cards
            valid_scores = [card_play_logits[i].item() for i in valid_indices]
            best_valid_idx = valid_indices[torch.argmax(torch.tensor(valid_scores)).item()]

            return player.hand[best_valid_idx]

    def record_trick_completion(
        self, played_cards: List[Card], player_ids: List[int], winner_team: int, winner_player_id: int
    ) -> None:
        """Record a completed trick (called by game after trick ends).

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
        self.tracker.record_trick(played_cards, player_ids, winner_team, winner_player_id)
        self.trick_history.append(
            {"cards": played_cards, "player_ids": player_ids, "winner_team": winner_team}
        )
        self.current_trick_number += 1

    def reset_for_new_hand(self) -> None:
        """Reset state for a new hand."""
        self.tracker.reset()
        self.trick_history.clear()
        self.current_trick_number = 0

    def update_scores(self, team_scores: List[int]) -> None:
        """Update team scores.

        Parameters
        ----------
        team_scores : List[int]
            Current team scores [team0, team1].
        """
        self.team_scores = team_scores.copy()

