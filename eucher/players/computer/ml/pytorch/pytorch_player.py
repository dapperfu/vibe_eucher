"""PyTorch-based strategic AI player profile."""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from eucher.cards import Card, Suit
from eucher.players.profiles import PlayerProfile
from eucher.rules import RulesEngine

from eucher.players.computer.ml.pytorch.feature_encoder import EuchreFeatureEncoder
from eucher.players.computer.ml.pytorch.game_state_tracker import BowerProbabilityEstimator, TrickHistoryTracker
from eucher.players.computer.ml.pytorch.model_manager import ModelManager
from eucher.players.computer.ml.pytorch.pytorch_networks import HybridNetwork, create_network
from eucher.players.computer.ml.pytorch.strategic_engine import BowerDrawingStrategy, CardEvaluationSystem

if TYPE_CHECKING:
    from eucher.players import Player


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

        # Get valid cards to prevent reneges
        valid_cards = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)
        if not valid_cards:
            # Fallback: if no valid cards (shouldn't happen), return first card
            return player.hand[0]

        # Check for strategic override (bower drawing) - but only if it's valid
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

            if should_draw and draw_card and draw_card in valid_cards:
                return draw_card

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

            # Select from valid cards only to prevent reneges
            valid_indices = [i for i, card in enumerate(player.hand) if card in valid_cards]
            if not valid_indices:
                return valid_cards[0]

            # Get scores for valid cards only
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
        # Simple heuristic: check for flush in trump or very strong hand
        trump_count = 0
        for card in player.hand:
            # Right Bower
            if card.rank.value == 11 and card.suit == trump_suit:
                trump_count += 1
            # Left Bower
            elif card.rank.value == 11:
                trump_card = Card(trump_suit, card.rank)
                if card.is_same_color(trump_card):
                    trump_count += 1
            # Regular trump
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
        # Simple heuristic: trade-in if eligible cards are low value (9s or 10s)
        # and we might get better cards from the kitty
        return False  # Conservative: don't trade-in by default

    def get_order_up_probabilities(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> Dict[bool, float]:
        """
        Return probability distribution for order up decision from model logits.

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
            # Convert logit to probability using sigmoid
            prob_order_up = 1.0 / (1.0 + torch.exp(-torch.tensor(order_up_logit))).item()
            return {True: float(prob_order_up), False: float(1.0 - prob_order_up)}

    def get_call_trump_probabilities(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Dict[Optional[Suit], float]:
        """
        Return probability distribution for call trump decision from model logits.

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

        forbidden_suit = turned_card.suit
        available_suits = [suit for suit in Suit if suit != forbidden_suit]

        # Encode game state for each possible trump suit
        suit_scores: Dict[Suit, float] = {}
        pass_score = 0.0

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
                suit_scores[suit] = trump_logits[suit_idx].item()

        # Get pass score
        if not must_choose:
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

            with torch.no_grad():
                outputs = self.model(
                    hand=features["hand"].unsqueeze(0).to(self.device),
                    trick_history=features["trick_history"].unsqueeze(0).to(self.device),
                    won_tricks_summary=features["won_tricks_summary"].unsqueeze(0).to(self.device),
                    game_context=features["game_context"].unsqueeze(0).to(self.device),
                )

                pass_score = outputs["trump_selection"][0, 4].item()  # Pass is index 4

        # Convert scores to probabilities using softmax
        all_scores = [pass_score] + [suit_scores[suit] for suit in available_suits]
        scores_tensor = torch.tensor(all_scores)
        probs_tensor = torch.softmax(scores_tensor, dim=0)

        result: Dict[Optional[Suit], float] = {}
        if not must_choose:
            result[None] = float(probs_tensor[0].item())
        for i, suit in enumerate(available_suits):
            result[suit] = float(probs_tensor[i + (0 if must_choose else 1)].item())

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
        Return probability distribution for play card decision from model logits.

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

            # Get valid indices
            valid_indices = [i for i, card in enumerate(player.hand) if card in valid_cards]
            if not valid_indices:
                return {valid_cards[0]: 1.0}

            # Get scores for valid cards
            valid_scores = [card_play_logits[i].item() for i in valid_indices]
            valid_scores_tensor = torch.tensor(valid_scores)
            valid_probs = torch.softmax(valid_scores_tensor, dim=0)

            result: Dict[Card, float] = {}
            for i, idx in enumerate(valid_indices):
                card = player.hand[idx]
                result[card] = float(valid_probs[i].item())
            return result

    def get_discard_probabilities(
        self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
    ) -> Dict[Card, float]:
        """
        Return probability distribution for discard decision from model logits.

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

        # Use model to get discard probabilities
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
            discard_probs = torch.softmax(discard_logits, dim=0)

            result: Dict[Card, float] = {}
            for i, card in enumerate(player.hand):
                if i < len(discard_probs):
                    result[card] = float(discard_probs[i].item())
                else:
                    result[card] = 0.0
            return result

