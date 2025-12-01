"""Transformer RL player profile for Euchre game integration.

This module provides a PlayerProfile implementation that uses the transformer
RL agent for decision-making, with perfect memory and deduction support.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import torch

from eucher.cards import Card, Suit
from eucher.players.profiles import PlayerProfile
from eucher.rules import RulesEngine

from eucher.players.computer.ml.pytorch.game_state_tracker import TrickHistoryTracker
from src.ai_players.transformer_rl.actor_critic_agent import ActorCriticAgent
from src.ai_players.transformer_rl.deduction_engine import DeductionEngine
from src.ai_players.transformer_rl.feature_encoder import TransformerRLFeatureEncoder

if TYPE_CHECKING:
    from eucher.players import Player


class TransformerRLPlayer(PlayerProfile):
    """Transformer RL player profile with perfect memory and deduction.

    Parameters
    ----------
    model_path : Optional[str]
        Path to pre-trained model. If None, uses random initialization.
    device : Optional[str]
        Device to run on ("cpu", "cuda", or None for auto).
    risk_factor : float
        Risk factor for decision-making (0.0-1.0).
    use_deduction : bool
        Whether to use deduction engine.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        risk_factor: float = 0.5,
        use_deduction: bool = True,
    ) -> None:
        """Initialize transformer RL player.

        Parameters
        ----------
        model_path : Optional[str]
            Path to model file.
        device : Optional[str]
            Device string.
        risk_factor : float
            Risk factor (0.0-1.0).
        use_deduction : bool
            Enable deduction engine.
        """
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize components
        self.feature_encoder = TransformerRLFeatureEncoder(device=self.device)
        self.tracker = TrickHistoryTracker()
        self.deduction_engine = DeductionEngine() if use_deduction else None
        self.rules = RulesEngine()
        self.risk_factor = risk_factor
        self.use_deduction = use_deduction

        # Create agent
        self.agent = ActorCriticAgent(device=self.device)

        # Load model if provided
        if model_path:
            try:
                self.agent.load(Path(model_path))
            except Exception:
                print(f"Failed to load model from {model_path}, using random initialization")

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
        state = self.feature_encoder.encode_full_state_with_deduction(
            hand=player.hand,
            trick_history=self.trick_history,
            tracker=self.tracker,
            deduction_engine=self.deduction_engine if self.use_deduction else None,
            trick_number=0,
            team_score=self.team_scores[player.team],
            opponent_score=self.team_scores[1 - player.team],
            player_position=player.player_id,
            dealer_id=dealer_id,
            trump_suit=turned_card.suit,  # Potential trump
            turned_card=turned_card,
            risk_factor=self.risk_factor,
        )

        # Get model prediction
        action, _, _ = self.agent.select_action(
            state=state,
            action_type="bidding",
            valid_actions=[0, 1],  # 0 = pass, 1 = order up
            training=False,
        )

        return action == 1

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
            state = self.feature_encoder.encode_full_state_with_deduction(
                hand=player.hand,
                trick_history=self.trick_history,
                tracker=self.tracker,
                deduction_engine=self.deduction_engine if self.use_deduction else None,
                trick_number=0,
                team_score=self.team_scores[player.team],
                opponent_score=self.team_scores[1 - player.team],
                player_position=player.player_id,
                dealer_id=self.dealer_id,
                trump_suit=suit,
                turned_card=turned_card,
                risk_factor=self.risk_factor,
            )

            # Get model prediction
            with torch.no_grad():
                # Move optional tensors to device if they exist
                deduction_maps = state.get("deduction_maps", None)
                if deduction_maps is not None and isinstance(deduction_maps, torch.Tensor):
                    deduction_maps = deduction_maps.to(self.device)
                
                current_trick = state.get("current_trick", None)
                if current_trick is not None and isinstance(current_trick, torch.Tensor):
                    current_trick = current_trick.to(self.device)
                
                outputs = self.agent.network(
                    hand=state["hand"].unsqueeze(0).to(self.device),
                    trick_history=state["trick_history"].unsqueeze(0).to(self.device),
                    game_context=state["game_context"].unsqueeze(0).to(self.device),
                    deduction_maps=deduction_maps,
                    current_trick=current_trick,
                )

                # Get trump selection logits (5 classes: 4 suits + pass)
                trump_logits = outputs["call_trump"][0]
                suit_idx = list(Suit).index(suit)
                score = trump_logits[suit_idx].item()

                if score > best_score:
                    best_score = score
                    best_suit = suit

        # Check if pass is better
        if not must_choose:
            state = self.feature_encoder.encode_full_state_with_deduction(
                hand=player.hand,
                trick_history=self.trick_history,
                tracker=self.tracker,
                deduction_engine=self.deduction_engine if self.use_deduction else None,
                trick_number=0,
                team_score=self.team_scores[player.team],
                opponent_score=self.team_scores[1 - player.team],
                player_position=player.player_id,
                dealer_id=self.dealer_id,
                trump_suit=None,
                turned_card=turned_card,
                risk_factor=self.risk_factor,
            )

            with torch.no_grad():
                # Move optional tensors to device if they exist
                deduction_maps = state.get("deduction_maps", None)
                if deduction_maps is not None and isinstance(deduction_maps, torch.Tensor):
                    deduction_maps = deduction_maps.to(self.device)
                
                current_trick = state.get("current_trick", None)
                if current_trick is not None and isinstance(current_trick, torch.Tensor):
                    current_trick = current_trick.to(self.device)
                
                outputs = self.agent.network(
                    hand=state["hand"].unsqueeze(0).to(self.device),
                    trick_history=state["trick_history"].unsqueeze(0).to(self.device),
                    game_context=state["game_context"].unsqueeze(0).to(self.device),
                    deduction_maps=deduction_maps,
                    current_trick=current_trick,
                )

                pass_score = outputs["call_trump"][0, 4].item()  # Pass is index 4

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

        # Encode game state
        state = self.feature_encoder.encode_full_state_with_deduction(
            hand=player.hand,
            trick_history=self.trick_history,
            tracker=self.tracker,
            deduction_engine=self.deduction_engine if self.use_deduction else None,
            trick_number=0,
            team_score=self.team_scores[player.team],
            opponent_score=self.team_scores[1 - player.team],
            player_position=player.player_id,
            dealer_id=self.dealer_id,
            trump_suit=turned_card.suit if turned_card else None,
            turned_card=turned_card,
            risk_factor=self.risk_factor,
        )

        # Get model prediction
        action, _, _ = self.agent.select_action(
            state=state,
            action_type="discard",
            valid_actions=list(range(len(player.hand))),
            training=False,
        )

        # Select card from hand
        if action >= len(player.hand):
            action = 0

        return player.hand[action]

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
        # Get valid plays
        valid_plays = self.rules.get_valid_plays(player.hand, led_suit, trump_suit)
        if not valid_plays:
            return player.hand[0]

        # Encode game state
        state = self.feature_encoder.encode_full_state_with_deduction(
            hand=player.hand,
            trick_history=self.trick_history,
            tracker=self.tracker,
            deduction_engine=self.deduction_engine if self.use_deduction else None,
            trick_number=self.current_trick_number,
            team_score=self.team_scores[player.team],
            opponent_score=self.team_scores[1 - player.team],
            player_position=player.player_id,
            dealer_id=self.dealer_id,
            trump_suit=trump_suit,
            current_trick=trick_cards,
            risk_factor=self.risk_factor,
        )

        # Get model prediction
        valid_indices = [i for i, card in enumerate(player.hand) if card in valid_plays]
        if not valid_indices:
            return valid_plays[0]

        # Ensure state tensors are on correct device
        if "deduction_maps" in state and state["deduction_maps"] is not None:
            if isinstance(state["deduction_maps"], torch.Tensor):
                state["deduction_maps"] = state["deduction_maps"].to(self.device)
        if "current_trick" in state and state["current_trick"] is not None:
            if isinstance(state["current_trick"], torch.Tensor):
                state["current_trick"] = state["current_trick"].to(self.device)

        action, _, _ = self.agent.select_action(
            state=state,
            action_type="play",
            valid_actions=valid_indices,
            training=False,
        )

        # Map action to card
        if action < len(player.hand) and player.hand[action] in valid_plays:
            return player.hand[action]

        # Fallback
        return valid_plays[0]

    def record_trick_completion(
        self, played_cards: List[Card], player_ids: List[int], winner_team: int, winner_player_id: int
    ) -> None:
        """Record a completed trick.

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

        # Update deduction engine
        if self.use_deduction and self.deduction_engine:
            for card, pid in zip(played_cards, player_ids):
                self.deduction_engine.record_play(pid, card)

        self.current_trick_number += 1

    def reset_for_new_hand(self) -> None:
        """Reset state for a new hand."""
        self.tracker.reset()
        self.trick_history.clear()
        self.current_trick_number = 0
        if self.use_deduction and self.deduction_engine:
            self.deduction_engine.reset()

    def update_scores(self, team_scores: List[int]) -> None:
        """Update team scores.

        Parameters
        ----------
        team_scores : List[int]
            Current team scores [team0, team1].
        """
        self.team_scores = team_scores.copy()

    def record_deal(self, player_id: int, cards: List[Card]) -> None:
        """Record cards dealt to a player.

        Parameters
        ----------
        player_id : int
            Player ID.
        cards : List[Card]
            Cards dealt.
        """
        if self.use_deduction and self.deduction_engine:
            self.deduction_engine.record_deal(player_id, cards)

    def decide_going_alone(self, player: "Player", trump_suit: Suit) -> bool:
        """
        Decide whether to go alone after making trump.

        Parameters
        ----------
        player : Player
            The player making the decision (must be the trump maker).
        trump_suit : Suit
            The trump suit that was selected.

        Returns
        -------
        bool
            True to go alone, False to play with partner.
        """
        # Encode game state
        state = self.feature_encoder.encode_full_state_with_deduction(
            hand=player.hand,
            trick_history=self.trick_history,
            tracker=self.tracker,
            deduction_engine=self.deduction_engine if self.use_deduction else None,
            trick_number=0,
            team_score=self.team_scores[player.team],
            opponent_score=self.team_scores[1 - player.team],
            player_position=player.player_id,
            dealer_id=self.dealer_id,
            trump_suit=trump_suit,
            turned_card=None,
            risk_factor=self.risk_factor,
        )

        # Get model prediction for going alone decision
        action, _, _ = self.agent.select_action(
            state=state,
            action_type="go_alone",
            valid_actions=[0, 1],  # 0 = play with partner, 1 = go alone
            training=False,
        )

        return action == 1

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
        # This can be improved with ML model prediction later
        # The transformer RL model would need to be extended to support trade-in decisions
        return False

