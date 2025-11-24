"""Extract probability distributions from different player types for decision-making.

This module provides functions to extract probability distributions for various
decision types (order up, call trump, discard, play card) from different player
profile types (Random, PyTorch, ML, AI).
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

from src.cards import Card, Suit
from src.ml_decision_weights import get_decision_weights_from_logits, get_decision_weights_from_probs

if TYPE_CHECKING:
    from src.players import Player
    from src.player_profiles import PlayerProfile


def get_order_up_probabilities(
    profile: "PlayerProfile",
    player: "Player",
    turned_card: Card,
    dealer_id: int,
    trump_suit: Optional[Suit],
) -> Dict[bool, float]:
    """
    Extract probability distribution for order up decision.

    Parameters
    ----------
    profile : PlayerProfile
        The player profile making the decision.
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
    from src.player_profiles import RandomPlayer

    # RandomPlayer: uniform distribution
    if isinstance(profile, RandomPlayer):
        return {True: 0.5, False: 0.5}

    # PyTorchStrategicPlayer: extract from logits
    from src.ai_players.pytorch_player import PyTorchStrategicPlayer

    if isinstance(profile, PyTorchStrategicPlayer):
        profile.dealer_id = dealer_id

        # Encode game state
        features = profile.feature_encoder.encode_full_state(
            hand=player.hand,
            trick_history=profile.trick_history,
            tracker=profile.tracker,
            trick_number=0,
            team_score=profile.team_scores[player.team],
            opponent_score=profile.team_scores[1 - player.team],
            player_position=player.player_id,
            dealer_id=dealer_id,
            trump_suit=turned_card.suit,
        )

        # Get model prediction
        with torch.no_grad():
            outputs = profile.model(
                hand=features["hand"].unsqueeze(0).to(profile.device),
                trick_history=features["trick_history"].unsqueeze(0).to(profile.device),
                won_tricks_summary=features["won_tricks_summary"].unsqueeze(0).to(profile.device),
                game_context=features["game_context"].unsqueeze(0).to(profile.device),
            )

            order_up_logit = outputs["order_up"][0, 0].item()
            # Convert logit to probability using sigmoid
            prob_order_up = 1.0 / (1.0 + np.exp(-order_up_logit))
            return {True: float(prob_order_up), False: float(1.0 - prob_order_up)}

    # MLPlayer: extract from predict_proba
    from src.ml_player import MLPlayer

    if isinstance(profile, MLPlayer):
        if profile.backend == "supervised" and profile._is_model_fitted(profile.order_up_classifier.model):
            trick_num, tricks_won_team0, tricks_won_team1 = profile._get_game_state()

            features = profile.encoder.encode_game_state(
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

            X = features.numpy().reshape(1, -1)
            probs = profile.order_up_classifier.predict_proba(X)[0]
            weights = get_decision_weights_from_probs(probs)
            # probs[0] = pass, probs[1] = order up
            return {False: float(weights[0]), True: float(weights[1])}

    # MLBasedProfile: similar to MLPlayer
    from src.player_profiles import MLBasedProfile

    if isinstance(profile, MLBasedProfile):
        trick_num, tricks_won_team0, tricks_won_team1 = profile._get_game_state()

        features = profile.encoder.encode_game_state(
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
            features_tensor = features.unsqueeze(0).to(profile.ml_model.device)
            order_up_logits, _ = profile.ml_model.trump_net(features_tensor)
            # order_up_logits is a single value, use sigmoid to get probability
            order_up_prob = torch.sigmoid(order_up_logits).item()
            return {
                False: float(1.0 - order_up_prob),
                True: float(order_up_prob),
            }

    # AIPlayer/HeuristicPlayer: not probabilistic, return uniform
    return {True: 0.5, False: 0.5}


def get_call_trump_probabilities(
    profile: "PlayerProfile",
    player: "Player",
    turned_card: Card,
    trump_suit: Optional[Suit],
    must_choose: bool = False,
) -> Dict[Optional[Suit], float]:
    """
    Extract probability distribution for call trump decision.

    Parameters
    ----------
    profile : PlayerProfile
        The player profile making the decision.
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
    Dict[Optional[Suit], float]
        Dictionary mapping suit (or None for pass) to probability.
    """
    from src.player_profiles import RandomPlayer

    # RandomPlayer: uniform distribution
    if isinstance(profile, RandomPlayer):
        forbidden_suit = turned_card.suit
        available_suits = [suit for suit in Suit if suit != forbidden_suit]
        if must_choose:
            prob = 1.0 / len(available_suits)
            return {suit: prob for suit in available_suits}
        else:
            choices: List[Optional[Suit]] = [None] + available_suits
            prob = 1.0 / len(choices)
            result: Dict[Optional[Suit], float] = {None: prob}
            for suit in available_suits:
                result[suit] = prob
            return result

    # PyTorchStrategicPlayer: extract from logits
    from src.ai_players.pytorch_player import PyTorchStrategicPlayer

    if isinstance(profile, PyTorchStrategicPlayer):
        forbidden_suit = turned_card.suit
        available_suits = [s for s in Suit if s != forbidden_suit]

        # Get scores for each suit
        suit_scores: Dict[Suit, float] = {}
        for suit in available_suits:
            features = profile.feature_encoder.encode_full_state(
                hand=player.hand,
                trick_history=profile.trick_history,
                tracker=profile.tracker,
                trick_number=0,
                team_score=profile.team_scores[player.team],
                opponent_score=profile.team_scores[1 - player.team],
                player_position=player.player_id,
                dealer_id=profile.dealer_id,
                trump_suit=suit,
            )

            with torch.no_grad():
                outputs = profile.model(
                    hand=features["hand"].unsqueeze(0).to(profile.device),
                    trick_history=features["trick_history"].unsqueeze(0).to(profile.device),
                    won_tricks_summary=features["won_tricks_summary"].unsqueeze(0).to(profile.device),
                    game_context=features["game_context"].unsqueeze(0).to(profile.device),
                )

                trump_logits = outputs["trump_selection"][0]
                suit_idx = list(Suit).index(suit)
                suit_scores[suit] = trump_logits[suit_idx].item()

        # Get pass score
        features = profile.feature_encoder.encode_full_state(
            hand=player.hand,
            trick_history=profile.trick_history,
            tracker=profile.tracker,
            trick_number=0,
            team_score=profile.team_scores[player.team],
            opponent_score=profile.team_scores[1 - player.team],
            player_position=player.player_id,
            dealer_id=profile.dealer_id,
            trump_suit=None,
        )

        with torch.no_grad():
            outputs = profile.model(
                hand=features["hand"].unsqueeze(0).to(profile.device),
                trick_history=features["trick_history"].unsqueeze(0).to(profile.device),
                won_tricks_summary=features["won_tricks_summary"].unsqueeze(0).to(profile.device),
                game_context=features["game_context"].unsqueeze(0).to(profile.device),
            )

            pass_score = outputs["trump_selection"][0, 4].item()  # Pass is index 4

        # Convert scores to probabilities using softmax
        all_scores = list(suit_scores.values()) + [pass_score]
        all_logits = torch.tensor(all_scores)
        probs = torch.softmax(all_logits, dim=0).numpy()

        result: Dict[Optional[Suit], float] = {}
        for i, suit in enumerate(available_suits):
            result[suit] = float(probs[i])
        if not must_choose:
            result[None] = float(probs[-1])
        else:
            result[None] = 0.0

        return result

    # MLPlayer: extract from predict_proba
    from src.ml_player import MLPlayer

    if isinstance(profile, MLPlayer):
        if profile.backend == "supervised" and profile._is_model_fitted(profile.call_trump_classifier.model):
            trick_num, tricks_won_team0, tricks_won_team1 = profile._get_game_state()

            features = profile.encoder.encode_game_state(
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

            X = features.numpy().reshape(1, -1)
            probs = profile.call_trump_classifier.predict_proba(X)[0]
            weights = get_decision_weights_from_probs(probs)

            # Map to suits: [pass, HEARTS, DIAMONDS, CLUBS, SPADES]
            suits = [None, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
            forbidden_suit = turned_card.suit

            result: Dict[Optional[Suit], float] = {}
            for i, suit in enumerate(suits):
                if suit != forbidden_suit:
                    result[suit] = float(weights[i])
            if must_choose:
                result[None] = 0.0

            return result

    # MLBasedProfile: similar to MLPlayer
    from src.player_profiles import MLBasedProfile

    if isinstance(profile, MLBasedProfile):
        trick_num, tricks_won_team0, tricks_won_team1 = profile._get_game_state()

        features = profile.encoder.encode_game_state(
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
            features_tensor = features.unsqueeze(0).to(profile.ml_model.device)
            _, call_trump_logits = profile.ml_model.trump_net(features_tensor)
            call_trump_probs = torch.softmax(call_trump_logits, dim=1).squeeze(0)

            # Map to suits: [pass, HEARTS, DIAMONDS, CLUBS, SPADES]
            suits = [None, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
            forbidden_suit = turned_card.suit

            result: Dict[Optional[Suit], float] = {}
            for i, suit in enumerate(suits):
                if suit != forbidden_suit:
                    result[suit] = float(call_trump_probs[i].item())
            if must_choose:
                result[None] = 0.0

            return result

    # AIPlayer/HeuristicPlayer: not probabilistic, return uniform
    forbidden_suit = turned_card.suit
    available_suits = [suit for suit in Suit if suit != forbidden_suit]
    if must_choose:
        prob = 1.0 / len(available_suits)
        return {suit: prob for suit in available_suits}
    else:
        choices: List[Optional[Suit]] = [None] + available_suits
        prob = 1.0 / len(choices)
        result = {None: prob}
        for suit in available_suits:
            result[suit] = prob
        return result


def get_discard_probabilities(
    profile: "PlayerProfile",
    player: "Player",
    turned_card: Optional[Card] = None,
    ordered_up_by: Optional[str] = None,
) -> Dict[Card, float]:
    """
    Extract probability distribution for discard decision.

    Parameters
    ----------
    profile : PlayerProfile
        The player profile making the decision.
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
    from src.player_profiles import RandomPlayer

    # RandomPlayer: uniform distribution
    if isinstance(profile, RandomPlayer):
        prob = 1.0 / len(player.hand)
        return {card: prob for card in player.hand}

    # PyTorchStrategicPlayer: extract from logits
    from src.ai_players.pytorch_player import PyTorchStrategicPlayer

    if isinstance(profile, PyTorchStrategicPlayer):
        features = profile.feature_encoder.encode_full_state(
            hand=player.hand,
            trick_history=profile.trick_history,
            tracker=profile.tracker,
            trick_number=0,
            team_score=profile.team_scores[player.team],
            opponent_score=profile.team_scores[1 - player.team],
            player_position=player.player_id,
            dealer_id=profile.dealer_id,
            trump_suit=turned_card.suit if turned_card else None,
        )

        with torch.no_grad():
            outputs = profile.model(
                hand=features["hand"].unsqueeze(0).to(profile.device),
                trick_history=features["trick_history"].unsqueeze(0).to(profile.device),
                won_tricks_summary=features["won_tricks_summary"].unsqueeze(0).to(profile.device),
                game_context=features["game_context"].unsqueeze(0).to(profile.device),
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

    # MLPlayer: extract from predict_proba
    from src.ml_player import MLPlayer

    if isinstance(profile, MLPlayer):
        if profile.backend == "supervised" and profile._is_model_fitted(profile.discard_classifier.model):
            trick_num, tricks_won_team0, tricks_won_team1 = profile._get_game_state()

            features = profile.encoder.encode_game_state(
                hand=player.hand,
                turned_card=turned_card,
                trump_suit=turned_card.suit if turned_card else None,
                led_suit=None,
                trick_cards=[],
                player_id=player.player_id,
                dealer_id=player.player_id,  # Dealer discards
                team=player.team,
                trick_number=trick_num,
                tricks_won_team0=tricks_won_team0,
                tricks_won_team1=tricks_won_team1,
            )

            X = features.numpy().reshape(1, -1)
            probs = profile.discard_classifier.predict_proba(X)[0]
            weights = get_decision_weights_from_probs(probs)

            result: Dict[Card, float] = {}
            for i, card in enumerate(player.hand):
                if i < len(weights):
                    result[card] = float(weights[i])
                else:
                    result[card] = 0.0
            return result

    # MLBasedProfile: similar to MLPlayer
    from src.player_profiles import MLBasedProfile

    if isinstance(profile, MLBasedProfile):
        trick_num, tricks_won_team0, tricks_won_team1 = profile._get_game_state()

        features = profile.encoder.encode_game_state(
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

        with torch.no_grad():
            features_tensor = features.unsqueeze(0).to(profile.ml_model.device)
            discard_logits = profile.ml_model.discard_net(features_tensor)
            discard_probs = torch.softmax(discard_logits, dim=1).squeeze(0)

            result: Dict[Card, float] = {}
            for i, card in enumerate(player.hand):
                if i < len(discard_probs):
                    result[card] = float(discard_probs[i].item())
                else:
                    result[card] = 0.0
            return result

    # AIPlayer/HeuristicPlayer: not probabilistic, return uniform
    prob = 1.0 / len(player.hand)
    return {card: prob for card in player.hand}


def get_play_card_probabilities(
    profile: "PlayerProfile",
    player: "Player",
    led_suit: Optional[Suit],
    trump_suit: Optional[Suit],
    trick_cards: List[Card],
    trick_player_ids: List[int],
) -> Dict[Card, float]:
    """
    Extract probability distribution for play card decision.

    Parameters
    ----------
    profile : PlayerProfile
        The player profile making the decision.
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
    from src.rules import RulesEngine

    rules = RulesEngine()
    valid_cards = rules.get_valid_plays(player.hand, led_suit, trump_suit)

    if not valid_cards:
        return {}

    from src.player_profiles import RandomPlayer

    # RandomPlayer: uniform distribution
    if isinstance(profile, RandomPlayer):
        prob = 1.0 / len(valid_cards)
        return {card: prob for card in valid_cards}

    # PyTorchStrategicPlayer: extract from logits
    from src.ai_players.pytorch_player import PyTorchStrategicPlayer

    if isinstance(profile, PyTorchStrategicPlayer):
        features = profile.feature_encoder.encode_full_state(
            hand=player.hand,
            trick_history=profile.trick_history,
            tracker=profile.tracker,
            trick_number=profile.current_trick_number,
            team_score=profile.team_scores[player.team],
            opponent_score=profile.team_scores[1 - player.team],
            player_position=player.player_id,
            dealer_id=profile.dealer_id,
            trump_suit=trump_suit,
            current_trick=trick_cards,
        )

        with torch.no_grad():
            outputs = profile.model(
                hand=features["hand"].unsqueeze(0).to(profile.device),
                trick_history=features["trick_history"].unsqueeze(0).to(profile.device),
                won_tricks_summary=features["won_tricks_summary"].unsqueeze(0).to(profile.device),
                game_context=features["game_context"].unsqueeze(0).to(profile.device),
                current_trick=features["current_trick"].unsqueeze(0).to(profile.device)
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

    # MLPlayer: extract from predict_proba
    from src.ml_player import MLPlayer

    if isinstance(profile, MLPlayer):
        if profile.backend == "supervised" and profile._is_model_fitted(profile.play_card_classifier.model):
            trick_num, tricks_won_team0, tricks_won_team1 = profile._get_game_state()

            features = profile.encoder.encode_game_state(
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

            X = features.numpy().reshape(1, -1)
            probs = profile.play_card_classifier.predict_proba(X)[0]
            weights = get_decision_weights_from_probs(probs)

            # Get valid card indices
            valid_indices = profile.encoder.get_valid_card_indices(valid_cards)

            result: Dict[Card, float] = {}
            total_prob = 0.0
            for idx in valid_indices:
                if idx < len(weights):
                    card = profile.encoder.decode_card_index(idx)
                    if card and card in valid_cards:
                        prob = float(weights[idx])
                        result[card] = prob
                        total_prob += prob

            # Normalize if needed
            if total_prob > 0 and abs(total_prob - 1.0) > 0.01:
                for card in result:
                    result[card] /= total_prob

            return result

    # MLBasedProfile: similar to MLPlayer
    from src.player_profiles import MLBasedProfile

    if isinstance(profile, MLBasedProfile):
        trick_num, tricks_won_team0, tricks_won_team1 = profile._get_game_state()

        features = profile.encoder.encode_game_state(
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
            features_tensor = features.unsqueeze(0).to(profile.ml_model.device)
            card_logits = profile.ml_model.card_play_net(features_tensor)
            card_probs = torch.softmax(card_logits, dim=1).squeeze(0)

            # Get valid card indices
            valid_indices = profile.encoder.get_valid_card_indices(valid_cards)

            result: Dict[Card, float] = {}
            total_prob = 0.0
            for idx in valid_indices:
                if idx < len(card_probs):
                    card = profile.encoder.decode_card_index(idx)
                    if card and card in valid_cards:
                        prob = float(card_probs[idx].item())
                        result[card] = prob
                        total_prob += prob

            # Normalize if needed
            if total_prob > 0 and abs(total_prob - 1.0) > 0.01:
                for card in result:
                    result[card] /= total_prob

            return result

    # AIPlayer/HeuristicPlayer: not probabilistic, return uniform
    prob = 1.0 / len(valid_cards)
    return {card: prob for card in valid_cards}

