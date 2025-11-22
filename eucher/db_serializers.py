"""Functions to convert database entities to dictionaries for easier serialization."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from eucher.cards import Card, Rank, Suit
from eucher.models import (
    CardPlay,
    Game,
    GamePlayer,
    Hand,
    Player,
    PlayerHand,
    Trick,
    TrumpDecision,
    rank_to_string,
    string_to_rank,
    suit_to_string,
    string_to_suit,
)


def player_to_dict(player: Player) -> Dict[str, Any]:
    """
    Convert a Player entity to a dictionary.

    Parameters
    ----------
    player : Player
        Player entity.

    Returns
    -------
    Dict[str, Any]
        Dictionary representation of the player.
    """
    return {
        "name": player.name,
        "created_at": player.created_at.isoformat() if player.created_at else None,
    }


def game_player_to_dict(game_player: GamePlayer) -> Dict[str, Any]:
    """
    Convert a GamePlayer entity to a dictionary.

    Parameters
    ----------
    game_player : GamePlayer
        GamePlayer entity.

    Returns
    -------
    Dict[str, Any]
        Dictionary representation of the game player.
    """
    return {
        "id": game_player.id,
        "player_name": game_player.player.name,
        "player_id": game_player.player_id,
        "team": game_player.team,
        "profile_type": game_player.profile_type,
    }


def game_to_dict(game: Game, include_hands: bool = False) -> Dict[str, Any]:
    """
    Convert a Game entity to a dictionary.

    Parameters
    ----------
    game : Game
        Game entity.
    include_hands : bool
        Whether to include hand data. Default is False.

    Returns
    -------
    Dict[str, Any]
        Dictionary representation of the game.
    """
    result: Dict[str, Any] = {
        "id": game.id,
        "started_at": game.started_at.isoformat() if game.started_at else None,
        "ended_at": game.ended_at.isoformat() if game.ended_at else None,
        "final_team0_score": game.final_team0_score,
        "final_team1_score": game.final_team1_score,
        "winner_team": game.winner_team,
        "players": [game_player_to_dict(gp) for gp in game.game_players],
    }

    if include_hands:
        result["hands"] = [hand_to_dict(h, include_tricks=True) for h in game.hands]

    return result


def player_hand_to_dict(player_hand: PlayerHand) -> Dict[str, Any]:
    """
    Convert a PlayerHand entity to a dictionary.

    Parameters
    ----------
    player_hand : PlayerHand
        PlayerHand entity.

    Returns
    -------
    Dict[str, Any]
        Dictionary representation of the player hand.
    """
    return {
        "id": player_hand.id,
        "player_id": player_hand.player_id,
        "card_suit": player_hand.card_suit,
        "card_rank": player_hand.card_rank,
        "position_in_hand": player_hand.position_in_hand,
    }


def card_play_to_dict(card_play: CardPlay) -> Dict[str, Any]:
    """
    Convert a CardPlay entity to a dictionary.

    Parameters
    ----------
    card_play : CardPlay
        CardPlay entity.

    Returns
    -------
    Dict[str, Any]
        Dictionary representation of the card play.
    """
    return {
        "id": card_play.id,
        "player_id": card_play.player_id,
        "play_order": card_play.play_order,
        "card_suit": card_play.card_suit,
        "card_rank": card_play.card_rank,
    }


def trick_to_dict(trick: Trick, include_card_plays: bool = True) -> Dict[str, Any]:
    """
    Convert a Trick entity to a dictionary.

    Parameters
    ----------
    trick : Trick
        Trick entity.
    include_card_plays : bool
        Whether to include card play data. Default is True.

    Returns
    -------
    Dict[str, Any]
        Dictionary representation of the trick.
    """
    result: Dict[str, Any] = {
        "id": trick.id,
        "trick_number": trick.trick_number,
        "leader_player_id": trick.leader_player_id,
        "led_suit": trick.led_suit,
        "winner_player_id": trick.winner_player_id,
        "winner_team": trick.winner_team,
    }

    if include_card_plays:
        # Sort card plays by play_order
        card_plays = sorted(trick.card_plays, key=lambda cp: cp.play_order)
        result["card_plays"] = [card_play_to_dict(cp) for cp in card_plays]

    return result


def trump_decision_to_dict(trump_decision: TrumpDecision) -> Dict[str, Any]:
    """
    Convert a TrumpDecision entity to a dictionary.

    Parameters
    ----------
    trump_decision : TrumpDecision
        TrumpDecision entity.

    Returns
    -------
    Dict[str, Any]
        Dictionary representation of the trump decision.
    """
    return {
        "id": trump_decision.id,
        "player_id": trump_decision.player_id,
        "round": trump_decision.round,
        "decision_type": trump_decision.decision_type,
        "suit_chosen": trump_decision.suit_chosen,
        "turned_card_suit": trump_decision.turned_card_suit,
        "turned_card_rank": trump_decision.turned_card_rank,
    }


def hand_to_dict(hand: Hand, include_tricks: bool = False) -> Dict[str, Any]:
    """
    Convert a Hand entity to a dictionary.

    Parameters
    ----------
    hand : Hand
        Hand entity.
    include_tricks : bool
        Whether to include trick data. Default is False.

    Returns
    -------
    Dict[str, Any]
        Dictionary representation of the hand.
    """
    result: Dict[str, Any] = {
        "id": hand.id,
        "hand_number": hand.hand_number,
        "dealer_player_id": hand.dealer_player_id,
        "turned_card_suit": hand.turned_card_suit,
        "turned_card_rank": hand.turned_card_rank,
        "trump_suit": hand.trump_suit,
        "trump_maker_player_id": hand.trump_maker_player_id,
        "trump_selection_round": hand.trump_selection_round,
        "tricks_won_team0": hand.tricks_won_team0,
        "tricks_won_team1": hand.tricks_won_team1,
        "points_team0": hand.points_team0,
        "points_team1": hand.points_team1,
        "team0_score_after": hand.team0_score_after,
        "team1_score_after": hand.team1_score_after,
        "player_hands": [
            player_hand_to_dict(ph)
            for ph in sorted(hand.player_hands, key=lambda ph: (ph.player_id, ph.position_in_hand))
        ],
        "trump_decisions": [
            trump_decision_to_dict(td)
            for td in sorted(hand.trump_decisions, key=lambda td: (td.round, td.player_id))
        ],
    }

    if include_tricks:
        # Sort tricks by trick_number
        tricks = sorted(hand.tricks, key=lambda t: t.trick_number)
        result["tricks"] = [trick_to_dict(t, include_card_plays=True) for t in tricks]

    return result


def card_play_to_card(card_play: CardPlay) -> Card:
    """
    Convert a CardPlay entity to a Card object.

    Parameters
    ----------
    card_play : CardPlay
        CardPlay entity.

    Returns
    -------
    Card
        Card object.
    """
    suit = string_to_suit(card_play.card_suit)
    rank = string_to_rank(card_play.card_rank)
    return Card(suit, rank)


def player_hand_to_card(player_hand: PlayerHand) -> Card:
    """
    Convert a PlayerHand entity to a Card object.

    Parameters
    ----------
    player_hand : PlayerHand
        PlayerHand entity.

    Returns
    -------
    Card
        Card object.
    """
    suit = string_to_suit(player_hand.card_suit)
    rank = string_to_rank(player_hand.card_rank)
    return Card(suit, rank)


def serialize_list(items: List[Any], serializer_func: Any) -> List[Dict[str, Any]]:
    """
    Serialize a list of entities using a serializer function.

    Parameters
    ----------
    items : List[Any]
        List of entities to serialize.
    serializer_func
        Function to convert each entity to a dictionary.

    Returns
    -------
    List[Dict[str, Any]]
        List of serialized dictionaries.
    """
    return [serializer_func(item) for item in items]
