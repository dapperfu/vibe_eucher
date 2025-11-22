"""Utility functions for saving and loading game state to/from database."""

from datetime import datetime
from typing import List, Optional, Tuple, cast

from pony.orm import commit, db_session

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


@db_session  # type: ignore[misc]
def get_or_create_player(name: str) -> Player:
    """
    Get an existing player or create a new one.

    Parameters
    ----------
    name : str
        Player name.

    Returns
    -------
    Player
        The player entity.
    """
    player = Player.get(name=name)
    if player is None:
        player = Player(name=name, created_at=datetime.now())
        commit()
    assert player is not None  # PonyORM guarantees this after commit
    return cast(Player, player)


@db_session  # type: ignore[misc]
def save_game(
    player_config: List[Tuple[str, str]],
    hands_data: List[dict],
    final_scores: Tuple[int, int],
    winner_team: Optional[int] = None,
    started_at: Optional[datetime] = None,
    ended_at: Optional[datetime] = None,
) -> Game:
    """
    Save a complete game to the database.

    Parameters
    ----------
    player_config : List[tuple[str, str]]
        List of (name, profile_type) tuples for each player.
    hands_data : List[dict]
        List of hand data dictionaries. Each dict should contain:
        - hand_number: int
        - dealer_player_id: int
        - turned_card: Card
        - trump_suit: Optional[Suit]
        - trump_maker_player_id: Optional[int]
        - trump_selection_round: Optional[int]
        - player_hands: List[List[Card]] - Cards for each player (indexed by player_id)
        - tricks_data: List[dict] - Trick data (see save_hand for structure)
        - tricks_won: List[int] - Tricks won by each team [team0, team1]
        - points: List[int] - Points awarded [team0, team1]
        - scores_after: List[int] - Scores after hand [team0, team1]
        - trump_decisions: Optional[List[dict]] - Trump decision data
    final_scores : tuple[int, int]
        Final scores (team0, team1).
    winner_team : Optional[int]
        Winning team (0 or 1), or None if game not finished.
    started_at : Optional[datetime]
        Game start time. If None, uses current time.
    ended_at : Optional[datetime]
        Game end time. If None, uses current time if winner_team is set.

    Returns
    -------
    Game
        The saved game entity.
    """
    if started_at is None:
        started_at = datetime.now()
    if ended_at is None and winner_team is not None:
        ended_at = datetime.now()

    # Create game
    game = Game(
        started_at=started_at,
        ended_at=ended_at,
        final_team0_score=final_scores[0],
        final_team1_score=final_scores[1],
        winner_team=winner_team,
    )

    # Create players and game players
    for player_id, (name, profile_type) in enumerate(player_config):
        player = get_or_create_player(name)
        team = player_id % 2
        GamePlayer(
            game=game,
            player=player,
            player_id=player_id,
            team=team,
            profile_type=profile_type,
        )

    # Save each hand
    for hand_data in hands_data:
        save_hand(game, hand_data)

    commit()
    return game


@db_session  # type: ignore[misc]
def save_hand(game: Game, hand_data: dict) -> Hand:
    """
    Save a single hand to the database.

    Parameters
    ----------
    game : Game
        The game entity this hand belongs to.
    hand_data : dict
        Dictionary containing hand data:
        - hand_number: int
        - dealer_player_id: int
        - turned_card: Card
        - trump_suit: Optional[Suit]
        - trump_maker_player_id: Optional[int]
        - trump_selection_round: Optional[int]
        - player_hands: List[List[Card]] - Cards for each player
        - tricks_data: List[dict] - Trick data
        - tricks_won: List[int] - [team0, team1]
        - points: List[int] - [team0, team1]
        - scores_after: List[int] - [team0, team1]
        - trump_decisions: Optional[List[dict]] - Trump decision data

    Returns
    -------
    Hand
        The saved hand entity.
    """
    turned_card: Card = hand_data["turned_card"]
    trump_suit: Optional[Suit] = hand_data.get("trump_suit")

    # Create hand
    hand = Hand(
        game=game,
        hand_number=hand_data["hand_number"],
        dealer_player_id=hand_data["dealer_player_id"],
        turned_card_suit=suit_to_string(turned_card.suit),
        turned_card_rank=rank_to_string(turned_card.rank),
        trump_suit=suit_to_string(trump_suit) if trump_suit else None,
        trump_maker_player_id=hand_data.get("trump_maker_player_id"),
        trump_selection_round=hand_data.get("trump_selection_round"),
        tricks_won_team0=hand_data["tricks_won"][0],
        tricks_won_team1=hand_data["tricks_won"][1],
        points_team0=hand_data["points"][0],
        points_team1=hand_data["points"][1],
        team0_score_after=hand_data["scores_after"][0],
        team1_score_after=hand_data["scores_after"][1],
    )

    # Save player hands
    player_hands: List[List[Card]] = hand_data.get("player_hands", [])
    for player_id, cards in enumerate(player_hands):
        for position, card in enumerate(cards):
            PlayerHand(
                hand=hand,
                player_id=player_id,
                card_suit=suit_to_string(card.suit),
                card_rank=rank_to_string(card.rank),
                position_in_hand=position,
            )

    # Save trump decisions if provided
    trump_decisions = hand_data.get("trump_decisions", [])
    for decision_data in trump_decisions:
        TrumpDecision(
            hand=hand,
            player_id=decision_data["player_id"],
            round=decision_data["round"],
            decision_type=decision_data["decision_type"],
            suit_chosen=(
                suit_to_string(decision_data["suit_chosen"])
                if decision_data.get("suit_chosen")
                else None
            ),
            turned_card_suit=suit_to_string(decision_data["turned_card"].suit),
            turned_card_rank=rank_to_string(decision_data["turned_card"].rank),
        )

    # Save tricks
    tricks_data = hand_data.get("tricks_data", [])
    for trick_data in tricks_data:
        save_trick(hand, trick_data)

    commit()
    return hand


@db_session  # type: ignore[misc]
def save_trick(hand: Hand, trick_data: dict) -> Trick:
    """
    Save a single trick to the database.

    Parameters
    ----------
    hand : Hand
        The hand entity this trick belongs to.
    trick_data : dict
        Dictionary containing trick data:
        - trick_number: int
        - leader_player_id: int
        - led_suit: Optional[Suit]
        - winner_player_id: int
        - winner_team: int
        - card_plays: List[tuple[int, Card]] - List of (player_id, card) tuples in play order

    Returns
    -------
    Trick
        The saved trick entity.
    """
    led_suit: Optional[Suit] = trick_data.get("led_suit")

    # Create trick
    trick = Trick(
        hand=hand,
        trick_number=trick_data["trick_number"],
        leader_player_id=trick_data["leader_player_id"],
        led_suit=suit_to_string(led_suit) if led_suit else None,
        winner_player_id=trick_data["winner_player_id"],
        winner_team=trick_data["winner_team"],
    )

    # Save card plays
    card_plays: List[Tuple[int, Card]] = trick_data.get("card_plays", [])
    for play_order, (player_id, card) in enumerate(card_plays):
        CardPlay(
            trick=trick,
            player_id=player_id,
            play_order=play_order,
            card_suit=suit_to_string(card.suit),
            card_rank=rank_to_string(card.rank),
        )

    commit()
    return trick


@db_session  # type: ignore[misc]
def load_player_hands(hand: Hand) -> List[List[Card]]:
    """
    Load player hands from a Hand entity.

    Parameters
    ----------
    hand : Hand
        The hand entity.

    Returns
    -------
    List[List[Card]]
        List of card lists, indexed by player_id.
    """
    player_hands: List[List[Tuple[int, Card]]] = [[], [], [], []]

    for ph in hand.player_hands:
        suit = string_to_suit(ph.card_suit)
        rank = string_to_rank(ph.card_rank)
        card = Card(suit, rank)
        player_hands[ph.player_id].append((ph.position_in_hand, card))

    # Sort by position and extract cards
    result: List[List[Card]] = [[], [], [], []]
    for player_id, cards_with_pos in enumerate(player_hands):
        cards_with_pos.sort(key=lambda x: x[0])
        result[player_id] = [card for _, card in cards_with_pos]

    return result


@db_session  # type: ignore[misc]
def load_trick_cards(trick: Trick) -> List[Tuple[int, Card]]:
    """
    Load card plays from a Trick entity.

    Parameters
    ----------
    trick : Trick
        The trick entity.

    Returns
    -------
    List[tuple[int, Card]]
        List of (player_id, card) tuples in play order.
    """
    card_plays = sorted(trick.card_plays, key=lambda cp: cp.play_order)
    result: List[Tuple[int, Card]] = []

    for cp in card_plays:
        suit = string_to_suit(cp.card_suit)
        rank = string_to_rank(cp.card_rank)
        card = Card(suit, rank)
        result.append((cp.player_id, card))

    return result
