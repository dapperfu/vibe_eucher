"""Database query endpoints for retrieving and analyzing game data."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pony.orm import db_session, desc, select

from eucher.models import (
    CardPlay,
    Game,
    GamePlayer,
    Hand,
    Player,
    PlayerHand,
    Trick,
    TrumpDecision,
)


# ============================================================================
# Player Queries
# ============================================================================


@db_session  # type: ignore[misc]
def get_player(name: str) -> Optional[Player]:
    """
    Get a player by name.

    Parameters
    ----------
    name : str
        Player name.

    Returns
    -------
    Optional[Player]
        The player entity if found, None otherwise.
    """
    player = Player.get(name=name)
    return player


@db_session  # type: ignore[misc]
def list_players(sort_by: str = "name") -> List[Player]:
    """
    List all players in database.

    Parameters
    ----------
    sort_by : str
        Sort order: "name" or "created_at". Default is "name".

    Returns
    -------
    List[Player]
        List of all players, sorted.
    """
    if sort_by == "created_at":
        return list(select(p for p in Player).order_by(Player.created_at))  # type: ignore[attr-defined]
    return list(select(p for p in Player).order_by(Player.name))  # type: ignore[attr-defined]


@db_session  # type: ignore[misc]
def get_player_games(player_name: str, limit: Optional[int] = None) -> List[Game]:
    """
    Get all games a player participated in.

    Parameters
    ----------
    player_name : str
        Player name.
    limit : Optional[int]
        Maximum number of games to return. If None, returns all.

    Returns
    -------
    List[Game]
        List of games, sorted by start time (newest first).
    """
    player = Player.get(name=player_name)
    if player is None:
        return []

    query = select(gp.game for gp in GamePlayer if gp.player == player).order_by(
        desc(Game.started_at)
    )

    if limit is not None:
        return list(query[:limit])
    return list(query)


@db_session  # type: ignore[misc]
def get_player_statistics(player_name: str) -> Dict[str, any]:
    """
    Get comprehensive statistics for a player.

    Parameters
    ----------
    player_name : str
        Player name.

    Returns
    -------
    Dict[str, any]
        Dictionary containing:
        - total_games: int
        - games_won: int
        - games_lost: int
        - win_rate: float
        - total_hands: int
        - profile_types: List[str]
    """
    player = Player.get(name=player_name)
    if player is None:
        return {
            "total_games": 0,
            "games_won": 0,
            "games_lost": 0,
            "win_rate": 0.0,
            "total_hands": 0,
            "profile_types": [],
        }

    # Get all games for this player
    game_players = list(select(gp for gp in GamePlayer if gp.player == player))
    total_games = len(game_players)

    # Count wins/losses
    games_won = 0
    games_lost = 0
    profile_types_set = set()

    for gp in game_players:
        profile_types_set.add(gp.profile_type)
        game = gp.game
        if game.winner_team is not None:
            if gp.team == game.winner_team:
                games_won += 1
            else:
                games_lost += 1

    # Count total hands
    total_hands = sum(len(gp.game.hands) for gp in game_players)

    # Calculate win rate
    win_rate = (games_won / total_games * 100.0) if total_games > 0 else 0.0

    return {
        "total_games": total_games,
        "games_won": games_won,
        "games_lost": games_lost,
        "win_rate": win_rate,
        "total_hands": total_hands,
        "profile_types": sorted(list(profile_types_set)),
    }


# ============================================================================
# Game Queries
# ============================================================================


@db_session  # type: ignore[misc]
def get_game(game_id: int) -> Optional[Game]:
    """
    Get a specific game by ID.

    Parameters
    ----------
    game_id : int
        Game ID.

    Returns
    -------
    Optional[Game]
        The game entity if found, None otherwise.
    """
    return Game.get(id=game_id)


@db_session  # type: ignore[misc]
def list_games(limit: Optional[int] = None, offset: int = 0) -> List[Game]:
    """
    List games with pagination.

    Parameters
    ----------
    limit : Optional[int]
        Maximum number of games to return. If None, returns all.
    offset : int
        Number of games to skip. Default is 0.

    Returns
    -------
    List[Game]
        List of games, sorted by start time (newest first).
    """
    query = select(g for g in Game).order_by(desc(Game.started_at))

    if limit is not None:
        return list(query[offset : offset + limit])
    if offset > 0:
        return list(query[offset:])
    return list(query)


@db_session  # type: ignore[misc]
def get_games_by_date_range(start_date: datetime, end_date: datetime) -> List[Game]:
    """
    Filter games by date range.

    Parameters
    ----------
    start_date : datetime
        Start date (inclusive).
    end_date : datetime
        End date (inclusive).

    Returns
    -------
    List[Game]
        List of games within the date range, sorted by start time.
    """
    return list(
        select(g for g in Game if g.started_at >= start_date and g.started_at <= end_date).order_by(
            desc(Game.started_at)
        )
    )


@db_session  # type: ignore[misc]
def get_games_by_player(player_name: str) -> List[Game]:
    """
    Get all games for a specific player.

    Parameters
    ----------
    player_name : str
        Player name.

    Returns
    -------
    List[Game]
        List of games, sorted by start time (newest first).
    """
    return get_player_games(player_name)


# ============================================================================
# Hand Queries
# ============================================================================


@db_session  # type: ignore[misc]
def get_hand(hand_id: int) -> Optional[Hand]:
    """
    Get a specific hand by ID.

    Parameters
    ----------
    hand_id : int
        Hand ID.

    Returns
    -------
    Optional[Hand]
        The hand entity if found, None otherwise.
    """
    return Hand.get(id=hand_id)


@db_session  # type: ignore[misc]
def get_hands_by_game(game_id: int) -> List[Hand]:
    """
    Get all hands for a game.

    Parameters
    ----------
    game_id : int
        Game ID.

    Returns
    -------
    List[Hand]
        List of hands, ordered by hand_number.
    """
    game = Game.get(id=game_id)
    if game is None:
        return []
    return list(select(h for h in Hand if h.game == game).order_by(Hand.hand_number))


@db_session  # type: ignore[misc]
def get_hands_by_player(player_name: str, limit: Optional[int] = None) -> List[Hand]:
    """
    Get hands where player participated.

    Parameters
    ----------
    player_name : str
        Player name.
    limit : Optional[int]
        Maximum number of hands to return. If None, returns all.

    Returns
    -------
    List[Hand]
        List of hands, sorted by game start time and hand number.
    """
    player = Player.get(name=player_name)
    if player is None:
        return []

    # Get all games for this player
    game_players = list(select(gp for gp in GamePlayer if gp.player == player))
    game_ids = [gp.game.id for gp in game_players]

    if not game_ids:
        return []

    query = select(h for h in Hand if h.game.id in game_ids).order_by(
        desc(Hand.game.started_at), Hand.hand_number
    )

    if limit is not None:
        return list(query[:limit])
    return list(query)


# ============================================================================
# Trick Queries
# ============================================================================


@db_session  # type: ignore[misc]
def get_trick(trick_id: int) -> Optional[Trick]:
    """
    Get a specific trick with all card plays.

    Parameters
    ----------
    trick_id : int
        Trick ID.

    Returns
    -------
    Optional[Trick]
        The trick entity if found, None otherwise.
    """
    return Trick.get(id=trick_id)


@db_session  # type: ignore[misc]
def get_tricks_by_hand(hand_id: int) -> List[Trick]:
    """
    Get all tricks for a hand.

    Parameters
    ----------
    hand_id : int
        Hand ID.

    Returns
    -------
    List[Trick]
        List of tricks, ordered by trick_number.
    """
    hand = Hand.get(id=hand_id)
    if hand is None:
        return []
    return list(select(t for t in Trick if t.hand == hand).order_by(Trick.trick_number))


@db_session  # type: ignore[misc]
def get_tricks_by_player(player_name: str, limit: Optional[int] = None) -> List[Trick]:
    """
    Get tricks where player played cards.

    Parameters
    ----------
    player_name : str
        Player name.
    limit : Optional[int]
        Maximum number of tricks to return. If None, returns all.

    Returns
    -------
    List[Trick]
        List of tricks, sorted by game start time, hand number, and trick number.
    """
    player = Player.get(name=player_name)
    if player is None:
        return []

    # Get all games for this player
    game_players = list(select(gp for gp in GamePlayer if gp.player == player))
    game_ids = [gp.game.id for gp in game_players]

    if not game_ids:
        return []

    # Get tricks where player played a card
    query = select(t for t in Trick if t.hand.game.id in game_ids).order_by(
        desc(Trick.hand.game.started_at), Trick.hand.hand_number, Trick.trick_number
    )

    if limit is not None:
        return list(query[:limit])
    return list(query)


# ============================================================================
# Statistics & Analytics
# ============================================================================


@db_session  # type: ignore[misc]
def get_game_statistics() -> Dict[str, any]:
    """
    Get overall game statistics.

    Returns
    -------
    Dict[str, any]
        Dictionary containing:
        - total_games: int
        - total_players: int
        - average_game_duration: Optional[float] (seconds)
        - profile_types: Dict[str, int] (count by type)
    """
    total_games = len(list(select(g for g in Game)))
    total_players = len(list(select(p for p in Player)))

    # Calculate average game duration
    games_with_end = list(select(g for g in Game if g.ended_at is not None))
    if games_with_end:
        durations = [(g.ended_at - g.started_at).total_seconds() for g in games_with_end]
        avg_duration = sum(durations) / len(durations)
    else:
        avg_duration = None

    # Count profile types
    profile_types: Dict[str, int] = {}
    game_players = select(gp for gp in GamePlayer)
    for gp in game_players:
        profile_type = gp.profile_type
        profile_types[profile_type] = profile_types.get(profile_type, 0) + 1

    return {
        "total_games": total_games,
        "total_players": total_players,
        "average_game_duration": avg_duration,
        "profile_types": profile_types,
    }


@db_session  # type: ignore[misc]
def get_player_win_rate(player_name: str) -> float:
    """
    Calculate win rate for a player.

    Parameters
    ----------
    player_name : str
        Player name.

    Returns
    -------
    float
        Win rate as a percentage (0-100).
    """
    stats = get_player_statistics(player_name)
    return stats["win_rate"]


@db_session  # type: ignore[misc]
def get_most_common_trump_suits(limit: int = 4) -> List[Tuple[str, int]]:
    """
    Get trump suit frequency.

    Parameters
    ----------
    limit : int
        Maximum number of results to return. Default is 4.

    Returns
    -------
    List[Tuple[str, int]]
        List of (suit, count) tuples, sorted by count (descending).
    """
    hands = select(h for h in Hand if h.trump_suit is not None)
    suit_counts: Dict[str, int] = {}

    for hand in hands:
        suit = hand.trump_suit
        if suit:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1

    # Sort by count descending
    sorted_suits = sorted(suit_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_suits[:limit]


@db_session  # type: ignore[misc]
def get_trick_win_statistics(player_name: str) -> Dict[str, any]:
    """
    Get trick win statistics for a player.

    Parameters
    ----------
    player_name : str
        Player name.

    Returns
    -------
    Dict[str, any]
        Dictionary containing:
        - tricks_won: int
        - tricks_lost: int
        - win_rate: float
        - tricks_by_position: Dict[int, int] (tricks won by player position)
    """
    player = Player.get(name=player_name)
    if player is None:
        return {
            "tricks_won": 0,
            "tricks_lost": 0,
            "win_rate": 0.0,
            "tricks_by_position": {},
        }

    # Get all games for this player
    game_players = list(select(gp for gp in GamePlayer if gp.player == player))
    player_team_map: Dict[int, int] = {}  # game_id -> team
    for gp in game_players:
        player_team_map[gp.game.id] = gp.team

    tricks_won = 0
    tricks_lost = 0
    tricks_by_position: Dict[int, int] = {}

    # Get all tricks from games this player participated in
    for game_id, team in player_team_map.items():
        hands = select(h for h in Hand if h.game.id == game_id)
        for hand in hands:
            tricks = select(t for t in Trick if t.hand == hand)
            for trick in tricks:
                if trick.winner_team == team:
                    tricks_won += 1
                    # Count by player position (0-3)
                    tricks_by_position[trick.winner_player_id] = (
                        tricks_by_position.get(trick.winner_player_id, 0) + 1
                    )
                else:
                    tricks_lost += 1

    total_tricks = tricks_won + tricks_lost
    win_rate = (tricks_won / total_tricks * 100.0) if total_tricks > 0 else 0.0

    return {
        "tricks_won": tricks_won,
        "tricks_lost": tricks_lost,
        "win_rate": win_rate,
        "tricks_by_position": tricks_by_position,
    }


# ============================================================================
# Search & Filter
# ============================================================================


@db_session  # type: ignore[misc]
def search_games(query: str) -> List[Game]:
    """
    Search games by player names.

    Parameters
    ----------
    query : str
        Search query (player name).

    Returns
    -------
    List[Game]
        List of games matching the query, sorted by start time.
    """
    query_lower = query.lower()
    # Find players matching the query
    matching_players = list(select(p for p in Player if query_lower in p.name.lower()))

    if not matching_players:
        return []

    # Get games for matching players
    game_ids = set()
    for player in matching_players:
        game_players = list(select(gp for gp in GamePlayer if gp.player == player))
        for gp in game_players:
            game_ids.add(gp.game.id)

    if not game_ids:
        return []

    return list(select(g for g in Game if g.id in game_ids).order_by(desc(Game.started_at)))


@db_session  # type: ignore[misc]
def filter_games_by_profile_type(profile_type: str) -> List[Game]:
    """
    Get games with specific profile types.

    Parameters
    ----------
    profile_type : str
        Profile type to filter by (e.g., "ai", "human", "ml_sklearn").

    Returns
    -------
    List[Game]
        List of games containing players with the specified profile type.
    """
    game_players = list(select(gp for gp in GamePlayer if gp.profile_type == profile_type))
    game_ids = {gp.game.id for gp in game_players}

    if not game_ids:
        return []

    return list(select(g for g in Game if g.id in game_ids).order_by(desc(Game.started_at)))
