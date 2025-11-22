"""PonyORM entity definitions for Euchre game database."""

from datetime import datetime

from pony.orm import Optional as PonyOptional
from pony.orm import PrimaryKey, Required, Set

from eucher.cards import Rank, Suit  # noqa: F401
from eucher.database import db


class Player(db.Entity):  # type: ignore[name-defined]
    """
    Persistent player records.

    Tracks players by name across multiple games.
    """

    name = PrimaryKey(str)
    created_at = Required(datetime, default=datetime.now)
    games = Set("GamePlayer")

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return f"Player(name={self.name!r}, created_at={self.created_at})"


class Game(db.Entity):  # type: ignore[name-defined]
    """
    Complete game session.

    Represents a full game with multiple hands.
    """

    id = PrimaryKey(int, auto=True)
    started_at = Required(datetime, default=datetime.now)
    ended_at = PonyOptional(datetime)
    final_team0_score = Required(int, default=0)
    final_team1_score = Required(int, default=0)
    winner_team = PonyOptional(int)  # 0 or 1
    hands = Set("Hand")
    game_players = Set("GamePlayer")

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return (
            f"Game(id={self.id}, started_at={self.started_at}, "
            f"final_team0_score={self.final_team0_score}, "
            f"final_team1_score={self.final_team1_score}, winner_team={self.winner_team})"
        )


class GamePlayer(db.Entity):  # type: ignore[name-defined]
    """
    Players in a specific game.

    Junction table linking players to games with game-specific information.
    """

    id = PrimaryKey(int, auto=True)
    game = Required(Game)
    player = Required(Player)
    player_id = Required(int)  # Position in game (0-3)
    team = Required(int)  # Team assignment (0 or 1)
    profile_type = Required(str)  # Player profile type (human, ai, ml, etc.)

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return (
            f"GamePlayer(id={self.id}, player={self.player.name}, "
            f"player_id={self.player_id}, team={self.team}, profile_type={self.profile_type!r})"
        )


class Hand(db.Entity):  # type: ignore[name-defined]
    """
    Single hand within a game.

    Represents one hand with trump selection, tricks, and scoring.
    """

    id = PrimaryKey(int, auto=True)
    game = Required(Game)
    hand_number = Required(int)  # Sequence number within game
    dealer_player_id = Required(int)  # Which player was dealer (0-3)
    turned_card_suit = Required(str)  # Suit of turned card (enum value)
    turned_card_rank = Required(str)  # Rank of turned card (enum value)
    trump_suit = PonyOptional(str)  # Selected trump suit (enum value)
    trump_maker_player_id = PonyOptional(int)  # Who made trump (0-3)
    trump_selection_round = PonyOptional(int)  # Round when trump was selected (1 or 2)
    tricks_won_team0 = Required(int, default=0)
    tricks_won_team1 = Required(int, default=0)
    points_team0 = Required(int, default=0)
    points_team1 = Required(int, default=0)
    team0_score_after = Required(int, default=0)
    team1_score_after = Required(int, default=0)
    tricks = Set("Trick")
    player_hands = Set("PlayerHand")
    trump_decisions = Set("TrumpDecision")

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return (
            f"Hand(id={self.id}, game_id={self.game.id}, hand_number={self.hand_number}, "
            f"trump_suit={self.trump_suit}, tricks_won_team0={self.tricks_won_team0}, "
            f"tricks_won_team1={self.tricks_won_team1})"
        )


class PlayerHand(db.Entity):  # type: ignore[name-defined]
    """
    Cards dealt to a player in a hand.

    Tracks the initial cards each player received.
    """

    id = PrimaryKey(int, auto=True)
    hand = Required(Hand)
    player_id = Required(int)  # Which player (0-3)
    card_suit = Required(str)  # Card suit (enum value)
    card_rank = Required(str)  # Card rank (enum value)
    position_in_hand = Required(int)  # Order in player's hand (0-4)

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return (
            f"PlayerHand(id={self.id}, hand_id={self.hand.id}, player_id={self.player_id}, "
            f"card={self.card_rank}{self.card_suit}, position={self.position_in_hand})"
        )


class Trick(db.Entity):  # type: ignore[name-defined]
    """
    Single trick within a hand.

    Represents one trick with leader, cards played, and winner.
    """

    id = PrimaryKey(int, auto=True)
    hand = Required(Hand)
    trick_number = Required(int)  # Trick number within hand (1-5)
    leader_player_id = Required(int)  # Who led the trick (0-3)
    led_suit = PonyOptional(str)  # Suit that was led (enum value)
    winner_player_id = Required(int)  # Who won the trick (0-3)
    winner_team = Required(int)  # Winning team (0 or 1)
    card_plays = Set("CardPlay")

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return (
            f"Trick(id={self.id}, hand_id={self.hand.id}, trick_number={self.trick_number}, "
            f"leader_player_id={self.leader_player_id}, winner_player_id={self.winner_player_id}, "
            f"winner_team={self.winner_team})"
        )


class CardPlay(db.Entity):  # type: ignore[name-defined]
    """
    Single card played in a trick.

    Tracks each card played with order and player information.
    """

    id = PrimaryKey(int, auto=True)
    trick = Required(Trick)
    player_id = Required(int)  # Which player played it (0-3)
    play_order = Required(int)  # Order within trick (0-3)
    card_suit = Required(str)  # Card suit (enum value)
    card_rank = Required(str)  # Card rank (enum value)

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return (
            f"CardPlay(id={self.id}, trick_id={self.trick.id}, player_id={self.player_id}, "
            f"play_order={self.play_order}, card={self.card_rank}{self.card_suit})"
        )


class TrumpDecision(db.Entity):  # type: ignore[name-defined]
    """
    Track trump selection decisions.

    Records each player's decision during trump selection rounds.
    """

    id = PrimaryKey(int, auto=True)
    hand = Required(Hand)
    player_id = Required(int)  # Which player made decision (0-3)
    round = Required(int)  # Round number (1 or 2)
    decision_type = Required(str)  # "order_up", "pass", "call_trump"
    suit_chosen = PonyOptional(str)  # Suit chosen if applicable
    turned_card_suit = Required(str)  # Suit of turned card
    turned_card_rank = Required(str)  # Rank of turned card

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return (
            f"TrumpDecision(id={self.id}, hand_id={self.hand.id}, player_id={self.player_id}, "
            f"round={self.round}, decision_type={self.decision_type!r}, "
            f"suit_chosen={self.suit_chosen})"
        )


def suit_to_string(suit: Suit) -> str:
    """
    Convert Suit enum to string for database storage.

    Parameters
    ----------
    suit : Suit
        The suit enum value.

    Returns
    -------
    str
        String representation of the suit.
    """
    return suit.value


def string_to_suit(suit_str: str) -> Suit:
    """
    Convert string to Suit enum from database.

    Parameters
    ----------
    suit_str : str
        String representation of the suit.

    Returns
    -------
    Suit
        The suit enum value.

    Raises
    ------
    ValueError
        If the string doesn't match any suit.
    """
    for suit in Suit:
        if suit.value == suit_str:
            return suit
    raise ValueError(f"Invalid suit string: {suit_str}")


def rank_to_string(rank: Rank) -> str:
    """
    Convert Rank enum to string for database storage.

    Parameters
    ----------
    rank : Rank
        The rank enum value.

    Returns
    -------
    str
        String representation of the rank.
    """
    return rank.name  # Use name (NINE, TEN, etc.) for consistency


def string_to_rank(rank_str: str) -> Rank:
    """
    Convert string to Rank enum from database.

    Parameters
    ----------
    rank_str : str
        String representation of the rank (enum name).

    Returns
    -------
    Rank
        The rank enum value.

    Raises
    ------
    ValueError
        If the string doesn't match any rank.
    """
    for rank in Rank:
        if rank.name == rank_str:
            return rank
    raise ValueError(f"Invalid rank string: {rank_str}")
