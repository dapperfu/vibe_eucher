"""Game configuration helpers for programmatic game setup."""

from typing import List, Tuple

from eucher.game import Game


def create_ai_vs_ai_game(player_names: List[str] = None) -> Game:
    """
    Create a game with all AI players.

    Parameters
    ----------
    player_names : List[str], optional
        List of 4 player names. If None, uses default names.

    Returns
    -------
    Game
        A game configured with all AI players.

    Raises
    ------
    ValueError
        If player_names is provided and doesn't have exactly 4 names.
    """
    if player_names is None:
        player_names = ["AI Player 1", "AI Player 2", "AI Player 3", "AI Player 4"]
    elif len(player_names) != 4:
        raise ValueError("Must provide exactly 4 player names")

    player_config = [(name, "ai") for name in player_names]
    return Game(player_config)


def create_heuristic_vs_heuristic_game(player_names: List[str] = None) -> Game:
    """
    Create a game with all heuristic rule-based players.

    Parameters
    ----------
    player_names : List[str], optional
        List of 4 player names. If None, uses default names.

    Returns
    -------
    Game
        A game configured with all heuristic rule-based players.

    Raises
    ------
    ValueError
        If player_names is provided and doesn't have exactly 4 names.
    """
    if player_names is None:
        player_names = ["Heuristic Player 1", "Heuristic Player 2", "Heuristic Player 3", "Heuristic Player 4"]
    elif len(player_names) != 4:
        raise ValueError("Must provide exactly 4 player names")

    player_config = [(name, "heuristic") for name in player_names]
    return Game(player_config)


def create_mixed_game(profile_types: List[str], player_names: List[str] = None) -> Game:
    """
    Create a game with specified profile types.

    Parameters
    ----------
    profile_types : List[str]
        List of 4 profile types: "human", "heuristic", "ai", or "random".
    player_names : List[str], optional
        List of 4 player names. If None, uses default names based on profile type.

    Returns
    -------
    Game
        A game configured with the specified profile types.

    Raises
    ------
    ValueError
        If profile_types doesn't have exactly 4 types, or if invalid profile type is provided.
    """
    if len(profile_types) != 4:
        raise ValueError("Must provide exactly 4 profile types")

    valid_types = {"human", "heuristic", "ai", "random"}
    for profile_type in profile_types:
        if profile_type not in valid_types:
            raise ValueError(f"Invalid profile type: {profile_type}. Must be one of {valid_types}")

    if player_names is None:
        player_names = []
        for i, profile_type in enumerate(profile_types):
            if profile_type == "human":
                player_names.append(f"Human Player {i + 1}")
            elif profile_type == "heuristic":
                player_names.append(f"Heuristic Player {i + 1}")
            elif profile_type == "ai":
                player_names.append(f"AI Player {i + 1}")
            else:  # random
                player_names.append(f"Random Player {i + 1}")
    elif len(player_names) != 4:
        raise ValueError("Must provide exactly 4 player names")

    player_config = list(zip(player_names, profile_types))
    return Game(player_config)


def create_custom_game(player_config: List[Tuple[str, str]]) -> Game:
    """
    Create a game with custom player configuration.

    Parameters
    ----------
    player_config : List[Tuple[str, str]]
        List of (name, profile_type) tuples for each player.
        profile_type can be: "human", "heuristic", "ai", "random"

    Returns
    -------
    Game
        A game configured with the provided player configuration.

    Raises
    ------
    ValueError
        If player_config doesn't have exactly 4 entries, or if invalid profile type is provided.
    """
    if len(player_config) != 4:
        raise ValueError("Must provide exactly 4 player configurations")

    valid_types = {"human", "heuristic", "ai", "random"}
    for name, profile_type in player_config:
        if profile_type not in valid_types:
            raise ValueError(f"Invalid profile type: {profile_type}. Must be one of {valid_types}")

    return Game(player_config)

