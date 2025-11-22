"""Test scenarios for ML model testing."""

from typing import Dict, List

from eucher.cards import Card, Rank, Suit
from tests.fixtures.ml_test_data import (
    create_test_scenario_all_trump,
    create_test_scenario_bowers,
    create_test_scenario_no_trump,
)


def get_common_test_scenarios() -> List[Dict]:
    """
    Get common test scenarios.

    Returns
    -------
    List[Dict]
        List of test scenario dictionaries.
    """
    scenarios = []

    # Test all trump scenarios
    for suit in Suit:
        scenarios.append(create_test_scenario_all_trump(suit))

    # Test no trump scenarios
    for suit in Suit:
        scenarios.append(create_test_scenario_no_trump(suit))

    # Test bower scenarios
    for suit in Suit:
        scenarios.append(create_test_scenario_bowers(suit))

    return scenarios


def get_tricky_decision_scenarios() -> List[Dict]:
    """
    Get tricky decision point scenarios.

    Returns
    -------
    List[Dict]
        List of tricky scenario dictionaries.
    """
    scenarios = []

    # Scenario: Should order up with weak hand?
    scenarios.append({
        "hand": [
            Card(Suit.HEARTS, Rank.NINE),
            Card(Suit.HEARTS, Rank.TEN),
            Card(Suit.DIAMONDS, Rank.ACE),
            Card(Suit.CLUBS, Rank.KING),
            Card(Suit.SPADES, Rank.QUEEN),
        ],
        "turned_card": Card(Suit.HEARTS, Rank.ACE),
        "trump_suit": None,
        "description": "Weak hand, should order up?",
    })

    # Scenario: Last trick, teammate winning
    scenarios.append({
        "hand": [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
        ],
        "trump_suit": Suit.HEARTS,
        "trick_cards": [
            Card(Suit.HEARTS, Rank.QUEEN),  # Teammate's card (winning)
            Card(Suit.DIAMONDS, Rank.ACE),
            Card(Suit.CLUBS, Rank.KING),
        ],
        "description": "Last trick, teammate winning - should duck",
    })

    return scenarios

