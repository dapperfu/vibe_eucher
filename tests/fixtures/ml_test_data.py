"""Generate synthetic test data for ML models."""

import random
from typing import List

import numpy as np

from eucher.cards import Card, Rank, Suit
from eucher.players.computer.ml.ml_features import GameStateEncoder


def generate_random_hand(num_cards: int = 5) -> List[Card]:
    """
    Generate a random hand.

    Parameters
    ----------
    num_cards : int
        Number of cards in hand.

    Returns
    -------
    List[Card]
        Random hand.
    """
    all_cards = []
    for suit in Suit:
        for rank in Rank:
            all_cards.append(Card(suit, rank))

    return random.sample(all_cards, num_cards)


def generate_game_state_features(
    hand: List[Card],
    turned_card: Card = None,
    trump_suit: Suit = None,
    led_suit: Suit = None,
    trick_cards: List[Card] = None,
    player_id: int = 0,
    dealer_id: int = 0,
    trick_number: int = 0,
    tricks_won_team0: int = 0,
    tricks_won_team1: int = 0,
) -> np.ndarray:
    """
    Generate game state features for testing.

    Parameters
    ----------
    hand : List[Card]
        Player's hand.
    turned_card : Card, optional
        Turned card.
    trump_suit : Suit, optional
        Trump suit.
    led_suit : Suit, optional
        Led suit.
    trick_cards : List[Card], optional
        Cards in current trick.
    player_id : int
        Player ID.
    dealer_id : int
        Dealer ID.
    trick_number : int
        Current trick number.
    tricks_won_team0 : int
        Tricks won by team 0.
    tricks_won_team1 : int
        Tricks won by team 1.

    Returns
    -------
    np.ndarray
        Feature vector.
    """
    encoder = GameStateEncoder()
    return encoder.encode_game_state(
        hand=hand,
        turned_card=turned_card,
        trump_suit=trump_suit,
        led_suit=led_suit,
        trick_cards=trick_cards or [],
        player_id=player_id,
        dealer_id=dealer_id,
        team=player_id % 2,
        trick_number=trick_number,
        tricks_won_team0=tricks_won_team0,
        tricks_won_team1=tricks_won_team1,
    ).numpy()


def generate_synthetic_decisions(
    num_samples: int = 100, decision_type: str = "play_card"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic decision data for testing.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    decision_type : str
        Type of decision: "play_card", "order_up", "call_trump", "discard"

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (features, decisions) tuple.
    """
    encoder = GameStateEncoder()
    features_list = []
    decisions_list = []

    for _ in range(num_samples):
        hand = generate_random_hand(5)
        turned_card = generate_random_hand(1)[0] if random.random() > 0.5 else None
        trump_suit = random.choice(list(Suit)) if random.random() > 0.5 else None

        features = generate_game_state_features(
            hand=hand,
            turned_card=turned_card,
            trump_suit=trump_suit,
        )
        features_list.append(features)

        # Generate synthetic decision
        if decision_type == "play_card":
            # Random card from hand
            decision = encoder.card_to_index[random.choice(hand)]
        elif decision_type == "order_up":
            decision = 1 if random.random() > 0.5 else 0
        elif decision_type == "call_trump":
            # 0 = pass, 1-4 = suit
            decision = random.randint(0, 4)
        elif decision_type == "discard":
            # Random card from hand (6 cards for discard)
            hand_6 = hand + [generate_random_hand(1)[0]]
            decision = encoder.card_to_index[random.choice(hand_6)]
        else:
            decision = 0

        decisions_list.append(decision)

    return np.array(features_list), np.array(decisions_list)


def create_test_scenario_all_trump(trump_suit: Suit) -> dict:
    """
    Create test scenario with all trump cards.

    Parameters
    ----------
    trump_suit : Suit
        Trump suit.

    Returns
    -------
    dict
        Test scenario dictionary.
    """
    hand = []
    # Add trump cards
    for rank in [Rank.ACE, Rank.KING, Rank.QUEEN, Rank.TEN, Rank.NINE]:
        hand.append(Card(trump_suit, rank))

    return {
        "hand": hand,
        "trump_suit": trump_suit,
        "description": f"All trump cards in {trump_suit.value}",
    }


def create_test_scenario_no_trump(trump_suit: Suit) -> dict:
    """
    Create test scenario with no trump cards.

    Parameters
    ----------
    trump_suit : Suit
        Trump suit.

    Returns
    -------
    dict
        Test scenario dictionary.
    """
    hand = []
    # Add non-trump cards
    other_suits = [s for s in Suit if s != trump_suit]
    for suit in other_suits[:2]:
        for rank in [Rank.ACE, Rank.KING, Rank.QUEEN]:
            hand.append(Card(suit, rank))
    # Add one more card
    hand.append(Card(other_suits[0], Rank.TEN))

    return {
        "hand": hand,
        "trump_suit": trump_suit,
        "description": f"No trump cards, trump is {trump_suit.value}",
    }


def create_test_scenario_bowers(trump_suit: Suit) -> dict:
    """
    Create test scenario with bowers.

    Parameters
    ----------
    trump_suit : Suit
        Trump suit.

    Returns
    -------
    dict
        Test scenario dictionary.
    """
    hand = []
    # Right bower
    hand.append(Card(trump_suit, Rank.JACK))
    # Left bower (same color)
    if trump_suit in [Suit.HEARTS, Suit.DIAMONDS]:
        left_bower_suit = Suit.CLUBS if trump_suit == Suit.HEARTS else Suit.SPADES
    else:
        left_bower_suit = Suit.HEARTS if trump_suit == Suit.CLUBS else Suit.DIAMONDS
    hand.append(Card(left_bower_suit, Rank.JACK))
    # Add some other cards
    for rank in [Rank.ACE, Rank.KING, Rank.QUEEN]:
        hand.append(Card(trump_suit, rank))

    return {
        "hand": hand,
        "trump_suit": trump_suit,
        "description": f"Has both bowers, trump is {trump_suit.value}",
    }

