"""Tests for rules module."""

from eucher.cards import Card, Rank, Suit
from eucher.rules import RulesEngine


class TestRulesEngine:
    """Tests for RulesEngine class."""

    def test_can_play_card_no_led_suit(self) -> None:
        """Test playing card when no suit was led."""
        rules = RulesEngine()
        hand = [Card(Suit.HEARTS, Rank.ACE), Card(Suit.DIAMONDS, Rank.KING)]
        card = Card(Suit.HEARTS, Rank.ACE)

        assert rules.can_play_card(card, hand, None, None)

    def test_must_follow_suit(self) -> None:
        """Test that player must follow suit if possible."""
        rules = RulesEngine()
        hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.KING),
            Card(Suit.HEARTS, Rank.QUEEN),
        ]
        led_suit = Suit.HEARTS

        # Can play hearts card
        assert rules.can_play_card(Card(Suit.HEARTS, Rank.ACE), hand, led_suit, None)
        # Cannot play diamonds when have hearts
        assert not rules.can_play_card(Card(Suit.DIAMONDS, Rank.KING), hand, led_suit, None)

    def test_can_play_any_if_cannot_follow(self) -> None:
        """Test that player can play any card if cannot follow suit."""
        rules = RulesEngine()
        hand = [Card(Suit.DIAMONDS, Rank.ACE), Card(Suit.CLUBS, Rank.KING)]
        led_suit = Suit.HEARTS

        # Can play any card since cannot follow
        assert rules.can_play_card(Card(Suit.DIAMONDS, Rank.ACE), hand, led_suit, None)
        assert rules.can_play_card(Card(Suit.CLUBS, Rank.KING), hand, led_suit, None)

    def test_trick_winner_no_trump(self) -> None:
        """Test trick winner determination without trump."""
        rules = RulesEngine()
        played_cards = [
            Card(Suit.HEARTS, Rank.NINE),
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.DIAMONDS, Rank.ACE),
        ]
        players = [0, 1, 2, 3]
        led_suit = Suit.HEARTS

        winner = rules.determine_trick_winner(played_cards, players, led_suit, None)
        assert winner == 1  # Ace of hearts wins

    def test_trick_winner_with_trump(self) -> None:
        """Test trick winner determination with trump."""
        rules = RulesEngine()
        played_cards = [
            Card(Suit.DIAMONDS, Rank.ACE),  # Non-trump
            Card(Suit.CLUBS, Rank.ACE),  # Non-trump
            Card(Suit.HEARTS, Rank.NINE),  # Trump (lowest trump)
            Card(Suit.HEARTS, Rank.ACE),  # Trump (higher)
        ]
        players = [0, 1, 2, 3]
        led_suit = Suit.DIAMONDS
        trump_suit = Suit.HEARTS

        winner = rules.determine_trick_winner(played_cards, players, led_suit, trump_suit)
        assert winner == 3  # Ace of trump beats all (trump beats non-trump, higher trump wins)

    def test_trick_winner_right_bower(self) -> None:
        """Test that Right Bower wins."""
        rules = RulesEngine()
        right_bower = Card(Suit.HEARTS, Rank.JACK)
        ace_trump = Card(Suit.HEARTS, Rank.ACE)
        played_cards = [ace_trump, right_bower]
        players = [0, 1]
        led_suit = Suit.HEARTS
        trump_suit = Suit.HEARTS

        winner = rules.determine_trick_winner(played_cards, players, led_suit, trump_suit)
        assert winner == 1  # Right Bower wins

    def test_get_valid_plays(self) -> None:
        """Test getting valid plays."""
        rules = RulesEngine()
        hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.KING),
            Card(Suit.HEARTS, Rank.QUEEN),
        ]
        led_suit = Suit.HEARTS

        valid = rules.get_valid_plays(hand, led_suit, None)
        assert len(valid) == 2  # Both hearts cards
        assert Card(Suit.HEARTS, Rank.ACE) in valid
        assert Card(Suit.HEARTS, Rank.QUEEN) in valid
        assert Card(Suit.DIAMONDS, Rank.KING) not in valid

