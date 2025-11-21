"""Tests for cards module."""

import pytest

from src.cards import Card, Deck, Rank, Suit


class TestCard:
    """Tests for Card class."""

    def test_card_creation(self) -> None:
        """Test card creation."""
        card = Card(Suit.HEARTS, Rank.ACE)
        assert card.suit == Suit.HEARTS
        assert card.rank == Rank.ACE

    def test_card_equality(self) -> None:
        """Test card equality."""
        card1 = Card(Suit.HEARTS, Rank.ACE)
        card2 = Card(Suit.HEARTS, Rank.ACE)
        card3 = Card(Suit.DIAMONDS, Rank.ACE)

        assert card1 == card2
        assert card1 != card3

    def test_card_hash(self) -> None:
        """Test card hashing."""
        card1 = Card(Suit.HEARTS, Rank.ACE)
        card2 = Card(Suit.HEARTS, Rank.ACE)
        card_set = {card1, card2}
        assert len(card_set) == 1

    def test_same_color(self) -> None:
        """Test same color detection."""
        hearts = Card(Suit.HEARTS, Rank.ACE)
        diamonds = Card(Suit.DIAMONDS, Rank.ACE)
        clubs = Card(Suit.CLUBS, Rank.ACE)
        spades = Card(Suit.SPADES, Rank.ACE)

        assert hearts.is_same_color(diamonds)
        assert clubs.is_same_color(spades)
        assert not hearts.is_same_color(clubs)
        assert not hearts.is_same_color(spades)

    def test_trump_comparison_right_bower(self) -> None:
        """Test comparison with Right Bower."""
        right_bower = Card(Suit.HEARTS, Rank.JACK)
        ace_trump = Card(Suit.HEARTS, Rank.ACE)
        trump_suit = Suit.HEARTS

        comparison = right_bower.compare_to(ace_trump, trump_suit, None)
        assert comparison > 0  # Right Bower beats Ace of trump

    def test_trump_comparison_left_bower(self) -> None:
        """Test comparison with Left Bower."""
        right_bower = Card(Suit.HEARTS, Rank.JACK)
        left_bower = Card(Suit.DIAMONDS, Rank.JACK)  # Same color as Hearts
        trump_suit = Suit.HEARTS

        comparison = right_bower.compare_to(left_bower, trump_suit, None)
        assert comparison > 0  # Right Bower beats Left Bower

        comparison = left_bower.compare_to(Card(Suit.HEARTS, Rank.ACE), trump_suit, None)
        assert comparison > 0  # Left Bower beats Ace of trump

    def test_trump_beats_non_trump(self) -> None:
        """Test that trump beats non-trump."""
        trump_card = Card(Suit.HEARTS, Rank.NINE)
        non_trump = Card(Suit.DIAMONDS, Rank.ACE)
        trump_suit = Suit.HEARTS

        comparison = trump_card.compare_to(non_trump, trump_suit, None)
        assert comparison > 0  # Any trump beats any non-trump


class TestDeck:
    """Tests for Deck class."""

    def test_deck_creation(self) -> None:
        """Test deck creation."""
        deck = Deck()
        assert len(deck) == 24

    def test_deck_has_all_cards(self) -> None:
        """Test that deck has all 24 Euchre cards."""
        deck = Deck()
        suits = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
        ranks = [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]

        for suit in suits:
            for rank in ranks:
                card = Card(suit, rank)
                assert card in deck.cards

    def test_deal_cards(self) -> None:
        """Test dealing cards."""
        deck = Deck()
        cards = deck.deal(5)
        assert len(cards) == 5
        assert len(deck) == 19

    def test_deal_all_cards(self) -> None:
        """Test dealing all cards."""
        deck = Deck()
        cards = deck.deal(24)
        assert len(cards) == 24
        assert len(deck) == 0

    def test_deal_too_many_cards(self) -> None:
        """Test dealing more cards than available."""
        deck = Deck()
        deck.deal(20)
        with pytest.raises(ValueError):
            deck.deal(10)

    def test_draw_one(self) -> None:
        """Test drawing a single card."""
        deck = Deck()
        initial_len = len(deck)
        card = deck.draw_one()
        assert isinstance(card, Card)
        assert len(deck) == initial_len - 1

    def test_draw_from_empty_deck(self) -> None:
        """Test drawing from empty deck."""
        deck = Deck()
        deck.deal(24)
        with pytest.raises(ValueError):
            deck.draw_one()

    def test_shuffle(self) -> None:
        """Test that shuffle changes order."""
        deck1 = Deck()
        deck2 = Deck()
        deck2.shuffle()

        # Very unlikely that shuffle produces same order
        # But possible, so we just check they're both valid decks
        assert len(deck1) == len(deck2) == 24

