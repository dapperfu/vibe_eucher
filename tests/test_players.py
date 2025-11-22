"""Tests for players module."""

import pytest

from eucher.ai import AIDecisionMaker
from eucher.cards import Card, Rank, Suit
from eucher.players.profiles import AIBasedProfile, HumanProfile, SimpleRuleBasedProfile
from eucher.players import Player


class TestPlayer:
    """Tests for Player class."""

    def test_player_creation(self) -> None:
        """Test player creation."""
        profile = SimpleRuleBasedProfile()
        player = Player("Test", 0, profile)
        assert player.name == "Test"
        assert player.player_id == 0
        assert player.team == 0
        assert player.profile == profile

    def test_player_teams(self) -> None:
        """Test player team assignment."""
        profile = SimpleRuleBasedProfile()
        player0 = Player("P0", 0, profile)
        player1 = Player("P1", 1, profile)
        player2 = Player("P2", 2, profile)
        player3 = Player("P3", 3, profile)

        assert player0.team == 0
        assert player1.team == 1
        assert player2.team == 0
        assert player3.team == 1

    def test_receive_card(self) -> None:
        """Test receiving a card."""
        profile = SimpleRuleBasedProfile()
        player = Player("Test", 0, profile)
        card = Card(Suit.HEARTS, Rank.ACE)
        player.receive_card(card)
        assert card in player.hand

    def test_receive_hand(self) -> None:
        """Test receiving a hand."""
        profile = SimpleRuleBasedProfile()
        player = Player("Test", 0, profile)
        cards = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.KING),
        ]
        player.receive_hand(cards)
        assert len(player.hand) == 2
        assert cards[0] in player.hand

    def test_remove_card(self) -> None:
        """Test removing a card."""
        profile = SimpleRuleBasedProfile()
        player = Player("Test", 0, profile)
        card = Card(Suit.HEARTS, Rank.ACE)
        player.receive_card(card)
        player.remove_card(card)
        assert card not in player.hand

    def test_remove_card_not_in_hand(self) -> None:
        """Test removing card not in hand."""
        profile = SimpleRuleBasedProfile()
        player = Player("Test", 0, profile)
        card = Card(Suit.HEARTS, Rank.ACE)
        with pytest.raises(ValueError):
            player.remove_card(card)

    def test_has_card(self) -> None:
        """Test checking if player has card."""
        profile = SimpleRuleBasedProfile()
        player = Player("Test", 0, profile)
        card = Card(Suit.HEARTS, Rank.ACE)
        assert not player.has_card(card)
        player.receive_card(card)
        assert player.has_card(card)


class TestAIBasedProfile:
    """Tests for AIBasedProfile."""

    def test_ai_profile_creation(self) -> None:
        """Test AI profile creation."""
        ai = AIDecisionMaker()
        profile = AIBasedProfile(ai)
        player = Player("AI", 0, profile)
        assert player.name == "AI"
        assert isinstance(player.profile, AIBasedProfile)

    def test_ai_decide_order_up(self) -> None:
        """Test AI order up decision."""
        ai = AIDecisionMaker()
        profile = AIBasedProfile(ai)
        player = Player("AI", 0, profile)
        # Give player strong trump hand
        player.receive_hand([
            Card(Suit.HEARTS, Rank.JACK),  # Right Bower
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.KING),
            Card(Suit.CLUBS, Rank.QUEEN),
            Card(Suit.SPADES, Rank.TEN),
        ])
        turned_card = Card(Suit.HEARTS, Rank.NINE)

        decision = player.decide_order_up(turned_card, 1, None)
        assert decision is True  # Should order up with Right Bower

    def test_ai_play_card(self) -> None:
        """Test AI card play."""
        ai = AIDecisionMaker()
        profile = AIBasedProfile(ai)
        player = Player("AI", 0, profile)
        player.receive_hand([
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.KING),
        ])

        card = player.play_card(None, None, [], [])
        assert card in player.hand


class TestHumanProfile:
    """Tests for HumanProfile."""

    def test_human_profile_creation(self) -> None:
        """Test human profile creation."""
        profile = HumanProfile()
        player = Player("Human", 0, profile)
        assert player.name == "Human"
        assert isinstance(player.profile, HumanProfile)

    def test_set_tui(self) -> None:
        """Test setting TUI."""
        from eucher.tui import TextTUI

        profile = HumanProfile()
        tui = TextTUI()
        profile.set_tui(tui)
        assert profile.tui == tui


class TestSimpleRuleBasedProfile:
    """Tests for SimpleRuleBasedProfile."""

    def test_simple_profile_creation(self) -> None:
        """Test simple profile creation."""
        profile = SimpleRuleBasedProfile()
        player = Player("Simple", 0, profile)
        assert isinstance(player.profile, SimpleRuleBasedProfile)

    def test_simple_decide_order_up(self) -> None:
        """Test simple profile order up decision."""
        profile = SimpleRuleBasedProfile()
        player = Player("Simple", 0, profile)
        player.receive_hand([
            Card(Suit.HEARTS, Rank.JACK),
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.KING),
        ])
        turned_card = Card(Suit.HEARTS, Rank.NINE)

        decision = player.decide_order_up(turned_card, 1, None)
        assert decision is True  # Should order up with 2+ trump
