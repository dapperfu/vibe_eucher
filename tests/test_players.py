"""Tests for players module."""

import pytest

from src.ai import AIDecisionMaker
from src.cards import Card, Rank, Suit
from src.players import AIPlayer, HumanPlayer, Player


class TestPlayer:
    """Tests for Player base class."""

    def test_player_creation(self) -> None:
        """Test player creation."""
        player = AIPlayer("Test", 0, AIDecisionMaker())
        assert player.name == "Test"
        assert player.player_id == 0
        assert player.team == 0

    def test_player_teams(self) -> None:
        """Test player team assignment."""
        ai = AIDecisionMaker()
        player0 = AIPlayer("P0", 0, ai)
        player1 = AIPlayer("P1", 1, ai)
        player2 = AIPlayer("P2", 2, ai)
        player3 = AIPlayer("P3", 3, ai)

        assert player0.team == 0
        assert player1.team == 1
        assert player2.team == 0
        assert player3.team == 1

    def test_receive_card(self) -> None:
        """Test receiving a card."""
        player = AIPlayer("Test", 0, AIDecisionMaker())
        card = Card(Suit.HEARTS, Rank.ACE)
        player.receive_card(card)
        assert card in player.hand

    def test_receive_hand(self) -> None:
        """Test receiving a hand."""
        player = AIPlayer("Test", 0, AIDecisionMaker())
        cards = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.KING),
        ]
        player.receive_hand(cards)
        assert len(player.hand) == 2
        assert cards[0] in player.hand

    def test_remove_card(self) -> None:
        """Test removing a card."""
        player = AIPlayer("Test", 0, AIDecisionMaker())
        card = Card(Suit.HEARTS, Rank.ACE)
        player.receive_card(card)
        player.remove_card(card)
        assert card not in player.hand

    def test_remove_card_not_in_hand(self) -> None:
        """Test removing card not in hand."""
        player = AIPlayer("Test", 0, AIDecisionMaker())
        card = Card(Suit.HEARTS, Rank.ACE)
        with pytest.raises(ValueError):
            player.remove_card(card)

    def test_has_card(self) -> None:
        """Test checking if player has card."""
        player = AIPlayer("Test", 0, AIDecisionMaker())
        card = Card(Suit.HEARTS, Rank.ACE)
        assert not player.has_card(card)
        player.receive_card(card)
        assert player.has_card(card)


class TestAIPlayer:
    """Tests for AIPlayer class."""

    def test_ai_player_creation(self) -> None:
        """Test AI player creation."""
        ai = AIDecisionMaker()
        player = AIPlayer("AI", 0, ai)
        assert player.name == "AI"
        assert player.ai == ai

    def test_ai_decide_order_up(self) -> None:
        """Test AI order up decision."""
        ai = AIDecisionMaker()
        player = AIPlayer("AI", 0, ai)
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
        player = AIPlayer("AI", 0, ai)
        player.receive_hand([
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.KING),
        ])

        card = player.play_card(None, None, [])
        assert card in player.hand


class TestHumanPlayer:
    """Tests for HumanPlayer class."""

    def test_human_player_creation(self) -> None:
        """Test human player creation."""
        player = HumanPlayer("Human", 0)
        assert player.name == "Human"
        assert isinstance(player, Player)

    def test_set_ui(self) -> None:
        """Test setting UI."""
        player = HumanPlayer("Human", 0)
        ui = object()
        player.set_ui(ui)
        assert player._ui == ui

