"""Tests for AI player profiles."""

import pytest
from euchre.ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI
from euchre.models import Card, Suit, Rank


class TestAIAIProfiles:
    """Test AI player profiles functionality."""
    
    def test_aggressive_ai_creation(self) -> None:
        """Test that aggressive AI can be created with different risk ratios."""
        ai = AggressiveAI("Test", 0.9)
        assert ai.name == "Test"
        assert ai.risk_ratio == 0.9
        assert ai.player_type.value == "ai"
        
    def test_conservative_ai_creation(self) -> None:
        """Test that conservative AI can be created with different risk ratios."""
        ai = ConservativeAI("Test", 0.1)
        assert ai.name == "Test"
        assert ai.risk_ratio == 0.1
        assert ai.player_type.value == "ai"
        
    def test_balanced_ai_creation(self) -> None:
        """Test that balanced AI can be created with different risk ratios."""
        ai = BalancedAI("Test", 0.5)
        assert ai.name == "Test"
        assert ai.risk_ratio == 0.5
        assert ai.player_type.value == "ai"
        
    def test_opportunistic_ai_creation(self) -> None:
        """Test that opportunistic AI can be created with different risk ratios."""
        ai = OpportunisticAI("Test", 0.7)
        assert ai.name == "Test"
        assert ai.risk_ratio == 0.7
        assert ai.player_type.value == "ai"
        
    def test_risk_ratio_clamping(self) -> None:
        """Test that risk ratios are clamped to [0, 1] range."""
        # Test values below 0
        ai = AggressiveAI("Test", -0.5)
        assert ai.risk_ratio == 0.0
        
        # Test values above 1
        ai = ConservativeAI("Test", 1.5)
        assert ai.risk_ratio == 1.0
        
        # Test valid values
        ai = BalancedAI("Test", 0.75)
        assert ai.risk_ratio == 0.75
        
    def test_aggressive_ai_trump_decision(self) -> None:
        """Test that aggressive AI makes appropriate trump decisions."""
        ai = AggressiveAI("Test", 0.8)
        
        # Add some cards to hand
        ai.add_card(Card(Rank.ACE, Suit.HEARTS))
        ai.add_card(Card(Rank.KING, Suit.HEARTS))
        ai.add_card(Card(Rank.JACK, Suit.DIAMONDS))  # Left bower for hearts
        
        # Test with top card of hearts (should order up with 3 potential trump cards)
        top_card = Card(Rank.QUEEN, Suit.HEARTS)
        assert ai.should_order_up(top_card) is True
        
        # Test with lower risk ratio
        ai.risk_ratio = 0.3
        assert ai.should_order_up(top_card) is True  # Still has 3+ cards
        
    def test_conservative_ai_trump_decision(self) -> None:
        """Test that conservative AI makes appropriate trump decisions."""
        ai = ConservativeAI("Test", 0.2)
        
        # Add some cards to hand
        ai.add_card(Card(Rank.ACE, Suit.HEARTS))
        ai.add_card(Card(Rank.KING, Suit.HEARTS))
        
        # Test with top card of hearts (should NOT order up with only 2 cards)
        top_card = Card(Rank.QUEEN, Suit.HEARTS)
        assert ai.should_order_up(top_card) is False
        
        # Add more cards
        ai.add_card(Card(Rank.JACK, Suit.DIAMONDS))  # Left bower
        ai.add_card(Card(Rank.TEN, Suit.HEARTS))
        
        # Now should order up with 4 potential trump cards
        assert ai.should_order_up(top_card) is True
        
    def test_ai_card_playing(self) -> None:
        """Test that AI profiles can choose cards to play."""
        ai = AggressiveAI("Test", 0.8)
        
        # Add cards to hand
        ai.add_card(Card(Rank.ACE, Suit.HEARTS))
        ai.add_card(Card(Rank.KING, Suit.CLUBS))
        ai.add_card(Card(Rank.NINE, Suit.SPADES))
        
        # Test leading (no lead suit)
        card = ai.choose_card_to_play(None, Suit.HEARTS)
        # Should play highest card (Ace of Hearts)
        assert card.rank == Rank.ACE
        assert card.suit == Suit.HEARTS
        
        # Test following suit
        card = ai.choose_card_to_play(Suit.HEARTS, Suit.CLUBS)
        # Should play highest card of hearts
        assert card.rank == Rank.ACE
        assert card.suit == Suit.HEARTS
        
    def test_ai_profiles_integration(self) -> None:
        """Test that AI profiles can be used in the game system."""
        from euchre.game import EuchreGame
        
        game = EuchreGame(quiet_mode=True)
        
        # Add different AI profiles
        game.add_ai_player("North", "aggressive", 0.8)
        game.add_ai_player("East", "conservative", 0.2)
        game.add_ai_player("South", "balanced", 0.5)
        game.add_ai_player("West", "opportunistic", 0.6)
        
        assert len(game.players) == 4
        assert all(p.player_type.value == "ai" for p in game.players)
        
        # Verify profile types
        assert isinstance(game.players[0], AggressiveAI)
        assert isinstance(game.players[1], ConservativeAI)
        assert isinstance(game.players[2], BalancedAI)
        assert isinstance(game.players[3], OpportunisticAI)
        
        # Verify risk ratios
        assert game.players[0].risk_ratio == 0.8
        assert game.players[1].risk_ratio == 0.2
        assert game.players[2].risk_ratio == 0.5
        assert game.players[3].risk_ratio == 0.6 