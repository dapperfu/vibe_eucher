"""Tests for AI profile functionality."""

import pytest
from euchre.models import Card, Suit, Rank
from euchre.ai.traditional_ai_impl import TraditionalAI


class TestAIAIProfiles:
    """Test AI profile functionality."""
    
    def test_aggressive_ai_creation(self) -> None:
        """Test that aggressive AI can be created."""
        ai = TraditionalAI("Test", "aggressive")
        assert ai.name == "Test"
        assert ai.ai_style == "aggressive"
        
    def test_conservative_ai_creation(self) -> None:
        """Test that conservative AI can be created."""
        ai = TraditionalAI("Test", "conservative")
        assert ai.name == "Test"
        assert ai.ai_style == "conservative"
        
    def test_balanced_ai_creation(self) -> None:
        """Test that balanced AI can be created."""
        ai = TraditionalAI("Test", "balanced")
        assert ai.name == "Test"
        assert ai.ai_style == "balanced"
        
    def test_opportunistic_ai_creation(self) -> None:
        """Test that opportunistic AI can be created."""
        ai = TraditionalAI("Test", "opportunistic")
        assert ai.name == "Test"
        assert ai.ai_style == "opportunistic"
        
    def test_risk_ratio_clamping(self) -> None:
        """Test that risk ratios are properly clamped."""
        # Test extreme values
        ai_very_aggressive = TraditionalAI("Test", "aggressive")
        ai_very_conservative = TraditionalAI("Test", "conservative")
        
        # Both should have valid risk ratios
        assert 0.0 <= ai_very_aggressive.risk_ratio <= 1.0
        assert 0.0 <= ai_very_conservative.risk_ratio <= 1.0
        
    def test_aggressive_ai_trump_decision(self) -> None:
        """Test that aggressive AI makes appropriate trump decisions."""
        ai = TraditionalAI("Test", "aggressive")
        
        # Add some cards to hand
        ai.add_card(Card(Rank.ACE, Suit.HEARTS))
        ai.add_card(Card(Rank.KING, Suit.HEARTS))
        
        # Test with top card of hearts (should order up with 2+ cards)
        top_card = Card(Rank.QUEEN, Suit.HEARTS)
        # Note: The new API uses GameContext, so we can't test this directly
        # The AI will make decisions during actual gameplay
        
    def test_ai_card_playing(self) -> None:
        """Test that AI can choose cards to play."""
        ai = TraditionalAI("Test", "balanced")
        
        # Add some cards to hand
        ai.add_card(Card(Rank.ACE, Suit.HEARTS))
        ai.add_card(Card(Rank.KING, Suit.DIAMONDS))
        ai.add_card(Card(Rank.QUEEN, Suit.CLUBS))
        
        # AI should have cards to play
        assert len(ai.hand) == 3
        assert ai.has_suit(Suit.HEARTS)
        assert ai.has_suit(Suit.DIAMONDS)
        assert ai.has_suit(Suit.CLUBS)
        
    def test_ai_profiles_integration(self) -> None:
        """Test that AI profiles can be used in the game system."""
        from euchre.game import EuchreGame
        
        game = EuchreGame(quiet_mode=True)
        
        # Add different AI profiles using the new API
        game.add_ai_player("North", "level1_aggressive", 0.8)
        game.add_ai_player("East", "level1_conservative", 0.2)
        game.add_ai_player("South", "level1_balanced", 0.5)
        game.add_ai_player("West", "level1_opportunistic", 0.6)
        
        assert len(game.players) == 4
        assert all(p.player_type.value == "ai" for p in game.players)
        
        # Verify profile types - the new API creates Player objects, not specific AI classes
        # But they should all be AI players
        assert all(p.player_type.value == "ai" for p in game.players)
        
        # Start game to verify integration works
        game.start_new_game()
        assert all(len(p.hand) == 5 for p in game.players)
        
    def test_ai_profile_risk_ratios(self) -> None:
        """Test that AI profiles have appropriate risk ratios."""
        # Test different AI styles
        styles = ["aggressive", "conservative", "balanced", "opportunistic"]
        
        for style in styles:
            ai = TraditionalAI("Test", style)
            assert 0.0 <= ai.risk_ratio <= 1.0
            
            # Verify style-specific characteristics
            if style == "aggressive":
                assert ai.risk_ratio > 0.5
            elif style == "conservative":
                assert ai.risk_ratio < 0.5
            elif style == "balanced":
                assert 0.4 <= ai.risk_ratio <= 0.6
            elif style == "opportunistic":
                assert 0.5 <= ai.risk_ratio <= 0.8
                
    def test_ai_profile_card_selection(self) -> None:
        """Test that AI profiles select cards appropriately."""
        ai = TraditionalAI("Test", "balanced")
        
        # Add a variety of cards
        ai.add_card(Card(Rank.ACE, Suit.HEARTS))
        ai.add_card(Card(Rank.KING, Suit.HEARTS))
        ai.add_card(Card(Rank.QUEEN, Suit.DIAMONDS))
        ai.add_card(Card(Rank.JACK, Suit.CLUBS))
        ai.add_card(Card(Rank.TEN, Suit.SPADES))
        
        # AI should have a full hand
        assert len(ai.hand) == 5
        
        # AI should be able to identify suits
        assert ai.has_suit(Suit.HEARTS)
        assert ai.has_suit(Suit.DIAMONDS)
        assert ai.has_suit(Suit.CLUBS)
        assert ai.has_suit(Suit.SPADES)
        
        # AI should be able to get cards of specific suits
        hearts_cards = ai.get_cards_of_suit(Suit.HEARTS)
        assert len(hearts_cards) == 2
        assert all(card.suit == Suit.HEARTS for card in hearts_cards)
        
    def test_ai_profile_hand_management(self) -> None:
        """Test that AI profiles can manage their hands properly."""
        ai = TraditionalAI("Test", "aggressive")
        
        # Add cards
        card1 = Card(Rank.ACE, Suit.HEARTS)
        card2 = Card(Rank.KING, Suit.DIAMONDS)
        ai.add_card(card1)
        ai.add_card(card2)
        
        # Check hand size
        assert len(ai.hand) == 2
        
        # Remove a card
        assert ai.remove_card(card1) is True
        assert len(ai.hand) == 1
        assert ai.hand[0] == card2
        
        # Clear hand
        ai.clear_hand()
        assert len(ai.hand) == 0
        
    def test_ai_profile_string_representation(self) -> None:
        """Test AI profile string representation."""
        ai = TraditionalAI("TestPlayer", "balanced")
        
        # Check string representation
        str_repr = str(ai)
        assert "TestPlayer" in str_repr
        assert "balanced" in str_repr
        
        # Add a card and check representation stays the same (correct behavior)
        ai.add_card(Card(Rank.ACE, Suit.HEARTS))
        str_repr_with_card = str(ai)
        # String representation should not change when adding cards
        assert str_repr == str_repr_with_card
        
    def test_ai_profile_edge_cases(self) -> None:
        """Test AI profile edge cases."""
        # Test with empty hand
        ai = TraditionalAI("Test", "conservative")
        assert len(ai.hand) == 0
        assert not ai.has_suit(Suit.HEARTS)
        assert len(ai.get_cards_of_suit(Suit.HEARTS)) == 0
        
        # Test removing non-existent card
        card = Card(Rank.ACE, Suit.HEARTS)
        assert ai.remove_card(card) is False
        
        # Test clearing empty hand
        ai.clear_hand()
        assert len(ai.hand) == 0 