"""Tests for the models module."""

import pytest
from euchre.models import (
    Player, PlayerType, Card, Suit, Rank, GameState
)


class TestCard:
    """Test the Card class."""
    
    def test_card_creation(self) -> None:
        """Test creating a card."""
        card = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        assert card.rank == Rank.ACE
        assert card.suit == Suit.HEARTS
        assert not card.is_trump
        
    def test_card_string_representation(self) -> None:
        """Test card string representation."""
        card = Card(rank=Rank.JACK, suit=Suit.CLUBS)
        assert str(card) == "Jack of Clubs"
        
        trump_card = Card(rank=Rank.ACE, suit=Suit.DIAMONDS, is_trump=True)
        assert str(trump_card) == "Ace of Diamonds (Trump)"
        
    def test_card_comparison(self) -> None:
        """Test card comparison."""
        card1 = Card(rank=Rank.NINE, suit=Suit.HEARTS)
        card2 = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        assert card1 < card2
        assert card2 > card1


class TestPlayer:
    """Test the Player class."""
    
    def test_player_creation(self) -> None:
        """Test creating a player."""
        player = Player("Alice", PlayerType.HUMAN)
        assert player.name == "Alice"
        assert player.player_type == PlayerType.HUMAN
        assert len(player.hand) == 0
        assert player.tricks_won == 0
        assert not player.is_dealer
        
    def test_add_and_remove_card(self) -> None:
        """Test adding and removing cards."""
        player = Player("Bob", PlayerType.AI)
        card = Card(rank=Rank.TEN, suit=Suit.SPADES)
        
        player.add_card(card)
        assert len(player.hand) == 1
        assert card in player.hand
        
        player.remove_card(card)
        assert len(player.hand) == 0
        
    def test_clear_hand(self) -> None:
        """Test clearing the hand."""
        player = Player("Charlie", PlayerType.HUMAN)
        card1 = Card(rank=Rank.JACK, suit=Suit.HEARTS)
        card2 = Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS)
        
        player.add_card(card1)
        player.add_card(card2)
        assert len(player.hand) == 2
        
        player.clear_hand()
        assert len(player.hand) == 0
        
    def test_has_suit(self) -> None:
        """Test checking if player has a specific suit."""
        player = Player("David", PlayerType.AI)
        card = Card(rank=Rank.KING, suit=Suit.CLUBS)
        player.add_card(card)
        
        assert player.has_suit(Suit.CLUBS)
        assert not player.has_suit(Suit.HEARTS)
        
    def test_get_cards_of_suit(self) -> None:
        """Test getting cards of a specific suit."""
        player = Player("Eve", PlayerType.HUMAN)
        card1 = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        card2 = Card(rank=Rank.KING, suit=Suit.HEARTS)
        card3 = Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS)
        
        player.add_card(card1)
        player.add_card(card2)
        player.add_card(card3)
        
        hearts_cards = player.get_cards_of_suit(Suit.HEARTS)
        assert len(hearts_cards) == 2
        assert card1 in hearts_cards
        assert card2 in hearts_cards
        
    def test_player_string_representation(self) -> None:
        """Test player string representation."""
        player = Player("Frank", PlayerType.AI)
        assert str(player) == "Frank (ai)"
        assert "Frank" in repr(player)
        assert "ai" in repr(player)


class TestGameState:
    """Test the GameState class."""
    
    def test_game_state_creation(self) -> None:
        """Test creating a game state."""
        players = [
            Player("Alice", PlayerType.HUMAN),
            Player("Bob", PlayerType.AI),
            Player("Charlie", PlayerType.AI),
            Player("David", PlayerType.AI)
        ]
        
        game_state = GameState(
            players=players,
            current_player_index=0,
            trump_suit=None,
            dealer_index=3,
            round_number=1,
            team1_score=0,
            team2_score=0
        )
        
        assert game_state.current_player_index == 0
        assert game_state.dealer_index == 3
        assert game_state.round_number == 1
        assert game_state.team1_score == 0
        assert game_state.team2_score == 0
        
    def test_get_current_player(self) -> None:
        """Test getting the current player."""
        players = [
            Player("Alice", PlayerType.HUMAN),
            Player("Bob", PlayerType.AI)
        ]
        
        game_state = GameState(
            players=players,
            current_player_index=1,
            trump_suit=None,
            dealer_index=0,
            round_number=1,
            team1_score=0,
            team2_score=0
        )
        
        current_player = game_state.get_current_player()
        assert current_player.name == "Bob"
        
    def test_next_player(self) -> None:
        """Test moving to the next player."""
        players = [
            Player("Alice", PlayerType.HUMAN),
            Player("Bob", PlayerType.AI),
            Player("Charlie", PlayerType.AI)
        ]
        
        game_state = GameState(
            players=players,
            current_player_index=0,
            trump_suit=None,
            dealer_index=2,
            round_number=1,
            team1_score=0,
            team2_score=0
        )
        
        assert game_state.current_player_index == 0
        game_state.next_player()
        assert game_state.current_player_index == 1
        game_state.next_player()
        assert game_state.current_player_index == 2
        game_state.next_player()
        assert game_state.current_player_index == 0  # Wraps around 