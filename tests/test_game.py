"""Tests for the game module."""

import pytest
from euchre.game import EuchreGame
from euchre.models import PlayerType, Suit, Rank, Card


class TestEuchreGame:
    """Test the EuchreGame class."""
    
    def test_game_initialization(self) -> None:
        """Test game initialization."""
        game = EuchreGame()
        assert len(game.players) == 0
        assert game.game_state is None
        assert game.deck.size == 24  # 6 ranks * 4 suits
        
    def test_add_player(self) -> None:
        """Test adding players."""
        game = EuchreGame()
        game.add_player("Alice", PlayerType.HUMAN)
        game.add_player("Bob", PlayerType.AI)
        
        assert len(game.players) == 2
        assert game.players[0].name == "Alice"
        assert game.players[0].player_type == PlayerType.HUMAN
        assert game.players[1].name == "Bob"
        assert game.players[1].player_type == PlayerType.AI
        
    def test_start_new_game_with_four_players(self) -> None:
        """Test starting a game with exactly 4 players."""
        game = EuchreGame()
        game.add_player("Alice", PlayerType.HUMAN)
        game.add_player("Bob", PlayerType.AI)
        game.add_player("Charlie", PlayerType.AI)
        game.add_player("David", PlayerType.AI)
        
        game.start_new_game()
        
        # Check that game state is initialized
        assert game.game_state_manager is not None
        assert len(game.players) == 4
        assert game.round_number == 1
        
        # Check that each player has 5 cards
        for player in game.players:
            assert len(player.hand) == 5
            
        # Check that dealer selection is available (dealer will be set during gameplay)
        assert hasattr(game, 'dealer_selection')
        
    def test_start_new_game_with_wrong_number_of_players(self) -> None:
        """Test that starting a game with wrong number of players raises error."""
        game = EuchreGame()
        game.add_player("Alice", PlayerType.HUMAN)
        game.add_player("Bob", PlayerType.AI)
        
        # The new API enforces exactly 4 players
        with pytest.raises(ValueError, match="Euchre requires exactly 4 players"):
            game.start_new_game()
        
    def test_deal_cards(self) -> None:
        """Test that cards are dealt correctly."""
        game = EuchreGame()
        game.add_player("Alice", PlayerType.HUMAN)
        game.add_player("Bob", PlayerType.AI)
        game.add_player("Charlie", PlayerType.AI)
        game.add_player("David", PlayerType.AI)
        
        # Store original deck size for comparison
        original_deck_size = game.deck.size
        
        game.start_new_game()
        
        # Check that 20 cards were dealt (5 per player)
        # The deck should have 3 cards remaining (24 - 20 = 4, but 1 is the top card)
        assert game.deck.size == 3  # 24 - 20 - 1 = 3
        
        # Check that each player has exactly 5 cards
        for player in game.players:
            assert len(player.hand) == 5
            
        # Check that all dealt cards are unique
        all_dealt_cards = []
        for player in game.players:
            all_dealt_cards.extend(player.hand)
            
        assert len(all_dealt_cards) == 20
        assert len(set(all_dealt_cards)) == 20  # All unique
        
    def test_get_player_hand(self) -> None:
        """Test getting a specific player's hand."""
        game = EuchreGame()
        game.add_player("Alice", PlayerType.HUMAN)
        game.add_player("Bob", PlayerType.AI)
        game.add_player("Charlie", PlayerType.AI)
        game.add_player("David", PlayerType.AI)
        
        game.start_new_game()
        
        # Test getting a specific player's hand
        alice_hand = game.get_player_hand("Alice")
        assert len(alice_hand) == 5
        assert all(isinstance(card, Card) for card in alice_hand)
        
    def test_is_game_over(self) -> None:
        """Test game over detection."""
        game = EuchreGame()
        
        # Game not started - should not be over
        assert not game.is_game_over()
        
        # Add players and start game
        game.add_player("Alice", PlayerType.HUMAN)
        game.add_player("Bob", PlayerType.AI)
        game.add_player("Charlie", PlayerType.AI)
        game.add_player("David", PlayerType.AI)
        
        game.start_new_game()
        
        # Game just started - should not be over
        assert not game.is_game_over()
        
    def test_get_winner(self) -> None:
        """Test getting the winner."""
        game = EuchreGame()
        
        # Game not started
        winner = game.get_winner()
        assert winner == "Game not over"
        
        # Add players and start game
        game.add_player("Alice", PlayerType.HUMAN)
        game.add_player("Bob", PlayerType.AI)
        game.add_player("Charlie", PlayerType.AI)
        game.add_player("David", PlayerType.AI)
        
        game.start_new_game()
        
        # Game just started - no winner yet
        winner = game.get_winner()
        assert winner == "Game not over"
        
    def test_add_ai_player(self) -> None:
        """Test adding AI players with specific types."""
        game = EuchreGame()
        
        # Add different AI types
        game.add_ai_player("Alice", "level1_aggressive", 0.8)
        game.add_ai_player("Bob", "level1_conservative", 0.2)
        game.add_ai_player("Charlie", "level1_balanced", 0.5)
        game.add_ai_player("David", "level1_opportunistic", 0.6)
        
        assert len(game.players) == 4
        assert all(p.player_type == PlayerType.AI for p in game.players)
        assert all(p.name in ["Alice", "Bob", "Charlie", "David"] for p in game.players)


class TestPlayer:
    """Test the Player class."""
    
    def test_player_creation(self) -> None:
        """Test player creation."""
        from euchre.models import Player, PlayerType
        
        player = Player("Bob", PlayerType.AI)
        assert player.name == "Bob"
        assert player.player_type == PlayerType.AI
        assert len(player.hand) == 0
        assert player.tricks_won == 0
        assert player.score == 0
        assert not player.is_dealer
        assert player.team is None
        
    def test_add_and_remove_card(self) -> None:
        """Test adding and removing cards."""
        from euchre.models import Player, PlayerType, Card, Suit, Rank
        
        player = Player("Bob", PlayerType.AI)
        card = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        
        # Add card
        player.add_card(card)
        assert len(player.hand) == 1
        assert player.hand[0] == card
        
        # Remove card
        assert player.remove_card(card) is True
        assert len(player.hand) == 0
        
        # Try to remove non-existent card
        assert player.remove_card(card) is False
        
    def test_clear_hand(self) -> None:
        """Test clearing hand."""
        from euchre.models import Player, PlayerType, Card, Suit, Rank
        
        player = Player("Bob", PlayerType.AI)
        player.add_card(Card(rank=Rank.ACE, suit=Suit.HEARTS))
        player.add_card(Card(rank=Rank.KING, suit=Suit.HEARTS))
        
        assert len(player.hand) == 2
        player.clear_hand()
        assert len(player.hand) == 0
        
    def test_has_suit(self) -> None:
        """Test checking if player has a specific suit."""
        from euchre.models import Player, PlayerType, Card, Suit, Rank
        
        player = Player("Bob", PlayerType.AI)
        player.add_card(Card(rank=Rank.ACE, suit=Suit.HEARTS))
        player.add_card(Card(rank=Rank.KING, suit=Suit.DIAMONDS))
        
        assert player.has_suit(Suit.HEARTS) is True
        assert player.has_suit(Suit.DIAMONDS) is True
        assert player.has_suit(Suit.CLUBS) is False
        assert player.has_suit(Suit.SPADES) is False
        
    def test_get_cards_of_suit(self) -> None:
        """Test getting cards of a specific suit."""
        from euchre.models import Player, PlayerType, Card, Suit, Rank
        
        player = Player("Bob", PlayerType.AI)
        hearts_card = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        diamonds_card = Card(rank=Rank.KING, suit=Suit.DIAMONDS)
        player.add_card(hearts_card)
        player.add_card(diamonds_card)
        
        hearts_cards = player.get_cards_of_suit(Suit.HEARTS)
        assert len(hearts_cards) == 1
        assert hearts_cards[0] == hearts_card
        
        diamonds_cards = player.get_cards_of_suit(Suit.DIAMONDS)
        assert len(diamonds_cards) == 1
        assert diamonds_cards[0] == diamonds_card
        
        clubs_cards = player.get_cards_of_suit(Suit.CLUBS)
        assert len(clubs_cards) == 0
        
    def test_player_string_representation(self) -> None:
        """Test player string representation."""
        from euchre.models import Player, PlayerType
        
        player = Player("Bob", PlayerType.AI)
        # The actual string representation is "Bob (ai)"
        assert str(player) == "Bob (ai)"
        
        # Add a card and check representation changes
        from euchre.models import Card, Suit, Rank
        player.add_card(Card(rank=Rank.ACE, suit=Suit.HEARTS))
        # String representation should still be the same
        assert str(player) == "Bob (ai)"


class TestGameState:
    """Test the GameState class."""
    
    def test_game_state_creation(self) -> None:
        """Test game state creation."""
        from euchre.models import GameState, Player, PlayerType, Suit
        
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
            dealer_index=0,
            round_number=1,
            team1_score=0,
            team2_score=0
        )
        assert len(game_state.players) == 4
        assert game_state.round_number == 1
        assert game_state.team1_score == 0
        assert game_state.team2_score == 0
        assert game_state.current_player_index == 0
        
    def test_get_current_player(self) -> None:
        """Test getting current player."""
        from euchre.models import GameState, Player, PlayerType, Suit
        
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
            dealer_index=0,
            round_number=1,
            team1_score=0,
            team2_score=0
        )
        current_player = game_state.get_current_player()
        assert current_player.name == "Alice"
        
    def test_next_player(self) -> None:
        """Test moving to next player."""
        from euchre.models import GameState, Player, PlayerType, Suit
        
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
            dealer_index=0,
            round_number=1,
            team1_score=0,
            team2_score=0
        )
        
        # Check initial player
        assert game_state.get_current_player().name == "Alice"
        
        # Move to next player
        game_state.next_player()
        assert game_state.get_current_player().name == "Bob"
        
        # Move to next player
        game_state.next_player()
        assert game_state.get_current_player().name == "Charlie"
        
        # Move to next player
        game_state.next_player()
        assert game_state.get_current_player().name == "David"
        
        # Move to next player (should wrap around)
        game_state.next_player()
        assert game_state.get_current_player().name == "Alice" 