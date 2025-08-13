"""Tests for the game module."""

import pytest
from euchre.game import EuchreGame, AIPlayer
from euchre.models import PlayerType, Suit, Rank, Card


class TestEuchreGame:
    """Test the EuchreGame class."""
    
    def test_game_initialization(self) -> None:
        """Test game initialization."""
        game = EuchreGame()
        assert len(game.players) == 0
        assert game.game_state is None
        assert len(game.deck) == 24  # 6 ranks * 4 suits
        
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
        
        assert game.game_state is not None
        assert len(game.game_state.players) == 4
        assert game.game_state.round_number == 1
        assert game.game_state.team1_score == 0
        assert game.game_state.team2_score == 0
        
        # Check that each player has 5 cards
        for player in game.players:
            assert len(player.hand) == 5
            
        # Check that dealer is set
        dealer_count = sum(1 for p in game.players if p.is_dealer)
        assert dealer_count == 1
        
    def test_start_new_game_with_wrong_number_of_players(self) -> None:
        """Test that starting a game with wrong number of players raises error."""
        game = EuchreGame()
        game.add_player("Alice", PlayerType.HUMAN)
        game.add_player("Bob", PlayerType.AI)
        
        with pytest.raises(ValueError, match="Euchre requires exactly 4 players"):
            game.start_new_game()
            
    def test_deal_cards(self) -> None:
        """Test that cards are dealt correctly."""
        game = EuchreGame()
        game.add_player("Alice", PlayerType.HUMAN)
        game.add_player("Bob", PlayerType.AI)
        game.add_player("Charlie", PlayerType.AI)
        game.add_player("David", PlayerType.AI)
        
        # Store original deck for comparison
        original_deck = game.deck.copy()
        
        game.start_new_game()
        
        # Check that 20 cards were dealt (5 per player)
        assert len(game.deck) == 4  # 24 - 20 = 4
        
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
        
        alice_hand = game.get_player_hand("Alice")
        assert len(alice_hand) == 5
        
        # Test getting hand for non-existent player
        non_existent_hand = game.get_player_hand("Eve")
        assert len(non_existent_hand) == 0
        
    def test_play_ai_turn(self) -> None:
        """Test AI turn playing."""
        game = EuchreGame()
        game.add_player("Alice", PlayerType.HUMAN)
        game.add_player("Bob", PlayerType.AI)
        game.add_player("Charlie", PlayerType.AI)
        game.add_player("David", PlayerType.AI)
        
        game.start_new_game()
        
        bob = game.players[1]  # Assuming Bob is the second player
        original_hand_size = len(bob.hand)
        
        card = game.play_ai_turn(bob)
        assert isinstance(card, Card)
        assert len(bob.hand) == original_hand_size - 1
        
    def test_play_ai_turn_with_empty_hand(self) -> None:
        """Test AI turn playing with empty hand raises error."""
        game = EuchreGame()
        game.add_player("Alice", PlayerType.HUMAN)
        game.add_player("Bob", PlayerType.AI)
        game.add_player("Charlie", PlayerType.AI)
        game.add_player("David", PlayerType.AI)
        
        game.start_new_game()
        
        bob = game.players[1]
        bob.clear_hand()  # Clear Bob's hand
        
        with pytest.raises(ValueError, match="AI player has no cards to play"):
            game.play_ai_turn(bob)
            
    def test_is_game_over(self) -> None:
        """Test game over detection."""
        game = EuchreGame()
        
        # Game not started
        assert game.is_game_over()
        
        # Start game
        game.add_player("Alice", PlayerType.HUMAN)
        game.add_player("Bob", PlayerType.AI)
        game.add_player("Charlie", PlayerType.AI)
        game.add_player("David", PlayerType.AI)
        game.start_new_game()
        
        # Game just started, not over
        assert not game.is_game_over()
        
        # Simulate game over by setting scores
        if game.game_state:
            game.game_state.team1_score = 10
            assert game.is_game_over()
            
            game.game_state.team1_score = 0
            game.game_state.team2_score = 10
            assert game.is_game_over()
            
    def test_get_winner(self) -> None:
        """Test getting the winner."""
        game = EuchreGame()
        
        # Game not started
        assert game.get_winner() is None
        
        # Start game
        game.add_player("Alice", PlayerType.HUMAN)
        game.add_player("Bob", PlayerType.AI)
        game.add_player("Charlie", PlayerType.AI)
        game.add_player("David", PlayerType.AI)
        game.start_new_game()
        
        # Game just started, no winner
        assert game.get_winner() is None
        
        # Simulate team 1 winning
        if game.game_state:
            game.game_state.team1_score = 10
            assert game.get_winner() == "Team 1"
            
            # Simulate team 2 winning
            game.game_state.team1_score = 0
            game.game_state.team2_score = 10
            assert game.get_winner() == "Team 2"


class TestAIPlayer:
    """Test the AIPlayer class."""
    
    def test_ai_player_creation(self) -> None:
        """Test AI player creation."""
        from euchre.models import Player, PlayerType
        
        player = Player("Bob", PlayerType.AI)
        ai_player = AIPlayer(player)
        
        assert ai_player.player == player
        assert ai_player.player.name == "Bob"
        
    def test_choose_card_to_play_with_lead_suit(self) -> None:
        """Test AI choosing card when it has the lead suit."""
        from euchre.models import Player, PlayerType, Card, Suit, Rank
        
        player = Player("Bob", PlayerType.AI)
        card1 = Card(rank=Rank.NINE, suit=Suit.HEARTS)
        card2 = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        card3 = Card(rank=Rank.KING, suit=Suit.DIAMONDS)
        
        player.add_card(card1)
        player.add_card(card2)
        player.add_card(card3)
        
        ai_player = AIPlayer(player)
        
        # AI should play highest card of lead suit
        chosen_card = ai_player.choose_card_to_play(Suit.HEARTS, None)
        assert chosen_card == card2  # Ace of Hearts
        
    def test_choose_card_to_play_without_lead_suit(self) -> None:
        """Test AI choosing card when it doesn't have the lead suit."""
        from euchre.models import Player, PlayerType, Card, Suit, Rank
        
        player = Player("Bob", PlayerType.AI)
        card1 = Card(rank=Rank.NINE, suit=Suit.HEARTS)
        card2 = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        card3 = Card(rank=Rank.KING, suit=Suit.DIAMONDS)
        
        player.add_card(card1)
        player.add_card(card2)
        player.add_card(card3)
        
        ai_player = AIPlayer(player)
        
        # AI should play lowest card when it can't follow suit
        chosen_card = ai_player.choose_card_to_play(Suit.CLUBS, None)
        assert chosen_card == card1  # Nine of Hearts (lowest)
        
    def test_should_order_up(self) -> None:
        """Test AI decision to order up."""
        from euchre.models import Player, PlayerType, Card, Suit, Rank
        
        player = Player("Bob", PlayerType.AI)
        card1 = Card(rank=Rank.NINE, suit=Suit.HEARTS)
        card2 = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        
        player.add_card(card1)
        player.add_card(card2)
        
        ai_player = AIPlayer(player)
        
        # AI has 2+ cards of the suit, should order up
        top_card = Card(rank=Rank.TEN, suit=Suit.HEARTS)
        assert ai_player.should_order_up(top_card)
        
        # AI has only 1 card of the suit, should not order up
        player.remove_card(card1)
        assert not ai_player.should_order_up(top_card) 