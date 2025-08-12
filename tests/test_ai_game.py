"""Tests for AI vs AI euchre game functionality."""

import pytest
from euchre.game import EuchreGame
from euchre.models import PlayerType, Suit, Rank, Card


class TestAIGame:
    """Test AI vs AI game functionality."""
    
    def test_ai_game_setup(self) -> None:
        """Test that AI game can be set up correctly."""
        game = EuchreGame()
        
        # Add 4 AI players
        game.add_player("North", PlayerType.AI)
        game.add_player("East", PlayerType.AI)
        game.add_player("South", PlayerType.AI)
        game.add_player("West", PlayerType.AI)
        
        assert len(game.players) == 4
        assert all(p.player_type == PlayerType.AI for p in game.players)
        
    def test_ai_game_start(self) -> None:
        """Test that AI game can start correctly."""
        game = EuchreGame()
        
        # Add 4 AI players
        game.add_player("North", PlayerType.AI)
        game.add_player("East", PlayerType.AI)
        game.add_player("South", PlayerType.AI)
        game.add_player("West", PlayerType.AI)
        
        game.start_new_game()
        
        assert game.game_state is not None
        assert game.game_state.trump_suit is not None
        assert all(len(p.hand) == 5 for p in game.players)
        
    def test_ai_round_play(self) -> None:
        """Test that AI can play a complete round."""
        game = EuchreGame()
        
        # Add 4 AI players
        game.add_player("North", PlayerType.AI)
        game.add_player("East", PlayerType.AI)
        game.add_player("South", PlayerType.AI)
        game.add_player("West", PlayerType.AI)
        
        game.start_new_game()
        
        # Play one round
        game.play_round()
        
        # Check that all hands are empty
        assert all(len(p.hand) == 0 for p in game.players)
        
        # Check that tricks were played
        assert len(game.tricks_this_round) == 5
        
    def test_ai_scoring(self) -> None:
        """Test that AI game scoring works correctly."""
        game = EuchreGame()
        
        # Add 4 AI players
        game.add_player("North", PlayerType.AI)
        game.add_player("East", PlayerType.AI)
        game.add_player("South", PlayerType.AI)
        game.add_player("West", PlayerType.AI)
        
        game.start_new_game()
        
        # Play multiple rounds until game ends
        rounds_played = 0
        while not game.is_game_over() and rounds_played < 20:  # Prevent infinite loop
            game.play_round()
            rounds_played += 1
            
        # Game should end with someone having 10+ points
        assert game.is_game_over()
        winner = game.get_winner()
        assert winner is not None
        
    def test_ai_trump_selection(self) -> None:
        """Test that AI can select trump suits."""
        game = EuchreGame()
        
        # Add 4 AI players
        game.add_player("North", PlayerType.AI)
        game.add_player("East", PlayerType.AI)
        game.add_player("South", PlayerType.AI)
        game.add_player("West", PlayerType.AI)
        
        game.start_new_game()
        
        # Trump should be selected
        assert game.game_state.trump_suit is not None
        assert game.game_state.trump_suit in list(Suit)
        
        # Some cards should be marked as trump
        trump_cards = []
        for player in game.players:
            trump_cards.extend([c for c in player.hand if c.is_trump])
            
        assert len(trump_cards) > 0
        
    def test_ai_trick_winning(self) -> None:
        """Test that AI can win tricks correctly."""
        game = EuchreGame()
        
        # Add 4 AI players
        game.add_player("North", PlayerType.AI)
        game.add_player("East", PlayerType.AI)
        game.add_player("South", PlayerType.AI)
        game.add_player("West", PlayerType.AI)
        
        game.start_new_game()
        
        # Play one round and get results
        round_results = game.play_round()
        
        # Check that tricks were won (total should be 5)
        total_tricks = sum(round_results)
        assert total_tricks == 5  # 5 tricks per round
        
        # Check that scores were updated
        if game.game_state:
            total_score = game.game_state.team1_score + game.game_state.team2_score
            assert total_score > 0  # At least one team should score 