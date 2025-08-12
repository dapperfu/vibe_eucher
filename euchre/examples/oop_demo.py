#!/usr/bin/env python3
"""
Demonstration of the enhanced OOP structure for the Euchre game.

This script shows how to use the various classes to create and manage
games, demonstrating the object-oriented design principles.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import euchre
sys.path.insert(0, str(Path(__file__).parent.parent))

from euchre.core import (
    Deck, GameFlow, GamePhase, GameSession, SessionConfig
)
from euchre.models import Player, PlayerType, Card, Suit, Rank
from euchre.ai.ai_factory import AIFactory
from euchre.utils.logging_config import setup_logging


def demonstrate_card_oop():
    """Demonstrate the enhanced Card class functionality."""
    print("=== Card OOP Demonstration ===")
    
    # Create some cards
    ace_hearts = Card(Rank.ACE, Suit.HEARTS)
    king_clubs = Card(Rank.KING, Suit.CLUBS)
    jack_spades = Card(Rank.JACK, Suit.SPADES)
    
    # Use the new properties and methods
    print(f"Ace of Hearts: value={ace_hearts.value}, is_high={ace_hearts.is_high_card}")
    print(f"King of Clubs: is_face_card={king_clubs.is_face_card}")
    print(f"Jack of Spades: is_low={jack_spades.is_low_card}")
    
    # Test trump card detection
    print(f"\nTrump card detection:")
    print(f"Ace of Hearts is trump for Hearts: {ace_hearts.is_trump_card(Suit.HEARTS)}")
    print(f"Jack of Spades is trump for Hearts (left bower): {jack_spades.is_trump_card(Suit.HEARTS)}")
    
    # Test card comparison
    print(f"\nCard comparison:")
    print(f"Ace beats King: {ace_hearts.beats(king_clubs, Suit.HEARTS, Suit.HEARTS)}")
    print(f"King beats Jack: {king_clubs.beats(jack_spades, Suit.HEARTS, Suit.HEARTS)}")


def demonstrate_deck_oop():
    """Demonstrate the enhanced Deck class functionality."""
    print("\n=== Deck OOP Demonstration ===")
    
    # Create and manipulate deck
    deck = Deck()
    print(f"Initial deck size: {deck.size}")
    print(f"Deck is full: {deck.is_full}")
    
    # Shuffle multiple times
    deck.shuffle_times(3)
    print("Deck shuffled 3 times")
    
    # Cut the deck
    deck.cut(12)
    print("Deck cut at position 12")
    
    # Peek at cards
    top_card = deck.peek_top_card()
    bottom_card = deck.peek_bottom_card()
    print(f"Top card: {top_card}")
    print(f"Bottom card: {bottom_card}")
    
    # Deal some cards
    hands = deck.deal_cards(4, 3)
    print(f"Dealt 3 cards to 4 players")
    print(f"Remaining deck size: {deck.size}")
    
    # Get card counts
    suit_counts = deck.get_card_count_by_suit()
    rank_counts = deck.get_card_count_by_rank()
    print(f"Cards by suit: {suit_counts}")
    print(f"Cards by rank: {rank_counts}")
    
    # Reset deck
    deck.reset_and_shuffle()
    print(f"Deck reset and shuffled, size: {deck.size}")


def demonstrate_player_oop():
    """Demonstrate the enhanced Player class functionality."""
    print("\n=== Player OOP Demonstration ===")
    
    # Create players
    alice = Player("Alice", PlayerType.AI)
    bob = Player("Bob", PlayerType.AI)
    
    # Set teams
    alice.set_team(0)
    bob.set_team(1)
    print(f"Alice: {alice.get_team_name()}")
    print(f"Bob: {bob.get_team_name()}")
    
    # Add cards to hands
    cards = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.DIAMONDS),
        Card(Rank.JACK, Suit.CLUBS),
        Card(Rank.TEN, Suit.SPADES)
    ]
    
    for card in cards:
        alice.add_card(card)
    
    print(f"Alice's hand: {[str(card) for card in alice.hand]}")
    print(f"Hand size: {alice.get_hand_size()}")
    print(f"Has cards: {alice.has_cards}")
    print(f"Hand is full: {alice.hand_is_full}")
    
    # Test suit checking
    print(f"Has hearts: {alice.has_suit(Suit.HEARTS)}")
    print(f"Has clubs: {alice.has_suit(Suit.CLUBS)}")
    
    # Test card retrieval
    hearts = alice.get_cards_of_suit(Suit.HEARTS)
    high_cards = alice.get_high_cards(11)
    print(f"Hearts: {[str(card) for card in hearts]}")
    print(f"High cards (11+): {[str(card) for card in high_cards]}")
    
    # Test legal plays
    legal_plays = alice.get_legal_plays(Suit.HEARTS, Suit.DIAMONDS)
    print(f"Legal plays when hearts led: {[str(card) for card in legal_plays]}")
    
    # Evaluate hand strength
    strength = alice.evaluate_hand_strength(Suit.DIAMONDS)
    print(f"Hand strength with Diamonds trump: {strength:.1f}")


def demonstrate_game_flow_oop():
    """Demonstrate the GameFlow class functionality."""
    print("\n=== Game Flow OOP Demonstration ===")
    
    # Create players
    players = [
        Player("Alice", PlayerType.AI),
        Player("Bob", PlayerType.AI),
        Player("Charlie", PlayerType.AI),
        Player("David", PlayerType.AI)
    ]
    
    # Create game flow
    game_flow = GameFlow(players, verbose=True)
    
    # Add phase callbacks
    def on_trump_selection(state):
        print(f"  → Entered {state.phase.name} phase")
    
    def on_playing_tricks(state):
        print(f"  → Entered {state.phase.name} phase")
    
    game_flow.add_phase_callback(GamePhase.TRUMP_SELECTION, on_trump_selection)
    game_flow.add_phase_callback(GamePhase.PLAYING_TRICKS, on_playing_tricks)
    
    # Start game flow
    print("Starting game flow...")
    game_flow.start_game()
    
    # Set dealer
    game_flow.set_dealer(0)
    print(f"Dealer: {players[0].name}")
    
    # Start round
    game_flow.start_round()
    
    # Set trump
    game_flow.set_trump_suit(Suit.HEARTS, players[1])
    
    # Start trick
    game_flow.start_trick()
    
    # Get current player
    current_player = game_flow.get_current_player()
    print(f"Current player: {current_player.name}")
    
    # Get game statistics
    stats = game_flow.get_game_statistics()
    print(f"Game statistics: {stats}")


def demonstrate_game_session_oop():
    """Demonstrate the GameSession class functionality."""
    print("\n=== Game Session OOP Demonstration ===")
    
    # Create session configuration
    config = SessionConfig(
        num_games=5,
        save_results=False,
        output_dir="demo_sessions",
        log_level="INFO",
        parallel_games=False
    )
    
    # Create game session
    session = GameSession(config)
    
    # Player factory function
    def create_players():
        return [
            Player("Alice", PlayerType.AI),
            Player("Bob", PlayerType.AI),
            Player("Charlie", PlayerType.AI),
            Player("David", PlayerType.AI)
        ]
    
    print("Running game session...")
    results = session.run_session(create_players)
    
    print(f"Session completed with {len(results)} games")
    
    # Get statistics
    stats = session.get_statistics()
    print(f"Session statistics: {stats}")
    
    # Print summary
    session.print_summary()


def main():
    """Run all demonstrations."""
    print("Euchre OOP Structure Demonstration")
    print("=" * 50)
    
    try:
        demonstrate_card_oop()
        demonstrate_deck_oop()
        demonstrate_player_oop()
        demonstrate_game_flow_oop()
        demonstrate_game_session_oop()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("\nThe enhanced OOP structure provides:")
        print("✅ Encapsulated card logic with properties and methods")
        print("✅ Robust deck management with multiple operations")
        print("✅ Enhanced player capabilities and team management")
        print("✅ Structured game flow with phase management")
        print("✅ Multi-game session support for AI training")
        print("✅ Clean interfaces and separation of concerns")
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 