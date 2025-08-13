#!/usr/bin/env python3
"""
Demo script for the enhanced ncurses euchre game.

This script demonstrates the new features:
- Human vs AI gameplay
- First-person perspective table layout
- Interactive card selection
- Proper table positioning (you at South, partner at North, opponents at East/West)
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import euchre modules
sys.path.append(str(Path.cwd()))

def demo_ai_vs_ai():
    """Demo the AI vs AI ncurses game."""
    print("🤖 AI vs AI Ncurses Game Demo")
    print("=" * 50)
    print("This will start a fully automated AI vs AI game")
    print("with a visual table representation.")
    print()
    print("Features:")
    print("  • Visual table layout with player positions")
    print("  • Color-coded cards and players")
    print("  • Real-time game state updates")
    print("  • Automatic gameplay with 1-second delays")
    print()
    
    input("Press Enter to start AI vs AI game (or Ctrl+C to cancel)...")
    
    try:
        from euchre.ncurses_game import NcursesGame
        game = NcursesGame(human_player_position=-1)  # AI vs AI mode
        game.run()
    except KeyboardInterrupt:
        print("\n❌ Demo cancelled by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_human_vs_ai():
    """Demo the human vs AI ncurses game."""
    print("🎮 Human vs AI Ncurses Game Demo")
    print("=" * 50)
    print("This will start an interactive human vs AI game")
    print("where you play against AI opponents.")
    print()
    print("Features:")
    print("  • First-person perspective (you at South)")
    print("  • Your partner across from you (North)")
    print("  • Opponents to your left and right (West/East)")
    print("  • Interactive card selection (press 1-5)")
    print("  • Visual table representation")
    print("  • Color-coded players and cards")
    print()
    print("Controls:")
    print("  • Press 1-5 to select cards from your hand")
    print("  • Press 'q' to quit the game")
    print()
    
    # Get player position preference
    print("Choose your position:")
    print("  0 = North (top of screen)")
    print("  1 = East (right side)")
    print("  2 = South (bottom - recommended)")
    print("  3 = West (left side)")
    print()
    
    while True:
        try:
            position = input("Enter position (0-3, default 2): ").strip()
            if not position:
                position = 2
            else:
                position = int(position)
            if 0 <= position <= 3:
                break
            else:
                print("❌ Invalid position. Please enter 0-3.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    print(f"\n🎯 Starting game with you at position {position}")
    print("Position names: 0=North, 1=East, 2=South, 3=West")
    print()
    
    input("Press Enter to start Human vs AI game (or Ctrl+C to cancel)...")
    
    try:
        from euchre.ncurses_game import NcursesGame
        game = NcursesGame(human_player_position=position)
        game.run()
    except KeyboardInterrupt:
        print("\n❌ Demo cancelled by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def main():
    """Main demo function."""
    print("🎴 Enhanced Ncurses Euchre Game Demo")
    print("=" * 60)
    print()
    print("This demo showcases the enhanced ncurses interface")
    print("with human player support and first-person perspective.")
    print()
    print("Choose a demo mode:")
    print("  1. AI vs AI Game (fully automated)")
    print("  2. Human vs AI Game (interactive)")
    print("  3. Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            demo_ai_vs_ai()
            break
        elif choice == "2":
            demo_human_vs_ai()
            break
        elif choice == "3":
            print("👋 Thanks for trying the demo!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main() 