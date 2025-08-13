#!/usr/bin/env python3
"""
Debug Single Game - Run one game with verbose output to see what's happening
"""

from euchre.game import EuchreGame
from euchre.ai.ai_factory import AIFactory


def main():
    """Run a single game with verbose output."""
    
    print("🔍 Debug Single Game")
    print("=" * 30)
    
    # Create AI players with different profiles
    players = [
        AIFactory.create_ai_player("Alice", "conservative", 0.15),
        AIFactory.create_ai_player("Bob", "balanced", 0.45),
        AIFactory.create_ai_player("Charlie", "aggressive", 0.75),
        AIFactory.create_ai_player("David", "opportunistic", 0.65),
    ]
    
    print("Created players:")
    for player in players:
        print(f"  {player.name}: {player.__class__.__name__} with risk ratio {player.risk_ratio}")
    
    print("\n🎯 Starting single game with verbose output...")
    print("=" * 50)
    
    # Create and run game with verbose output
    game = EuchreGame(players, verbose=True, very_verbose=True)
    
    try:
        game.start_new_game()
        game.run_full_game()
        
        print("\n" + "=" * 50)
        print("🎉 GAME COMPLETE!")
        print("=" * 50)
        
        # Show final results
        print(f"Final Scores:")
        for player in game.players:
            print(f"  {player.name}: {player.score}")
        
        # Determine winner
        team1_score = game.players[0].score + game.players[2].score
        team2_score = game.players[1].score + game.players[3].score
        
        print(f"\nTeam 1 (Alice & Charlie): {team1_score}")
        print(f"Team 2 (Bob & David): {team2_score}")
        
        if team1_score > team2_score:
            print("🏆 Team 1 (Alice & Charlie) wins!")
        else:
            print("🏆 Team 2 (Bob & David) wins!")
            
    except Exception as e:
        print(f"❌ Error during game: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 