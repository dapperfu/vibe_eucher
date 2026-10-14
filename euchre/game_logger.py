"""Game logger for recording euchre game details to a text file."""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from .models import Player, Card, Trick, GameState


class GameLogger:
    """Logs euchre game details to a structured text file."""
    
    def __init__(self, filename: Optional[str] = None) -> None:
        """Initialize the game logger.
        
        Parameters
        ----------
        filename : Optional[str]
            The filename to log to. If None, generates a timestamped filename.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"euchre_game_{timestamp}.txt"
        
        self.filename = filename
        self.game_start_time = datetime.now()
        self.round_logs: List[Dict[str, Any]] = []
        
    def log_game_start(self, players: List[Player], dealer: Player) -> None:
        """Log the start of a new game.
        
        Parameters
        ----------
        players : List[Player]
            List of all players in the game
        dealer : Player
            The dealer for this game
        """
        with open(self.filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EUCHE GAME LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Game started: {self.game_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Players: {', '.join(p.name for p in players)}\n")
            f.write(f"Dealer: {dealer.name}\n")
            f.write("=" * 80 + "\n\n")
            
    def log_round_start(self, round_num: int, trump_suit: str, hands: Dict[str, List[Card]]) -> None:
        """Log the start of a new round.
        
        Parameters
        ----------
        round_num : int
            The round number
        trump_suit : str
            The trump suit for this round
        hands : Dict[str, List[Card]]
            Dictionary mapping player names to their hands
        """
        round_log = {
            'round_num': round_num,
            'trump_suit': trump_suit,
            'hands': hands,
            'tricks': [],
            'final_scores': {}
        }
        self.round_logs.append(round_log)
        
        with open(self.filename, 'a') as f:
            f.write(f"ROUND {round_num}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Trump: {trump_suit}\n\n")
            
            # Log hands
            f.write("Initial Hands:\n")
            for player_name, hand in hands.items():
                f.write(f"  {player_name}: {' '.join(card.short_str() for card in hand)}\n")
            f.write("\n")
            
    def log_trick(self, trick_num: int, trick: Trick, winner: Player, winning_card: Card) -> None:
        """Log a completed trick.
        
        Parameters
        ----------
        trick_num : int
            The trick number within the round
        trick : Trick
            The completed trick
        winner : Player
            The player who won the trick
        winning_card : Card
            The card that won the trick
        """
        if not self.round_logs:
            return
            
        current_round = self.round_logs[-1]
        trick_info = {
            'trick_num': trick_num,
            'cards_played': [(p.name, c.short_str()) for p, c in trick.cards_played],
            'winner': winner.name,
            'winning_card': winning_card.short_str()
        }
        current_round['tricks'].append(trick_info)
        
        with open(self.filename, 'a') as f:
            f.write(f"Trick {trick_num}:\n")
            for player_name, card_str in trick_info['cards_played']:
                f.write(f"  {player_name}: {card_str}")
                if player_name == winner.name:
                    f.write(" ← WINNER")
                f.write("\n")
            f.write(f"Winner: {winner.name} with {winning_card.short_str()}\n\n")
            
    def log_round_end(self, round_num: int, final_scores: Dict[str, int], team_scores: Dict[str, int]) -> None:
        """Log the end of a round with final scores.
        
        Parameters
        ----------
        round_num : int
            The round number
        final_scores : Dict[str, int]
            Dictionary mapping player names to their trick counts
        team_scores : Dict[str, int]
            Dictionary mapping team names to their scores
        """
        if not self.round_logs:
            return
            
        current_round = self.round_logs[-1]
        current_round['final_scores'] = final_scores
        
        with open(self.filename, 'a') as f:
            f.write("Round Summary:\n")
            f.write("  Player Trick Counts:\n")
            for player_name, tricks in final_scores.items():
                f.write(f"    {player_name}: {tricks} tricks\n")
            f.write("  Team Scores:\n")
            for team_name, score in team_scores.items():
                f.write(f"    {team_name}: {score} points\n")
            f.write("\n")
            
    def log_game_end(self, winner: str, final_team_scores: Dict[str, int]) -> None:
        """Log the end of the game.
        
        Parameters
        ----------
        winner : str
            The winning team
        final_team_scores : Dict[str, int]
            Final team scores
        """
        game_end_time = datetime.now()
        duration = game_end_time - self.game_start_time
        
        with open(self.filename, 'a') as f:
            f.write("=" * 80 + "\n")
            f.write("GAME END\n")
            f.write("=" * 80 + "\n")
            f.write(f"Winner: {winner}\n")
            f.write(f"Final Scores:\n")
            for team_name, score in final_team_scores.items():
                f.write(f"  {team_name}: {score} points\n")
            f.write(f"Game Duration: {duration}\n")
            f.write(f"Game ended: {game_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
    def write_summary_table(self) -> None:
        """Write a summary table showing all rounds and results."""
        if not self.round_logs:
            return
            
        with open(self.filename, 'a') as f:
            f.write("SUMMARY TABLE\n")
            f.write("=" * 80 + "\n")
            
            # Header
            f.write(f"{'Round':<6} {'Trump':<8}")
            for player_name in self.round_logs[0]['hands'].keys():
                f.write(f"{player_name:<8}")
            f.write(f"{'Team1':<6} {'Team2':<6}\n")
            
            # Separator
            f.write("-" * 80 + "\n")
            
            # Round data
            for round_log in self.round_logs:
                round_num = round_log['round_num']
                trump = round_log['trump_suit']
                
                # Get trick counts for each player
                trick_counts = []
                for player_name in round_log['hands'].keys():
                    count = round_log['final_scores'].get(player_name, 0)
                    trick_counts.append(str(count))
                
                # Get team scores (North+South vs East+West)
                if len(trick_counts) >= 4:
                    team1_score = int(trick_counts[0]) + int(trick_counts[2])  # North + South
                    team2_score = int(trick_counts[1]) + int(trick_counts[3])  # East + West
                else:
                    team1_score = 0
                    team2_score = 0
                
                f.write(f"{round_num:<6} {trump:<8}")
                for count in trick_counts:
                    f.write(f"{count:<8}")
                f.write(f"{team1_score:<6} {team2_score:<6}\n")
                
            f.write("=" * 80 + "\n\n")
            
    def get_log_filename(self) -> str:
        """Get the current log filename.
        
        Returns
        -------
        str
            The filename being used for logging
        """
        return self.filename 