"""
Logging configuration for the Euchre game.

This module provides centralized logging configuration with verbosity control
for different levels of detail in game output.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional, List, Dict, Any


class GameLogger:
    """Centralized logger for the Euchre game with verbosity control."""
    
    def __init__(self, verbose: bool = False, very_verbose: bool = False):
        """Initialize the game logger.
        
        Parameters
        ----------
        verbose : bool
            Enable verbose logging (INFO level)
        very_verbose : bool
            Enable very verbose logging (DEBUG level)
        """
        self.verbose = verbose
        self.very_verbose = very_verbose
        
        # Configure root logger
        self.logger = logging.getLogger('euchre')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set appropriate log level based on verbosity
        if very_verbose:
            console_handler.setLevel(logging.DEBUG)
        elif verbose:
            console_handler.setLevel(logging.INFO)
        else:
            console_handler.setLevel(logging.WARNING)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
        
        # Create child loggers for different components
        self.game_logger = logging.getLogger('euchre.game')
        self.ai_logger = logging.getLogger('euchre.ai')
        self.core_logger = logging.getLogger('euchre.core')
        self.cli_logger = logging.getLogger('euchre.cli')
        
        # Generate timestamped filename for file logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"euchre_game_{timestamp}.txt"
        self.game_start_time = datetime.now()
        
        # Initialize file logging
        self._init_file_logging()
        
        # Game state tracking for logging
        self.round_logs: List[Dict[str, Any]] = []
        self.current_round: Optional[Dict[str, Any]] = None
    
    def debug(self, message: str, component: str = 'game') -> None:
        """Log a debug message if very verbose mode is enabled.
        
        Parameters
        ----------
        message : str
            The debug message to log
        component : str
            The component logging the message ('game', 'ai', 'core', 'cli')
        """
        if self.very_verbose:
            if component == 'ai':
                self.ai_logger.debug(message)
            elif component == 'core':
                self.core_logger.debug(message)
            elif component == 'cli':
                self.cli_logger.debug(message)
            else:
                self.game_logger.debug(message)
            # Also write to file for debug messages
            self._write_to_file(f"DEBUG: {message}")
    
    def info(self, message: str, component: str = 'game') -> None:
        """Log an info message if verbose mode is enabled.
        
        Parameters
        ----------
        message : str
            The info message to log
        component : str
            The component logging the message ('game', 'ai', 'core', 'cli')
        """
        if self.verbose or self.very_verbose:
            if component == 'ai':
                self.ai_logger.info(message)
            elif component == 'core':
                self.core_logger.info(message)
            elif component == 'cli':
                self.cli_logger.info(message)
            else:
                self.game_logger.info(message)
            # Also write to file for info messages
            self._write_to_file(message)
    
    def warning(self, message: str, component: str = 'game') -> None:
        """Log a warning message (always shown).
        
        Parameters
        ----------
        message : str
            The warning message to log
        component : str
            The component logging the message ('game', 'ai', 'core', 'cli')
        """
        if component == 'ai':
            self.ai_logger.warning(message)
        elif component == 'core':
            self.core_logger.warning(message)
        elif component == 'cli':
            self.cli_logger.warning(message)
        else:
            self.game_logger.warning(message)
        # Always write warnings to file
        self._write_to_file(f"WARNING: {message}")
    
    def error(self, message: str, component: str = 'game') -> None:
        """Log an error message (always shown).
        
        Parameters
        ----------
        message : str
            The error message to log
        component : str
            The component logging the message ('game', 'ai', 'core', 'cli')
        """
        if component == 'ai':
            self.ai_logger.error(message)
        elif component == 'core':
            self.core_logger.error(message)
        elif component == 'cli':
            self.cli_logger.error(message)
        else:
            self.game_logger.error(message)
        # Always write errors to file
        self._write_to_file(f"ERROR: {message}")
    
    def game_info(self, message: str) -> None:
        """Log a game-specific info message.
        
        Parameters
        ----------
        message : str
            The game info message to log
        """
        self.info(message, 'game')
    
    def game_debug(self, message: str) -> None:
        """Log a game-specific debug message.
        
        Parameters
        ----------
        message : str
            The game debug message to log
        """
        self.debug(message, 'game')
    
    def ai_info(self, message: str) -> None:
        """Log an AI-specific info message.
        
        Parameters
        ----------
        message : str
            The AI info message to log
        """
        self.info(message, 'ai')
    
    def ai_debug(self, message: str) -> None:
        """Log an AI-specific debug message.
        
        Parameters
        ----------
        message : str
            The AI debug message to log
        """
        self.debug(message, 'ai')
    
    def log_game_start(self, players: List, dealer) -> None:
        """Log the start of a new game.
        
        Parameters
        ----------
        players : List
            List of all players in the game
        dealer : Player
            The dealer for this game
        """
        player_names = [p.name for p in players] if hasattr(players[0], 'name') else [str(p) for p in players]
        dealer_name = dealer.name if hasattr(dealer, 'name') else str(dealer)
        
        message = f"Game started with players: {', '.join(player_names)}"
        self.info(message)
        
        message = f"Dealer: {dealer_name}"
        self.info(message)
        
        # Write detailed game start info to file
        self._write_to_file("=" * 80)
        self._write_to_file("EUCHRE GAME LOG")
        self._write_to_file("=" * 80)
        self._write_to_file(f"Game started: {self.game_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_to_file(f"Players: {', '.join(player_names)}")
        self._write_to_file(f"Dealer: {dealer_name}")
        self._write_to_file("=" * 80)
        self._write_to_file("")
    
    def log_round_start(self, round_num: int, trump_suit: str, hands: Dict[str, List]) -> None:
        """Log the start of a new round.
        
        Parameters
        ----------
        round_num : int
            The round number
        trump_suit : str
            The trump suit for this round
        hands : Dict[str, List]
            Dictionary mapping player names to their hands
        """
        self.info(f"Starting Round {round_num}")
        self.info(f"Trump suit: {trump_suit}")
        
        # Write round start info to file
        self._write_to_file(f"ROUND {round_num}")
        self._write_to_file("-" * 40)
        self._write_to_file(f"Trump: {trump_suit}")
        self._write_to_file("")
        
        # Log hands
        self._write_to_file("Initial Hands:")
        for player_name, hand in hands.items():
            if hasattr(hand[0], 'unicode_str'):
                hand_str = ' '.join(card.unicode_str() for card in hand)
            else:
                hand_str = ' '.join(str(card) for card in hand)
            self._write_to_file(f"  {player_name}: {hand_str}")
        self._write_to_file("")
        
        # Store round info
        self.current_round = {
            'round_num': round_num,
            'trump_suit': trump_suit,
            'hands': hands,
            'tricks': [],
            'final_scores': {}
        }
        self.round_logs.append(self.current_round)
    
    def log_trick(self, trick_num: int, trick, winner, winning_card) -> None:
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
        if not self.current_round:
            return
            
        winner_name = winner.name if hasattr(winner, 'name') else str(winner)
        winning_card_str = winning_card.unicode_str() if hasattr(winning_card, 'unicode_str') else str(winning_card)
        
        self.info(f"Trick {trick_num} won by {winner_name} with {winning_card_str}")
        
        # Get trick details
        if hasattr(trick, 'cards_played'):
            cards_played = trick.cards_played
        else:
            cards_played = []
        
        # Write trick info to file
        self._write_to_file(f"Trick {trick_num}:")
        for player, card in cards_played:
            player_name = player.name if hasattr(player, 'name') else str(player)
            card_str = card.unicode_str() if hasattr(card, 'unicode_str') else str(card)
            if player_name == winner_name:
                self._write_to_file(f"  {player_name}: {card_str} ← WINNER")
            else:
                self._write_to_file(f"  {player_name}: {card_str}")
        self._write_to_file(f"Winner: {winner_name} with {winning_card_str}")
        self._write_to_file("")
        
        # Store trick info
        trick_info = {
            'trick_num': trick_num,
            'cards_played': [(p.name if hasattr(p, 'name') else str(p), 
                             c.unicode_str() if hasattr(c, 'unicode_str') else str(c)) 
                            for p, c in cards_played],
            'winner': winner_name,
            'winning_card': winning_card_str
        }
        self.current_round['tricks'].append(trick_info)
    
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
        if not self.current_round:
            return
            
        self.current_round['final_scores'] = final_scores
        
        self.info(f"Round {round_num} completed")
        
        # Write round summary to file
        self._write_to_file("Round Summary:")
        self._write_to_file("  Player Trick Counts:")
        for player_name, tricks in final_scores.items():
            self._write_to_file(f"    {player_name}: {tricks} tricks")
        self._write_to_file("  Team Scores:")
        for team_name, score in team_scores.items():
            self._write_to_file(f"    {team_name}: {score} points")
        self._write_to_file("")
    
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
        
        self.info(f"Game ended. Winner: {winner}")
        
        # Write game end info to file
        self._write_to_file("=" * 80)
        self._write_to_file("GAME END")
        self._write_to_file("=" * 80)
        self._write_to_file(f"Winner: {winner}")
        self._write_to_file(f"Final Scores:")
        for team_name, score in final_team_scores.items():
            self._write_to_file(f"  {team_name}: {score} points")
        self._write_to_file(f"Game Duration: {duration}")
        self._write_to_file(f"Game ended: {game_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_to_file("=" * 80)
        self._write_to_file("")
        
        # Write summary table
        self._write_summary_table()
    
    def _write_summary_table(self) -> None:
        """Write a summary table showing all rounds and results."""
        if not self.round_logs:
            return
            
        self._write_to_file("SUMMARY TABLE")
        self._write_to_file("=" * 80)
        
        # Header
        if self.round_logs and self.round_logs[0]['hands']:
            header = f"{'Round':<6} {'Trump':<8}"
            for player_name in self.round_logs[0]['hands'].keys():
                header += f"{player_name:<8}"
            header += f"{'Team1':<6} {'Team2':<6}"
            self._write_to_file(header)
            
            # Separator
            self._write_to_file("-" * 80)
            
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
                
                row = f"{round_num:<6} {trump:<8}"
                for count in trick_counts:
                    row += f"{count:<8}"
                row += f"{team1_score:<6} {team2_score:<6}"
                self._write_to_file(row)
            
            self._write_to_file("=" * 80)
            self._write_to_file("")
    
    def _write_to_file(self, message: str) -> None:
        """Write a message to the log file.
        
        Parameters
        ----------
        message : str
            The message to write to the file
        """
        try:
            with open(self.log_filename, 'a', encoding='utf-8') as f:
                f.write(message + "\n")
        except Exception as e:
            # If we can't write to file, at least log the error
            print(f"Error writing to log file: {e}")
    
    def _init_file_logging(self) -> None:
        """Initialize file logging with a timestamped filename."""
        # Create file handler
        file_handler = logging.FileHandler(self.log_filename, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter for file (more detailed than console)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add file handler to root logger
        self.logger.addHandler(file_handler)
        
        # Log game start header
        self._write_game_header()
    
    def _write_game_header(self) -> None:
        """Write the game header to the log file."""
        with open(self.log_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EUCHRE GAME LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Game started: {self.game_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def get_log_filename(self) -> str:
        """Get the current log filename.
        
        Returns
        -------
        str
            The filename being used for logging
        """
        return self.log_filename


def setup_logging(verbose: bool = False, very_verbose: bool = False) -> GameLogger:
    """Set up logging for the Euchre game.
    
    Parameters
    ----------
    verbose : bool
        Enable verbose logging (INFO level)
    very_verbose : bool
        Enable very verbose logging (DEBUG level)
        
    Returns
    -------
    GameLogger
        Configured game logger instance
    """
    return GameLogger(verbose, very_verbose) 