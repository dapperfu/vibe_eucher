"""Ncurses interface for AI vs AI euchre game."""

import curses
import time
from typing import List, Optional, Tuple
from .game import EuchreGame
from .models import PlayerType, Suit, Card, Trick


class NcursesGame:
    """Ncurses interface for the euchre game."""
    
    def __init__(self) -> None:
        """Initialize the ncurses game interface."""
        self.game = EuchreGame()
        self.screen = None
        self.max_y = 0
        self.max_x = 0
        
    def setup_game(self) -> None:
        """Set up the game with 4 AI players."""
        # Add AI players
        self.game.add_player("North", PlayerType.AI)
        self.game.add_player("East", PlayerType.AI)
        self.game.add_player("South", PlayerType.AI)
        self.game.add_player("West", PlayerType.AI)
        
        # Start the game
        self.game.start_new_game()
        
    def run(self) -> None:
        """Run the ncurses game."""
        try:
            curses.wrapper(self._main_loop)
        except KeyboardInterrupt:
            pass
            
    def _main_loop(self, screen) -> None:
        """Main game loop."""
        self.screen = screen
        curses.start_color()
        curses.use_default_colors()
        
        # Initialize colors
        curses.init_pair(1, curses.COLOR_RED, -1)      # Hearts, Diamonds
        curses.init_pair(2, curses.COLOR_BLACK, -1)    # Clubs, Spades
        curses.init_pair(3, curses.COLOR_YELLOW, -1)   # Trump cards
        curses.init_pair(4, curses.COLOR_GREEN, -1)    # Winner highlight
        curses.init_pair(5, curses.COLOR_CYAN, -1)     # Current player
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # Dealer
        
        # Hide cursor
        curses.curs_set(0)
        
        # Get screen dimensions
        self.max_y, self.max_x = screen.getmaxyx()
        
        # Setup game
        self.setup_game()
        
        # Main game loop
        while not self.game.is_game_over():
            self._display_game()
            self._play_round()
            time.sleep(1)  # Pause to see what happened
            
        # Show final result
        self._display_final_result()
        self.screen.getch()  # Wait for key press
        
    def _display_game(self) -> None:
        """Display the current game state."""
        self.screen.clear()
        
        # Draw title
        title = "Euchre - AI vs AI"
        self.screen.addstr(0, (self.max_x - len(title)) // 2, title, curses.A_BOLD)
        
        # Draw scores
        self._draw_scores()
        
        # Draw trump suit
        self._draw_trump_suit()
        
        # Draw table layout
        self._draw_table()
        
        # Draw hands
        self._draw_hands()
        
        # Draw current trick
        if self.game.current_trick:
            self._draw_current_trick()
            
        # Draw round info
        self._draw_round_info()
        
        self.screen.refresh()
        
    def _draw_scores(self) -> None:
        """Draw the current scores."""
        if not self.game.game_state:
            return
            
        # Check if we have enough space
        if self.max_y < 3 or self.max_x < 50:
            return
            
        score_line = f"Team 1 (North/South): {self.game.game_state.team1_score} | Team 2 (East/West): {self.game.game_state.team2_score}"
        
        # Truncate if too long for screen
        if len(score_line) > self.max_x - 4:
            score_line = score_line[:self.max_x - 7] + "..."
            
        try:
            self.screen.addstr(2, 2, score_line)
        except curses.error:
            pass
        
    def _draw_trump_suit(self) -> None:
        """Draw the current trump suit."""
        if not self.game.game_state or not self.game.game_state.trump_suit:
            return
            
        # Check if we have enough space
        if self.max_y < 4 or self.max_x < 20:
            return
            
        trump_text = f"Trump: {self.game.game_state.trump_suit.name.title()}"
        
        # Truncate if too long for screen
        if len(trump_text) > self.max_x - 4:
            trump_text = trump_text[:self.max_x - 7] + "..."
            
        try:
            self.screen.addstr(3, 2, trump_text, curses.color_pair(3) | curses.A_BOLD)
        except curses.error:
            pass
        
    def _draw_table(self) -> None:
        """Draw the table layout with player positions."""
        # Calculate center positions
        center_y = self.max_y // 2
        center_x = self.max_x // 2
        
        # Calculate spacing based on screen size
        spacing_y = min(8, self.max_y // 4)
        spacing_x = min(20, self.max_x // 4)
        
        # North (top)
        north_pos = (center_y - spacing_y, center_x)
        self._draw_player_position("North", north_pos, 0)
        
        # East (right)
        east_pos = (center_y, center_x + spacing_x)
        self._draw_player_position("East", east_pos, 1)
        
        # South (bottom)
        south_pos = (center_y + spacing_y, center_x)
        self._draw_player_position("South", south_pos, 2)
        
        # West (left)
        west_pos = (center_y, center_x - spacing_x)
        self._draw_player_position("West", west_pos, 3)
        
        # Draw table center
        try:
            self.screen.addstr(center_y, center_x - 2, "TABLE", curses.A_BOLD)
        except curses.error:
            pass  # Skip if we can't write here
        
    def _draw_player_position(self, name: str, pos: Tuple[int, int], player_index: int) -> None:
        """Draw a player position on the table."""
        y, x = pos
        
        # Check bounds
        if y < 0 or y >= self.max_y or x < 0 or x >= self.max_x:
            return
            
        # Get player info
        player = self.game.players[player_index]
        is_current = (self.game.game_state and 
                     self.game.game_state.current_player_index == player_index)
        is_dealer = player.is_dealer
        
        # Choose color
        color = curses.color_pair(5) if is_current else curses.color_pair(6) if is_dealer else 0
        attr = curses.A_BOLD if is_current or is_dealer else 0
        
        # Draw player name (with bounds checking)
        try:
            self.screen.addstr(y, x - len(name) // 2, name, color | attr)
        except curses.error:
            return
            
        # Draw hand size
        hand_size = f"({len(player.hand)} cards)"
        try:
            self.screen.addstr(y + 1, x - len(hand_size) // 2, hand_size)
        except curses.error:
            return
            
        # Draw tricks won
        tricks = f"Tricks: {player.tricks_won}"
        try:
            self.screen.addstr(y + 2, x - len(tricks) // 2, tricks)
        except curses.error:
            return
        
    def _draw_hands(self) -> None:
        """Draw the hands of all players."""
        if not self.game.game_state:
            return
            
        # Calculate available space for hands
        hand_width = min(25, (self.max_x - 10) // 4)  # Max 25 chars per hand
        
        # Draw hands in corners with bounds checking
        self._draw_hand("North", 5, 5, 0, hand_width)
        self._draw_hand("East", 5, self.max_x - hand_width - 5, 1, hand_width)
        self._draw_hand("South", self.max_y - 8, 5, 2, hand_width)
        self._draw_hand("West", 5, 5, 3, hand_width)
        
    def _draw_hand(self, name: str, y: int, x: int, player_index: int, max_width: int) -> None:
        """Draw a player's hand."""
        player = self.game.players[player_index]
        
        # Check bounds
        if y >= self.max_y or x >= self.max_x:
            return
            
        # Draw hand label
        try:
            self.screen.addstr(y, x, f"{name}'s Hand:", curses.A_BOLD)
        except curses.error:
            return
            
        # Draw cards (with bounds checking)
        for i, card in enumerate(player.hand):
            card_y = y + 1
            card_x = x + (i * 4)
            
            # Skip if card would be off screen
            if card_y >= self.max_y or card_x >= self.max_x:
                continue
                
            # Choose color based on suit and trump status
            if card.is_trump:
                color = curses.color_pair(3)
            elif card.suit in [Suit.HEARTS, Suit.DIAMONDS]:
                color = curses.color_pair(1)
            else:
                color = curses.color_pair(2)
                
            # Draw card
            card_str = card.short_str()
            try:
                self.screen.addstr(card_y, card_x, card_str, color)
            except curses.error:
                continue  # Skip if we can't write here
            
    def _draw_current_trick(self) -> None:
        """Draw the current trick being played."""
        if not self.game.current_trick or not self.game.current_trick.cards_played:
            return
            
        # Draw trick area (with bounds checking)
        trick_y = self.max_y // 2
        trick_x = max(2, self.max_x // 2 - 10)
        
        # Check if we have enough space
        if trick_y < 2 or trick_x < 0:
            return
            
        try:
            self.screen.addstr(trick_y - 2, trick_x, "Current Trick:", curses.A_BOLD)
        except curses.error:
            return
            
        # Draw cards in the trick
        for i, (player, card) in enumerate(self.game.current_trick.cards_played):
            card_y = trick_y
            card_x = trick_x + (i * 6)
            
            # Skip if card would be off screen
            if card_y >= self.max_y or card_x >= self.max_x:
                continue
                
            # Choose color
            if card.is_trump:
                color = curses.color_pair(3)
            elif card.suit in [Suit.HEARTS, Suit.DIAMONDS]:
                color = curses.color_pair(1)
            else:
                color = curses.color_pair(2)
                
            # Draw card
            card_str = card.short_str()
            try:
                self.screen.addstr(card_y, card_x, card_str, color)
            except curses.error:
                continue
                
            # Draw player name below card
            player_name = player.name
            try:
                self.screen.addstr(card_y + 1, card_x, player_name, curses.A_DIM)
            except curses.error:
                continue
            
    def _draw_round_info(self) -> None:
        """Draw round information."""
        if not self.game.game_state:
            return
            
        # Check if we have enough space at the bottom
        if self.max_y < 3:
            return
            
        round_info = f"Round: {self.game.game_state.round_number}"
        try:
            self.screen.addstr(self.max_y - 3, 2, round_info)
        except curses.error:
            pass
            
        # Draw dealer info
        dealer = self.game.game_state.get_dealer()
        dealer_info = f"Dealer: {dealer.name}"
        try:
            self.screen.addstr(self.max_y - 2, 2, dealer_info)
        except curses.error:
            pass
            
        # Draw current player info
        current_player = self.game.game_state.get_current_player()
        current_info = f"Current: {current_player.name}"
        try:
            self.screen.addstr(self.max_y - 1, 2, current_info)
        except curses.error:
            pass
        
    def _play_round(self) -> None:
        """Play a complete round."""
        if not self.game.game_state:
            return
            
        # Play the round
        self.game.play_round()
        
        # Show results
        self._display_round_results()
        
        # Increment round number
        self.game.game_state.round_number += 1
        
    def _display_round_results(self) -> None:
        """Display the results of the round."""
        if not self.game.tricks_this_round:
            return
            
        # Show each trick result (with bounds checking)
        start_y = 10
        for i, trick in enumerate(self.game.tricks_this_round):
            if start_y + i >= self.max_y - 5:  # Leave space for other info
                break
                
            if trick.cards_played:
                winner, winning_card = trick.get_winner(self.game.game_state.trump_suit)
                
                # Highlight winner
                result_text = f"Trick {i+1}: {winner.name} wins with {winning_card}"
                # Truncate if too long for screen
                if len(result_text) > self.max_x - 4:
                    result_text = result_text[:self.max_x - 7] + "..."
                    
                try:
                    self.screen.addstr(start_y + i, 2, result_text, curses.color_pair(4))
                except curses.error:
                    # If we can't write to that position, skip it
                    continue
                    
        # Show round scoring (with bounds checking)
        score_y = start_y + len(self.game.tricks_this_round) + 2
        if score_y < self.max_y - 3:
            team1_tricks = (self.game.players[0].tricks_won + self.game.players[2].tricks_won)
            team2_tricks = (self.game.players[1].tricks_won + self.game.players[3].tricks_won)
            
            round_score = f"Round Score - Team 1: {team1_tricks}, Team 2: {team2_tricks}"
            # Truncate if too long
            if len(round_score) > self.max_x - 4:
                round_score = round_score[:self.max_x - 7] + "..."
                
            try:
                self.screen.addstr(score_y, 2, round_score, curses.A_BOLD)
            except curses.error:
                pass  # Skip if we can't write here
        
        self.screen.refresh()
        time.sleep(2)  # Show results for 2 seconds
        
    def _display_final_result(self) -> None:
        """Display the final game result."""
        self.screen.clear()
        
        winner = self.game.get_winner()
        if winner:
            title = f"Game Over! {winner} Wins!"
        else:
            title = "Game Over!"
            
        # Check if we have enough space
        if self.max_y < 5 or self.max_x < len(title):
            # Fallback for very small screens
            try:
                self.screen.addstr(0, 0, "Game Over!", curses.color_pair(4) | curses.A_BOLD)
            except curses.error:
                pass
            return
            
        try:
            self.screen.addstr(self.max_y // 2 - 2, (self.max_x - len(title)) // 2, 
                              title, curses.color_pair(4) | curses.A_BOLD)
        except curses.error:
            pass
            
        # Show final scores
        if self.game.game_state:
            final_score = f"Final Score - Team 1: {self.game.game_state.team1_score}, Team 2: {self.game.game_state.team2_score}"
            try:
                self.screen.addstr(self.max_y // 2, (self.max_x - len(final_score)) // 2, final_score)
            except curses.error:
                pass
                
        # Show log filename
        log_filename = self.game.get_log_filename()
        if log_filename:
            log_info = f"Game log saved to: {log_filename}"
            try:
                self.screen.addstr(self.max_y // 2 + 1, (self.max_x - len(log_info)) // 2, log_info)
            except curses.error:
                pass
                
        try:
            self.screen.addstr(self.max_y // 2 + 2, (self.max_x - 20) // 2, 
                              "Press any key to exit...")
        except curses.error:
            pass
        
        self.screen.refresh()


def main() -> None:
    """Main function to run the ncurses game."""
    game = NcursesGame()
    game.run()


if __name__ == "__main__":
    main() 