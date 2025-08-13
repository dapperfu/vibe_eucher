"""Enhanced Ncurses interface for Euchre game with human player support."""

import curses
import time
from typing import List, Optional, Tuple, Union
from .game import EuchreGame
from .models import PlayerType, Suit, Card, Trick, Player


class NcursesGame:
    """Enhanced Ncurses interface for the euchre game with human player support."""
    
    def __init__(self, human_player_position: int = 2, enable_logging: bool = True) -> None:
        """Initialize the ncurses game interface.
        
        Parameters
        ----------
        human_player_position : int
            Position of human player (0=North, 1=East, 2=South, 3=West)
        enable_logging : bool
            Whether to enable game logging
        """
        self.game = EuchreGame(enable_logging=enable_logging)
        self.screen = None
        self.max_y = 0
        self.max_x = 0
        self.human_player_position = human_player_position
        self.human_player = None
        self.game_mode = "human_vs_ai" if human_player_position >= 0 else "ai_vs_ai"
        
    def setup_game(self) -> None:
        """Set up the game with human and AI players."""
        if self.game_mode == "human_vs_ai":
            # Add human player
            self.human_player = self.game.add_player("You", PlayerType.HUMAN)
            
            # Add AI players in other positions
            positions = ["North", "East", "South", "West"]
            ai_names = ["Alice", "Bob", "Charlie", "David"]
            
            for i, (pos, name) in enumerate(zip(positions, ai_names)):
                if i != self.human_player_position:
                    self.game.add_player(name, PlayerType.AI)
                else:
                    # This position is already taken by human player
                    continue
        else:
            # AI vs AI mode
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
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Human player highlight
        
        # Hide cursor
        curses.curs_set(0)
        
        # Get screen dimensions
        self.max_y, self.max_x = screen.getmaxyx()
        
        # Setup game
        self.setup_game()
        
        # Main game loop
        while not self.game.is_game_over():
            self._display_game()
            
            if self.game_mode == "human_vs_ai" and self._is_human_turn():
                self._handle_human_turn()
            else:
                self._play_ai_turn()
                time.sleep(1)  # Pause to see what happened
            
        # Show final result
        self._display_final_result()
        self.screen.getch()  # Wait for key press
        
    def _is_human_turn(self) -> bool:
        """Check if it's the human player's turn."""
        if not self.game.game_state or not self.human_player:
            return False
        return self.game.game_state.current_player == self.human_player
        
    def _handle_human_turn(self) -> None:
        """Handle human player's turn."""
        # Show human player's hand prominently
        self._display_human_hand()
        
        # Get human player's choice
        choice = self._get_human_card_choice()
        
        if choice is not None:
            # Play the chosen card
            self.game.play_card(self.human_player, choice)
            
    def _display_human_hand(self) -> None:
        """Display the human player's hand prominently."""
        if not self.human_player or not self.human_player.hand:
            return
            
        # Clear bottom area for hand display
        hand_y = self.max_y - 8
        hand_x = 2
        
        try:
            self.screen.addstr(hand_y, hand_x, "YOUR HAND:", curses.color_pair(7) | curses.A_BOLD)
            
            # Display cards with selection indicators
            for i, card in enumerate(self.human_player.hand):
                card_y = hand_y + 1
                card_x = hand_x + (i * 8)
                
                # Choose color based on suit and trump status
                if hasattr(card, 'is_trump') and card.is_trump:
                    color = curses.color_pair(3)
                elif card.suit in [Suit.HEARTS, Suit.DIAMONDS]:
                    color = curses.color_pair(1)
                else:
                    color = curses.color_pair(2)
                
                # Add selection number
                card_str = f"{i+1}: {card.short_str()}"
                self.screen.addstr(card_y, card_x, card_str, color)
                
            # Show instructions
            instructions = "Press 1-5 to select a card, 'q' to quit"
            self.screen.addstr(hand_y + 2, hand_x, instructions, curses.A_DIM)
            
        except curses.error:
            pass
            
        self.screen.refresh()
        
    def _get_human_card_choice(self) -> Optional[Card]:
        """Get the human player's card choice."""
        while True:
            try:
                key = self.screen.getch()
                
                if key == ord('q') or key == ord('Q'):
                    # Quit game
                    raise KeyboardInterrupt
                    
                # Check for number keys 1-5
                for i in range(5):
                    if key == ord(str(i + 1)):
                        if i < len(self.human_player.hand):
                            return self.human_player.hand[i]
                        break
                        
            except curses.error:
                continue
                
    def _play_ai_turn(self) -> None:
        """Play an AI player's turn."""
        if not self.game.game_state:
            return
            
        # Play the round
        self.game.play_round()
        
        # Show results
        self._display_round_results()
        
    def _display_game(self) -> None:
        """Display the current game state."""
        self.screen.clear()
        
        # Draw title
        title = f"Euchre - {'Human vs AI' if self.game_mode == 'human_vs_ai' else 'AI vs AI'}"
        self.screen.addstr(0, (self.max_x - len(title)) // 2, title, curses.A_BOLD)
        
        # Draw scores
        self._draw_scores()
        
        # Draw trump suit
        self._draw_trump_suit()
        
        # Draw table layout (first-person perspective)
        self._draw_table_first_person()
        
        # Draw hands
        self._draw_hands()
        
        # Draw current trick
        if hasattr(self.game, 'current_trick') and self.game.current_trick:
            self._draw_current_trick()
            
        # Draw round info
        self._draw_round_info()
        
        self.screen.refresh()
        
    def _draw_table_first_person(self) -> None:
        """Draw the table layout from first-person perspective (South position)."""
        # Calculate center positions
        center_y = self.max_y // 2
        center_x = self.max_x // 2
        
        # Calculate spacing based on screen size
        spacing_y = min(8, self.max_y // 4)
        spacing_x = min(20, self.max_x // 4)
        
        # South (bottom - YOU - front of screen)
        south_pos = (center_y + spacing_y, center_x)
        self._draw_player_position("YOU", south_pos, 2, is_human=True)
        
        # North (top - your partner - across from you)
        north_pos = (center_y - spacing_y, center_x)
        self._draw_player_position("Partner", north_pos, 0, is_partner=True)
        
        # West (left - opponent)
        west_pos = (center_y, center_x - spacing_x)
        self._draw_player_position("Opponent", west_pos, 3, is_opponent=True)
        
        # East (right - opponent)
        east_pos = (center_y, center_x + spacing_x)
        self._draw_player_position("Opponent", east_pos, 1, is_opponent=True)
        
        # Draw table center
        try:
            self.screen.addstr(center_y, center_x - 2, "TABLE", curses.A_BOLD)
        except curses.error:
            pass
        
    def _draw_player_position(self, name: str, pos: Tuple[int, int], player_index: int, 
                            is_human: bool = False, is_partner: bool = False, 
                            is_opponent: bool = False) -> None:
        """Draw a player position on the table."""
        y, x = pos
        
        # Check bounds
        if y < 0 or y >= self.max_y or x < 0 or x >= self.max_x:
            return
            
        # Get player info
        if player_index < len(self.game.players):
            player = self.game.players[player_index]
            is_current = (self.game.game_state and 
                         self.game.game_state.current_player_index == player_index)
            is_dealer = hasattr(player, 'is_dealer') and player.is_dealer
        else:
            player = None
            is_current = False
            is_dealer = False
        
        # Choose color based on player type
        if is_human:
            color = curses.color_pair(7)  # Human player highlight
        elif is_partner:
            color = curses.color_pair(4)  # Partner (green)
        elif is_opponent:
            color = curses.color_pair(1)  # Opponent (red)
        elif is_current:
            color = curses.color_pair(5)  # Current player (cyan)
        elif is_dealer:
            color = curses.color_pair(6)  # Dealer (magenta)
        else:
            color = 0
            
        # Choose attributes
        attr = curses.A_BOLD if is_current or is_dealer or is_human else 0
        
        # Draw player name (with bounds checking)
        try:
            self.screen.addstr(y, x - len(name) // 2, name, color | attr)
        except curses.error:
            return
            
        # Draw hand size if player exists
        if player and hasattr(player, 'hand'):
            hand_size = f"({len(player.hand)} cards)"
            try:
                self.screen.addstr(y + 1, x - len(hand_size) // 2, hand_size)
            except curses.error:
                return
                
            # Draw tricks won
            if hasattr(player, 'tricks_won'):
                tricks = f"Tricks: {player.tricks_won}"
                try:
                    self.screen.addstr(y + 2, x - len(tricks) // 2, tricks)
                except curses.error:
                    return
        else:
            # Placeholder for missing player
            try:
                self.screen.addstr(y + 1, x - 2, "(0 cards)")
                self.screen.addstr(y + 2, x - 3, "Tricks: 0")
            except curses.error:
                pass
        
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


def main() -> None:
    """Main function to run the ncurses game."""
    game = NcursesGame()
    game.run()


if __name__ == "__main__":
    main() 