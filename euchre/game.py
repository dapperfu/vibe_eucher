"""
Main game controller for Euchre.

This module orchestrates the game flow by coordinating interactions between
different components like deck management, game state, trick management,
scoring, and trump selection.
"""

from typing import List, Optional, Tuple
from .core.deck import Deck
from .core.game_state import GameStateManager
from .core.trick_manager import TrickManager
from .core.scoring import ScoringManager
from .core.dealer_selection import DealerSelection
from .game_logic.trump_selection import TrumpSelectionManager
from .ai.ai_factory import AIFactory
from .models import Player, PlayerType, Card, Suit, Trick
from .utils.logging_config import GameLogger
import click


class EuchreGame:
    """Main game controller for Euchre."""
    
    def __init__(self, players: List[Player] = None, quiet_mode: bool = False, verbose: bool = False, very_verbose: bool = False):
        """Initialize the Euchre game.
        
        Parameters
        ----------
        players : List[Player], optional
            List of players in the game (can be added later)
        quiet_mode : bool
            If True, suppress most game output
        verbose : bool
            Enable verbose logging (INFO level)
        very_verbose : bool
            Enable very verbose logging (DEBUG level)
        """
        self.players = players or []
        self.quiet_mode = quiet_mode
        self.verbose = verbose
        self.very_verbose = very_verbose
        
        # Initialize logging
        self.logger = GameLogger(verbose, very_verbose)
        
        # Initialize game components
        self.deck = Deck()
        self.game_state_manager = GameStateManager()
        self.trick_manager = TrickManager()
        self.scoring_manager = ScoringManager()
        self.trump_selection_manager = TrumpSelectionManager(self.logger)
        
        # Game state
        self.current_trick: Optional[Trick] = None
        self.top_card: Optional[Card] = None
        self.trump_suit: Optional[Suit] = None
        self.tricks_won = {}
        self.round_number = 1
        self._top_card_picked_up = False  # Track if top card was picked up
        
        # Initialize dealer selection (will be done in start_new_game)
        self.dealer_selection = None
        self.dealer_selection_method = 'black_jack'  # Default method
        
        # Initialize game state
        self.game_state = None
        self.trump_caller = None
        self.renege_count = 0
        
        # Initialize tricks_won if players are provided
        if self.players:
            self.tricks_won = {player.name: 0 for player in self.players}
            self.dealer_selection = DealerSelection(self.players)
    
    def start_new_game(self) -> None:
        """Start a new game."""
        self.logger.info("Starting new game")
        self.logger.info(f"Number of players: {len(self.players)}")
        self.logger.debug("Player objects:")
        for i, player in enumerate(self.players):
            self.logger.debug(f"Player {i}: {player.name} (id: {id(player)}) type: {player.player_type}")
            self.logger.debug(f"Player {i}: {player.name} hand: {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Reset game state
        self.game_state_manager.initialize_game(self.players)
        self.deck.reset_and_shuffle()
        self.trick_manager.reset()
        
        # Use traditional dealer selection for the first game
        dealer_method = getattr(self, 'dealer_selection_method', 'black_jack')
        first_dealer, selection_cards = self.dealer_selection.select_first_dealer(method=dealer_method, verbose=self.verbose)
        
        # Set the selected dealer
        self.game_state_manager.set_dealer(first_dealer)
        
        # Log the dealer selection process
        if self.verbose:
            self.logger.info("🎲 Traditional dealer selection completed:")
            for i, card in enumerate(selection_cards):
                player_index = i % 4
                player = self.players[player_index]
                self.logger.info(f"  {player.name} received: {card.unicode_str()}")
            self.logger.info(f"👑 {first_dealer.name} selected as first dealer!")
        
        # Reset the deck after dealer selection
        self.dealer_selection.reset_deck()
        
        self.logger.debug("After resetting components")
        self.logger.debug("Player hands after reset:")
        for player in self.players:
            self.logger.debug(f"{player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Deal cards
        self._deal_cards()
        
        # Show dealer and initial hands
        dealer = self.game_state_manager.get_dealer()
        self.logger.info(f"Dealer: {dealer.name}")
        self.logger.info("Initial hands:")
        for player in self.players:
            self.logger.info(f"{player.name}: {[card.unicode_str() for card in player.hand]}")
        
        self.logger.debug("After dealing cards")
        self.logger.debug("Player hands after dealing:")
        for player in self.players:
            self.logger.debug(f"{player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Reset round number
        self.round_number = 1
        
        # Log game start to file
        self.logger.log_game_start(self.players, dealer)
        
        self.logger.debug("Game initialization complete")
    
    def _deal_cards(self):
        """Deal 5 cards to each player and set the top card."""
        self.logger.info("🎴 Dealing cards using traditional Euchre pattern...")
        self.logger.info("   First round: Deal 2 cards to each player")
        self.logger.info("   Second round: Deal 3 cards to each player")
        
        hands = self.deck.deal_cards(len(self.players))
        
        # Debug: Print player objects and their hands
        self.logger.debug(f"Number of players: {len(self.players)}")
        self.logger.debug(f"Number of hands: {len(hands)}")
        for i, (player, hand) in enumerate(zip(self.players, hands)):
            self.logger.debug(f"Player {i}: {player.name} (id: {id(player)}) got {len(hand)} cards: {[card.unicode_str() for card in hand]}")
            player.hand = hand
            self.logger.debug(f"After assignment: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Set the top card
        self.top_card = self.deck.draw_top_card()
        self.logger.debug(f"Top card: {self.top_card}")
        
        # Debug: Verify all players have cards after dealing
        self.logger.debug("Final hand verification after dealing:")
        for player in self.players:
            self.logger.debug(f"{player.name} (id: {id(player)}) final hand: {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Show kitty size
        self.logger.info(f"Kitty: {self.deck.size} cards remaining")
    
    def _start_new_round(self) -> None:
        """Start a new round of the game."""
        # Rotate dealer for new rounds (but not the first round)
        if self.round_number > 1:
            self.game_state_manager.start_new_round()
        
        self.logger.info(f"\n=== Starting Round {self.round_number} ===")
        self.logger.debug("Player hands at start of round:")
        for player in self.players:
            self.logger.debug(f"{player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Reset round state
        self.current_trick = None
        self.tricks_won = {player.name: 0 for player in self.players}
        self.trump_suit = None
        self._top_card_picked_up = False  # Reset top card picked up flag
        
        # Deal new cards for this round (unless it's the first round which was already dealt)
        if self.round_number > 1:
            self.logger.debug(f"Dealing new cards for round {self.round_number}")
            # Reset and shuffle the deck for the new round
            self.deck.reset_and_shuffle()
            self._deal_cards()
            self.logger.debug("Player hands after dealing new cards:")
            for player in self.players:
                self.logger.debug(f"{player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Select trump suit
        self.logger.info(f"Dealer: {self.game_state_manager.get_dealer().name}")
        if self.top_card and not self._top_card_picked_up:
            self.logger.info(f"Top Card: {self.top_card.unicode_str()}")
        else:
            self.logger.info("Top Card: Picked up by player")
        
        trump_suit, caller = self.trump_selection_manager.select_trump_suit(
            self.players, 
            self.top_card, 
            self.game_state_manager.get_dealer()
        )
        
        # Set the trump suit in the game state
        if trump_suit:
            self.trump_suit = trump_suit
            self.game_state_manager.set_trump_suit(trump_suit, caller or self.game_state_manager.get_dealer())
            if caller:
                self.logger.info(f"{caller.name} called {trump_suit.name} as trump!")
            else:
                self.logger.info(f"Trump suit is {trump_suit.name}")
        else:
            # If no trump was selected, dealer picks
            dealer = self.game_state_manager.get_dealer()
            trump_suit = self.trump_selection_manager._dealer_suit_selection(dealer, self.top_card)
            self.trump_suit = trump_suit
            self.game_state_manager.set_trump_suit(trump_suit, dealer)
            # Note: The detailed logging is now handled in the TrumpSelectionManager
        
        # Debug: Check hands after trump selection
        self.logger.debug("Player hands after trump selection:")
        for player in self.players:
            self.logger.debug(f"{player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Log round start to file
        hands_dict = {player.name: player.hand for player in self.players}
        trump_suit_name = trump_suit.name if trump_suit else "None"
        self.logger.log_round_start(self.round_number, trump_suit_name, hands_dict)
        
        # Play 5 tricks
        for trick_number in range(1, 6):
            self.logger.info(f"\n--- Trick {trick_number} ---")
            self.logger.debug("Before starting trick - Player hands:")
            for player in self.players:
                self.logger.debug(f"{player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
            
            self._play_trick(trick_number)
            
            self.logger.debug("After completing trick - Player hands:")
            for player in self.players:
                self.logger.debug(f"{player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Score the round
        # Determine which team called trump
        trump_caller_team = 0 if caller and self.players.index(caller) % 2 == 0 else 1
        team1_score, team2_score = self.scoring_manager.score_round(self.players, trump_caller_team)
        
        # Update player scores (assuming team 1 is players 0,2 and team 2 is players 1,3)
        for i in range(0, 4, 2):  # Team 1
            self.players[i].score += team1_score
        for i in range(1, 4, 2):  # Team 2
            self.players[i].score += team2_score
        
        # Display round results
        self.logger.info(f"\n=== Round {self.round_number} Complete ===")
        self.logger.info(f"Final trick counts: Alice: {self.tricks_won['Alice']}, Bob: {self.tricks_won['Bob']}, Charlie: {self.tricks_won['Charlie']}, David: {self.tricks_won['David']}")
        self.logger.info(f"Team 1 (Alice & Charlie): {team1_score} points")
        self.logger.info(f"Team 2 (Bob & David): {team2_score} points")
        
        # Show current game scores
        team1_total = sum(self.players[i].score for i in range(0, 4, 2))
        team2_total = sum(self.players[i].score for i in range(1, 4, 2))
        self.logger.info(f"Game Score - Team 1: {team1_total}, Team 2: {team2_total}")
        
        # Log round end to file
        final_scores = {player.name: self.tricks_won[player.name] for player in self.players}
        team_scores = {"Team 1 (Alice & Charlie)": team1_score, "Team 2 (Bob & David)": team2_score}
        self.logger.log_round_end(self.round_number, final_scores, team_scores)
        
        # Check if game is over
        if self.scoring_manager.is_game_over(self.players):
            self.logger.info("\n🎉 GAME OVER! 🎉")
            if team1_total > team2_total:
                self.logger.info("Team 1 (Alice & Charlie) wins!")
                winner = "Team 1 (Alice & Charlie)"
            else:
                self.logger.info("Team 2 (Bob & David) wins!")
                winner = "Team 2 (Bob & David)"
            
            # Log game end to file
            final_team_scores = {"Team 1 (Alice & Charlie)": team1_total, "Team 2 (Bob & David)": team2_total}
            self.logger.log_game_end(winner, final_team_scores)
    
    def _update_trump_status(self, trump_suit: Suit) -> None:
        """Update the trump status of all cards.
        
        Parameters
        ----------
        trump_suit : Suit
            The trump suit for this round
        """
        for player in self.players:
            for card in player.hand:
                card.is_trump = (card.suit == trump_suit or 
                               (card.suit == self.trump_selection_manager._get_left_bower_suit(trump_suit) and 
                                card.rank.value == 11))  # Jack
    
    def run_full_game(self) -> None:
        """Run a complete game of Euchre."""
        self.logger.info("Starting full game")
        self.logger.info(f"Number of players: {len(self.players)}")
        self.logger.debug("Player objects:")
        for i, player in enumerate(self.players):
            self.logger.debug(f"Player {i}: {player.name} (id: {id(player)}) type: {player.player_type}")
        
        # Start new game
        self.start_new_game()
        
        self.logger.debug("After start_new_game")
        self.logger.debug("Player hands:")
        for player in self.players:
            self.logger.debug(f"{player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Start the first round
        self._start_new_round()
        
        self.logger.debug("After _start_new_round")
        self.logger.debug("Player hands:")
        for player in self.players:
            self.logger.debug(f"{player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Continue rounds until game is over
        while not self.scoring_manager.is_game_over(self.players):
            self.round_number += 1
            self.logger.info(f"Starting round {self.round_number}")
            self.logger.debug("Player hands before new round:")
            for player in self.players:
                self.logger.debug(f"{player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
            
            # Start new round
            self._start_new_round()
            
            self.logger.info(f"Completed round {self.round_number}")
            self.logger.debug("Player hands after round:")
            for player in self.players:
                self.logger.debug(f"{player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Game is over
        self.logger.info("Game over")
        final_scores = self.scoring_manager.get_team_scores(self.players)
        self.logger.debug(f"Final scores: {final_scores}")
    
    def run_interactive_game(self, show_ai_hands: bool = False) -> None:
        """Run an interactive game with human players.
        
        Parameters
        ----------
        show_ai_hands : bool
            Whether to show AI player hands (for debugging)
        """
        while not self.game_state_manager.is_game_over():
            self._play_round()
            self._start_new_round()
        
        self._show_final_results()
    
    def _play_round(self) -> None:
        """Play a single round."""
        # Play 5 tricks
        for trick_num in range(5):
            self._play_trick(trick_num + 1)
        
        # Score the round
        self._score_round()
        
        # Start new round
        self.game_state_manager.start_new_round()
    
    def _play_trick(self, trick_number: int = 1) -> None:
        """Play a single trick."""
        self.logger.debug("Starting new trick")
        self.logger.debug("Player hands at start of trick:")
        for player in self.players:
            self.logger.debug(f"{player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Start new trick
        self.trick_manager.start_new_trick()
        self.current_trick = self.trick_manager.get_current_trick()
        
        # Determine starting player (first trick: player after dealer, subsequent tricks: winner of previous trick)
        if trick_number == 1:
            # First trick of the round
            dealer_index = next(i for i, p in enumerate(self.players) if p.name == self.game_state_manager.get_dealer().name)
            current_player_index = (dealer_index + 1) % 4
            self.logger.info(f"First trick - starting with {self.players[current_player_index].name}")
        else:
            # Subsequent tricks: winner of previous trick goes first
            # Find the player with the most tricks won so far
            max_tricks = max(self.tricks_won.values())
            winners = [name for name, count in self.tricks_won.items() if count == max_tricks]
            if winners:
                # If multiple players have the same number of tricks, use the first one
                winner_name = winners[0]
                current_player_index = next(i for i, p in enumerate(self.players) if p.name == winner_name)
                self.logger.info(f"Starting trick with {self.players[current_player_index].name} (winner of previous trick)")
            else:
                # Fallback to player after dealer
                dealer_index = next(i for i, p in enumerate(self.players) if p.name == self.game_state_manager.get_dealer().name)
                current_player_index = (dealer_index + 1) % 4
                self.logger.info(f"Fallback - starting with {self.players[current_player_index].name}")
        
        # Play cards for this trick
        for _ in range(4):
            current_player = self.players[current_player_index]
            
            self.logger.debug(f"Current player: {current_player.name} (id: {id(current_player)})")
            self.logger.debug(f"{current_player.name} hand before playing: {len(current_player.hand)} cards: {[card.unicode_str() for card in current_player.hand]}")
            
            # Get card to play
            if current_player.player_type == PlayerType.AI:
                card = self._ai_play_card(current_player)
            else:
                card = self._human_play_card(current_player, trick_number)
            
            self.logger.debug(f"{current_player.name} played: {card}")
            self.logger.debug(f"{current_player.name} hand after playing: {len(current_player.hand)} cards: {[card.unicode_str() for card in current_player.hand]}")
            
            # Play the card
            self.trick_manager.play_card(current_player, card)
            
            # Display the play
            self.logger.info(f"{current_player.name} plays {card.unicode_str()}")
            
            # Show current trick state (removed verbose "Trick so far" logging)
            
            # Move to next player
            current_player_index = (current_player_index + 1) % 4
        
        # Complete the trick
        winner = self.trick_manager.complete_trick(self.trump_suit)
        self.logger.debug(f"Trick won by: {winner.name}")
        
        # Show what each player played in this trick
        if self.current_trick and hasattr(self.current_trick, 'cards_played'):
            self.logger.info("Trick summary:")
            for player, card in self.current_trick.cards_played:
                winner_indicator = " ← WINNER" if player.name == winner.name else ""
                self.logger.info(f"  {player.name}: {card.unicode_str()}{winner_indicator}")
        
        # Show remaining cards in hands
        self.logger.debug("Player hands after completing trick:")
        for player in self.players:
            self.logger.debug(f"{player.name} (id: {id(player)}) has {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        
        # Update trick count
        self.tricks_won[winner.name] += 1
        
        # Display trick result
        self.logger.info(f"Trick won by {winner.name}!")
        
        # Show tricks won for all players in the game
        tricks_summary = []
        for player in self.players:
            tricks_summary.append(f"{player.name}: {self.tricks_won.get(player.name, 0)}")
        self.logger.info(f"Tricks won so far: {', '.join(tricks_summary)}")
        
        # Log trick completion to file
        winning_card = self.current_trick.cards_played[-1][1] if self.current_trick and self.current_trick.cards_played else None
        if winning_card:
            self.logger.log_trick(trick_number, self.current_trick, winner, winning_card)
    
    def _ai_play_card(self, player: Player) -> Card:
        """Get a card from an AI player."""
        self.logger.debug(f"Player: {player.name} (id: {id(player)})")
        self.logger.debug(f"Player type: {player.player_type}")
        self.logger.debug(f"Player hand: {len(player.hand)} cards: {[card.unicode_str() for card in player.hand]}")
        self.logger.debug(f"Current trick: {self.current_trick}")
        self.logger.debug(f"Trump suit: {self.trump_suit}")
        
        # Check if player has cards
        if not player.hand:
            self.logger.error(f"ERROR: Player {player.name} has no cards!")
            self.logger.debug(f"Player object: {player}")
            self.logger.debug(f"Player hand attribute: {player.hand}")
            self.logger.debug("All players and their hands:")
            for p in self.players:
                self.logger.debug(f"{p.name} (id: {id(p)}) has {len(p.hand)} cards: {[card.unicode_str() for card in p.hand]}")
            raise ValueError(f"AI player {player.name} has no cards to play")
        
        # Get the current trick state
        current_trick = self.trick_manager.get_current_trick()
        
        # Choose card based on AI profile
        if hasattr(player, 'choose_card_to_play'):
            card = player.choose_card_to_play(current_trick, self.trump_suit)
        else:
            # Fallback for basic AI
            card = player.hand[0]
        
        self.logger.debug(f"Chosen card: {card}")
        
        # Don't remove card from hand yet - let trick manager handle it
        self.logger.debug(f"Selected card: {card}")
        
        return card
    
    def _basic_ai_card_choice(self, player: Player, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Basic AI card choice logic.
        
        Parameters
        ----------
        player : Player
            The AI player
        lead_suit : Optional[Suit]
            The lead suit of the trick
        trump_suit : Optional[Suit]
            The trump suit for this round
            
        Returns
        -------
        Card
            The card to play
        """
        if not lead_suit:
            # Leading - play highest card
            return max(player.hand, key=lambda c: c.rank.value)
        
        # Must follow suit if possible
        cards_of_suit = [card for card in player.hand if card.suit == lead_suit]
        if cards_of_suit:
            return max(cards_of_suit, key=lambda c: c.rank.value)
        else:
            # Can't follow suit - play any card
            return max(player.hand, key=lambda c: c.rank.value)
    
    def _human_play_card(self, player: Player, trick_number: int) -> Card:
        """Get a human player's card choice.
        
        Parameters
        ----------
        player : Player
            The human player
        trick_number : int
            The current trick number
            
        Returns
        -------
        Card
            The card chosen by the human player
        """
        # This would prompt the human player
        # For now, use basic AI logic
        current_trick = self.trick_manager.get_current_trick()
        lead_suit = current_trick.lead_suit if current_trick else None
        trump_suit = self.trump_suit
        
        return self._basic_ai_card_choice(player, lead_suit, trump_suit)
    
    def _score_round(self) -> None:
        """Score the current round."""
        if not self.game_state_manager.trump_caller_team:
            return
        
        # Get round results
        round_results = self.trick_manager.get_round_results()
        
        # Score the round
        team1_score, team2_score = self.scoring_manager.score_round(
            self.players, self.game_state_manager.trump_caller_team
        )
        
        # Update scores
        current_team1_score = self.game_state_manager.game_scores["Team 1"]
        current_team2_score = self.game_state_manager.game_scores["Team 2"]
        new_team1_score = current_team1_score + team1_score
        new_team2_score = current_team2_score + team2_score
        self.game_state_manager.update_scores(new_team1_score, new_team2_score)
        
        # Show round results
        if not self.quiet_mode:
            self._show_round_results(round_results, new_team1_score, new_team2_score)
    
    def _show_round_results(self, round_results: List[int], team1_score: int, team2_score: int) -> None:
        """Show the results of a round.
        
        Parameters
        ----------
        round_results : List[int]
            Trick counts for each player
        team1_score : int
            Team 1's score
        team2_score : int
            Team 2's score
        """
        self.logger.info(f"Round complete! Trick counts: {round_results}")
        self.logger.info(f"Team 1: {team1_score}, Team 2: {team2_score}")
    
    def _show_final_results(self) -> None:
        """Show the final game results."""
        if self.quiet_mode:
            return
        
        winner = self.game_state_manager.get_winner()
        self.logger.info(f"\n🎉 GAME OVER! {winner} wins! 🎉")
        
        team1_score = self.game_state_manager.game_scores["Team 1"]
        team2_score = self.game_state_manager.game_scores["Team 2"]
        self.logger.info(f"Final Score - Team 1: {team1_score}, Team 2: {team2_score}")
        
        # Check if team gets set
        if self.game_state_manager.trump_caller_team is not None:
            if self.scoring_manager.is_team_set(self.players, self.game_state_manager.trump_caller_team):
                self.logger.info("\n🚨 TEAM SET! The trump calling team lost after calling trump!")
    
    def get_player_hand(self, player_name: str) -> List[Card]:
        """Get a player's hand.
        
        Parameters
        ----------
        player_name : str
            The name of the player
            
        Returns
        -------
        List[Card]
            The player's hand
        """
        for player in self.players:
            if player.name == player_name:
                return player.hand
        return []
    
    def is_game_over(self) -> bool:
        """Check if the game is over.
        
        Returns
        -------
        bool
            True if game is over
        """
        return self.game_state_manager.is_game_over()
    
    def get_winner(self) -> str:
        """Get the winning team.
        
        Returns
        -------
        str
            The winning team
        """
        return self.game_state_manager.get_winner()
    
    def is_team_set(self) -> bool:
        """Check if the trump calling team got set.
        
        Returns
        -------
        bool
            True if trump calling team got set
        """
        if self.game_state_manager.trump_caller_team is None:
            return False
        return self.scoring_manager.is_team_set(self.players, self.game_state_manager.trump_caller_team)
    
    def run_tournament(self, num_games: int) -> None:
        """Run a tournament with multiple games.
        
        Parameters
        ----------
        num_games : int
            Number of games to play
        """
        for game_num in range(num_games):
            self.logger.info(f"\n=== TOURNAMENT GAME {game_num + 1}/{num_games} ===")
            
            self.start_new_game()
            self.run_full_game()
            
            # Reset for next game
            self.game_state_manager.reset()
            self.trick_manager.reset()
    
    def get_log_filename(self) -> Optional[str]:
        """Get the log filename if logging is enabled.
        
        Returns
        -------
        Optional[str]
            The log filename or None
        """
        # Get the log filename from the GameLogger
        if hasattr(self.logger, 'get_log_filename'):
            return self.logger.get_log_filename()
        return None
    
    def add_player(self, name: str, player_type: PlayerType) -> None:
        """Add a player to the game.
        
        Parameters
        ----------
        name : str
            The player's name
        player_type : PlayerType
            The type of player (Human or AI)
        """
        from .models import Player
        player = Player(name, player_type)
        self.players.append(player)
        
        # Update tricks_won dictionary
        self.tricks_won[player.name] = 0
        
        # Initialize dealer selection if this is the 4th player
        if len(self.players) == 4 and self.dealer_selection is None:
            self.dealer_selection = DealerSelection(self.players)
    
    def add_ai_player(self, name: str, ai_type: str, risk_ratio: float = 0.5) -> None:
        """Add an AI player to the game.
        
        Parameters
        ----------
        name : str
            The player's name
        ai_type : str
            The AI type: "aggressive", "conservative", "balanced", "opportunistic"
        risk_ratio : float
            The risk ratio for the AI (0.0 = conservative, 1.0 = aggressive)
        """
        from .ai.ai_factory import AIFactory
        player = AIFactory.create_ai_player(name, ai_type, risk_ratio)
        self.players.append(player)
        
        # Update tricks_won dictionary
        self.tricks_won[player.name] = 0
        
        # Initialize dealer selection if this is the 4th player
        if len(self.players) == 4 and self.dealer_selection is None:
            self.dealer_selection = DealerSelection(self.players)