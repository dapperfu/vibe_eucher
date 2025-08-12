"""Main game controller for the euchre card game."""

from typing import List, Optional
from .core.deck import Deck
from .core.game_state import GameStateManager
from .core.trick_manager import TrickManager
from .core.scoring import ScoringManager
from .game_logic.trump_selection import TrumpSelectionManager
from .ai.ai_factory import AIFactory
from .models import Player, PlayerType, Card, Suit
from .game_logger import GameLogger
import click


class EuchreGame:
    """Main euchre game controller."""
    
    def __init__(self, enable_logging: bool = True, quiet_mode: bool = False) -> None:
        """Initialize a new euchre game.
        
        Parameters
        ----------
        enable_logging : bool
            Whether to enable game logging to file
        quiet_mode : bool
            Whether to suppress console output (useful for training)
        """
        self.players: List[Player] = []
        self.deck = Deck()
        self.game_state_manager = GameStateManager()
        self.trick_manager = TrickManager()
        self.scoring_manager = ScoringManager()
        self.trump_selection_manager = TrumpSelectionManager()
        self.logger: Optional[GameLogger] = None
        self.top_card: Optional[Card] = None
        self.quiet_mode = quiet_mode
        
        if enable_logging:
            self.logger = GameLogger()
    
    def add_ai_player(self, name: str, ai_type: str = "balanced", risk_ratio: float = 0.5) -> None:
        """Add an AI player with a specific profile.
        
        Parameters
        ----------
        name : str
            The player's name
        ai_type : str
            Type of AI: "aggressive", "conservative", "balanced", "opportunistic"
        risk_ratio : float
            Risk tolerance (0.0 = conservative, 1.0 = aggressive)
        """
        player = AIFactory.create_ai_player(name, ai_type, risk_ratio)
        self.players.append(player)
    
    def add_player(self, name: str, player_type: PlayerType) -> None:
        """Add a player to the game.
        
        Parameters
        ----------
        name : str
            The player's name
        player_type : PlayerType
            Whether the player is human or AI
        """
        player = Player(name=name, player_type=player_type)
        self.players.append(player)
    
    def start_new_game(self) -> None:
        """Start a new game."""
        if not self.quiet_mode:
            print("DEBUG: start_new_game - Starting new game")
            print("DEBUG: start_new_game - Number of players:", len(self.players))
            print("DEBUG: start_new_game - Player objects:")
            for i, player in enumerate(self.players):
                print(f"DEBUG: Player {i}: {player.name} (id: {id(player)}) type: {player.player_type}")
                print(f"DEBUG: Player {i}: {player.name} hand: {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
        # Reset game state
        self.game_state_manager.initialize_game(self.players)
        self.deck.reset()
        self.trick_manager.reset()
        
        if not self.quiet_mode:
            print("DEBUG: start_new_game - After resetting components")
            print("DEBUG: start_new_game - Player hands after reset:")
            for player in self.players:
                print(f"DEBUG: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
        # Deal cards
        self._deal_cards()
        
        if not self.quiet_mode:
            print("DEBUG: start_new_game - After dealing cards")
            print("DEBUG: start_new_game - Player hands after dealing:")
            for player in self.players:
                print(f"DEBUG: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
        # Reset round number
        self.round_number = 1
        
        if not self.quiet_mode:
            print("DEBUG: start_new_game - Game initialization complete")
    
    def _deal_cards(self):
        """Deal 5 cards to each player and set the top card."""
        hands = self.deck.deal_cards(len(self.players))
        
        # Debug: Print player objects and their hands
        if not self.quiet_mode:
            print(f"DEBUG: _deal_cards - Number of players: {len(self.players)}")
            print(f"DEBUG: _deal_cards - Number of hands: {len(hands)}")
            for i, (player, hand) in enumerate(zip(self.players, hands)):
                print(f"DEBUG: _deal_cards - Player {i}: {player.name} (id: {id(player)}) got {len(hand)} cards: {[str(card) for card in hand]}")
                player.hand = hand
                print(f"DEBUG: _deal_cards - After assignment: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
        # Set the top card
        self.top_card = self.deck.draw_top_card()
        if not self.quiet_mode:
            print(f"DEBUG: Top card: {self.top_card}")
        
        # Debug: Verify all players have cards after dealing
        if not self.quiet_mode:
            print("DEBUG: Final hand verification after dealing:")
            for player in self.players:
                print(f"DEBUG: {player.name} (id: {id(player)}) final hand: {len(player.hand)} cards: {[str(card) for card in player.hand]}")
    
    def _start_new_round(self) -> None:
        """Start a new round of the game."""
        if not self.quiet_mode:
            print(f"\n=== Starting Round {self.round_number} ===")
            print("DEBUG: _start_new_round - Player hands at start of round:")
            for player in self.players:
                print(f"DEBUG: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
        # Reset round state
        self.current_trick = None
        self.tricks_won = {player.name: 0 for player in self.players}
        self.trump_suit = None
        
        # Deal new cards for this round (unless it's the first round which was already dealt)
        if self.round_number > 1:
            if not self.quiet_mode:
                print("DEBUG: _start_new_round - Dealing new cards for round", self.round_number)
            # Reset the deck for the new round
            self.deck.reset()
            self._deal_cards()
            if not self.quiet_mode:
                print("DEBUG: _start_new_round - Player hands after dealing new cards:")
                for player in self.players:
                    print(f"DEBUG: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
        # Select trump suit
        trump_suit, caller = self.trump_selection_manager.select_trump_suit(
            self.players, 
            self.top_card, 
            self.game_state_manager.get_dealer()
        )
        
        # Set the trump suit in the game state
        if trump_suit:
            self.trump_suit = trump_suit
            self.game_state_manager.set_trump_suit(trump_suit, caller or self.game_state_manager.get_dealer())
        else:
            # If no trump was selected, dealer picks
            dealer = self.game_state_manager.get_dealer()
            trump_suit = self.trump_selection_manager._dealer_suit_selection(dealer, self.top_card)
            self.trump_suit = trump_suit
            self.game_state_manager.set_trump_suit(trump_suit, dealer)
        
        # Debug: Check hands after trump selection
        if not self.quiet_mode:
            print("DEBUG: _start_new_round - Player hands after trump selection:")
            for player in self.players:
                print(f"DEBUG: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
        # Play 5 tricks
        for trick_number in range(1, 6):
            if not self.quiet_mode:
                print(f"\n--- Trick {trick_number} ---")
                print("DEBUG: Before starting trick - Player hands:")
                for player in self.players:
                    print(f"DEBUG: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
            
            self._play_trick()
            
            if not self.quiet_mode:
                print("DEBUG: After completing trick - Player hands:")
                for player in self.players:
                    print(f"DEBUG: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
        # Score the round
        # Determine which team called trump
        trump_caller_team = 0 if caller and self.players.index(caller) % 2 == 0 else 1
        team1_score, team2_score = self.scoring_manager.score_round(self.players, trump_caller_team)
        
        # Update player scores (assuming team 1 is players 0,2 and team 2 is players 1,3)
        for i in range(0, 4, 2):  # Team 1
            self.players[i].score += team1_score
        for i in range(1, 4, 2):  # Team 2
            self.players[i].score += team2_score
    
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
        if not self.quiet_mode:
            print("DEBUG: run_full_game - Starting full game")
            print("DEBUG: run_full_game - Number of players:", len(self.players))
            print("DEBUG: run_full_game - Player objects:")
            for i, player in enumerate(self.players):
                print(f"DEBUG: Player {i}: {player.name} (id: {id(player)}) type: {player.player_type}")
        
        # Start new game
        self.start_new_game()
        
        if not self.quiet_mode:
            print("DEBUG: run_full_game - After start_new_game")
            print("DEBUG: run_full_game - Player hands:")
            for player in self.players:
                print(f"DEBUG: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
        # Start the first round
        self._start_new_round()
        
        if not self.quiet_mode:
            print("DEBUG: run_full_game - After _start_new_round")
            print("DEBUG: run_full_game - Player hands:")
            for player in self.players:
                print(f"DEBUG: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
        # Continue rounds until game is over
        while not self.scoring_manager.is_game_over(self.players):
            self.round_number += 1
            if not self.quiet_mode:
                print(f"DEBUG: run_full_game - Starting round {self.round_number}")
                print("DEBUG: run_full_game - Player hands before new round:")
                for player in self.players:
                    print(f"DEBUG: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
            
            # Start new round
            self._start_new_round()
            
            if not self.quiet_mode:
                print(f"DEBUG: run_full_game - Completed round {self.round_number}")
                print("DEBUG: run_full_game - Player hands after round:")
                for player in self.players:
                    print(f"DEBUG: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
        # Game is over
        if not self.quiet_mode:
            print("DEBUG: run_full_game - Game over")
            final_scores = self.scoring_manager.get_team_scores(self.players)
            print(f"DEBUG: Final scores: {final_scores}")
    
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
    
    def _play_trick(self) -> None:
        """Play a single trick."""
        if not self.quiet_mode:
            print("DEBUG: _play_trick - Starting new trick")
            print("DEBUG: _play_trick - Player hands at start of trick:")
            for player in self.players:
                print(f"DEBUG: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
        # Start new trick
        self.trick_manager.start_new_trick()
        
        # Determine starting player (first trick: player after dealer, subsequent tricks: winner of previous trick)
        if not self.current_trick or self.current_trick.is_complete():
            # First trick of the round
            dealer_index = next(i for i, p in enumerate(self.players) if p.name == self.game_state_manager.get_dealer().name)
            starting_player_index = (dealer_index + 1) % len(self.players)
        else:
            # Subsequent tricks
            starting_player_index = next(i for i, p in enumerate(self.players) if p.name == self.current_trick.winner.name)
        
        # Play cards in order
        current_player_index = starting_player_index
        for _ in range(len(self.players)):
            current_player = self.players[current_player_index]
            
            if not self.quiet_mode:
                print(f"DEBUG: _play_trick - Current player: {current_player.name} (id: {id(current_player)})")
                print(f"DEBUG: _play_trick - {current_player.name} hand before playing: {len(current_player.hand)} cards: {[str(card) for card in current_player.hand]}")
            
            # Get card to play
            if current_player.player_type == PlayerType.AI:
                card = self._ai_play_card(current_player)
            else:
                card = self._human_play_card(current_player, trick_number)
            
            if not self.quiet_mode:
                print(f"DEBUG: _play_trick - {current_player.name} played: {card}")
                print(f"DEBUG: _play_trick - {current_player.name} hand after playing: {len(current_player.hand)} cards: {[str(card) for card in current_player.hand]}")
            
            # Play the card
            self.trick_manager.play_card(current_player, card)
            
            # Move to next player
            current_player_index = (current_player_index + 1) % len(self.players)
        
        # Complete the trick
        winner = self.trick_manager.complete_trick()
        if not self.quiet_mode:
            print(f"DEBUG: _play_trick - Trick won by: {winner.name}")
            print("DEBUG: _play_trick - Player hands after completing trick:")
            for player in self.players:
                print(f"DEBUG: {player.name} (id: {id(player)}) has {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
        # Update trick count
        self.tricks_won[winner.name] += 1
    
    def _ai_play_card(self, player: Player) -> Card:
        """Get a card from an AI player."""
        if not self.quiet_mode:
            print(f"DEBUG: _ai_play_card - Player: {player.name} (id: {id(player)})")
            print(f"DEBUG: _ai_play_card - Player type: {player.player_type}")
            print(f"DEBUG: _ai_play_card - Player hand: {len(player.hand)} cards: {[str(card) for card in player.hand]}")
            print(f"DEBUG: _ai_play_card - Current trick: {self.current_trick}")
            print(f"DEBUG: _ai_play_card - Trump suit: {self.trump_suit}")
        
        # Check if player has cards
        if not player.hand:
            if not self.quiet_mode:
                print(f"DEBUG: _ai_play_card - ERROR: Player {player.name} has no cards!")
                print(f"DEBUG: _ai_play_card - Player object: {player}")
                print(f"DEBUG: _ai_play_card - Player hand attribute: {player.hand}")
                print(f"DEBUG: _ai_play_card - All players and their hands:")
                for p in self.players:
                    print(f"DEBUG: {p.name} (id: {id(p)}) has {len(p.hand)} cards: {[str(card) for card in p.hand]}")
            raise ValueError(f"AI player {player.name} has no cards to play")
        
        # Get the current trick state
        current_trick = self.trick_manager.get_current_trick()
        
        # Choose card based on AI profile
        if hasattr(player, 'choose_card_to_play'):
            card = player.choose_card_to_play(current_trick, self.trump_suit)
        else:
            # Fallback for basic AI
            card = player.hand[0]
        
        if not self.quiet_mode:
            print(f"DEBUG: _ai_play_card - Chosen card: {card}")
        
        # Remove card from hand
        player.hand.remove(card)
        
        if not self.quiet_mode:
            print(f"DEBUG: _ai_play_card - After removing card: {len(player.hand)} cards: {[str(card) for card in player.hand]}")
        
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
        return self._basic_ai_card_choice(player, 
                                         self.trick_manager.current_trick.lead_suit if self.trick_manager.current_trick else None,
                                         self.game_state_manager.get_state().trump_suit if self.game_state_manager.get_state() else None)
    
    def _score_round(self) -> None:
        """Score the current round."""
        game_state = self.game_state_manager.get_state()
        if not game_state or not self.game_state_manager.trump_caller_team:
            return
        
        # Get round results
        round_results = self.trick_manager.get_round_results()
        
        # Score the round
        team1_score, team2_score = self.scoring_manager.score_round(
            self.players, self.game_state_manager.trump_caller_team
        )
        
        # Update scores
        new_team1_score = game_state.team1_score + team1_score
        new_team2_score = game_state.team2_score + team2_score
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
        click.echo(f"\nRound complete! Trick counts: {round_results}")
        click.echo(f"Team 1: {team1_score}, Team 2: {team2_score}")
    
    def _show_final_results(self) -> None:
        """Show the final game results."""
        if self.quiet_mode:
            return
        
        winner = self.game_state_manager.get_winner()
        click.echo(f"\n🎉 GAME OVER! {winner} wins! 🎉")
        
        game_state = self.game_state_manager.get_state()
        if game_state:
            click.echo(f"Final Score - Team 1: {game_state.team1_score}, Team 2: {game_state.team2_score}")
        
        # Check if team gets set
        if self.game_state_manager.trump_caller_team is not None:
            if self.scoring_manager.is_team_set(self.players, self.game_state_manager.trump_caller_team):
                click.echo("\n🚨 TEAM SET! The trump calling team lost after calling trump!")
    
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
    
    def get_log_filename(self) -> Optional[str]:
        """Get the log filename if logging is enabled.
        
        Returns
        -------
        Optional[str]
            The log filename or None
        """
        if self.logger:
            return self.logger.get_log_filename()
        return None
    
    def run_tournament(self, num_games: int) -> None:
        """Run a tournament with multiple games.
        
        Parameters
        ----------
        num_games : int
            Number of games to play
        """
        for game_num in range(num_games):
            if not self.quiet_mode:
                click.echo(f"\n=== TOURNAMENT GAME {game_num + 1}/{num_games} ===")
            
            self.start_new_game()
            self.run_full_game()
            
            # Reset for next game
            self.game_state_manager.reset()
            self.trick_manager.reset()
    
    @property
    def game_state(self):
        """Get the current game state."""
        return self.game_state_manager.get_state()
    
    @property
    def trump_caller(self):
        """Get the player who called trump."""
        return self.game_state_manager.trump_caller
    
    @property
    def trump_caller_team(self):
        """Get the team that called trump."""
        return self.game_state_manager.trump_caller_team