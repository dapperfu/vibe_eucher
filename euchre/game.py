"""Main game logic for the euchre card game."""

from typing import List, Optional, Tuple, Dict
import random
from .models import (
    Player, PlayerType, Card, Suit, Rank, GameState, Trick
)
from .game_logger import GameLogger
from .ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI


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
        self.game_state: Optional[GameState] = None
        self.deck: List[Card] = []
        self.current_trick: Optional[Trick] = None
        self.tricks_this_round: List[Trick] = []
        self.top_card: Optional[Card] = None
        self.logger: Optional[GameLogger] = None
        self.trump_caller: Optional[Player] = None  # Track who called trump
        self.trump_caller_team: Optional[int] = None  # Track which team called trump
        self.quiet_mode = quiet_mode
        
        if enable_logging:
            self.logger = GameLogger()
            
        self._initialize_deck()
        
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
        ai_type = ai_type.lower()
        
        if ai_type == "aggressive":
            player = AggressiveAI(name, risk_ratio)
        elif ai_type == "conservative":
            player = ConservativeAI(name, risk_ratio)
        elif ai_type == "opportunistic":
            player = OpportunisticAI(name, risk_ratio)
        else:  # balanced or unknown
            player = BalancedAI(name, risk_ratio)
            
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
        
    def add_model_player(self, name: str, model, device, risk_profile: str = "balanced") -> None:
        """Add a trained AI model player to the game.
        
        Parameters
        ----------
        name : str
            The player's name
        model : torch.nn.Module
            The trained PyTorch model
        device : torch.device
            Device the model is running on
        risk_profile : str
            Risk profile for the model player
        """
        from .ai_model.model_player import ModelPlayer
        player = ModelPlayer(name=name, model=model, device=device, risk_profile=risk_profile)
        self.players.append(player)
        
        # Update game state to track model players
        if not hasattr(self, 'model_players'):
            self.model_players = []
        self.model_players.append(player)
        
    def start_new_game(self) -> None:
        """Start a new game with the current players."""
        if len(self.players) != 4:
            raise ValueError("Euchre requires exactly 4 players")
            
        # Reset all players
        for player in self.players:
            player.clear_hand()
            player.tricks_won = 0
            player.is_dealer = False
            
        # Set dealer (rotate each game)
        dealer_index = random.randint(0, 3)
        self.players[dealer_index].is_dealer = True
        
        # Initialize game state
        self.game_state = GameState(
            players=self.players,
            current_player_index=(dealer_index + 1) % 4,  # Start with player after dealer
            trump_suit=None,
            dealer_index=dealer_index,
            round_number=1,
            team1_score=0,
            team2_score=0
        )
        
        self._deal_cards()
        self._select_trump()
        
        # Log game start
        if self.logger:
            dealer = self.game_state.get_dealer()
            self.logger.log_game_start(self.players, dealer)
            
    def _initialize_deck(self) -> None:
        """Initialize the deck with euchre cards (9-A of each suit)."""
        self.deck.clear()
        for suit in Suit:
            for rank in Rank:
                self.deck.append(Card(rank=rank, suit=suit))
                
    def _deal_cards(self) -> None:
        """Deal 5 cards to each player."""
        # Shuffle deck
        random.shuffle(self.deck)
        
        # Deal 5 cards to each player
        for i, player in enumerate(self.players):
            start_idx = i * 5
            end_idx = start_idx + 5
            player.hand = self.deck[start_idx:end_idx]
            
        # Remove dealt cards from deck
        self.deck = self.deck[20:]
        
        # Set top card (for trump selection)
        if self.deck:
            self.top_card = self.deck[0]
            
    def _select_trump(self) -> None:
        """Handle trump suit selection through proper euchre bidding."""
        if not self.top_card:
            return
            
        # Start with player after dealer (left of dealer)
        current_player_idx = self.game_state.current_player_index
        trump_selected = False
        bidding_log = []
        
        # First round: players can order up the top card
        for round_num in range(4):  # Each player gets one chance
            player = self.players[current_player_idx]
            
            if player.player_type == PlayerType.AI:
                # AI decision
                if hasattr(player, 'should_order_up'):
                    should_order = player.should_order_up(self.top_card)
                else:
                    # Fallback to generic AI
                    ai_player = AIPlayer(player)
                    should_order = ai_player.should_order_up(self.top_card)
                
                if should_order:
                    self.game_state.trump_suit = self.top_card.suit
                    trump_selected = True
                    self.trump_caller = player
                    self.trump_caller_team = (current_player_idx % 2)  # 0 = team 1, 1 = team 2
                    bidding_log.append(f"{player.name} orders up {self.top_card}")
                    break
                else:
                    bidding_log.append(f"{player.name} passes")
            else:
                # Human player - for now, always pass (can be enhanced later)
                bidding_log.append(f"{player.name} passes")
                
            current_player_idx = (current_player_idx + 1) % 4
            
        # If no one ordered up, dealer must choose trump
        if not trump_selected:
            dealer = self.game_state.get_dealer()
            if dealer.player_type == PlayerType.AI:
                # Dealer AI chooses trump based on their hand
                ai_player = AIPlayer(dealer)
                chosen_suit = ai_player.choose_trump_suit(dealer.hand)
                self.game_state.trump_suit = chosen_suit
                self.trump_caller = dealer
                self.trump_caller_team = (self.game_state.dealer_index % 2)  # 0 = team 1, 1 = team 2
                bidding_log.append(f"{dealer.name} (dealer) calls {chosen_suit.value}")
            else:
                # Human dealer - for now, pick based on hand strength
                chosen_suit = self._choose_trump_for_human_dealer(dealer.hand)
                self.game_state.trump_suit = chosen_suit
                self.trump_caller = dealer
                self.trump_caller_team = (self.game_state.dealer_index % 2)  # 0 = team 1, 1 = team 2
                bidding_log.append(f"{dealer.name} (dealer) calls {chosen_suit.value}")
                
        # Print the bidding process to console (unless in quiet mode)
        if not self.quiet_mode:
            print(f"\nTrump Selection:")
            for bid in bidding_log:
                print(f"  {bid}")
            print(f"Trump suit: {self.game_state.trump_suit.value}")
            
        # Mark trump cards
        self._mark_trump_cards()
        
    def is_team_set(self) -> bool:
        """Check if the trump calling team gets set (loses after calling trump).
        
        Returns
        -------
        bool
            True if the trump calling team gets set
        """
        if not self.trump_caller_team or not self.game_state:
            return False
            
        # Count tricks won by each team
        team1_tricks = 0
        team2_tricks = 0
        
        for i, player in enumerate(self.players):
            if i % 2 == 0:  # Team 1 (players 0 and 2)
                team1_tricks += player.tricks_won
            else:  # Team 2 (players 1 and 3)
                team2_tricks += player.tricks_won
                
        # Check if trump calling team won less than 3 tricks
        if self.trump_caller_team == 0:  # Team 1 called trump
            return team1_tricks < 3
        else:  # Team 2 called trump
            return team2_tricks < 3
        
    def _mark_trump_cards(self) -> None:
        """Mark all trump cards in the game."""
        if not self.game_state.trump_suit:
            return
            
        trump_suit = self.game_state.trump_suit
        
        for player in self.players:
            for card in player.hand:
                if self._is_trump_card(card, trump_suit):
                    card.is_trump = True
                    
        if self.top_card and self._is_trump_card(self.top_card, trump_suit):
            self.top_card.is_trump = True
            
    def _deal_new_round(self) -> None:
        """Deal new cards for a new round."""
        # Reinitialize deck
        self._initialize_deck()
        
        # Deal new cards
        self._deal_cards()
        
        # Select new trump
        self._select_trump()
        
        # Reset trick counts
        for player in self.players:
            player.tricks_won = 0
        
    def play_round(self) -> None:
        """Play a complete round (5 tricks)."""
        # If this is not the first round, deal new cards
        if self.game_state and self.game_state.round_number > 1:
            self._deal_new_round()
            
        # Log round start
        if self.logger and self.game_state:
            hands = {player.name: player.hand.copy() for player in self.players}
            trump_suit = self.game_state.trump_suit.name.title() if self.game_state.trump_suit else "None"
            self.logger.log_round_start(self.game_state.round_number, trump_suit, hands)
            
        self.tricks_this_round.clear()
        
        for trick_num in range(5):
            # Create new trick
            self.current_trick = Trick()
            
            # Play the trick (each player plays one card)
            for player_idx in range(4):
                current_player = self.game_state.get_current_player()
                
                if current_player.player_type == PlayerType.AI:
                    # Use the AI profile's method directly
                    if hasattr(current_player, 'choose_card_to_play'):
                        card = current_player.choose_card_to_play(
                            self.current_trick.lead_suit,
                            self.game_state.trump_suit
                        )
                    else:
                        # Fallback to generic AI if not a profile
                        ai_player = AIPlayer(current_player)
                        card = ai_player.choose_card_to_play(
                            self.current_trick.lead_suit,
                            self.game_state.trump_suit
                        )
                else:
                    # Human player - for now, play first card
                    card = current_player.hand[0]
                    
                # Remove card from hand
                current_player.remove_card(card)
                
                # Add to trick
                if not self.current_trick.lead_suit:
                    self.current_trick.lead_suit = card.suit
                self.current_trick.cards_played.append((current_player, card))
                
                # Move to next player
                self.game_state.next_player()
            
            # Add completed trick to round
            self.tricks_this_round.append(self.current_trick)
            
            # Award trick to winner
            winner = self._determine_trick_winner()
            winner.tricks_won += 1
            
            # Log trick completion
            if self.logger:
                winning_card = self._get_winning_card_from_trick(self.current_trick)
                self.logger.log_trick(trick_num + 1, self.current_trick, winner, winning_card)
            
            # Set next trick leader
            self.game_state.current_player_index = self.players.index(winner)
        
        # Get round results before scoring
        round_results = [player.tricks_won for player in self.players]
        
        # Log round end (before scoring resets trick counts)
        if self.logger:
            final_scores = {player.name: player.tricks_won for player in self.players}
            team_scores = {
                "Team 1": self.game_state.team1_score,
                "Team 2": self.game_state.team2_score
            }
            self.logger.log_round_end(self.game_state.round_number, final_scores, team_scores)
        
        # Score the round
        self._score_round()
        
        # Update model players' game context
        if hasattr(self, 'model_players'):
            # Check if team was set
            was_set = any(player.tricks_won == 0 for player in self.players)
            self._update_model_players_context(self.game_state.round_number, was_set)
        
        # Increment round number
        if self.game_state:
            self.game_state.round_number += 1
        
        # Return the round results for testing/debugging
        return round_results
        
    def get_round_results(self) -> List[int]:
        """Get the trick counts for each player before scoring.
        
        Returns
        -------
        List[int]
            List of trick counts for each player
        """
        return [player.tricks_won for player in self.players]
        
    def _get_winning_card_from_trick(self, trick: Trick) -> Card:
        """Get the winning card from a completed trick.
        
        Parameters
        ----------
        trick : Trick
            The completed trick
            
        Returns
        -------
        Card
            The card that won the trick
        """
        if not trick.cards_played:
            raise ValueError("Trick has no cards played")
            
        winner, winning_card = trick.get_winner(self.game_state.trump_suit if self.game_state else None)
        return winning_card
        
    def _determine_trick_winner(self) -> Player:
        """Determine who won the current trick."""
        if not self.current_trick or not self.current_trick.cards_played:
            raise ValueError("No cards played in trick")
            
        winner = self.current_trick.cards_played[0][0]
        winning_card = self.current_trick.cards_played[0][1]
        
        for player, card in self.current_trick.cards_played[1:]:
            if self._card_beats(card, winning_card):
                winner = player
                winning_card = card
                
        return winner
        
    def _card_beats(self, card1: Card, card2: Card) -> bool:
        """Determine if card1 beats card2."""
        # Get current trump suit
        trump_suit = self.game_state.trump_suit if self.game_state else None
        
        # Mark trump cards (including left bower)
        is_trump1 = self._is_trump_card(card1, trump_suit)
        is_trump2 = self._is_trump_card(card2, trump_suit)
        
        # Trump cards beat non-trump cards
        if is_trump1 and not is_trump2:
            return True
        if not is_trump1 and is_trump2:
            return False
            
        # If both are trump or both are non-trump, compare ranks
        if is_trump1 == is_trump2:
            return card1.rank.value > card2.rank.value
            
        # If one follows lead suit and other doesn't, lead suit wins
        if self.current_trick and self.current_trick.lead_suit:
            if card1.suit == self.current_trick.lead_suit and card2.suit != self.current_trick.lead_suit:
                return True
            if card2.suit == self.current_trick.lead_suit and card1.suit != self.current_trick.lead_suit:
                return False
                
        # Same suit, compare ranks
        if card1.suit == card2.suit:
            return card1.rank.value > card2.rank.value
            
        # Different suits, neither trump, neither follows lead - first card wins
        return False
    
    def _is_trump_card(self, card: Card, trump_suit: Optional[Suit]) -> bool:
        """Check if a card is a trump card (including left bower)."""
        if not trump_suit:
            return False
            
        # Right bower (jack of trump suit)
        if card.rank == Rank.JACK and card.suit == trump_suit:
            return True
            
        # Left bower (jack of same color as trump)
        if card.rank == Rank.JACK:
            if (trump_suit == Suit.HEARTS and card.suit == Suit.DIAMONDS) or \
               (trump_suit == Suit.DIAMONDS and card.suit == Suit.HEARTS) or \
               (trump_suit == Suit.CLUBS and card.suit == Suit.SPADES) or \
               (trump_suit == Suit.SPADES and card.suit == Suit.CLUBS):
                return True
                
        # Regular trump suit cards
        return card.suit == trump_suit
        
    def _score_round(self) -> None:
        """Score the current round."""
        team1_tricks = (self.players[0].tricks_won + self.players[2].tricks_won)
        team2_tricks = (self.players[1].tricks_won + self.players[3].tricks_won)
        
        if team1_tricks >= 3:
            self.game_state.team1_score += 1
        if team2_tricks >= 3:
            self.game_state.team2_score += 1
            
        # Check if game just ended and log it
        self._check_and_log_game_end()
            
        # Reset trick counts for next round
        for player in self.players:
            player.tricks_won = 0
            
    def get_player_hand(self, player_name: str) -> List[Card]:
        """Get the hand of a specific player.
        
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
                return player.hand.copy()
        return []
        
    def play_ai_turn(self, player: Player) -> Card:
        """Play a turn for an AI player.
        
        Parameters
        ----------
        player : Player
            The AI player
            
        Returns
        -------
        Card
            The card the AI chose to play
        """
        # Simple AI: just play the first card in hand
        # In a real implementation, this would be much more sophisticated
        if not player.hand:
            raise ValueError("AI player has no cards to play")
            
        card = player.hand[0]
        player.remove_card(card)
        return card
        
    def is_game_over(self) -> bool:
        """Check if the game is over.
        
        Returns
        -------
        bool
            True if the game is over
        """
        if not self.game_state:
            return True
            
        return (self.game_state.team1_score >= 10 or 
                self.game_state.team2_score >= 10)
        
    def _check_and_log_game_end(self) -> None:
        """Check if game just ended and log it if so."""
        if not self.game_state or not self.logger:
            return
            
        # Check if game just ended
        if (self.game_state.team1_score >= 10 or 
            self.game_state.team2_score >= 10):
            
            winner = self.get_winner()
            final_team_scores = {
                "Team 1": self.game_state.team1_score,
                "Team 2": self.game_state.team2_score
            }
            self.logger.log_game_end(winner or "Unknown", final_team_scores)
            self.logger.write_summary_table()
        
    def get_winner(self) -> Optional[str]:
        """Get the winning team.
        
        Returns
        -------
        Optional[str]
            The name of the winning team, or None if game not over
        """
        if not self.is_game_over():
            return None
            
        if self.game_state and self.game_state.team1_score >= 10:
            return "Team 1"
        elif self.game_state and self.game_state.team2_score >= 10:
            return "Team 2"
        return None
        
    def get_log_filename(self) -> Optional[str]:
        """Get the filename of the game log.
        
        Returns
        -------
        Optional[str]
            The log filename, or None if logging is disabled
        """
        return self.logger.get_log_filename() if self.logger else None

    def _update_model_players_context(self, round_num: int, was_set: bool = False):
        """Update game context for all model players."""
        if hasattr(self, 'model_players'):
            for player in self.model_players:
                if hasattr(player, 'update_game_context'):
                    # Determine team scores
                    if player.name in ['North', 'South']:
                        team_score = self.game_state.team1_score
                        opponent_score = self.game_state.team2_score
                    else:
                        team_score = self.game_state.team2_score
                        opponent_score = self.game_state.team1_score
                    
                    player.update_game_context(
                        team_score=team_score,
                        opponent_score=opponent_score,
                        current_round=round_num,
                        was_set=was_set
                    )


class AIPlayer:
    """AI player logic for euchre."""
    
    def __init__(self, player: Player) -> None:
        """Initialize AI player.
        
        Parameters
        ----------
        player : Player
            The player this AI controls
        """
        self.player = player
        
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play.
        
        Parameters
        ----------
        lead_suit : Optional[Suit]
            The suit that was led (if any)
        trump_suit : Optional[Suit]
            The current trump suit
            
        Returns
        -------
        Card
            The card to play
        """
        if not self.player.hand:
            raise ValueError("AI player has no cards to play")
            
        # Must follow suit if possible
        if lead_suit and self.player.has_suit(lead_suit):
            cards_of_suit = self.player.get_cards_of_suit(lead_suit)
            # Play highest card of lead suit
            return max(cards_of_suit, key=lambda c: c.rank.value)
        else:
            # Can play any card - choose strategically
            # If we're leading, play highest trump or highest card
            if not lead_suit:
                trump_cards = [c for c in self.player.hand if c.is_trump]
                if trump_cards:
                    return max(trump_cards, key=lambda c: c.rank.value)
                else:
                    # Lead with highest non-trump
                    return max(self.player.hand, key=lambda c: c.rank.value)
            else:
                # Not leading - play lowest non-trump if possible
                non_trump_cards = [c for c in self.player.hand if not c.is_trump]
                if non_trump_cards:
                    return min(non_trump_cards, key=lambda c: c.rank.value)
                else:
                    # Only trump cards left - play lowest
                    return min(self.player.hand, key=lambda c: c.rank.value)
            
    def should_order_up(self, top_card: Card) -> bool:
        """Decide whether to order up the top card.
        
        Parameters
        ----------
        top_card : Card
            The top card that could be ordered up
            
        Returns
        -------
        bool
            True if the AI should order up the card
        """
        # Count cards of the potential trump suit
        cards_of_suit = self.player.get_cards_of_suit(top_card.suit)
        
        # Also count left bower (jack of same color)
        left_bower_suit = self._get_left_bower_suit(top_card.suit)
        left_bower_cards = self.player.get_cards_of_suit(left_bower_suit)
        
        # Count high cards (J, Q, K, A) of the potential trump suit
        high_cards = [c for c in cards_of_suit if c.rank.value >= 11]
        left_bower_high = [c for c in left_bower_cards if c.rank == Rank.JACK]
        
        total_trump_potential = len(cards_of_suit) + len(left_bower_cards)
        high_card_bonus = len(high_cards) + len(left_bower_high)
        
        # Order up if we have 2+ potential trump cards, or 1+ with multiple high cards
        return total_trump_potential >= 2 or (total_trump_potential >= 1 and high_card_bonus >= 2)
        
    def _get_left_bower_suit(self, trump_suit: Suit) -> Suit:
        """Get the left bower suit for a given trump suit."""
        if trump_suit == Suit.HEARTS:
            return Suit.DIAMONDS
        elif trump_suit == Suit.DIAMONDS:
            return Suit.HEARTS
        elif trump_suit == Suit.CLUBS:
            return Suit.SPADES
        else:  # SPADES
            return Suit.CLUBS 

    def choose_trump_suit(self, hand: List[Card]) -> Suit:
        """Choose the trump suit for the dealer.
        
        Parameters
        ----------
        hand : List[Card]
            The dealer's hand
            
        Returns
        -------
        Suit
            The chosen trump suit
        """
        # Count cards by suit
        suit_counts = {}
        for suit in Suit:
            suit_counts[suit] = len([card for card in hand if card.suit == suit])
            
        # Choose suit with most cards, or highest cards if tied
        best_suit = max(suit_counts.keys(), key=lambda s: (suit_counts[s], 
                                                          max([card.rank.value for card in hand if card.suit == s] or [0])))
        return best_suit 