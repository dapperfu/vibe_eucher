"""Main game logic and state management for Euchre."""

import random
import uuid
from typing import List, Optional, Tuple, Union

import numpy as np

from eucher.cards import Card, Deck, Suit
from eucher.players import Player
from eucher.players.computer.ai import AIDecisionMaker
from eucher.players.computer.ml.ml_config import MLConfig
from eucher.players.computer.ml.ml_features import GameStateEncoder
from eucher.players.computer.ml.ml_model import EuchreMLModel
from eucher.players.computer.ml.player import MLPlayer
from eucher.players.computer import AIPlayer, HeuristicPlayer, RandomPlayer
from eucher.players.profiles import HumanProfile, MLBasedProfile, PlayerProfile
from eucher.rules import RulesEngine
from eucher.game_stats import GameStatistics
from eucher.training.profiling import timed_operation
from eucher.trade_in import TradeInHandler
from eucher.trump import TrumpSelector


class Game:
    """Manages the Euchre game state and flow."""

    def __init__(
        self,
        player_config: List[Tuple[str, str]],
        trump_selection_risk: Optional[float] = None,
        gameplay_risk: Optional[float] = None,
        seed: Optional[Union[int, str]] = None,
        trade_in_enabled: bool = False,
    ) -> None:
        """
        Initialize a new game.

        Parameters
        ----------
        player_config : List[Tuple[str, str]]
            List of (name, profile_type) tuples for each player.
            profile_type can be: "human", "heuristic", "ai", "random", "ml_sklearn", "ml_pytorch"
        trump_selection_risk : Optional[float]
            Risk factor for trump selection decisions (0.0-1.0) for ML players.
            If None, uses MLConfig default.
        gameplay_risk : Optional[float]
            Risk factor for gameplay decisions (0.0-1.0) for ML players.
            If None, uses MLConfig default.
        seed : Optional[Union[int, str]]
            Random seed for reproducibility. Can be an integer or a UUID string.
            If a UUID string is provided, it will be converted to a numeric seed.
            If None, a new UUID will be generated and used as the seed.
        trade_in_enabled : bool
            If True, enables the Midwest 9-10 trade-in variant.
            Default is False.
        """
        if len(player_config) != 4:
            raise ValueError("Euchre requires exactly 4 players")

        # Handle seed/UUID
        if seed is None:
            # Generate a new UUID
            self.game_uuid = str(uuid.uuid4())
            numeric_seed = self._uuid_to_seed(self.game_uuid)
        elif isinstance(seed, str):
            # UUID string provided
            self.game_uuid = seed
            numeric_seed = self._uuid_to_seed(seed)
        else:
            # Integer seed provided - convert to UUID for consistency
            # Use the integer as the seed directly, but generate a deterministic UUID
            numeric_seed = seed
            # Create a deterministic UUID from the seed for file naming
            # We'll use a namespace UUID approach
            namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace
            self.game_uuid = str(uuid.uuid5(namespace, str(seed)))

        # Set random seeds
        random.seed(numeric_seed)
        np.random.seed(numeric_seed)

        self.players: List[Player] = []
        self.dealer_id: int = 0
        self.scores: List[int] = [0, 0]  # Team 0 and Team 1 scores
        self.trump_suit: Optional[Suit] = None
        self.turned_card: Optional[Card] = None
        self.rules = RulesEngine()
        self.trump_selector: Optional[TrumpSelector] = None
        self.ai_decision_maker = AIDecisionMaker()
        self.tui = None
        
        # Game statistics tracking
        self.stats = GameStatistics()

        # ML model setup (lazy initialization)
        self._ml_model: Optional[EuchreMLModel] = None
        self._ml_encoder: Optional[GameStateEncoder] = None

        # Game state tracking for ML profiles
        self._current_trick_number: int = 0
        self._current_tricks_won: List[int] = [0, 0]
        self._renege_occurred: bool = False
        self._renege_player_id: Optional[int] = None
        self._renege_team: Optional[int] = None

        # Store risk factors for ML players
        self.trump_selection_risk = trump_selection_risk
        self.gameplay_risk = gameplay_risk

        # Store trade-in configuration
        self.trade_in_enabled = trade_in_enabled

        # Create players with profiles
        for i, (name, profile_type) in enumerate(player_config):
            profile = self._create_profile(profile_type, i)
            player = Player(name, i, profile)
            self.players.append(player)

    def _create_profile(self, profile_type: str, player_id: int) -> PlayerProfile:
        """
        Create a player profile based on type.

        Parameters
        ----------
        profile_type : str
            Type of profile: "human", "heuristic", "ai", "random", "ml_sklearn"
        player_id : int
            ID of the player (for ML profile game state access).

        Returns
        -------
        PlayerProfile
            The created profile.
        """
        if profile_type == "human":
            return HumanProfile()
        elif profile_type == "heuristic":
            return HeuristicPlayer()
        elif profile_type == "ai":
            return AIPlayer(self.ai_decision_maker)
        elif profile_type == "random":
            return RandomPlayer()
        elif profile_type == "ml_sklearn":
            return self._create_ml_player_profile(player_id, backend="supervised")
        elif profile_type == "ml_rl":
            return self._create_ml_player_profile(player_id, backend="rl")
        elif profile_type == "ml_gan":
            return self._create_ml_player_profile(player_id, backend="gan")
        elif profile_type == "ml_pytorch":
            return self._create_ml_profile(player_id)
        elif profile_type == "pytorch_ai" or profile_type == "pytorch_strategic":
            from eucher.players.computer.ml.pytorch.pytorch_player import PyTorchStrategicPlayer

            return PyTorchStrategicPlayer()
        elif profile_type == "euchre_zero":
            from eucher.players.computer.euchre_zero.player import EuchreZeroPlayer
            from eucher.players.computer.euchre_zero.config import EuchreZeroConfig

            config = EuchreZeroConfig()
            player = EuchreZeroPlayer(config=config, game=self)
            return player
        elif profile_type.startswith("perceiver_muzero"):
            from eucher.players.computer.perceiver_muzero.player import (
                EuchrePerceiverMuZeroPlayer,
            )
            from eucher.players.computer.perceiver_muzero.config import (
                PerceiverMuZeroConfig,
            )

            config = PerceiverMuZeroConfig()
            
            # Parse simulation count from profile type (e.g., "perceiver_muzero_16" -> 16)
            num_simulations = None
            if "_" in profile_type:
                parts = profile_type.split("_")
                if len(parts) >= 3:
                    try:
                        num_simulations = int(parts[-1])
                    except ValueError:
                        pass
            
            # If no simulation count specified or 1, use fast mode (equivalent to 1 simulation)
            if num_simulations is None or num_simulations == 1:
                player = EuchrePerceiverMuZeroPlayer(config=config, game=self, fast_mode=True)
            else:
                # Use specified number of simulations (not fast mode)
                player = EuchrePerceiverMuZeroPlayer(
                    config=config, game=self, num_simulations=num_simulations, fast_mode=False
                )
            return player
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")

    def _create_ml_profile(self, player_id: int) -> MLBasedProfile:
        """
        Create an ML-based profile.

        Parameters
        ----------
        player_id : int
            ID of the player.

        Returns
        -------
        MLBasedProfile
            The ML profile.
        """
        # Lazy initialization of ML components
        if self._ml_model is None:
            config = MLConfig()
            self._ml_model = EuchreMLModel(config)
            self._ml_encoder = GameStateEncoder()

            # Try to load existing weights
            trump_path = config.get_model_path("order_up.pth")
            card_path = config.get_model_path("play_card.pth")
            discard_path = config.get_model_path("discard.pth")

            if trump_path.exists() or card_path.exists() or discard_path.exists():
                self._ml_model.load_weights(
                    trump_path=str(trump_path) if trump_path.exists() else None,
                    card_play_path=str(card_path) if card_path.exists() else None,
                    discard_path=str(discard_path) if discard_path.exists() else None,
                )

        # Create game state provider function
        def get_game_state() -> tuple[int, int, int]:
            """Get current game state for ML profile."""
            trick_num = getattr(self, "_current_trick_number", 0)
            tricks_won = getattr(self, "_current_tricks_won", [0, 0])
            return (trick_num, tricks_won[0], tricks_won[1])

        return MLBasedProfile(
            self._ml_model,
            self._ml_encoder,
            get_game_state,
            trump_selection_risk=self.trump_selection_risk,
            gameplay_risk=self.gameplay_risk,
        )

    def _create_ml_player_profile(self, player_id: int, backend: str = "supervised") -> MLPlayer:
        """
        Create an ML player profile.

        Parameters
        ----------
        player_id : int
            ID of the player.
        backend : str
            ML backend: "supervised", "gan", or "rl"

        Returns
        -------
        MLPlayer
            The ML player profile.
        """
        # Create game state provider function
        def get_game_state() -> tuple[int, int, int]:
            """Get current game state for ML player."""
            trick_num = getattr(self, "_current_trick_number", 0)
            tricks_won = getattr(self, "_current_tricks_won", [0, 0])
            return (trick_num, tricks_won[0], tricks_won[1])

        return MLPlayer(
            backend=backend,
            model_type="random_forest",
            game_state_provider=get_game_state,
            trump_selection_risk=self.trump_selection_risk,
            gameplay_risk=self.gameplay_risk,
        )

    def set_tui(self, tui) -> None:
        """
        Set the TUI object for human players.

        Parameters
        ----------
        tui
            The TUI object.
        """
        # Set players list in TUI for table display
        if hasattr(tui, "set_players"):
            tui.set_players(self.players, self.dealer_id)
        self.tui = tui

        for player in self.players:
            if isinstance(player.profile, HumanProfile):
                player.profile.set_tui(tui)

    @timed_operation("Game.play_hand")
    def play_hand(self) -> bool:
        """
        Play a single hand.

        Returns
        -------
        bool
            True if game should continue, False if game is over.
        """
        # Reset trick winner for new hand (first trick always starts with player left of dealer)
        if hasattr(self, "_last_trick_winner"):
            delattr(self, "_last_trick_winner")

        # Reset game state tracking for ML profiles
        self._current_trick_number = 0
        self._current_tricks_won = [0, 0]
        # Reset renege state
        self._renege_occurred = False
        self._renege_player_id = None
        self._renege_team = None
        # Reset going alone state
        self.going_alone = False
        self.going_alone_player_id: Optional[int] = None

        # Start new hand log
        if self.tui is not None and hasattr(self.tui, "start_new_hand"):
            self.tui.start_new_hand()

        # Update TUI with initial scores
        if self.tui is not None and hasattr(self.tui, "display_scores"):
            scores = self.get_scores()
            self.tui.display_scores(scores[0], scores[1])

        # Deal cards
        deck = Deck()
        deck.shuffle()

        # Deal 5 cards to each player
        for player in self.players:
            cards = deck.deal(5)
            player.receive_hand(cards)

        # Log initial hands
        if self.tui is not None and hasattr(self.tui, "log_initial_hands"):
            self.tui.log_initial_hands(self.players)

        # Turn up one card
        self.turned_card = deck.draw_one()

        # Store kitty (remaining 3 cards in deck)
        self.kitty = deck.cards.copy()

        # Log turned card
        if self.tui is not None and hasattr(self.tui, "log_turned_card"):
            self.tui.log_turned_card(self.turned_card)

        # Display turned card prominently after deal
        if self.tui is not None and hasattr(self.tui, "display_turned_card"):
            self.tui.display_turned_card(self.turned_card)

        # Process trade-in if enabled (before trump selection)
        if self.trade_in_enabled:
            trade_in_handler = TradeInHandler(self.players, self.kitty, self.dealer_id, self.tui)
            trade_in_player = trade_in_handler.process_trade_in()
            # Update kitty after trade-in (may be empty if trade-in occurred)
            self.kitty = trade_in_handler.kitty.copy()

        # Select trump
        self.trump_selector = TrumpSelector(self.players, self.tui)
        self.trump_suit = self.trump_selector.select_trump(self.turned_card, self.dealer_id)
        
        # Record trump selection statistics
        trump_maker_id = None
        if self.trump_selector.trump_maker_name:
            for player in self.players:
                if player.name == self.trump_selector.trump_maker_name:
                    trump_maker_id = player.player_id
                    break
        self.stats.record_trump_selection(self.trump_suit, trump_maker_id)

        # Log full trump-decision history into TUI for easier post-hand inspection
        if self.tui is not None and hasattr(self.tui, "log_trump_decision_history"):
            history = self.trump_selector.get_decision_history()
            self.tui.log_trump_decision_history(
                history.get("order_up", []), history.get("call_trump", [])
            )

        # If all passed, redeal
        if self.trump_suit is None:
            return True  # Continue game, redeal

        # Check if going alone
        if self.trump_selector.going_alone:
            self.going_alone = True
            self.going_alone_player_id = self.trump_selector.going_alone_player_id

            # Log partner sitting out
            if self.going_alone_player_id is not None:
                lone_player = self.players[self.going_alone_player_id]
                partner_id = (self.going_alone_player_id + 2) % 4  # Partner is 2 positions away
                partner = self.players[partner_id]
                if self.tui is not None and hasattr(self.tui, "log_message"):
                    self.tui.log_message(f"{partner.name} sits out (partner going alone)")
                if self.tui is not None and hasattr(self.tui, "current_hand_log"):
                    self.tui.current_hand_log.append(f"{partner.name} sits out (partner going alone)")

        # Update TUI with dealer info
        if self.tui is not None and hasattr(self.tui, "dealer_id"):
            self.tui.dealer_id = self.dealer_id

        # Display trump decision summary before gameplay starts
        if self.tui is not None and hasattr(self.tui, "display_trump_decision_summary"):
            history = self.trump_selector.get_decision_history()
            maker_name = getattr(self.trump_selector, "trump_maker_name", None)
            self.tui.display_trump_decision_summary(
                history.get("order_up", []),
                history.get("call_trump", []),
                self.trump_suit,
                maker_name
            )

        # Play 5 tricks
        tricks_won = [0, 0]  # Team 0 and Team 1
        for trick_num in range(5):
            self._current_trick_number = trick_num
            # Update TUI with trick number
            if self.tui is not None and hasattr(self.tui, "update_trick_number"):
                self.tui.update_trick_number(trick_num + 1)
            # Note: Scores are displayed after hand completes, not during each trick

            winner_id = self._play_trick()
            winner = self.players[winner_id]
            tricks_won[winner.team] += 1
            self._current_tricks_won = tricks_won.copy()
            
            # Record statistics
            self.stats.record_trick_winner(winner_id, winner.team)

            # Display trick winner and wait for spacebar
            if self.tui is not None and hasattr(self.tui, "display_trick_winner"):
                self.tui.display_trick_winner(winner.name)

            # Log trick
            if self.tui is not None and hasattr(self.tui, "log_trick"):
                if hasattr(self, "_last_trick_cards") and hasattr(self, "_last_trick_player_ids"):
                    self.tui.log_trick(
                        trick_num + 1,
                        self._last_trick_cards,
                        self._last_trick_player_ids,
                        winner_id,
                    )

        # Score the hand (get scores before to calculate points awarded)
        scores_before = self.get_scores()
        self._score_hand(tricks_won)
        scores_after = self.get_scores()
        points_awarded = [scores_after[0] - scores_before[0], scores_after[1] - scores_before[1]]
        
        # Record hand completion statistics
        making_team = None
        if self.trump_selector and self.trump_selector.trump_maker_name:
            for player in self.players:
                if player.name == self.trump_selector.trump_maker_name:
                    making_team = player.team
                    break
        self.stats.record_hand_complete(tricks_won, points_awarded, self.players, making_team)
        
        # Record going alone statistics
        if self.going_alone and self.going_alone_player_id is not None:
            # Successful if making team won 5 tricks
            successful = tricks_won[making_team] == 5 if making_team is not None else False
            self.stats.record_going_alone(self.going_alone_player_id, successful)

        # Log hand score
        if self.tui is not None and hasattr(self.tui, "log_hand_score"):
            scores = self.get_scores()
            self.tui.log_hand_score(tricks_won, scores)

        # Display hand log
        if self.tui is not None and hasattr(self.tui, "display_hand_log"):
            self.tui.display_hand_log()

        # Rotate dealer
        self.dealer_id = (self.dealer_id + 1) % 4

        # Check for game end
        return not self._is_game_over()

    @timed_operation("Game._play_trick")
    def _play_trick(self) -> int:
        """
        Play a single trick.

        Returns
        -------
        int
            ID of the winning player.
        """
        # Determine who leads
        if not hasattr(self, "_last_trick_winner"):
            # First trick: player left of dealer leads (or lone player if going alone)
            if self.going_alone and self.going_alone_player_id is not None:
                # When going alone, player to the left of lone player leads
                leader_id = (self.going_alone_player_id + 1) % 4
            else:
                leader_id = (self.dealer_id + 1) % 4
        else:
            # Subsequent tricks: last trick winner leads
            leader_id = self._last_trick_winner

        played_cards: List[Card] = []
        player_ids: List[int] = []
        led_suit: Optional[Suit] = None

        # Reset trick state in TUI
        if self.tui is not None and hasattr(self.tui, "update_trick_state"):
            self.tui.update_trick_state([], [])

        # Determine which players participate
        if self.going_alone and self.going_alone_player_id is not None:
            # Going alone: skip partner (partner is 2 positions away)
            partner_id = (self.going_alone_player_id + 2) % 4
            participating_players = [i for i in range(4) if i != partner_id]
            num_players = 3
        else:
            # Normal play: all 4 players
            participating_players = list(range(4))
            num_players = 4

        # Each participating player plays a card
        player_idx_in_trick = 0
        for i in range(4):
            player_idx = (leader_id + i) % 4

            # Skip partner if going alone
            if self.going_alone and self.going_alone_player_id is not None:
                partner_id = (self.going_alone_player_id + 2) % 4
                if player_idx == partner_id:
                    continue  # Skip partner's turn

            player = self.players[player_idx]

            # Get card to play
            card = player.play_card(led_suit, self.trump_suit, played_cards, player_ids)

            # Check for renege (invalid play) - don't raise error, mark it and apply penalties
            is_renege = not self.rules.can_play_card(card, player.hand, led_suit, self.trump_suit)
            if is_renege:
                # Mark renege occurred - this will result in heavy penalties
                self._renege_occurred = True
                self._renege_player_id = player_idx
                self._renege_team = player.team
                # If calling team reneged, they automatically lose the hand
                if hasattr(self, "trump_selector") and self.trump_selector and self.trump_selector.trump_maker_name:
                    # Find calling team from trump maker name
                    calling_team = None
                    for p in self.players:
                        if p.name == self.trump_selector.trump_maker_name:
                            calling_team = p.team
                            break
                    if calling_team is not None and player.team == calling_team:
                        # Calling team reneged - they lose the hand (set to 0 tricks)
                        self._current_tricks_won[calling_team] = 0
                        self._current_tricks_won[1 - calling_team] = 5  # Opponents get all tricks

            # Play the card (even if it's a renege - let the model learn from the penalty)
            player.remove_card(card)
            played_cards.append(card)
            player_ids.append(player_idx)

            # Update TUI with current trick state
            if self.tui is not None and hasattr(self.tui, "update_trick_state"):
                self.tui.update_trick_state(played_cards, player_ids)

            # Set led suit if first card
            if player_idx_in_trick == 0:
                led_suit = self._get_card_suit_for_led(card)

            player_idx_in_trick += 1

            # Stop if we've had all participating players play
            if len(played_cards) >= num_players:
                break

        # Store trick info for logging
        self._last_trick_cards = played_cards.copy()
        self._last_trick_player_ids = player_ids.copy()

        # Determine winner
        winner_id = self.rules.determine_trick_winner(
            played_cards, player_ids, led_suit, self.trump_suit
        )
        self._last_trick_winner = winner_id
        return winner_id

    def _get_card_suit_for_led(self, card: Card) -> Suit:
        """
        Get the suit that a card represents when leading.

        Parameters
        ----------
        card : Card
            The card that was led.

        Returns
        -------
        Suit
            The suit for following purposes.
        """
        if self.trump_suit is None:
            return card.suit

        # Right Bower leads as trump
        if card.rank.value == 11 and card.suit == self.trump_suit:
            return self.trump_suit

        # Left Bower leads as trump
        if card.rank.value == 11:
            trump_card = Card(self.trump_suit, card.rank)
            if card.is_same_color(trump_card):
                return self.trump_suit

        # Regular cards lead as their suit
        return card.suit

    def _score_hand(self, tricks_won: List[int]) -> None:
        """
        Score a completed hand.

        Parameters
        ----------
        tricks_won : List[int]
            Number of tricks won by each team [team0, team1].
        """
        # Determine which team made trump
        # Track which player/team called trump
        if self.trump_selector is None:
            return

        # Find which player called trump
        making_team = 0  # Default
        if self.trump_selector.trump_maker_name:
            for player in self.players:
                if player.name == self.trump_selector.trump_maker_name:
                    making_team = player.team
                    break
        defending_team = 1 - making_team

        making_tricks = tricks_won[making_team]

        # Check if going alone
        if self.going_alone and self.going_alone_player_id is not None:
            # Going alone scoring (Midwestern rules):
            # - Sweep (5 tricks): 4 points
            # - Win (3-4 tricks): 1 point (same as normal make)
            # - Euchred (<3 tricks): 2 points to opponents
            if making_tricks >= 5:
                # Sweep: 4 points
                self.scores[making_team] += 4
            elif making_tricks >= 3:
                # Win: 1 point (same as normal make)
                self.scores[making_team] += 1
            else:
                # Euchred: defending team gets 2 points
                self.scores[defending_team] += 2
        else:
            # Normal play scoring
            if making_tricks >= 5:
                # March: 2 points
                self.scores[making_team] += 2
            elif making_tricks >= 3:
                # 3-4 tricks: 1 point
                self.scores[making_team] += 1
            else:
                # Euchred: defending team gets 2 points
                self.scores[defending_team] += 2

    def _is_game_over(self) -> bool:
        """
        Check if the game is over (a team has 10+ points).

        Returns
        -------
        bool
            True if game is over, False otherwise.
        """
        return self.scores[0] >= 10 or self.scores[1] >= 10

    def get_winner(self) -> Optional[int]:
        """
        Get the winning team ID.

        Returns
        -------
        Optional[int]
            Team ID (0 or 1) if game is over, None otherwise.
        """
        if not self._is_game_over():
            return None
        if self.scores[0] >= 10:
            return 0
        return 1

    def get_scores(self) -> Tuple[int, int]:
        """
        Get current team scores.

        Returns
        -------
        Tuple[int, int]
            (Team 0 score, Team 1 score).
        """
        return (self.scores[0], self.scores[1])

    @staticmethod
    def _uuid_to_seed(uuid_str: str) -> int:
        """
        Convert a UUID string to a numeric seed.

        Parameters
        ----------
        uuid_str : str
            UUID string to convert.

        Returns
        -------
        int
            Numeric seed derived from the UUID.
        """
        try:
            uuid_obj = uuid.UUID(uuid_str)
            # Use the integer representation of the UUID
            # This gives us a large integer that we can use as a seed
            return uuid_obj.int % (2**31)  # Keep within int32 range for compatibility
        except ValueError:
            # If it's not a valid UUID, try to hash it
            return hash(uuid_str) % (2**31)

